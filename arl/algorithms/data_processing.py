"""Data processing for RL training: advantages, trajectory->datum conversion,
and batch assembly.

Vendored from tinker_cookbook.rl.data_processing (with the one needed helper
from supervised.common inlined), imports repointed to the local backend.
"""

from __future__ import annotations

import logging

import torch

from arl.models import types as tinker
from arl.models.types import TensorData
from arl.utils.misc_utils import all_same, safezip
from arl.algorithms.rl_types import Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


def create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """Right-shifted inputs and left-shifted targets for next-token prediction."""
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.EncodedTextChunk):
        raise DataValidationError("The last chunk must be a text chunk.")

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise DataValidationError("need at least 2 tokens for input/target split")

    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


def compute_advantages(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Per-group advantages, mean-centered within each group (Dr-GRPO).

    Dr-GRPO (improvement #1): ``A = r - mean``, WITHOUT the ``/std`` that original
    GRPO applies. Dropping ``/std`` removes the difficulty bias -- otherwise easy
    groups (small std) get their advantages blown up and hard groups (large std)
    get theirs shrunk, over-weighting easy questions. Uniform-reward groups give
    advantage 0 and are already filtered upstream by remove_constant_reward_groups.
    """
    advantages_P: list[torch.Tensor] = []
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)
    return advantages_P


FlatObElem = "int | tinker.ModelInputChunk"
FlatOb = list


def _is_prefix(seq1: list, seq2: list) -> bool:
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: list) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: list) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=list(current_text_chunk)))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> list:
    out: list = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """Convert a trajectory into one or more training Datums.

    Consecutive observation-action pairs whose observation extends the previous
    sequence are merged into a single Datum; a non-extending observation starts
    a new Datum.
    """

    class SequenceAccumulator:
        full_sequence: list = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []
        # 最后一个动作的采样路由 (见 TokensWithLogprobs.maybe_routing)。它那次请求的 prompt
        # 就是此前的整条序列, 所以它一份就覆盖整个 datum 的 input。
        routing = None

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []
            cls.routing = None

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        loss_fn_inputs = {
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
            "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
            "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
            "mask": TensorData.from_torch(torch.tensor(mask_T)),
        }
        routing = SequenceAccumulator.routing
        if routing is not None and routing.shape[0] >= input_tokens_T.length:
            loss_fn_inputs["routed_experts"] = TensorData.from_numpy(routing[: input_tokens_T.length])
        return tinker.Datum(model_input=input_tokens_T, loss_fn_inputs=loss_fn_inputs)

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.routing = getattr(ac_with_logprobs, "maybe_routing", None)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        # 动作里也可能有不该进 loss 的 token (思考预算强塞的过渡语, 见 maybe_mask):
        # mask 记 0, advantage 跟着 0 —— 训练侧按 adv != 0 认 active token。
        ac_mask = ac_with_logprobs.maybe_mask or [1.0] * len(ac_with_logprobs.tokens)
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [traj_advantage * m for m in ac_mask]
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + list(ac_mask))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Convert trajectory groups to a flat list of Datums plus per-datum metadata."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([{"group_idx": i_group, "traj_idx": i_traj} for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    """Drop groups whose trajectories all share the same reward (zero gradient)."""
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]
    return new_groups
