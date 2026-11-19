"""Validate the environment/harness -> GRPO boundary before policy updates."""
import math


def validate_sample(sequence):
    tokens, logprobs = sequence.tokens, sequence.logprobs
    mask = getattr(sequence, 'loss_mask', None)
    if not tokens or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens):
        raise ValueError('sample needs nonempty integer token IDs')
    if logprobs is None or len(tokens) != len(logprobs):
        raise ValueError('sample token/old-policy logprob lengths differ')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in logprobs):
        raise ValueError('sample old-policy logprobs must be finite')
    if mask is not None and (len(mask) != len(tokens) or any(v not in (0, 1) for v in mask)):
        raise ValueError('sample loss mask must align with tokens and contain 0/1')


def validate_rollout(result):
    """All segments from one trajectory share a group, return and correctness.

    Prompt/tool/feedback tokens have mask=0. Group baselines count trajectory
    units, so a long self-repair episode is not counted as several siblings.
    """
    lengths = [len(getattr(result, key)) for key in ('datums', 'returns', 'groups', 'units', 'correctness')]
    if len(set(lengths)) != 1:
        raise ValueError('GRPO rollout datum/reward/group/unit/correctness alignment differs')
    trajs = {t.tid: t for t in result.trajs}
    units = {}
    for datum, reward, group, unit, correct in zip(result.datums, result.returns, result.groups, result.units, result.correctness):
        if not math.isfinite(reward) or (correct is not None and not isinstance(correct, bool)):
            raise ValueError('GRPO requires finite returns and boolean/unknown correctness')
        if not unit or unit[0] not in trajs or trajs[unit[0]].dropped or trajs[unit[0]].infra:
            raise ValueError('GRPO datum belongs to an invalid/unscored trajectory')
        if group != (trajs[unit[0]].group,):
            raise ValueError('GRPO sibling group does not match the source task')
        signature = (group, reward, correct)
        if tuple(unit) in units and units[tuple(unit)] != signature:
            raise ValueError('segments of one trajectory have inconsistent returns/groups/labels')
        units[tuple(unit)] = signature
        values = datum.loss_fn_inputs
        fields = {key: values[key].to_torch().reshape(-1) for key in ('target_tokens', 'logprobs', 'mask')}
        if not datum.model_input.length or any(len(v) != datum.model_input.length for v in fields.values()):
            raise ValueError('GRPO token/target/logprob/mask alignment differs')
        import torch
        if not torch.isfinite(fields['logprobs']).all() or not ((fields['mask'] == 0) | (fields['mask'] == 1)).all():
            raise ValueError('GRPO trace has nonfinite logprobs or nonbinary loss mask')
        if not fields['mask'].any():
            raise ValueError('GRPO datum has no sampled action tokens')
    return result
