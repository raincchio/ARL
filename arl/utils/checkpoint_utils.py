"""Local checkpointing, replacing tinker_cookbook.checkpoint_utils.

Checkpoints are directories under ``{log_path}/checkpoints/{name}`` containing a
full HF ``policy/`` checkpoint (the sampler weights) and, for ``kind="both"``,
DeepSpeed optimizer/engine state. Only the latest optimizer state is retained;
older policy weights remain available. A ``{log_path}/checkpoints.jsonl`` index
records each checkpoint so the loop can resume.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)


def _index_path(log_path: str) -> str:
    return os.path.join(log_path, "checkpoints.jsonl")


def load_checkpoints_file(log_path: str) -> list[dict[str, Any]]:
    path = _index_path(log_path)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_last_checkpoint(log_path: str) -> dict[str, Any] | None:
    ckpts = load_checkpoints_file(log_path)
    # Prefer the highest batch with a resumable optimizer state.
    resumable = [c for c in ckpts if c.get("state_path")
                 and os.path.isdir(os.path.join(c["state_path"], "ds_state"))]
    if not resumable:
        return None
    return max(resumable, key=lambda c: c.get("batch", -1))


def _append_index(log_path: str, record: dict[str, Any]) -> None:
    # Under SPMD only rank 0 writes the index (all ranks save their own DeepSpeed
    # shard + participate in the collective weight consolidation, but a single
    # process owns the shared jsonl to avoid interleaved/corrupt lines).
    if os.environ.get("RANK", "0") != "0":
        return
    os.makedirs(log_path, exist_ok=True)
    with open(_index_path(log_path), "a") as f:
        f.write(json.dumps(record) + "\n")


def _rewrite_index(log_path: str, records: list[dict[str, Any]]) -> None:
    """Overwrite the index with ``records`` (rank 0 only)."""
    if os.environ.get("RANK", "0") != "0":
        return
    path = _index_path(log_path)
    with open(path + ".tmp", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)


def _prune_optimizer_states(log_path: str, state_path: str) -> None:
    """Drop older DeepSpeed state only after the new checkpoint is complete."""
    if os.environ.get("RANK", "0") != "0":
        return
    records = load_checkpoints_file(log_path)
    previous = {c["state_path"] for c in records
                if c.get("state_path") and c["state_path"] != state_path}
    for path in previous:
        state_dir = os.path.join(path, "ds_state")
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)
        latest = os.path.join(path, "latest")
        if os.path.isfile(latest):
            os.remove(latest)
        logger.info(f"[checkpoint] removed old optimizer state -> {path}")
    for record in records:
        if record.get("state_path") in previous:
            record["state_path"] = ""
    _rewrite_index(log_path, records)


def _prune_keep_best(log_path: str, keep_best_n: int) -> None:
    """Keep top-scored checkpoints plus the latest resumable checkpoint.

    Only checkpoints saved with a numeric ``score`` are pruning candidates;
    unscored ones (e.g. ``final``) and the latest resumable one are always kept.
    The losers' whole checkpoint
    directory (policy + any optimizer state) is removed and the index rewritten.
    Rank 0 only (it owns the shared index + on-disk files).
    """
    if os.environ.get("RANK", "0") != "0" or keep_best_n <= 0:
        return
    ckpts = load_checkpoints_file(log_path)
    # The latest resumable checkpoint must survive even if its score is low.
    ranked = [c for c in ckpts if isinstance(c.get("score"), (int, float))
              and not c.get("state_path")]
    others = [c for c in ckpts if not isinstance(c.get("score"), (int, float))
              or c.get("state_path")]
    ranked.sort(key=lambda c: c["score"], reverse=True)
    keep, drop = ranked[:keep_best_n], ranked[keep_best_n:]
    for c in drop:
        ckpt_dir = os.path.dirname(c.get("sampler_path", ""))  # parent of policy/
        if ckpt_dir:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.info(f"[checkpoint] pruned {c.get('name')} (score={c['score']:.4f})")
    _rewrite_index(log_path, others + keep)


async def save_checkpoint_async(
    training_client,
    name: str,
    log_path: str,
    loop_state: dict[str, Any] | None = None,
    kind: str = "both",
    score: float | None = None,
    keep_best_n: int = 0,
    background: bool = False,
) -> dict[str, str]:
    """Save a checkpoint and return ``{"sampler_path": ..., "state_path": ...}``.

    ``kind="sampler"`` saves only the policy weights (~1x model size); ``kind="both"``
    also saves the DeepSpeed optimizer/engine state (``ds_state``, ~6x) for exact resume.
    Once the new checkpoint is complete, older ``ds_state`` directories are
    removed and their index entries become weights-only checkpoints.

    If ``score`` is given, it's recorded so retention can rank checkpoints; if
    ``keep_best_n > 0``, the best-``keep_best_n`` scored checkpoints are kept
    alongside the latest resumable checkpoint (worse ones are deleted).

    ``background=True``: the serialize+write half of the weight save runs in a
    background thread on rank 0 and this call returns after the (collective,
    unavoidable) ZeRO-3 consolidation -- training overlaps the disk write. The
    index line is appended, "saved" logged and pruning run only AFTER the files
    are complete (from the writer thread), so anything trusting
    ``checkpoints.jsonl`` never sees a half-written checkpoint. The next save --
    or teardown -- joins the in-flight write first; a synchronous ``final`` save
    therefore also guarantees every earlier background write has landed.
    """
    # Join before touching optimizer files or the index, including when a name
    # is reused. The previous writer may still be publishing/pruning its save.
    training_client.wait_pending_save()
    ckpt_dir = os.path.join(log_path, "checkpoints", name)
    policy_dir = os.path.join(ckpt_dir, "policy")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optimizer state first (when asked for): DeepSpeed's save_checkpoint is a
    # collective every rank enters, so it can't be backgrounded -- and doing it
    # before the weight save keeps the collective order identical on all ranks
    # while letting `record` below carry the final state_path either way.
    state_path = ""
    if kind in ("both", "state"):
        training_client.save_optimizer_state(ckpt_dir)
        state_path = ckpt_dir

    record = {
        "name": name,
        "batch": (loop_state or {}).get("batch"),
        "sampler_path": policy_dir,
        "state_path": state_path,
    }
    if score is not None:
        record["score"] = float(score)

    def _publish() -> None:
        _append_index(log_path, record)
        if state_path:
            _prune_optimizer_states(log_path, state_path)
        logger.info(
            f"[checkpoint] saved {name} -> {ckpt_dir} (kind={kind}"
            + (f", score={score:.4f}" if score is not None else "")
            + ")"
        )
        if keep_best_n and keep_best_n > 0:
            _prune_keep_best(log_path, keep_best_n)

    training_client.save_policy_weights(
        policy_dir, background=background, on_done=_publish if background else None)
    if not background:
        _publish()
    return {"sampler_path": policy_dir, "state_path": state_path}
