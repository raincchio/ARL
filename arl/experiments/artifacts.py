"""Identity manifests and aligned, token-level collection artifacts."""
import json
import math
from pathlib import Path


def write_identity(path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as f:
            json.dump(identity, f, ensure_ascii=False, indent=2)
    except FileExistsError:
        if json.loads(path.read_text()) != identity:
            raise ValueError(f'existing run identity differs from requested env/harness/model: {path}')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def append_datums(path, result, identity, batch=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(result.datums), len(result.returns), len(result.groups), len(result.units), len(result.correctness)]
    if len(set(lengths)) != 1:
        raise ValueError('collection datum/reward/group/unit/correctness alignment differs')
    trajs = {traj.tid: traj for traj in result.trajs}
    records = []
    for datum, reward, group, unit, correct in zip(result.datums, result.returns, result.groups, result.units, result.correctness):
        values = datum.loss_fn_inputs
        record = {'schema': 'arl.datum.v1', 'identity': identity, 'batch': batch,
                  'task': trajs[unit[0]].task_key, 'trajectory_id': [batch, *unit], 'group': [batch, *group],
                  'reward': reward, 'correct': correct, 'token_ids': datum.model_input.to_ints(),
                  'target_tokens': values['target_tokens'].to_torch().tolist(),
                  'logprobs': values['logprobs'].to_torch().tolist(),
                  'loss_mask': values['mask'].to_torch().tolist()}
        if "routed_experts" in values:
            record["routed_experts"] = values["routed_experts"].to_torch().tolist()
        validate_datum(record)
        records.append(record)
    with path.open('a') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return len(records)


def validate_datum(record):
    fields = ('token_ids', 'target_tokens', 'logprobs', 'loss_mask')
    if any(not isinstance(record.get(field), list) for field in fields):
        raise ValueError('training datum requires token_ids, target_tokens, logprobs and loss_mask lists')
    if len({len(record[field]) for field in fields}) != 1 or not record['token_ids']:
        raise ValueError('training datum token/logprob/mask alignment differs')
    if any(not isinstance(token, int) or isinstance(token, bool) for field in ('token_ids', 'target_tokens') for token in record[field]):
        raise ValueError('training tokens must be integer IDs')
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for field in ('logprobs', 'loss_mask') for value in record[field]):
        raise ValueError('training logprobs/masks must be finite')
    if any(value not in (0, 1) for value in record['loss_mask']):
        raise ValueError('loss_mask must contain 0/1')


def validate_collection(path, *, env, harness, model=None, require_training_trace=False):
    from arl.harnesses import resolve_pair
    pair = resolve_pair(env, harness, operation='collect')
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('collection is not complete')
    identity = manifest['identity']
    if (identity['env'], identity['harness']) != (pair.env, pair.harness):
        raise ValueError('collection env/harness do not match the requested pair')
    if model is not None and identity['model'] != model:
        raise ValueError('collection model does not match the requested model')
    count = 0
    if require_training_trace:
        from arl.harnesses.compatibility import TRAIN_TRACE_FIELDS
        if not TRAIN_TRACE_FIELDS.issubset(set(manifest.get('trace_fields', ()))):
            raise ValueError('collection has no token/logprob/mask training trace')
        with (root / 'datums.jsonl').open() as f:
            for line in f:
                record = json.loads(line)
                if record['identity'] != identity:
                    raise ValueError('mixed env/harness/model identities in collection')
                validate_datum(record)
                count += 1
        if count == 0:
            raise ValueError('collection contains no training datums')
    return {'identity': identity, 'datums_validated': count}
