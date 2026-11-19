"""Validated IFEval records and deterministic experiment splits."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Task:
    key: str
    prompt: str
    instruction_id_list: tuple[str, ...]
    kwargs: tuple[dict, ...]

    @classmethod
    def from_record(cls, record):
        key, prompt = record.get('key'), record.get('prompt')
        ids, args = record.get('instruction_id_list'), record.get('kwargs')
        if isinstance(key, bool) or not isinstance(key, (str, int)) or not str(key):
            raise ValueError('IFEval key must be a nonempty string/integer')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('IFEval prompt must be nonempty')
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('IFEval instruction_id_list must be nonempty')
        if not isinstance(args, list) or len(args) != len(ids) or any(not isinstance(a, dict) for a in args):
            raise ValueError('IFEval kwargs must align with instruction_id_list')
        return cls(str(key), prompt, tuple(ids), tuple(dict(a) for a in args))

    def record(self):
        return dict(key=self.key, prompt=self.prompt, instruction_id_list=list(self.instruction_id_list),
                    kwargs=list(self.kwargs))

    @property
    def prompt_id(self):
        return hashlib.sha256(self.prompt.strip().encode()).hexdigest()


def load_tasks(path):
    tasks = [Task.from_record(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if not tasks:
        raise ValueError('no IFEval tasks')
    if len({t.key for t in tasks}) != len(tasks):
        raise ValueError('duplicate IFEval task keys')
    # The same prompt with different labels must not cross train/test.
    if len({t.prompt_id for t in tasks}) != len(tasks):
        raise ValueError('duplicate IFEval prompts')
    return tasks


def split_tasks(tasks, fraction=.1, seed=0):
    if not 0 <= fraction < 1:
        raise ValueError('test_fraction must be in [0, 1)')
    # Sort by identity: shuffling input records must not change membership.
    keys = sorted(t.key for t in tasks)
    random.Random(seed).shuffle(keys)
    n = int(len(keys)*fraction)
    if fraction and len(keys) > 1:
        n = max(1, n)
    heldout = set(keys[:n])
    return ([t for t in tasks if t.key not in heldout], [t for t in tasks if t.key in heldout])
