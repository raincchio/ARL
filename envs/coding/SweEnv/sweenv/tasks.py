"""SWE-bench records. Gold patches/tests stay in the trusted evaluator only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


@dataclass(frozen=True)
class SweTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    record: dict[str, Any]

    @classmethod
    def from_record(cls, row):
        for name in ("instance_id", "repo", "base_commit", "problem_statement"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"SWE-bench record requires nonempty {name}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["instance_id"]):
            raise ValueError("invalid SWE-bench instance_id")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", row["base_commit"]):
            raise ValueError("invalid SWE-bench base_commit")
        return cls(*(row[k] for k in ("instance_id", "repo", "base_commit", "problem_statement")),
                   record=dict(row))


def load_task_set(name_or_path: str, *, split="test", limit=0) -> list[SweTask]:
    path = Path(name_or_path).expanduser()
    if path.is_file():
        with path.open() as f:
            rows = ([json.loads(line) for line in f if line.strip()]
                    if path.suffix == ".jsonl" else json.load(f))
        if not isinstance(rows, list):
            raise ValueError("SWE-bench file must contain a JSON array or JSONL records")
    else:
        from datasets import load_dataset
        dataset = DATASETS.get(name_or_path, name_or_path)
        rows = load_dataset(dataset, split=split)
    if limit:
        rows = list(rows)[:limit]
    tasks = [SweTask.from_record(row) for row in rows]
    if len({t.instance_id for t in tasks}) != len(tasks):
        raise ValueError("duplicate SWE-bench instance_id")
    return tasks
