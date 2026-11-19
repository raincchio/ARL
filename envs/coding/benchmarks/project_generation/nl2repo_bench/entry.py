"""NL2Repo-Bench: benchmark and training entry points (pending)."""
from arl.benchmarks.registry import BenchmarkPendingError


def benchmark(**kwargs):
    """TODO: connect the benchmark dataset, harness and scoring protocol."""
    raise BenchmarkPendingError('coding.nl2repo_bench', mode="benchmark")


def train(**kwargs):
    """TODO: implement the training task source, rewards and data split."""
    raise BenchmarkPendingError('coding.nl2repo_bench', mode="train")
