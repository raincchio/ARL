"""Codeforces Rating: benchmark and training entry points (pending)."""
from arl.benchmarks.registry import BenchmarkPendingError


def benchmark(**kwargs):
    """TODO: connect the benchmark dataset, harness and scoring protocol."""
    raise BenchmarkPendingError('coding.codeforces_rating', mode="benchmark")


def train(**kwargs):
    """TODO: implement the training task source, rewards and data split."""
    raise BenchmarkPendingError('coding.codeforces_rating', mode="train")
