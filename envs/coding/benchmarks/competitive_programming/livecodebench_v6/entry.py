"""LiveCodeBench v6: benchmark and training entry points (pending)."""
from arl.benchmarks.registry import BenchmarkPendingError


def benchmark(**kwargs):
    """TODO: connect the benchmark dataset, harness and scoring protocol."""
    raise BenchmarkPendingError('coding.livecodebench_v6', mode="benchmark")


def train(**kwargs):
    """TODO: implement the training task source, rewards and data split."""
    raise BenchmarkPendingError('coding.livecodebench_v6', mode="train")
