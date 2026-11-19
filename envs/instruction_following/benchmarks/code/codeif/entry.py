"""CodeIF: benchmark and training entry points (pending)."""
from arl.benchmarks.registry import BenchmarkPendingError


def benchmark(**kwargs):
    """TODO: connect the benchmark dataset, harness and scoring protocol."""
    raise BenchmarkPendingError('instruction_following.codeif', mode="benchmark")


def train(**kwargs):
    """TODO: implement the training task source, rewards and data split."""
    raise BenchmarkPendingError('instruction_following.codeif', mode="train")
