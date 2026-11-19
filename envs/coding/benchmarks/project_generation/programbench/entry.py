"""ProgramBench official offline grading and native GRPO environment."""


def benchmark(**kwargs):
    from arl.benchmarks.programbench import benchmark
    return benchmark(**kwargs)


def train(**kwargs):
    from arl.benchmarks.registry import BenchmarkPendingError
    raise BenchmarkPendingError('coding.programbench', mode='train')
