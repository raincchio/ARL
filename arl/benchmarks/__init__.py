"""Benchmark entry points; default to evaluation and keep training opt-in."""
from .registry import (BenchmarkPendingError, BenchmarkSpec, find_benchmark,
                       get_benchmark, list_benchmarks, run_benchmark)

__all__ = ["BenchmarkPendingError", "BenchmarkSpec", "find_benchmark", "get_benchmark",
           "list_benchmarks", "run_benchmark"]
