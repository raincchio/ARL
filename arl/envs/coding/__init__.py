"""Coding tasks: repository repair and CUDA operator optimization."""
from ..registry import EnvironmentSpec

ENVIRONMENTS = (
    EnvironmentSpec("coding.cuda", "coding", "arl.envs.coding.cuda",
                    "CudaSourceConfig", "CudaTaskSource", aliases=("cuda",), default_harness="arl_agent", supported_harnesses=("arl_agent",)),
    EnvironmentSpec("coding.swe", "coding", "arl.envs.coding.swe",
                    "SweSourceConfig", "SweTaskSource", aliases=("swe",), default_harness="claude_code", supported_harnesses=("claude_code", "mini_swe_agent")),
    EnvironmentSpec("coding.programbench", "coding", "arl.envs.coding.programbench",
                    "ProgramBenchSourceConfig", "ProgramBenchTaskSource", aliases=("programbench",),
                    default_harness="mini_swe_agent", supported_harnesses=("mini_swe_agent",)),
)
