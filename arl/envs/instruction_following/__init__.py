"""Instruction-following environments with verifiable training rewards."""
from ..registry import EnvironmentSpec

ENVIRONMENTS = (
    EnvironmentSpec('instruction_following.multicodeif', 'instruction_following',
                    'arl.envs.instruction_following.multicodeif', 'MultiCodeIFSourceConfig', 'MultiCodeIFTaskSource',
                    aliases=('multicodeif',), default_harness='multicodeif_self_repair',
                    supported_harnesses=('multicodeif_self_repair',),
                    backend_interfaces=(('openai_compatible', 'text_constraints'), ('offline', 'text_constraints'))),
    EnvironmentSpec('instruction_following.ifeval', 'instruction_following',
                    'arl.envs.instruction_following.ifeval', 'IFEvalSourceConfig', 'IFEvalTaskSource',
                    aliases=('ifeval',), default_harness='direct', supported_harnesses=('direct',)),
)
