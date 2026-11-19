"""Upstream mini-swe-agent over the local token sampler and Docker backends."""
from arl.harnesses.mini_swe import check_dependencies, run


def make_renderer(tokenizer, **kwargs):
    check_dependencies()
    from arl.agent.render import AgentRenderer
    return AgentRenderer(tokenizer, **kwargs)
