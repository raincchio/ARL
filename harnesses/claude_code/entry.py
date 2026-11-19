"""Claude Code CLI over ARL's local on-policy sampler and remote SWE MCP tool."""
from arl.harnesses.claude_code import check_dependencies, run


def make_renderer(tokenizer, **kwargs):
    check_dependencies()
    from arl.agent.render import AgentRenderer
    return AgentRenderer(tokenizer, **kwargs)
