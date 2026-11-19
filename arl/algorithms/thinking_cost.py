"""Linear cost of sampled reasoning tokens, independent of task reward."""
import math


def validate_thinking_cost(cfg):
    free = getattr(cfg, "think_cost_free_tokens", 0)
    if not isinstance(free, int) or isinstance(free, bool) or free < 0:
        raise ValueError("think_cost_free_tokens must be an integer >= 0")
    lam = cfg.think_cost_lambda
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("think_cost_lambda must be finite and >= 0")
    if lam and not cfg.enable_thinking:
        raise ValueError("thinking cost requires enable_thinking=True")


def thinking_cost(resp, tokenizer, lam, free_tokens=0):
    """Return (sampled thinking token count, positive penalty).

    Qwen thinking renderers open <think> in the prompt. Count completion tokens
    before </think>; when truncated without a closing tag, count the whole prefix.
    Exclude forced transition text via the sampler's loss mask. No decoding or
    re-tokenization of generated text, so counts retain the sampled token boundaries.
    """
    if lam == 0:
        return 0, 0.0
    if tokenizer is None:
        raise ValueError("thinking cost requires a tokenizer")
    end = list(tokenizer.encode("</think>", add_special_tokens=False))
    if not end:
        raise ValueError("tokenizer returned an empty </think> sequence")
    tokens = resp.tokens
    stop = next((i for i in range(len(tokens) - len(end) + 1)
                 if tokens[i:i + len(end)] == end), len(tokens))
    start = 0
    opening = list(tokenizer.encode("<think>", add_special_tokens=False))
    if opening and tokens[:len(opening)] == opening:
        start = len(opening)
    mask = resp.maybe_mask
    if mask is not None and len(mask) != len(tokens):
        raise ValueError("thinking cost: loss mask and response token lengths differ")
    count = sum(1 for i in range(start, stop) if mask is None or mask[i] > 0)
    return count, float(lam) * max(0, count - free_tokens) / 8192.0
