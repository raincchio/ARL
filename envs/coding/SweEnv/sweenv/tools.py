"""Qwen XML and Hermes JSON calls; independent of CudaEnv."""
import json
import re

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in /testbed. Each call uses a fresh shell; files persist. "
        "Inspect code, edit files, and run your own tests here. Use explicit cd for other directories.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
        }, "required": ["command"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the current repository changes as the final patch and end the task.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def split_reasoning(text):
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.removeprefix("<think>"), content
    # Non-thinking models also emit tool calls directly.
    if "<think>" in text or not re.search(r"<tool_call>|<function=", text):
        return text.removeprefix("<think>"), ""
    return "", text


def parse_tool_calls(text):
    _, content = split_reasoning(text)
    calls, errors = [], []
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S)
    if content.count("<tool_call>") != len(blocks):
        errors.append("Incomplete tool_call; emit a complete bash or submit call.")
    for block in blocks:
        try:
            if block.lstrip().startswith("{"):
                call = json.loads(block)
                name, args = call["name"], call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
            else:
                match = re.fullmatch(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", block, re.S)
                if not match:
                    raise ValueError("expected a function block or JSON object")
                name = match[1]
                args = {k: v for k, v in re.findall(
                    r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", match[2], re.S)}
            if name not in {"bash", "submit"} or not isinstance(args, dict):
                raise ValueError("available tools: bash, submit")
            if name == "bash" and (set(args) != {"command"} or not isinstance(args["command"], str)
                                   or not args["command"].strip()):
                raise ValueError("bash requires a nonempty string command")
            if name == "submit" and args:
                raise ValueError("submit takes no arguments")
            calls.append({"name": name, "arguments": args})
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"Invalid tool call: {e}")
    return calls, errors
