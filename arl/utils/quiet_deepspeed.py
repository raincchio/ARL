"""Import deepspeed without its bogus op-compatibility spam.

``import deepspeed`` runs ``git_version_info.py``, which probes EVERY optional op
with ``is_compatible()`` just to build a report. Ops we never use print warnings
during that probe -- notably ``sparse_attn``, whose check demands ``triton==1.0.0``
and ``torch<2.0`` (both from ~2021; this env runs triton 3.5 / torch 2.10). The op
is simply marked unavailable and never built, so the warning is pure noise -- and
it cannot be "fixed" by changing versions: triton 1.0.0 is incompatible with the
torch this project needs.

It also can't be silenced the normal ways: ``OpBuilder.warning()`` is a bare
``print()`` to stdout (deepspeed/ops/op_builder/builder.py), so it ignores the
logging module and ``warnings`` filters alike.

So: import once, up front, with stdout captured, and drop only the lines matching
known-bogus patterns. Everything else is re-emitted verbatim -- this hides noise,
not information. Later ``import deepspeed`` calls hit sys.modules and stay quiet.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys

# Only lines we've confirmed are harmless probe output for ops we don't use.
_BOGUS = (
    re.compile(r"using untested triton version"),
    re.compile(r"please install triton==1\.0\.0"),
    re.compile(r"sparse_attn requires a torch version"),
    re.compile(r"async_io requires the dev libaio"),
    re.compile(r"please specify the CUTLASS repo directory", re.I),
)


def _is_bogus(line: str) -> bool:
    stripped = re.sub(r"\x1b\[[0-9;]*m", "", line)  # deepspeed colors its prefix
    return any(p.search(stripped) for p in _BOGUS)


def import_deepspeed_quietly():
    """Import deepspeed, dropping only known-bogus op-probe lines. Returns the module."""
    if "deepspeed" in sys.modules:
        return sys.modules["deepspeed"]

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            import deepspeed  # noqa: F401
    finally:
        # Re-emit anything that wasn't on the known-bogus list, so a real import
        # problem still reaches the user.
        kept = [ln for ln in buf.getvalue().splitlines() if ln.strip() and not _is_bogus(ln)]
        if kept:
            print("\n".join(kept))
    return sys.modules.get("deepspeed")
