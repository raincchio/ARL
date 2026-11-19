from copy import deepcopy
import math

from sweenv.env import SweWorkspaceEnv, StepOutput
from sweenv.backend import BackendError
from sweenv.tools import TOOLS

PROGRAM_TOOLS = deepcopy(TOOLS)
PROGRAM_TOOLS[0]['function']['description'] = 'Run bash in /workspace as agent. Files persist; each call starts a fresh shell. Inspect the reference program, write source code, and test your implementation.'
PROGRAM_TOOLS[1]['function']['description'] = 'Submit the complete /workspace source tree and compile.sh for official ProgramBench evaluation.'

SYSTEM = '''Rebuild the reference program from scratch. In /workspace, ./executable is the reference program;
use its normal CLI/stdin/stdout and bundled documentation to discover behavior. Write original source
and an executable ./compile.sh that builds your own ./executable on a fresh workspace.
Internet access is disabled. Do not obtain original source, install the original program, wrap/reuse
reference binaries, decompile, disassemble, trace or instrument the reference executable.
Do not access or modify grading infrastructure. Analysis tools on your own binaries are allowed.
Use bash to inspect, implement and test. Each command starts a fresh shell; files persist.
Call submit when finished; a final answer without tools also submits. The entire workspace is archived
and evaluated in fresh containers using the official tests. You do not see hidden tests or test feedback.
'''


def classify_summary(summary):
    """Model build failures are zero; missing/broken evaluation is not a negative label."""
    error = summary.get('error_code')
    if error in ('compile_failed', 'copy_executable_failed'):
        return 0., False
    if error or summary.get('test_branch_errors') or summary.get('n_system_errors'):
        raise BackendError(f'official ProgramBench evaluation incomplete: {error or summary}')
    score, passed, total = summary.get('score'), summary.get('n_resolved'), summary.get('n_tests')
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (passed, total)) or total <= 0 or not 0 <= passed <= total:
        raise BackendError('invalid/missing ProgramBench test counts')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 or not math.isclose(score, passed/total, abs_tol=1e-8):
        raise BackendError('invalid/inconsistent ProgramBench score')
    return float(score), passed == total


class ProgramBenchEnv(SweWorkspaceEnv):
    tools = PROGRAM_TOOLS
    workspace_dir = '/workspace'
    submit_message = 'Submitting the complete workspace for official ProgramBench evaluation.'

    def __init__(self, task, workspace, cfg, *, reward_mode='fraction'):
        if reward_mode not in ('fraction', 'strict'): raise ValueError('ProgramBench reward_mode must be fraction or strict')
        super().__init__(task, workspace, cfg)
        self.reward_mode = reward_mode

    def initial_messages(self):
        return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
            f'Task: {self.task.instance_id}\nWork in /workspace. Inspect the bundled documentation and run ./executable '
            'to understand the reference behavior. Build your replacement from original source using ./compile.sh. '
            f'You have at most {self.cfg.max_turns} turns.'}]

    async def _finish(self, infra=False):
        self.done, self._infra = True, infra
        score, resolved = 0., False
        try:
            if not infra:
                self._grade = await self.workspace.submit(self.cfg.eval_timeout)
                score, resolved = classify_summary(self._grade['summary'])
                self._reward = score if self.reward_mode == 'fraction' else float(resolved)
        except (BackendError, KeyError, TypeError, ValueError) as exc:
            self._infra = True
            self._grade = {'error': str(exc)}
        finally:
            await self.close()
        summary = self._grade.get('summary', {})
        return StepOutput(self._reward, True, [], {'correct': None if self._infra else resolved,
            'resolved': float(resolved), 'programbench/test_pass_fraction': score,
            'programbench/tests_passed': summary.get('n_resolved', 0), 'programbench/tests_total': summary.get('n_tests', 0),
            'episode/turns': self._turns}, self._infra)

    def summary(self):
        return {'instance_id': self.task.instance_id, 'turns': self._turns, 'reward': self._reward,
                'infra_error': self._infra, 'grade': self._grade}
