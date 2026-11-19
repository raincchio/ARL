"""Single-answer IFEval environment using official strict/loose checks."""
from dataclasses import dataclass, field


@dataclass
class StepOutput:
    reward: float = 0.
    done: bool = True
    metrics: dict = field(default_factory=dict)
    infra_error: bool = False
    tool_results: list = field(default_factory=list)


def final_answer(text):
    # Qwen templates may open <think> in the prompt, so only </think> is decoded.
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text:
        return ''
    return text.strip()


class IFEvalEnv:
    tools = []

    def __init__(self, task, grader, *, reward_mode='strict'):
        if reward_mode not in ('strict', 'fraction'):
            raise ValueError('IFEval reward_mode must be strict or fraction')
        self.task, self.grader, self.reward_mode = task, grader, reward_mode
        self.done = False
        self.response = ''
        self.result = None
        self.error = ''

    async def reset(self):
        if self.done:
            raise RuntimeError('cannot reset a completed IFEval environment')

    def messages(self, aggressiveness=0):
        # Never reveal instruction IDs/kwargs or replace the original task to fit.
        return [{'role': 'user', 'content': self.task.prompt}]

    async def step(self, text, *, truncated=False):
        if self.done:
            raise RuntimeError('IFEval environment already completed')
        self.done = True
        self.response = final_answer(text)
        if truncated:
            return StepOutput(metrics={'correct': False, 'truncated': True})
        try:
            self.result = await self.grader.score(self.task, self.response)
            strict, loose = self.result['strict'], self.result['loose']
            if any(len(result['checks']) != len(self.task.instruction_id_list) or
                   not all(isinstance(x, bool) for x in result['checks']) or
                   result['all'] != all(result['checks']) for result in (strict, loose)):
                raise ValueError('invalid/misaligned IFEval checker output')
            fraction = sum(strict['checks']) / len(strict['checks'])
            reward = float(strict['all']) if self.reward_mode == 'strict' else fraction
            return StepOutput(reward=reward, metrics={'correct': strict['all'],
                'ifeval/strict_prompt': float(strict['all']), 'ifeval/loose_prompt': float(loose['all']),
                'ifeval/strict_instructions_passed': sum(strict['checks']),
                'ifeval/loose_instructions_passed': sum(loose['checks']),
                'ifeval/instructions_total': len(strict['checks'])})
        except Exception as e:
            self.error = f'{type(e).__name__}: {e}'
            return StepOutput(infra_error=True, metrics={'correct': None, 'grading_error': self.error})

    async def finish(self, reason=''):
        self.done = True
        return StepOutput(metrics={'correct': False, 'finish_reason': reason})

    async def close(self):
        pass  # grader lifetime belongs to TaskSource

    def summary(self):
        return {'task': self.task.key, 'response': self.response, 'checks': self.result, 'error': self.error}
