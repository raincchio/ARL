"""JSON worker; imports unmodified official IFEval only inside its isolated env."""
import json
import random
import sys


def main(payload):
    import nltk
    from langdetect import DetectorFactory
    from instruction_following_eval import evaluation_lib as official
    from instruction_following_eval import instructions_registry
    # Fail preflight if any checks would require missing sentence resources.
    nltk.data.find('tokenizers/punkt_tab/english/')
    random.seed(0)
    DetectorFactory.seed = 0
    if payload['action'] == 'check':
        # Exercise build_description/check_following for every selected task,
        # catching unknown instruction IDs, invalid kwargs and missing resources
        # before the model is loaded. False scores here are expected.
        for row in payload['tasks']:
            official.test_instruction_following_strict(official.InputExample(**row), {row['prompt']: 'Preflight response.'})
        return {'ready': True, 'tasks': len(payload['tasks']), 'instruction_types': sorted(instructions_registry.INSTRUCTION_DICT)}
    row = payload['task']
    inp = official.InputExample(**row)
    result = {}
    for name in ('strict', 'loose'):
        random.seed(0)
        out = getattr(official, 'test_instruction_following_'+name)(inp, {inp.prompt: payload['response']})
        result[name] = {'all': bool(out.follow_all_instructions),
                        'checks': [bool(x) for x in out.follow_instruction_list]}
    return result


if __name__ == '__main__':
    try:
        print(json.dumps(main(json.load(sys.stdin))))
    except Exception as e:
        print(json.dumps({'error': f'{type(e).__name__}: {e}'}))
