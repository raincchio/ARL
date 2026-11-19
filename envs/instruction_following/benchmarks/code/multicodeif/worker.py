"""Isolated bridge to the pinned official MultiCodeIF constraint checks.

No generated code is executed. Source is not rewritten on disk. Only the eager
Guesslang import is made lazy so unrelated rule checks do not require TensorFlow.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from arl.benchmarks.multicodeif import chat


class Unavailable(RuntimeError):
    pass


def tree_sitter_legacy_api():
    """Keep upstream queries/captures unchanged with ABI-15 grammar wheels."""
    import tree_sitter as ts
    native_language, native_parser = ts.Language, ts.Parser
    native_query, native_cursor = ts.Query, ts.QueryCursor

    class Query:
        def __init__(self, language, text):
            self.native = native_query(language.native, text)

        def captures(self, node):
            return native_cursor(self.native).captures(node)

    class Language:
        def __init__(self, capsule):
            self.native = native_language(capsule)

        def query(self, text):
            return Query(self, text)

    class Parser:
        def __init__(self, language):
            self.language = language
            self.native = native_parser(language.native)

        def parse(self, code):
            return self.native.parse(code)

    ts.Language, ts.Parser = Language, Parser


def load_syntax(root):
    if 'syntax' in sys.modules:
        return sys.modules['syntax']
    path = root / 'src/evaluation/syntax.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == 'guesslang')]
    module = types.ModuleType('syntax')
    module.__file__ = str(path)
    def guess():
        try:
            from guesslang import Guess
        except ImportError as e:
            raise Unavailable('language_type needs the official Guesslang/TensorFlow environment') from e
        return Guess()
    module.Guess = guess
    sys.modules['syntax'] = module
    try:
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    except Exception:
        sys.modules.pop('syntax', None)
        raise
    return module


def configure(root, config):
    tree_sitter_legacy_api()
    sys.path.insert(0, str(root / 'src/evaluation'))
    bridge = types.ModuleType('generation')
    def generation(prompts, model_name):
        if not config.get('judge_base_url') or not config.get('judge_model'):
            raise Unavailable('this official check requires judge_base_url and judge_model')
        if isinstance(prompts, str):
            prompts = [prompts]
        return [(prompt, chat(config, [{'role': 'user', 'content': prompt}], judge=True)) for prompt in prompts]
    bridge.generation_in_parallel = generation
    sys.modules['generation'] = bridge


def check_single(root, row, code):
    category, constraint = row['category'], row.get('constraint', row['category'])
    if category == 'syntax' and constraint == 'language_version':
        raise Unavailable('language_version requires pinned compilation toolchains; compilation is not enabled')
    if constraint in ('declaration_style', 'framework', 'input_range', 'output_range'):
        module = importlib.import_module('llm_based_eval')
        return bool(getattr(module, f'check_{constraint}_by_llm')(code, row['prompt'], row['answer']))
    if category in ('syntax', 'data_structure'):
        load_syntax(root)
    if category in ('syntax', 'input_output', 'data_structure', 'code_convention'):
        module = importlib.import_module(category)
        return bool(getattr(module, f'evaluate_{category}')(code, row['answer'], constraint))
    if category in ('situation', 'example', 'code_contextual', 'non_functional_requirements', 'algorithm'):
        module = importlib.import_module('llm_based_eval')
        if category == 'algorithm':
            prompt = module.generate_algorithm_evaluation_prompt(constraint, row['prompt'], code, row['answer'])
        else:
            prompt = module.genearte_evaluation_prompt(category, row['prompt'], row['answer'], code)
        response = sys.modules['generation'].generation_in_parallel(prompt, module.BASE_MODEL)
        return 'yes' in response[0][1].lower()
    raise Unavailable(f'no released checker for category {category!r}')


def evaluate_request(root, request, repair_template):
    from util import extract_code
    results = []
    raw = request.get('response')
    # Thinking text is not code; strip it before the official code extractor.
    code = extract_code(raw.split('</think>', 1)[-1]) if isinstance(raw, str) else ''
    for row in request['checks']:
        constraint = row.get('constraint', row['category'])
        result = {'category': row['category'], 'constraint': constraint, 'answer': row['answer']}
        if request.get('generation_error'):
            result.update(status='generation_error', passed=None, error=request['generation_error'])
            results.append(result)
            continue
        try:
            if request['suite'] == 'multi_level':
                load_syntax(root)
                module = importlib.import_module('multi_eval')
                passed = bool(module.check_data_structure_constraint(code, row['answer']))
            else:
                passed = check_single(root, row, code)
            result.update(status='scored', passed=passed)
        except SyntaxError as e:
            result.update(status='scored', passed=False, error=f'invalid generated syntax: {e}')
        except (Unavailable, ImportError) as e:
            result.update(status='unavailable', passed=None, error=str(e))
        except Exception as e:
            result.update(status='error', passed=None, error=f'{type(e).__name__}: {e}')
        results.append(result)
    complete = all(r['status'] == 'scored' for r in results)
    return {'task_id': request['task_id'], 'status': 'scored' if complete else 'incomplete',
            'passed': all(r['passed'] for r in results) if complete else None,
            'checks': results, 'repair_template': repair_template}


def main():
    request = json.load(sys.stdin)
    try:
        root = Path(request['repo'])
        configure(root, request['config'])
        # Use the actual official self-repair template, not a rewritten variant.
        repair_path = root / 'src/self_repair/repair.py'
        tree = ast.parse(repair_path.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'base_prompt' for t in node.targets))
        template = ast.literal_eval(assignment.value)
        with contextlib.redirect_stdout(sys.stderr):
            results = [evaluate_request(root, item, template) for item in request['requests']]
        print(json.dumps({'results': results}))
    except Exception as e:
        print(json.dumps({'error': f'{type(e).__name__}: {e}'}))


if __name__ == '__main__':
    main()
