"""Official IFEval inventory and offline strict/loose prediction evaluation."""
import asyncio
from collections import Counter
import json
from pathlib import Path

from arl.envs.instruction_following.ifeval import IFEvalSourceConfig, IFEvalTaskSource


def benchmark(**options):
    allowed = {'action', 'predictions', 'output_dir', 'train_set', 'limit', 'harness', 'python', 'upstream', 'timeout'}
    if set(options)-allowed:
        raise ValueError(f'unknown IFEval options: {sorted(set(options)-allowed)}')
    from arl.harnesses import resolve_pair
    pair = resolve_pair('ifeval', options.get('harness'), operation='evaluate')
    action = options.get('action', 'evaluate' if options.get('predictions') else 'inspect')
    if action not in ('inspect', 'evaluate'):
        raise ValueError('IFEval action must be inspect or evaluate')
    cfg = IFEvalSourceConfig(train_set=options.get('train_set', 'ifeval'), test_fraction=0,
        limit=options.get('limit', 0), python=options.get('python', ''), upstream=options.get('upstream', ''),
        timeout=options.get('timeout', 60))

    async def run():
        source = IFEvalTaskSource(cfg)
        try:
            await source.initialize()
            health = await source.check()
            tasks = source.train_tasks()
            if action == 'inspect':
                return {'env': pair.env, 'harness': pair.harness, 'tasks': len(tasks),
                        'instruction_types': dict(Counter(i for t in tasks for i in t.instruction_id_list)),
                        'commit': health['commit'], 'training': 'ready', 'benchmark': 'official strict/loose offline scoring'}
            if not options.get('predictions'):
                raise ValueError('IFEval evaluate requires predictions JSONL; local model runs use arl.experiments')
            records = [json.loads(line) for line in Path(options['predictions']).read_text().splitlines() if line.strip()]
            by_id, by_prompt = {}, {}
            for row in records:
                response = row.get('response')
                if not isinstance(response, str):
                    raise ValueError('IFEval predictions require string response')
                if 'key' in row:
                    key = str(row['key'])
                    if key in by_id: raise ValueError(f'duplicate prediction key {key}')
                    by_id[key] = response
                elif isinstance(row.get('prompt'), str):
                    if row['prompt'] in by_prompt: raise ValueError('duplicate prediction prompt')
                    by_prompt[row['prompt']] = response
                else:
                    raise ValueError('prediction requires key or exact prompt')
            rows = []
            for task in tasks:
                response = by_id.get(task.key, by_prompt.get(task.prompt))
                if response is None:
                    raise ValueError(f'missing prediction for {task.key}')
                scores = await source.grader.score(task, response)
                rows.append({'key': task.key, 'response': response, **scores})
            totals = sum(len(t.instruction_id_list) for t in tasks)
            result = {'tasks': len(tasks), 'instructions': totals, 'commit': health['commit'], 'harness': pair.harness}
            for mode in ('strict', 'loose'):
                result[mode] = {'prompt_accuracy': sum(row[mode]['all'] for row in rows)/len(rows),
                    'instruction_accuracy': sum(sum(row[mode]['checks']) for row in rows)/totals}
            if options.get('output_dir'):
                output = Path(options['output_dir']); output.mkdir(parents=True, exist_ok=False)
                (output/'scores.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                (output/'report.json').write_text(json.dumps(result, indent=2))
                result['report'] = str(output/'report.json')
            return result
        finally:
            await source.close()
    return asyncio.run(run())
