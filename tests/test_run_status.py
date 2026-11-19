import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from script.run_status import outcome_counts, render, snapshot


class RunStatusTests(unittest.TestCase):
    def test_old_run_reports_saved_results_without_inventing_batch_progress(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            output = run / 'baseline'
            output.mkdir()
            (run / 'baseline.json').write_text(json.dumps({'evaluate': {
                'output_dir': str(output), 'options': {'task_limit': 10, 'group_size': 8}}}))
            (output / 'outcomes.jsonl').write_text('{"reward":1}\n{"dropped":"infra"}\n{"partial":')
            self.assertEqual(outcome_counts(output / 'outcomes.jsonl'), (2, 1))
            with patch('script.run_status.local_processes', return_value=[123]):
                state = snapshot(run)
            text = render(state)
            self.assertIn('2/80', text)
            self.assertIn('PID: 123', text)
            self.assertIn('暂无批内快照', text)

    def test_render_live_batch_and_keep_generation_separate_from_environment(self):
        live = dict(hostname='node', pid=123, rank=0, batch='evaluate-1-', status='running',
            updated_at=time.time(), completed=7, total=16, batch_size=2, samples_per_task=8,
            dropped=1, initialized=16, generating=5, environment=4,
            turns_completed=20, generated_tokens=4000)
        text = render(dict(hostname='node', run='/tmp/run', phases=[dict(
            phase='baseline', pids=[123], status='running', outcomes_saved=16,
            outcomes_dropped=0, total=80, live=[live])]))
        self.assertIn('本批完成: 7/16（2 × 8）', text)
        self.assertIn('生成/排队 5 | 环境执行/判分 4', text)
        self.assertIn('累计已落盘: 16/80', text)
        live.update(total=4, completed=1, shard_trajectories=True)
        text = render(dict(hostname='node', run='/tmp/run', phases=[dict(
            phase='train', pids=[123], status='running', outcomes_saved=0,
            outcomes_dropped=0, total=64, live=[live])]))
        self.assertIn('本批完成: 1/4（涉及 2 题，轨迹跨 rank 分配）', text)
