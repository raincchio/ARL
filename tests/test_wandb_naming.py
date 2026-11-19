"""ARM-style run allocation and W&B wiring without network or GPU startup."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from script import swe10
from arl.utils import ml_log


class WandbNamingTests(unittest.TestCase):
    now=datetime(2026,11,16)

    def test_preview_and_atomic_version_allocation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(swe10,'ROOT',Path(directory)):
            args=swe10.parser().parse_args([])
            preview=swe10.resolve_run_dir(args,now=self.now)
            self.assertEqual(preview.name,'qwen36_1116_v1')
            self.assertFalse(preview.parent.exists())
            def allocate(_):
                return swe10.resolve_run_dir(args,reserve=True,now=self.now)
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths=list(pool.map(allocate,range(4)))
            self.assertEqual(len(set(paths)),4)
            self.assertTrue(all(p.is_dir() for p in paths))
            self.assertEqual(swe10.resolve_run_dir(args,now=self.now).name,'qwen36_1116_v5')
            args.model='/models/Other/Policy-9B'
            self.assertEqual(swe10.resolve_run_dir(args,now=self.now).name,'policy_9b_1116_v1')

    def test_explicit_path_dates_and_training_name(self):
        with tempfile.TemporaryDirectory() as directory:
            args=swe10.parser().parse_args(swe10.normalize_key_values([f'log_path={directory}/my_model_%m%d_v9']))
            run=swe10.resolve_run_dir(args,reserve=True,now=self.now)
            self.assertEqual(run.name,'my_model_1116_v9')
            self.assertFalse(run.exists())
            run.mkdir()
            self.assertEqual(swe10.resolve_run_dir(args,reserve=True,now=self.now),run)
            configs=swe10.configurations(args,run)
            options=configs['train']['train']['options']
            self.assertEqual(options['wandb_project'],'ARL')
            self.assertEqual(options['wandb_name'],run.name)
            for key in ('baseline','evaluate'):
                self.assertNotIn('wandb_project',configs[key]['evaluate']['options'])
            args.wandb=False
            self.assertIsNone(swe10.configurations(args,run)['train']['train']['options']['wandb_project'])

    def test_logger_mirrors_metrics_with_path_name_and_project(self):
        sdk=MagicMock()
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',{'wandb':sdk}),patch.object(ml_log,'_host_ip',return_value='127.0.0.1'):
            log=ml_log.setup_logging(directory,wandb_project='ARL',config={'harness':'mini_swe_agent'})
            kwargs=sdk.init.call_args.kwargs
            self.assertEqual(kwargs['project'],'ARL')
            self.assertEqual(kwargs['name'],Path(directory).name)
            self.assertEqual(kwargs['dir'],directory)
            log.log_metrics({'ppo/grad_norm':.5},step=3)
            sdk.init.return_value.log.assert_called_once_with({'ppo/grad_norm':.5},step=3)
            log.close()
            sdk.init.return_value.finish.assert_called_once()
            self.assertEqual(json.loads((Path(directory)/'metrics.jsonl').read_text())['step'],3)

    def test_all_training_defaults_share_arl_project(self):
        from arl.grpo import GRPOConfig
        self.assertEqual(GRPOConfig(model_name='local',log_path='/tmp/check').wandb_project,'ARL')
        root=Path(__file__).resolve().parents[1]
        for conf in (root/'script').glob('*.conf'):
            for line in conf.read_text().splitlines():
                if line.startswith('wandb_project='):
                    self.assertEqual(line,'wandb_project=ARL',str(conf))
