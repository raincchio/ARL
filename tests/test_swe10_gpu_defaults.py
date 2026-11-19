"""All-visible-GPU defaults and small-batch distributed datum rebalancing."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from script import swe10
from arl.grpo import GRPOConfig, _validate


def _rebalance_worker(rank, init_file, directory):
    from datetime import timedelta
    import torch.distributed as dist
    from arl.algorithms.loop import rebalance_substep, _make_dummy_datum
    from arl.models.types import Datum, ModelInput, TensorData
    os.environ.update(RANK=str(rank), WORLD_SIZE='4', LOCAL_RANK=str(rank), GLOO_SOCKET_IFNAME='lo')
    dist.init_process_group('gloo',rank=rank,world_size=4,init_method='file://'+init_file,timeout=timedelta(seconds=30))
    try:
        batch=[]
        for i in range(4):
            if rank<2:
                n=40+rank*4+i
                batch.append(Datum(ModelInput.from_ints([1]*n),{
                    'advantages':TensorData.from_torch(torch.ones(n))}))
            else:batch.append(_make_dummy_datum(0))
        original=sum(float(d.loss_fn_inputs['advantages'].to_torch().sum()) for d in batch)/max(1,sum(bool(d.loss_fn_inputs['advantages'].to_torch().any()) for d in batch))
        moved,_=rebalance_substep(batch)
        active=sum(bool(d.loss_fn_inputs['advantages'].to_torch().any()) for d in moved)
        weighted=sum(float(d.loss_fn_inputs['advantages'].to_torch().sum())*float(d.loss_fn_inputs['resp_weight'].to_torch()[0]) for d in moved)
        totals=torch.tensor([original,weighted],dtype=torch.float64)
        dist.all_reduce(totals)
        assert torch.allclose(totals[0],totals[1]),totals
        assert active>0, (rank,active)
        Path(directory,f'rank{rank}').write_text(str(active))
    finally:
        dist.destroy_process_group()


class GpuDefaultTests(unittest.TestCase):
    def test_all_visible_gpus_and_dp4_are_defaults(self):
        env={k:v for k,v in os.environ.items() if k not in ('TRAIN_GPUS','SAMPLER_GPUS','CUDA_VISIBLE_DEVICES','KERNEL_RL_PACK_TOKENS')}
        with patch.dict(os.environ,env,clear=True), patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,stdout='\n'.join(map(str,range(16))))):
            args=swe10.parser().parse_args([])
            self.assertEqual(args.train_gpus,'all');self.assertEqual(args.sampler_gpus,'all')
            self.assertTrue(args.colocate);self.assertEqual(args.sampler_dp,4)
            swe10.validate_args(args)
            self.assertEqual(args.train_gpus,','.join(map(str,range(16))))
            self.assertEqual(args.sampler_gpus,args.train_gpus)
            env=swe10.runtime_env(args,True)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],args.train_gpus)
            self.assertEqual(env['KERNEL_RL_SAMPLER_DP'],'4')
            self.assertIn('--nproc_per_node=16',swe10.command(args,Path('/tmp/preview'),'train'))

    def test_explicit_lists_and_visibility_mask_are_respected(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2,3,6,7'}), patch('subprocess.run') as query:
            self.assertEqual(swe10.gpu_ids('all'),['2','3','6','7'])
            self.assertEqual(swe10.gpu_ids('0,1'),['0','1'])
            query.assert_not_called()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError):swe10.gpu_ids('all')

    def test_fewer_trajectories_than_ranks_requires_packing_and_rebalancing(self):
        cfg=GRPOConfig(model_name='policy',log_path='/tmp/test-all-gpus',env='swe',batch_size=1,group_size=2)
        for packing,rebalance in [('0','1'),('32768','0')]:
            with patch.dict(os.environ,{'KERNEL_RL_PACK_TOKENS':packing,'KERNEL_RL_REBALANCE':rebalance}):
                with self.assertRaises(SystemExit):_validate(cfg,16)
        with patch.dict(os.environ,{'KERNEL_RL_PACK_TOKENS':'32768','KERNEL_RL_REBALANCE':'1'}):
            _validate(cfg,16)

    def test_four_cpu_ranks_rebalance_two_task_ranks_without_changing_objective(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_rebalance_worker,args=(str(Path(directory)/'init'),directory),nprocs=4,join=True)
            self.assertEqual(len(list(Path(directory).glob('rank*'))),4)


if __name__=='__main__':unittest.main()
