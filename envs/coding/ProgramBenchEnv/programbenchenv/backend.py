import json
from pathlib import Path
import uuid

from sweenv.backend import BackendConfig, BackendError, DockerBackend, DockerWorkspace

ENTRY = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ENTRY/'upstream.json').read_text())


class ProgramBenchBackend(DockerBackend):
    def __init__(self, cfg, *, blob_dir, eval_cpus=4, branch_workers=1):
        if eval_cpus < 1 or branch_workers < 1:
            raise ValueError('ProgramBench evaluator CPUs/branch workers must be positive')
        super().__init__(cfg, worker_path=ENTRY/'worker.py', kind='ProgramBench')
        self.blob_dir, self.eval_cpus, self.branch_workers = blob_dir, eval_cpus, branch_workers

    async def rpc(self, action, **payload):
        return await super().rpc(action, manifest=MANIFEST, blob_dir=self.blob_dir, **payload)

    async def inventory(self):
        return (await self.rpc('inventory', timeout=120))['tasks']

    async def health(self, instance_ids=()):
        instance_ids = list(instance_ids)
        return await self.rpc('health', instance_ids=instance_ids, timeout=max(60, 20*len(instance_ids)))

    def workspace(self, task): return ProgramWorkspace(self, task)

    async def evaluate(self, task, submissions_dir, timeout):
        name = 'arl-pb-'+uuid.uuid4().hex
        try:
            return await self.rpc('evaluate', name=name, instance_id=task.instance_id,
                submissions_dir=submissions_dir, work_dir=self.cfg.work_dir,
                grade_timeout=timeout, eval_cpus=self.eval_cpus, branch_workers=self.branch_workers,
                timeout=timeout+180)
        finally:
            await self.rpc('close', name=name)


class ProgramWorkspace(DockerWorkspace):
    def __init__(self, backend, task):
        super().__init__(backend, task)
        self.name = 'arl-pb-'+uuid.uuid4().hex

    async def start(self):
        if self._closed: raise BackendError('ProgramBench workspace is closed')
        if self._started: return
        await self.backend._slots.acquire()
        self._slot, self._started = True, True
        try:
            await self.backend.rpc('start', name=self.name, instance_id=self.task.instance_id,
                memory=self.backend.cfg.memory, cpus=self.backend.cfg.cpus)
        except BaseException:
            await self.close()
            raise

    async def submit(self, timeout):
        await self.start()
        return await self.backend.rpc('grade', name=self.name, instance_id=self.task.instance_id,
            work_dir=self.backend.cfg.work_dir, grade_timeout=timeout, eval_cpus=self.backend.eval_cpus,
            branch_workers=self.backend.branch_workers, timeout=timeout+180)

    async def close(self):
        if self._closed: return
        try:
            if self._started: await self.backend.rpc('close', name=self.name)
            self._closed = True
        finally:
            if self._slot:
                self._slot = False
                self.backend._slots.release()
