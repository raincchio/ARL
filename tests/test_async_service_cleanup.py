"""Exercise orphan cleanup with real process groups, without GPUs."""
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, stdout='10.0.0.1\n')):
    service = importlib.import_module('arl.experiments.eval_service')


class CleanupTest(unittest.TestCase):
    def test_exited_leader_with_term_resistant_worker(self):
        with tempfile.TemporaryDirectory() as d:
            ready = Path(d) / 'ready'
            worker = 'import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(%r).touch(); time.sleep(60)' % str(ready)
            parent = ('import subprocess,sys,time,pathlib; subprocess.Popen([sys.executable,"-c",%r]); '
                      '\nwhile not pathlib.Path(%r).exists(): time.sleep(.01)') % (worker, str(ready))
            proc = subprocess.Popen([sys.executable, '-c', parent], start_new_session=True)
            try:
                proc.wait(timeout=5)
                self.assertTrue(service._group_alive(proc.pid))
                service._kill(proc, grace=.1)
                self.assertFalse(service._group_alive(proc.pid))
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

    def test_reap_cleans_before_releasing_lease_for_success_and_failure(self):
        for rc in (0, 1):
            with self.subTest(rc=rc), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / 'failed').mkdir()
                lock = root / 'running'
                lock.touch()
                run = SimpleNamespace(path=root, out=root, giveup_streak=0)
                job = SimpleNamespace(proc=Mock(), lease=Mock(), run=run, name='000005', t0=0, log_file=root / 'log')
                job.proc.poll.return_value = rc
                daemon = object.__new__(service.Daemon)
                daemon.jobs = [job]
                daemon.a = SimpleNamespace(retries=2)
                daemon.save_jobs = Mock()
                events = []
                job.lease.release.side_effect = lambda: events.append('release')
                with patch.object(service, '_kill', side_effect=lambda p: events.append('kill')), patch.object(service, 'lock_path', return_value=lock):
                    daemon.reap()
                self.assertEqual(events, ['kill', 'release'])
                self.assertEqual(daemon.jobs, [])
                self.assertFalse(lock.exists())

    def test_failed_cleanup_keeps_job_and_lease(self):
        job = SimpleNamespace(proc=Mock(), lease=Mock())
        job.proc.poll.return_value = 0
        daemon = object.__new__(service.Daemon)
        daemon.jobs = [job]
        with patch.object(service, '_kill', side_effect=RuntimeError('still alive')):
            with self.assertRaises(RuntimeError):
                daemon.reap()
        job.lease.release.assert_not_called()
        self.assertEqual(daemon.jobs, [job])

    def test_refuses_own_group(self):
        with self.assertRaises(ValueError):
            service._kill_group(os.getpgrp())

    def test_orphan_record_without_process_still_removes_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lock = root / 'async_eval' / 'running' / 'base'
            lock.parent.mkdir(parents=True)
            lock.touch()
            (root / 'jobs.json').write_text(json.dumps([dict(scope=service.SCOPE, pgid=99999999, run=d, name='base')]))
            daemon = object.__new__(service.Daemon)
            daemon.save_jobs = Mock()
            with patch.object(service, 'STATE', root), patch.object(service, '_group_alive', return_value=False):
                daemon.reap_orphans()
            self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main()
