"""Regression coverage for containers sharing a host IP but not a PID namespace."""
import importlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, stdout='10.0.0.1\n')):
    service = importlib.import_module('arl.experiments.eval_service')


class ServiceIdentityTest(unittest.TestCase):
    def test_foreign_pid_is_not_probed(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / 'checkpoint'
            lock.write_text(json.dumps({'host': service.HOST, 'scope': 'another-container', 'pid': 123}))
            with patch.object(service.os, 'kill') as kill:
                self.assertEqual(service._lock_dead(lock), '')
                kill.assert_not_called()

    def test_dead_pid_in_same_scope_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / 'checkpoint'
            lock.write_text(json.dumps({'scope': service.SCOPE, 'pid': -123}))
            with patch.object(service.os, 'kill', side_effect=ProcessLookupError):
                self.assertIn('gone', service._lock_dead(lock))

    def test_stop_refuses_foreign_pid_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)
            (state / 'daemon.json').write_text(json.dumps({'scope': 'foreign'}))
            with patch.object(service, 'STATE', state), patch.object(service, 'daemon_pid', return_value=123), patch.object(service.os, 'kill') as kill:
                with self.assertRaises(SystemExit):
                    service.stop_daemon()
                kill.assert_not_called()

    def test_gpu_recheck_failure_releases_lease(self):
        from types import SimpleNamespace
        daemon = object.__new__(service.Daemon)
        daemon.a = SimpleNamespace(util=.85, ngpus=1)
        daemon.bad = set()
        daemon.jobs = []
        daemon.pools = [['0']]
        lease = service.gpu_lease.Lease(['0'], [], 'test', 'job')
        with patch.object(service.gpu_lease, 'holders', return_value={}), patch.object(service, 'free_devices', side_effect=[{'0'}, set()]), patch.object(service.gpu_lease, 'try_acquire', return_value=lease), patch.object(lease, 'release') as release:
            self.assertIsNone(daemon.pick('job'))
            release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
