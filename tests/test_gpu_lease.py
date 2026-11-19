"""CPU-only regression tests for the shared GPU lease protocol.
Run: python3 -m unittest script.test_gpu_lease
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arl.utils import gpu_lease


class SharedLeaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, GPU_LEASE_DIR=self.tmp.name)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.mapping = patch.object(gpu_lease, 'device_uuids', return_value={'0': 'GPU-test-a', '1': 'GPU-test-b'})
        self.mapping.start()
        self.addCleanup(self.mapping.stop)

    def acquire(self, cards):
        lease = gpu_lease.try_acquire(cards, 'test', 'job')
        if lease:
            self.addCleanup(lease.release)
        return lease

    def test_remapped_indices_contend_for_same_uuid(self):
        lease = self.acquire(['0'])
        with patch.object(gpu_lease, 'device_uuids', return_value={'7': 'GPU-test-a'}):
            self.assertIsNone(self.acquire(['7']))
            self.assertEqual(gpu_lease.holders()['7']['owner'], 'test')
        lease.release()
        self.assertIsNotNone(self.acquire(['0']))

    def test_partial_acquisition_rolls_back(self):
        self.acquire(['1'])
        self.assertIsNone(self.acquire(['0', '1']))
        self.assertIsNotNone(self.acquire(['0']))

    def test_child_keeps_lock_after_parent_releases(self):
        lease = self.acquire(['0'])
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, pass_fds=lease.fds)
        try:
            lease.release()
            self.assertIsNone(self.acquire(['0']))
        finally:
            child.communicate(timeout=5)
        self.assertIsNotNone(self.acquire(['0']))

    def test_metadata_write_error_releases_all_fds(self):
        with patch.object(os, 'pwrite', side_effect=OSError('storage unavailable')):
            with self.assertRaises(OSError):
                self.acquire(['0', '1'])
        self.assertIsNotNone(self.acquire(['0', '1']))

    def test_umask_does_not_exclude_other_users(self):
        old = os.umask(0o077)
        try:
            self.acquire(['0'])
        finally:
            os.umask(old)
        self.assertEqual((Path(self.tmp.name) / 'GPU-test-a.lock').stat().st_mode & 0o777, 0o666)


if __name__ == '__main__':
    unittest.main()
