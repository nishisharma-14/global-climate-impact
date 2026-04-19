import os
import subprocess


def test_data_quality_runs():
    # sanity check: script should exit with code 0 when dataset exists
    if not os.path.exists('global_warming_dataset.csv'):
        # skip test if dataset not present in CI
        return
    res = subprocess.run(['python', 'data_quality_checks.py'], capture_output=True, text=True)
    assert res.returncode == 0
    assert 'DATA QUALITY REPORT' in res.stdout
