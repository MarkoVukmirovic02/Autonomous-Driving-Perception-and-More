"""Fail-fast runner for core IMM checks and the stop/restart scenario."""
from pathlib import Path
import subprocess
import sys

CHECKS = ('check_imm_single_model', 'check_imm_conversions', 'check_imm_interaction',
          'check_imm_low_speed', 'check_imm_measurements',
          'imm_stop_restart')


def main():
    project_root = Path(__file__).resolve().parents[2]
    for name in CHECKS:
        print(f'\nRunning {name}', flush=True)
        subprocess.run([sys.executable, '-m', 'Perception.experiments.' + name],
                       cwd=project_root, check=True)
    print('\nAll IMM check groups and the stop/restart scenario passed.')


if __name__ == '__main__':
    main()
