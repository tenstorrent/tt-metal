"""pytest plugin: run a suite with the bank_local_small rule applied in-process (the op's files untouched).
Use: PYTHONPATH=<this dir>:$PYTHONPATH scripts/run_safe_pytest.sh <tests> -p bls_rule_plugin"""
import os

os.environ.setdefault("BLS_VARIANT", "rule")

import bls_variants  # noqa: E402

print(f"BLS_INSTALLED {bls_variants.install()}", flush=True)
