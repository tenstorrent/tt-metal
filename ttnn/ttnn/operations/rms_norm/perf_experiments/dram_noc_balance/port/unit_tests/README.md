The port was validated by running a copy of `tests/ttnn/unit_tests/operations/rms_norm/` (all test files) with
`conftest.py` here swapping the port's descriptor in for the session (`--import-mode=prepend`). The copied test
files were removed after graduation (they duplicated the canonical suite, which now runs against the graduated op);
to re-run, copy the canonical test files next to this conftest.
