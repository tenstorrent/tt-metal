# Softmax verification artifacts

Artifacts from the Phase-0 verifier pass. The large files (`junit.xml`,
`test_results.json`, `verifier_report.json`, `pytest_stdout.log`,
`test_axes.json`) are gitignored — regenerate them locally with:

```bash
# 1. Run the golden suite (produces junit.xml, test_results.json, golden_results.txt)
eval/eval_test_runner.sh eval/golden_tests/softmax/ <results_dir>

# 2. Collect test axes alongside (axes_plugin doesn't run inside the runner)
PYTEST_AXES_JSON=<results_dir>/test_axes.json \
  pytest eval/golden_tests/softmax/ -p eval.axes_plugin --collect-only -q

# 3. Run the verifier
python3 -m eval.verify_supported <results_dir> ttnn.operations.softmax \
  --output <results_dir>/verifier_report.json
```

Committed here:
- `golden_results.txt` — one-line summary (PASSED/FAILED counts).
- `verifier_summary.json` — slim version of `verifier_report.json` keeping
  only the summary, by_category sizes, and the loud-category entries
  (supported_fail, xpass_drift, xfail_wrong_mode, no_axes_found). The
  full `xfail_expected` list of 1360 entries lives in the regenerated
  `verifier_report.json` if you need it.
