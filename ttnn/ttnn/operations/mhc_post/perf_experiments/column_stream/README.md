# mhc_post Perf 1 — column_stream (REGRESSION, not graduated)
Idea: a column-granular reader -> compute -> writer hand-off with trid-tracked reads, to cut the pipeline tail.
Bench: `tests/ttnn/unit_tests/operations/mhc_post/test_perf_experiments_column_stream.py`; `parse_perf.py` parses the perf CSVs.
Focus bf16 (3 repeats): baseline 223.1 / 67.4 / 269.9 us; G=1 299.8 / 76.6 / 335.5; G=B equals the baseline.
Details are in `../../changelog.md` (## Perf 1).
