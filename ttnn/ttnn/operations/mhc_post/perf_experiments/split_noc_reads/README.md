# mhc_post Perf 1 — split_noc_reads (GRADUATED as kernels/mhc_post_dm.cpp)
Idea: BRISC (the writer RISC, mostly idle) issues a share of the input reads, to even out per-core DRAM service.
The bench is `tests/ttnn/unit_tests/operations/mhc_post/perf_experiments_split_noc_reads_test.py`. `orig` is the pre-Perf-1 op kernels, verbatim.

To run it (put every argument inside -k with no spaces; the wrapper splits on spaces):

    SPLIT_PERF_SHAPES=... SPLIT_PERF_VARIANTS=orig,hF_n0_hf1_inc SPLIT_PERF_MODES=full \
      scripts/run_safe_pytest.sh --profile --run-all <test> -k test_perf
    python report.py <ops_perf_results.csv> run_order.jsonl

Results: `repeat_results.txt` (focus 3x), `sweep_results.txt` (domain), `fp32x_results.txt` (fp32-X carve-out),
`guard_orig.txt`, `graduated_whole_op.txt`. The verdict and domain are in `../../changelog.md` (## Perf 1).
