# graduation_r19 (Perf 19)

Round-start op vs the graduated op, in one session, per FLUX.2 bh2x4 case: device ns per call, PCC vs torch on
output-row bands, and bit-identity (sha1 of the output bytes, ulp diff) against the first variant run.

- `op_base/`: the working tree at the round start (HEAD 14f9b142734 kernels + the uncommitted `l1_small_bytes` /
  `prepare_conv2d_weights_host` edits).
- `op_cand/`: the op after the round's four graduations (1bc58e8d770, f3c260c339d, 60b05b20c42, 91f90f36961).
- `test_final.py` (from `ring_fill_steady_split/test_rfss.py`):
  `GR_LOG=<label> GR_CASES=41,36,t34,x2 GR_VARIANTS=base,cand [GR_ORDER=alt GR_ITERS=3] scripts/run_safe_pytest.sh
  [--profile] --run-all ttnn/ttnn/operations/conv2d_3x3/perf_experiments/graduation_r19/test_final.py`;
  results in `runs/<label>.jsonl`. Case keys: `<n>` FLUX bh2x4 idx n (ROW_MAJOR), `t<n>` the same on TILE input,
  `x<n>` single-device (1x1 group) cases.
- `runs/bit1.jsonl`, `runs/bit2.jsonl`: all 17 focus cases and TILE P15, base vs cand: bit-identical.
  `runs/x2ab.jsonl`: TILE 1024² 128->3 and 256->128, alternating, 3 calls each (bimodal timing in both builds).
- Profiler zone hashes are 16-bit over "name,path,line": keep the two copies' locations collision-free
  (`../ablation_r19/zone_hash_check.py op_base op_cand`), or a `--profile` run throws.
