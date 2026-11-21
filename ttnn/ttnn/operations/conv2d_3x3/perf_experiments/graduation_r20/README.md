# graduation_r20 (Perf 20)

Round-start op vs the graduated op, in one session, per case: device ns per call, PCC vs torch on output-row bands, and
bit-identity (sha1 of the output bytes, ulp diff) against the first variant run (with `GR_ORDER=alt` the first variant
alternates per case).

- `op_base/`: the working tree at the round start (HEAD 6b8031bb8b8 + the uncommitted `l1_small_bytes` /
  `prepare_conv2d_weights_host` edits; two comment lines added at the top of its compute kernel so its profiler zone
  hashes do not collide with the other copies').
- `op_prev/`: after the ragged tail and the leftover-L1 output CB (e0918479b0c, e14b9f0ecc6).
- `op_cand/`: the final op (through eb9963f22f6).
- `test_final.py` (from `graduation_r19/test_final.py`, plus `op_prev` and case `x4` = TILE 256x256 32->512);
  `gr.sh <label> <cases> [variants]` runs it with `--profile` and appends window medians to `runs/gr_results.txt`.
- Runs: `g1a` / `g1b` base vs ragged + CB (all 17); `g2a` / `g2b` the row stream vs `op_prev`; `g3` P14 / P7 repeat;
  `g4` kernel-row K-blocks on the kh-reuse cases; `g5` the TILE 256x256 32->512 regression; `g6` the TILE C_in 32
  twins; `fin_a` / `fin_b` round start vs final (the changelog table).
