# expand_fused (Perf 1, graduated)

Idea: replace the expand stage (two transposed map matmuls for per-channel rstd / mean, then W-tile eltwise chains
a = rstd*gamma, m = mean*a, b' = m - beta) with two matmuls that emit a and b' directly against a reader-built
gamma map (+ beta map): a = rstd x gmap, b' = [mean*rstd, e] x [gmap; bmap].

Variants (each a full kernel set under its dir; `bench_expand.py` / `run_bench.sh` drive them, `results.jsonl` holds
the measured rows): v0 / v0s baseline, v3a..v3s candidates. v3b / v3d overflowed the kernel-config ring under --dev;
v3s (raw-LLK expand_matmul: overlapped last subblock, one pack reconfig per product) was the subagent's pick.

Graduation (coordinator) changed one thing from v3s: the subagent's gamma map was [channel x group lane]
(unpacked transposed), built by halfword stores, W*32 iterations of 2 loads + 2 stores. That reader loop
cost ~2 us more than the old row-0 copies, and on short shapes (one tile row per core) the combine round trip no
longer hid it (refiner_rm_ip_256x1536 10083 -> 12812 ns, c_wait_affine 3.2 us). The graduated gamma map is
[group lane x channel] (no transpose; the plain group map without gamma still runs transposed), so adjacent channels
of a group are adjacent halfwords and both maps are word copies: 256x1536 -> 9830 ns, every focus case faster.
See changelog.md "Perf 1".
