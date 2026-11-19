# all_gather_rms_norm changelog

## Phase 0 (ttnn-implementer)
- Built R1 mirror_resident_gather end to end: fabric plan (probe placement, balanced ring / line schedules,
  ports_union excluded from the compute grid of every chip), port reader / sender (ready fence, relay, one fused
  arrival increment per (wave, shard) entry, landing notifications), member reader / compute / writer (base-op local
  norm with symmetric partial exchange), LN pivot + Chan combine, dense assembly for `w_non_aligned`.
- SUPPORTED: the full from-base list (fp32 / bf16 / bf8b, fp32_dest_acc_en both, all three alignments, rank 2-4,
  gamma TILE / ROW_MAJOR fp32 / bf16, beta, INTERLEAVED / HEIGHT / WIDTH sharded) × norm_type both × cluster_axis
  0/1 × Linear / Ring × 1-2 links. EXCLUSIONS: fp32 ∧ fp32_dest_acc_en=False; bf8b ∧ {w,h}_non_aligned.
- Tests: acceptance 19/19; golden sample (~430 cells over every INPUTS shape) pass; test_ring_mock 128/128;
  loose (perf) cases pass.
- Perf (eval ccl_perf, 5 exp, fastest chip): focus 640x384 rms 22.23 -> 21.25 us (direct gamma reads finished
  after the exchange + out unit 4); indexer 640x32 ln 22.68 -> 16.16 us (LN FPU fast path + k = 1 combine).
- Deferred: owner combine for k > 8 (k capped at 8); compact gamma pages; zero-copy own shard; credited landing.
- Known environment issue: with `run_safe_pytest.sh --dev` (watcher), re-opening the fabric mesh between tests times
  out in the Ethernet-core teardown ("Timed out while waiting for active ethernet core"); the same tests pass without
  `--dev` and a single `--dev` test passes.
