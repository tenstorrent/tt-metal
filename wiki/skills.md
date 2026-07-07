# Skills & lessons — LTX VAE decode / neighbor_pad halo (BH 4x8)

Distilled, transferable lessons from optimizing the LTX-2 VAE decode's halo path on Blackhole 4x8.
Audience: TT dev touching conv3d halo exchange, copy-free activation pipelines, or fabric ops.

## Performance reasoning

- **A coalesced bulk DRAM copy is CHEAP; strided copy-avoidance is often SLOWER.** The whole
  "eliminate the interior copy" premise was disproven by measurement: full literal copy elimination
  in the decode is achievable and bit-exact (PCC=1.0) but **452ms vs 435ms shipped** (+17ms). The
  copy-avoidance machinery (border-only scatters, RM↔TILE layout churn, per-block chaining) costs more
  than the coalesced copies it removes. Ladder: 477 (halo-read) → 443 (repack everywhere) → 435
  (conv2-direct copy-free, SHIPPED) → 452 (full elimination). **Measure before assuming a copy is the cost.**

- **The compact halo buffer exists FOR fabric coalescing, not by accident.** `np_w_mux_writer.cpp`
  ships contiguous compact pages as one fabric packet (1.4→12.5 GB/s). The padded border is strided,
  so "fabric writes padded directly" (Architecture A) loses coalescing → slower. Don't remove the
  compact staging buffer to "save a round-trip"; you'd break the bandwidth path.

- **Profile before optimizing the wrong thing.** In this decode: Conv3d 32%, Permute (depth-to-space)
  19%, LayerNorm 18%, BinaryNg 15% = **84%** of device work. Halo transport (np_halo 1.9% +
  halo_scatter 4.1%) = 6%. Optimizing the halo fold has a hard <~2% ceiling. `tt:profiler` on the
  traced decode gives the per-op device-FW breakdown that settles this in one run.

- **What CAN overlap for free: work with no dependency on the fabric exchange.** The interior copy
  (input→padded interior) doesn't depend on the halo exchange, so it runs concurrently on free cores
  in the same program — no barrier, no hang risk. That's the one real win here (435→432.7ms, ~0.5%).
  The border DOES depend on the exchange, so it can't overlap without a barrier.

## Fabric op mechanics (neighbor_pad_halo)

- **Core layout:** fabric cores occupy **column 0**; the NP sender workers + mux cores occupy
  **columns ≥1** (`np_core_grid(1, num_h_fabric_cores)`). Free cores for a co-resident kernel =
  `cols≥1` minus (`np_worker_core_ranges` + mux h/w workers + mux cores) — subtract ALL of them via
  `CoreRangeSet::subtract` chain, or you hit a CB-index-0 conflict (workers already own c_0 there).

- **Corner routing is a two-hop baked to the compact layout:** corners go H-exchange → compact
  H-section → W-fabric → compact W-section. Any "write padded directly" scheme must re-route corners;
  this is what killed Architecture A twice.

- **mux (recv-authority `H_SIGNAL_W_RECV`) barrier is race-free; non-mux (send-done) barrier races
  the corner two-hop under perturbed timing.** Production decode uses mux → fine. If you ever need
  non-mux copy-free, switch it to the recv-authority barrier.

- **Op plumbing checklist** for adding a param/mode to a device op: params struct (+ `attribute_names`
  AND `attribute_values` — keep tuple sizes equal or it won't compile) → tensor_args → public op
  (.cpp/.hpp) → prim entry → **nanobind kwargs** (forgetting this = Python `TypeError` at call, not
  build) → `override_runtime_arguments` (refresh per-dispatch DRAM addresses, else stale on traced replay).

## Traps that cost real time this session

- **A gated code path can silently never fire.** The chaining/copy-free-decode work gated on
  `_FUSE_MIDBLOCK_NORM_ADD` (default off) — every "PCC=1.0, 435ms" run for many iterations was
  re-measuring the *baseline against itself*. **Confirm the path actually executes** (a one-shot debug
  print) before trusting a measurement, especially "it didn't change" results.

- **Fused-norm-add × copy-free had a pre-existing FATAL:** `forward_residual_sum` is TILE-only but got
  `h` in ROW_MAJOR → layernorm padded-shape mismatch on non-tile-aligned W. Fix: `to_layout(h, TILE)`
  before the residual sum. Two features that each "work" can be broken in combination and never
  co-tested.

- **Traced-op semaphore reset** is the hard part of any in-program barrier: the sem must zero between
  dispatches without racing the producers' increments (ops self-reset their sems at kernel end — mirror
  that). This, plus the **total-T problem** (a reader core knows only its row-slice, not global T, so it
  can't compute global compact section bases `2·T·pH·Wd + …` without extra args), are the two concrete
  blockers to folding the *border* scatter into np_halo. See `np-halo-4x8.md` for both routes.

## Build / workflow

- `cmake --build build --target ttnn` then install **both** `--component tt_pybinds` AND
  `--component tar` (from `build_Release`) or Python won't pick up the new op.
- Pre-commit hooks need the venv on PATH: `PATH="…/python_env/bin:$PATH" git commit …`. clang-format /
  black reformat then abort the first commit — re-stage and re-commit.
- `wiki/` is in `.git/info/exclude` on this branch; force-add docs (`git add -f wiki/…`).
- Serialize all device jobs through tt-device-mcp; a SIGKILL'd/wedged job leaks the chip lock and the
  next job hangs at device-open (reset required — user owns resets).

## Process lesson

When a directive's premise ("if faster") is disproven by measurement, say so with the numbers and
stop — don't grind the same conclusion, and don't ship unvalidated fabric-kernel surgery to satisfy a
literal checklist. Deliver the measured win (shipped) + a documented completion plan with its blockers.
