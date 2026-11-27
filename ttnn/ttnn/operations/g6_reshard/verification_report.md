# Verification Report: g6_reshard

Phase 0 verification of the PULL reshard (`op_design.md`). The op is correct and clean: the
acceptance tests, the golden suite and the verifier CLI pass, every metric is exact, and
`TARGET − SUPPORTED = ∅`. The refinement queue therefore holds only measured perf
refinements, all on the flagged production case P.

## Code Review

Fixes made:

| # | Finding | Fix |
|---|---------|-----|
| 1 | `_build` looped `range(ISSUERS_PER_CORE)` while `describe_plan` read the same module constant on its own. The plan and the kernel list each read the knob separately, so they could disagree if the knob changed between the two reads. | `_build` now loops over `plan["issuers_per_core"]`. The knob is read once per build, and the issuer rows and the kernel descriptors come from that one value. |
| 2 | Uncommitted working-tree change to `g6_reshard_reader.cpp` moved the RT/common-arg loads and `noc` read inside the `RESHARD-ISSUE` zone. | Reviewed and kept. BB-21 requires every piece of work to fall inside a named zone, and the arg loads are per-block fixed cost. Passes acceptance (plain, `--dev`, profiler zones, NoC-event tracing) and golden. |
| 3 | `tests/.../g6_reshard/probes/test_floor_scratch.py` was an untracked scratch test. It built a kernel from `/tmp/g6_floor/empty_issuer.cpp`, which no longer exists, so 4 cases failed whenever the op's test directory ran. | Deleted. Its measured empty-kernel floor is recorded below (from the profiler reports it produced). |

Checked and found correct, no change:

- **Kernel API**: `void kernel_main()`, `api/dataflow/dataflow_api.h`, `TensorAccessor` (not `InterleavedAddrGen`), `noc_async_read` with the issuer's NoC, `noc_async_read_barrier(noc)` on that same NoC (BB-03).
- **Helpers**: no compute kernel and no CB. The dataflow helpers in `kernel_lib` (`local_copy_helpers_dataflow.hpp`, trid helpers, `McastArgs`) cannot express a remote, multi-packet, one-destination pull. The design's rejection table holds. No raw multicast or semaphore code exists that the mcast helper should replace.
- **Correctness corners**: the merge rule compares against both **shard** widths (`shard_widths_match && piece_width == in_block_width_tiles`), so cut-short tail shards are read at the shard stride. An empty issuer range (1-row block) skips the walk and runs a no-op barrier.
- **Generality beyond the suite**: four extra exact probes (`probes/probe_026..028.py`):
  - `(2,1,48,96)`, which has per-image tile padding (`d2 % 32 ≠ 0`, `H % 32 = 0`), into a COL_MAJOR output off the origin.
  - `(1,2,16,128)` into one core.
  - `(1,1,320,224)` with tails on both sides into a COL_MAJOR output at (5,0).
  - The same tensor as a COL_MAJOR input at (1,1), going into 70 one-tile shards.

  All four were bit-exact. The probe-026 third case was refused with `ValueError` for an illegal grid, which is the correct refusal.
- **Host cache (BB-07)**: the key holds padded shape, dtype, layout, both placements, device id, `ISSUERS_PER_CORE` and `NOC_POLICY`. On a hit only the two address common-args are rewritten. `custom_program_hash` is set. The `placements_one_process` and `repeat_determinism` loose cases pass.

Deferred: none. Nothing found needs architectural rework.

## Prompt Rules (`eval/prompts/g6_reshard.txt` → `## Rules`)

Every rule is MUST-class, and every condition applies at Phase 0. All hold:

| Rule | Evidence |
|------|----------|
| New tensor, every tile written once, as stored; no delegation, DRAM staging, third-core staging, unpack/convert | One `ttnn.generic_op`; `allocate_tensor_on_device` allocates only. Issuer row ranges partition each out block. No CB, no compute kernel. `device_num_programs = 1` per call in golden. The `input_unchanged` loose case passes. |
| Placements read at run time; every legal placement correct; no refusal of a legal one | Golden P/I0/I3/I5/O1 plus the four off-suite probes above. Refusals are the spec's validation list plus BB-05 only. |
| Core count is the blocking decision's | The out block holders run the kernel (P: 64 cores, O1: 56 cores). |
| BB-01 one program | `device_num_programs = 1`. |
| BB-03 nothing in flight at return | Barrier on the issuer's NoC. PULL issues no writes, atomics or semaphores. |
| BB-04 fits in every build flavour | Acceptance passes in the plain build, under `--profile` (zones), and with `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` (NoC tracing). Per-core RT args are 6 words; common args are 2. |
| BB-05 L1 ≤ 1347776 B | Out shard 34816 B (P) / 43520 B (O1). The host refuses with `RuntimeError` above the bound. |
| BB-06 deterministic | `repeat_determinism` loose case and `test_g6_reshard_repeat`. |
| BB-07 no rebuild, complete key | See the host-cache item above. |
| BB-08 plan report | `last_plan()`. |
| BB-17 few rectangles | One kernel core range, which is the out block-holder rectangle. |
| BB-21 zones + role dump | `RESHARD-ISSUE` (all work, now including arg loads) and `RESHARD-WAIT` (the barrier). `dump_roles()`. |
| BB-22 test scope | Production shape only. Placements vary. |

No soft (prefer/consider) rules exist in this prompt.

## Registry Conformance

- `INPUT_TAGGERS = {}`, `SUPPORTED = {dtype: [bfloat8_b], layout: [TILE]}`, `EXCLUSIONS = []`, `validate()` checks SUPPORTED then EXCLUSIONS and raises `UnsupportedAxisValue` / `ExcludedCell`. `g6_reshard()` calls `validate()` on its first line. The op file declares no `INVALID`.
- No drift fixes were needed. No XPASS and no wrong-mode xfails.
- Latent inconsistency, noted only: `validate()` hands taggers `(input_tensor.shape,)`, while `axes.classify_call` hands them `(scenario_dict,)`. With `INPUT_TAGGERS = {}` this has no effect. A future tagger must read the same `inputs` form in both places.
- **INVALID audit** (`eval/golden_tests/g6_reshard/feature_spec.py`): `INVALID = []`, which is correct. TARGET holds only `TILE_LAYOUT`, so the canonical bf8b + ROW_MAJOR entry cannot arise. There are no weights, no cross-tensor couplings and no "not yet" entries.

## Design Conformance

- **Algorithm / topology**: PULL. Out holders read their pieces straight into the out shard, with the piece walk on device and maximal runs. One dataflow source runs as two instances (RISCV_1, RISCV_0). This matches the design.
- **Work distribution**: the kernel range is out's block-holder rectangle, with one block per core and rows split over two issuers. The in-only holders (46 cores on P) run nothing, as designed. The route model (below) says P is link-bound rather than issue-bound, so they are not idle silicon that a design change would profit from.
- **NoC assignment**: the design shipped NoC0/NoC1 per issuer as a default plus a "NoC assignment" perf lamp. The implementer turned that lamp into a `NOC_POLICY` knob (`balanced`: greedy per-issuer moves over a dimension-order route model, `DM_DYNAMIC_NOC` when an issuer leaves its default NoC). This is measured (P 2407 → ~1600 ns) and in scope of the lamp.
- **Blocking-model fidelity**:
  - The block is the out shard, read from `memory_config` once on the host.
  - `issuers_per_core` has a single source (`ISSUERS_PER_CORE`, now read once per build, fix 1).
  - The run extent is derived on device and is never a literal.
  - Every geometry value is one CT arg derived from the placements.
  - No collapsed knob, no CB sized to an op dimension, no duplicated block literal.
- **Expression**: each issuer issues every run of its sub-block and then waits once. No per-unit synchronization exists on either kernel instance.

## L1 Ledger Audit

- **Currency**: no CBs are declared in the code (`cbs=[]`). The ledger's single row is the out shard reached by address, and its capacity expression `block_height_tiles · block_width_tiles · aligned_page_bytes` matches the allocation. Golden `device_l1_peak_bytes` confirms it: P 57664 = in shard 22848 + out shard 34816; I0 69632 = 2 × 34816.
- **Capacity vs live set**: the out shard is the result; its live set is every slot. No over- or under-sized buffer exists. The axis accounting spans both block axes, and so does the capacity.
- **Page format vs DEST**: no DEST on the path. Bfp8_b is stored as is. N/A, and stated in the ledger.
- **Disjoint lifetime**: there is no second buffer.
- **Bounds / closed form**: the symbol table bounds every symbol. The total is closed-form. BB-05 is enforced on the host.
- **Data-movement budget**: present. DRAM 0 B. It matches the implemented split: P crosses 1880 tiles and keeps 168 local, which the route-model probe also counts (`local_tiles=168`). PULL is a cheapest-traffic split. PUSH and BALANCED are `deferred` with a positive reason: 0 B delta and full coverage by PULL.
- **Block-size default (sharded → one block)**: holds. The block is the whole out shard. The only mechanism cap (one-packet reads ≤ 15 tiles) is unused, because `noc_async_read` splits packets itself.
- **Per-core footprint**: `block_height_tiles · block_width_tiles · aligned_page_bytes`. It scales only with `memory_config`'s shard. Every knob adds 0 B.

Nothing to fix and nothing to fold into a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_reshard/test_g6_reshard_precision_baseline.py`. BB-22 pins the shape to `(1,1,2048,1024)` bfloat8_b, so the baseline varies the placement pair instead of the shape. It covers the production tails (P), a COL_MAJOR input (I3), every tile crossing (I5), and a COL_MAJOR output off the origin with 5-wide over 3-wide shards (O1). PCC is from `comp_pcc`, the abs errors from `comp_allclose` plus direct reduction, and relative RMS is the RMS error over the reference's std (as in `eval/metrics.py`).

| Placement | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | Bit mismatches | got/true ratio median (p5, p95) |
|-----------|-----|-------------|--------------|------------------|---------|----------------|---------------------------------|
| P  | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| I3 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| I5 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| O1 | 1.000000 | 0 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |

**Assessment**: bit-exact on every element. The op moves stored tiles, shared exponents included, so there is no rounding to measure and no scale signature (ratio exactly 1.0 everywhere).
**Recommended tolerances**: exact (`helpers.TOLERANCES[bfloat8_b] = ("exact", None)`). No PCC/RMS gate applies.

## Verifier CLI Summary

Results: `eval_results/g6_reshard_verifier/` (`verifier_report.json`, `test_results.json`). Golden run: 12/12 passed, 0 hangs.

- supported_pass: 9 (5 cells P/I0/I3/I5/O1 + 4 loose cases)
- xfail_expected: 0 (TARGET == SUPPORTED, so there is no refused cell)
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (`test_regression.py` data-distribution tests: not registry-driven, all passed)

## Performance Baseline (Blackhole, AICLK 1350 MHz)

Device ns per placement, two metrics. The lead's stock figure for P (1.99 µs) is labelled "device kernel duration", which matches the Tracy `DEVICE KERNEL DURATION [ns]` column.

| Placement | Tracy DEVICE KERNEL DURATION, warm median of calls 2–6 (ns) | Golden `device_kernel_ns`, realtime profiler, single call (ns) | Busiest modeled link (tiles) | Max ingress port (tiles) |
|-----------|------|------|-----|----|
| P (perf target) | 1637 | 1980–2041 | 64 | 32 |
| I0 | 899 | 971–982 | 0 | 16 |
| I3 | 2227 | 2394–2416 | 112 | 16 |
| I5 | 1661 | 1780–1822 | 64 | 32 |
| O1 | 2873 | 3115–3118 | 128 | 40 |

Where P's time goes. Zones are in cycles, from the Tracy device log; probe-024 provides the route model.

- **Launch skew**: kernel start offsets across the 64 cores reach ~480 cycles (NCRISC) and ~610 cycles (BRISC). The skew tracks FW-start (go-signal arrival), so it is dispatch-side. The earlier empty-kernel profiler reports show the same floor: 1 core ~95 ns, 64 cores ~530–550 ns, 110 cores ~550 ns, with one or two kernels. BRISC kernels start ~90–140 cycles after NCRISC on the same core.
- **RESHARD-ISSUE**: median 770 (NCRISC) / 890 (BRISC), max ~1035. I0, with one 16-tile run per issuer, still spends 310–330 cycles here, so ~300 cycles per issuer is fixed cost before reads are in flight.
- **RESHARD-WAIT**: median 520–650, max ~1160.
- **Route model** (dimension-order, BH virtual coordinates):
  - Sources lie east/south of their targets, so the `balanced` policy puts 121 of 128 issuers on NoC1.
  - The busiest NoC1 link carries 64 tiles, about 1088 cycles at 64 B/cycle. That is the largest single term of P's ~2200-cycle span.
  - Per-piece NoC choice (probe-025, a host model only) lowers it to 56. Per-issuer choice cannot do better than 64.

Headroom estimate for P: ~15–25% in Tracy kernel time (skew ~500 + fixed ~300 + link ~950–1090 + tail latency). The launch skew is not an op lever.

## Recommendations

- **Metric alignment**: on the golden harness's realtime-profiler metric, P sits at 1.98–2.04 µs, which is parity with the stock 1.99 µs. On the Tracy kernel-duration metric it is 1.64 µs (18% faster). Perf refinements must gate on the same metric as the stock figure (Tracy `DEVICE KERNEL DURATION`) and also report golden `device_kernel_ns`. A win that shows on only one metric is not a win.
- **Not filed, no modeled headroom**:
  - BALANCED/PUSH (in-only holders issuing). Link load is set by (src, dst, NoC), not by the issuer, and P is link-bound, not issue-bound.
  - `issuers_per_core = 1`. Measured by the implementer: P unchanged, others worse.
  - Launch skew. It comes from dispatch, and PULL fixes which core holds which block.
- **Guard cases with the heaviest links**: I3 (112 tiles, a transpose pattern) and O1 (128) are the slowest cells. They are not perf targets, but any NoC-shaping change must not regress them, and they are where a link-load lever should show most clearly.
- **Environment**: `libtt_metal.so` needs `LD_PRELOAD=generated/mpix_stub/libmpix_stub.so` for direct `python3 -m eval.verify_supported` (undefined `MPIX_Comm_shrink`). `run_safe_pytest.sh` / `eval_test_runner.sh` handle it. `tracy-capture` exits with code 3 under `--profile`, but the device CSV (`generated/profiler/.logs/cpp_device_perf_report.csv`, `profile_log_device.csv`) is still written and usable.
