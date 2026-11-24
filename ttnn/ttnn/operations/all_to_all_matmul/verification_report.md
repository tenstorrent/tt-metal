# Verification Report: all_to_all_matmul

Verifier pass on 2026-11-24, QuietBox 2 (Blackhole, 1x4 snake ring under FABRIC_2D_TORUS_XY, 2 links).
Phase 0 is regime R1 (head-resident, L1 landing) with transport T1 (direct far quarter). It runs as one
`ttnn.generic_op` on a `MeshProgramDescriptor` per call.

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **`cb_weight` over capacity (L1 audit 2).** It was sized `WEIGHT_DEPTH (2) * Kt * block_n` on every active core. On Kimi and GLM every core processes exactly one weight block per call (1 head x 1 n-block), so the second buffer was never live: 128 KiB (Kimi) or 192 KiB (GLM) of dead L1 per compute core. The L1-output refinement must place its per-bank share of the output on the same cores. | `weight_depth = min(WEIGHT_DEPTH, heads_per_core_max * num_n_blocks)`, derived from the single `WEIGHT_DEPTH` knob. Harness `compare` on the Phase 0 Kimi case: fastest chip 94.473 → 94.421 us, slowest 103.779 → 104.191 us (verdict null, within noise). Measured L1 peak: Kimi 368,672 → 270,368 B; GLM 458,752 → 262,144 B. Ledger updated. |
| 2 | **`validate()` order.** Two caller errors (weight not TILE / interleaved; input not in DRAM) were raised *after* the SUPPORTED / EXCLUSIONS refusals. A call with both an unsupported axis and a contract violation therefore raised `NotImplementedError` instead of `ValueError`. | Moved both into `_check_caller_errors` (runs before the support check), as `op_design.md` "Registry contract" requires. |
| 3 | **Acceptance-test fixture duplication.** Each test module imported the golden `_fabric_mesh_cache` / `mesh_device` fixtures. A second module in the directory got its own session cache and tried to re-open the fabric under the first (`SetFabricConfig ... not allowed while devices are still open`). | The fixtures are imported once in the directory `conftest.py`. The acceptance test's import line was removed; its test logic is unchanged. |
| 4 | Stale docstring: the compute reader was documented on NoC0, but it runs on NoC1 (the implementer's measured NoC swap). | Docstring corrected. |

Reviewed with no change needed:
- **Helper usage.** Compute uses `compute_kernel_lib::matmul_block` with `WaitAndRetainOnLastBlock` A and `WaitAndPopPerKBlock` weight, so A is retained across n-blocks and popped once per head. There is one K-block, so `cb_out` doubles as the interm placeholder. The dataflow kernels use raw fabric / TensorAccessor; the design's "helpers considered and rejected" table holds (no shared operand across cores, so `McastArgs` does not apply).
- **CB sync.** `cb_a_landed`: reader pushes `slot_tiles` per slot; the helper waits `slot_tiles` and compute pops `slot_tiles`. `cb_weight`: `Kt*block_n` pushed and popped per (slot, n-block). `cb_out`: `out_subblock_h*block_n` per row group on both sides. `cb_port_chunks`: reader and sender share one batch rule (`need = min(left, batch - pos % batch)`), so their reserve / wait sizes match.
- **APIs.** `TensorAccessor` everywhere, `void kernel_main()`, `api/dataflow/dataflow_api.h` includes.
- **Blocking-model fidelity.** Every knob is a host constant defined once (`BLOCK_HEADS`, `WEIGHT_BLOCK_BYTES`→`block_n`, `WEIGHT_DEPTH`, `OUT_DEPTH`, `PORT_BATCH_CHUNKS`, `A_SLOT_BUDGET_BYTES`). It reaches the kernels as compile-time args, and every dependent CB size and trip count is derived from it. No CB scales unconditionally with an op dimension. `cb_a_landed` is the sanctioned predicate-guarded residency (`A_RESIDENT`, raises naming R3 otherwise).
  - `BLOCK_HEADS` is honestly asserted `== 1` ("not yet wired"). The design fixes it at 1 with a reason: heads share no operand, so a 2-head block only doubles `cb_weight`. I left it as is; no refinement depends on it.
- **Expression (does the schedule read as blocks?).** The reader issues one barrier per weight block and one per local slot. The writer flushes once per row group (`out_subblock_h x block_n`, 16 tiles at Kimi), which is compute's own push granularity. The port reader and sender use one barrier / flush per 8-packet batch. No kernel acquires, synchronizes and retires single tiles.
- **Parallelization.** Heads are spread row-major over the compute grid (grid minus the union of every chip's port cores = 101 cores): 96 busy at Kimi, all 101 at DeepSeek (27 with 2 heads), 64 at GLM. 2 x num_links = 4 ports per chip. The harness reports **not dispatch-bound** (max gap 0.52 us).
- **One dispatch per call.** One `generic_op`. Persistent resources (landing scratch, global semaphores) are allocated, and `synchronize_device` runs, only on the first call for a mesh or on a grow — never on a cached call.

### Prompt rules (`eval/prompts/all_to_all_matmul.txt` `## Rules`)
| Rule | Status |
|------|--------|
| MUST be one fused native implementation (no matmul / CCL ops) | ✓ one `generic_op`, own kernels, raw fabric |
| Head h = source h // Hl's local head h % Hl, times weight head h; rows `[q*S/G, (q+1)*S/G)` | ✓ `test_head_provenance`, `test_row_provenance` pass |
| Trace / back to back: no host sync inside the op; no stale rows / counts / fence state; no write before the destination started the call; deterministic | ✓ `test_back_to_back`, `test_deterministic` pass. The per-call ready fence is sent from the destination's own ports at kernel start. Arrival counters are re-armed by the consumer. *Advisory:* a call that **grows** the cached landing scratch (a larger shape than any before on that mesh) does one `synchronize_device` and reallocates. A trace must therefore be captured after a warm-up call at the largest shape; this is documented in `op_design.md` Risks. |
| Packets sized from the active max payload, never hard-coded | ✓ `packet_tiles = payload // tile_bytes`. Verified at 7232 B (3 tiles, perf cases) and 14400 B (7 tiles, golden cells). |

### Environment findings (not op defects; not refinements)
- **`test_small_packet_size` (4160 B payload) cannot run on this board/build.** `open_mesh_device` itself fails while the fabric firmware is compiled: `Program size (26848) too large for kernel config buffer (26624) on ACTIVE_ETH`. The fabric router's ERISC program does not fit at that payload, so the op is never reached. In the graded golden run the same test fails earlier, with `L1ProfilingError: ... has no device or mesh_device fixture`: the test asks for `_fabric_mesh_cache` directly, and the eval's L1 profiling plugin requires a `mesh_device` fixture. Both are golden-harness / fabric issues; `eval/golden_tests` is read-only for agents. **Please fix via `/golden-tests`** (give the test a `mesh_device`-named fixture) and check the 4160 B router build. Packet sizing is covered at 7232 B and 14400 B.

## Registry Conformance
- `INPUT_TAGGERS = {"alignment": tag_alignment}`, signature `(inputs, axes)`. `inputs = ((B, Hl, S, K), (1, H, K, N), G)`, with G read from the mesh in `validate()`. ✓
- `SUPPORTED` covers all 10 axes of the prompt (dtype, weight_dtype, layout, fp32_dest_acc_en, alignment, output_buffer, memory_layout, cluster_axis, topology, num_links). ✓
- `EXCLUSIONS = []`. ✓
- `validate()`: caller errors (`ValueError` / `RuntimeError`), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`); now in that order (fix #2). The entry point calls it first. ✓
- The op file declares no `INVALID`. ✓
- No XPASS drift, so no SUPPORTED auto-fixes were needed.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`. Correct: TILE is the only layout, so the canonical bf8b x ROW_MAJOR entry has no cell to cover. There are no cross-tensor couplings and no weightless canonicalization (the op is not norm-like). No issue.

## L1 Ledger Audit
- **Currency.** 4 CBs in code (`cb_a_landed`, `cb_weight`, `cb_out` on compute cores; `cb_port_chunks` on port cores) and 4 rows. Size expressions match the allocation; the `cb_weight` expression was updated for fix #1.
- **Capacity vs live set.**
  - *Over*: `cb_weight` was 2x its live set on 1-block cores; **fixed in place** (fix #1).
  - `cb_a_landed` above its live set on cores with fewer heads than `heads_per_core_max`, or after a grow: justified (one shard size per sharded tensor; persistent, grow-only).
  - *Under*: no buffer spans an axis whose capacity does not scale with it. `cb_a_landed` spans head x rows x K and scales with all three. `cb_out` streams rows in row groups and spans n.
- **Page format vs DEST width.** `cb_out` is a bf16 page with fp32 DEST: output ABI (the final rounding of the result), not an intermediate. The other CBs are dataflow payloads that never traverse DEST. No finding.
- **Disjoint lifetimes.** All CBs are whole-call and concurrent; each row carries a "cannot share" reason. No unjustified pair.
- **Bounds / closed form.** Every capacity term is in the symbol table with its bound and predicate (`A_RESIDENT`, the R2 bound on `Kt`, the semaphore cap, the DEST limit, `packet_tiles`). The total is closed-form.
- **Data-movement budget.** Present and consistent with R1: input / weight / output each cross DRAM once. The busiest directed link carries input/2. The cheapest-DRAM split (R6, output exchange) is a `deferred` regime row with a positive reason: it costs 4x the link bytes and only pays when N < K, which no perf case has.
- **Block-size defaults.** Held. The interleaved split spreads heads over the full remaining grid, and each core takes the coarsest block (whole head, `block_n = Nt` at every perf case).
- **Per-core footprint.** `L1_compute = cap_tiles*2048 + weight_depth*Kt*block_n*w_tile + OUT_DEPTH*sbh*block_n*2048`. The A term scales with `heads_per_core` and `block_m`; the weight term with `Kt*block_n` and `weight_depth`; the out term with `block_n` and `out_subblock_h`. Kimi 232 KiB, DeepSeek 400 KiB, GLM 316 KiB, deep-K / wide-N ≤ 772 KiB. Port core: `2*PORT_BATCH_CHUNKS*packet_bytes` (96 KiB at 7232 B, 224 KiB at 14400 B).

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_to_all_matmul/test_all_to_all_matmul_precision_baseline.py`: bf16 x bf16, fp32 DEST, HiFi4, DRAM out, randn input, weight randn/sqrt(K). Measured over all 4 devices.

| Shape (x / w) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 (bf16) | got/true p5 / median / p95 |
|---------------|-----|-------------|--------------|------------------|----------------|----------------------------|
| (1,4,128,64) / (1,16,64,128) | 0.9999986 | 0.0156 | 0.00117 | 0.00169 | 3 | 0.9968 / 1.0000 / 1.0033 |
| (1,8,256,128) / (1,32,128,256) | 0.9999986 | 0.0161 | 0.00117 | 0.00169 | 4 | 0.9968 / 1.0000 / 1.0033 |
| (1,24,640,128) / (1,96,128,512) Kimi | 0.9999986 | 0.0163 | 0.00117 | 0.00169 | 4 | 0.9968 / 1.0000 / 1.0033 |
| (1,4,128,4096) / (1,16,4096,128) deep K | 0.9999986 | 0.0132 | 0.00117 | 0.00169 | 4 | 0.9968 / 1.0000 / 1.0033 |

**Assessment:** The error is the bf16 output rounding of an fp32-accumulated K-sum, the same at every K up to 4096. The ratio is centred on 1.0 with a ±0.33% spread (bf16 half-ULP), so there is no scale bug. The exchange is bit-exact (`test_all_to_all_matmul_exchange_is_bit_exact` passes with identity weights). The golden cells agree: every passing cell has PCC 0.9999986 and rel-RMS 0.0017.
**Recommended tolerances:** defaults hold. bf16 / fp32 DEST: PCC ≥ 0.995, rel-RMS ≤ 0.04 (measured 0.0017, 24x margin). Re-measure under 16-bit DEST and bfloat8_b weights when Refinement 1 lands.

## Verifier CLI Summary
Golden run: `eval/eval_test_runner.sh eval/golden_tests/all_to_all_matmul/` (after the fixes), report `verifier_report.json` (copied next to this file).
- supported_pass: 16 (15 `test_op` tile-aligned shapes + the Phase 0 loose case)
- xfail_expected: 132 (weight_dtype=bfloat8_b, fp32_dest_acc_en=False, output_buffer=L1, alignment=m_non_aligned, plus the 3 model-config loose cases). All of them are covered by Refinements 1 and 2.
- invalid_skipped: 0 (INVALID empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 8 (`test_regression.py`: 7 pass; `test_small_packet_size` fails in the harness, see Environment findings)

Acceptance + precision suite: 14 / 14 pass (`scripts/run_safe_pytest.sh --dev`).

## Performance (Phase 0 config, Kimi, 7232 B; the eval harness, fastest chip per call)

| Measurement | Fastest chip | Slowest chip |
|-------------|--------------|--------------|
| Phase 0, 5 experiments (`perf/phase0_kimi_after_verifier.json`) | **94.4 us** | 104.2 us |
| roofline / target (LOOSE_CASES[1]) | 63.0 / 84.0 us | — |

Per-stage ablation (`A2AMM_ABLATE` payload stubs, 3 experiments each, `perf/abl_*.json`):

| Variant (stubbed) | Fastest | Reading |
|-------------------|---------|---------|
| none | 94.5 | full op |
| output DRAM write | 84.1 | the write costs ~10 us (≈ what an L1 output removes from DRAM) |
| weight reads | 74.1 | the weight reads cost ~20 us even though they were scheduled to hide behind the exchange |
| matmul | 87.1 | compute ~7 us exposed |
| write + weight | 64.8 | 15 us above the bare exchange: the compute tail after the last landing |
| write + weight + matmul (**exchange + fence alone**) | **49.3** | TRANSPORT in-op. The prompt measured a standalone 4-ring all-to-all of the same tensor at 53-56 us at 7232 B, so transport is already at the examples' measured rate. |
| port DRAM reads | 92.7 | port reads are not the bottleneck |

**Breakdown verdict:** ~45 us of the 94.5 us is local work *not hidden* behind the 49 us transport (**OVERLAP** is the first lever). The DRAM weight reads contend with the exchange, and the matmul plus write tail runs after the last landing. Transport alone is ~2.4x the links roofline (20.3 us), but at this packet size that is the router's fixed ~235 ns per packet (busiest link ~47-51% of 48.5 GB/s). Its own headroom is the fence (~5-6 us, 2-hop wait) and partial packets (~5%), modest next to OVERLAP. For the perf-flagged model config (L1 out, HiFi2, 16-bit DEST; roofline 32.3, target 43.0 us), the bare exchange (49 us) already exceeds the target. That case can only approach its target if the local work hides *fully* behind the exchange *and* the exchange itself shrinks.

## Recommendations
- Run the queue in order (op_requirements.md). Refinement 1 lands the full perf-focus contract (fp32_dest_acc_en=False, output_buffer=L1, plus bf8b weights in the same numeric bundle). It must ship with the grid filled and dataflow batched, because the perf rounds optimize that exact path.
- **L1 output footprint:** an L1-interleaved output places ~15.7 MB / (number of L1 banks) on *every* core, including port cores (≈ 140 KiB at Kimi). Add that share to the per-core budget check. Compute cores are at 232 KiB of CBs plus 40 KiB of landing scratch at Kimi; DeepSeek (400 KiB) is the tightest perf case. Wide-N cells with L1 output (`N = 8192`: 8 MB per chip) may be skipped as infeasible by the harness (`eval.oom`); that is uncharged, not a defect.
- **Unmeasured observations (no refinement):** (a) the compute reader on NoC1 / writer on NoC0 inverts the catalog default (`noc_placement`: reads NoC0 / writes NoC1). The implementer measured it a win *here* (it keeps NoC0 read responses for the port readers). Re-check it once the output goes to L1 and the write pattern changes. (b) `BLOCK_HEADS` is unwired (asserted 1) by design.
- `_topology_plan` falls back silently to row-0 port placement on any exception. The design measured that placement at 40.6 / 30.5 GB/s, so a silent fallback would look like a perf regression. Consider logging the fallback if it ever triggers (it does not on this board).
