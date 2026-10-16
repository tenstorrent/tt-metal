# Operation Requirements: matmul_reduce_scatter

## Definition
- **Formula**: `P = Σ_g A[g] @ W[g]` over the G devices of the collective group along `cluster_axis`; device at group position p keeps block p of the scattered dim. `scatter_dim=-2`: `out[p] = P[..., p·M/G:(p+1)·M/G, :]`; `scatter_dim=-1`: `out[p] = P[..., :, p·N/G:(p+1)·N/G]`.
- **PyTorch Reference**: `eval/golden_tests/matmul_reduce_scatter/helpers.py::pytorch_matmul_reduce_scatter(a_stacked, w_stacked, cluster_axis, scatter_dim)` (stacked `(R, C, ...)` per-device tensors, fp32 partials summed over the group axis, `torch.chunk(total, G, dim=scatter_dim)[p]`).
- **Import Path**: `from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter`
- **Function Signature**:
  ```python
  matmul_reduce_scatter(
      input_tensor: ttnn.Tensor,            # per device (..., M, K), rank 2-4, leading dims 1, TILE, DRAM interleaved
      weight: ttnn.Tensor,                  # per device (K, N), TILE, DRAM interleaved
      *,
      cluster_axis: int,                    # 0 or 1
      scatter_dim: int = -2,                # -2 (rows) or -1 (columns)
      topology: ttnn.Topology = ttnn.Topology.Linear,   # Linear | Ring
      num_links: int | None = None,         # None = all usable links per hop
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # default HiFi2, fp32_dest_acc_en=True (a floor)
      memory_config: ttnn.MemoryConfig | None = None,   # output; default DRAM interleaved
  ) -> ttnn.Tensor                          # per device (..., M/G, N) or (..., M, N/G), bfloat16, TILE
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Running the golden suite on this op (read before any refinement).** Run it as
> `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/matmul_reduce_scatter/ <dir>`. With the default
> L1 capture on, every executed cell except the first call on a freshly opened mesh fails with
> `L1ProfilingError: peak_total=0`. That is an instrumentation gap, not an op failure: this op is one heterogeneous
> MeshWorkload (a distinct program per chip), whose programs are compiled and their CBs allocated on thread-pool
> workers (`tt_metal/distributed/mesh_workload.cpp:150-157`), while graph-capture processors are `thread_local`
> (`tt_metal/impl/graph/graph_tracking.cpp:23`), so no CB allocation reaches the capture. See
> `verification_report.md` → "Golden-harness L1 capture". Also: after a `--dev` fabric run the next mesh open can
> fail on an Ethernet heartbeat; `touch /tmp/tt-device.dirty` before the next `run_safe_pytest.sh` / runner
> invocation so the wrapper resets the board under its lock. The acceptance file needs two invocations
> (`-k "not test_ring"` and `-k test_ring`) because its line and ring mesh fixtures cannot both be open.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED weight_dtype**: [bfloat16, bfloat8_b]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **SUPPORTED op-specific axes**: cluster_axis ∈ {0, 1}, scatter_dim ∈ {-1, -2}, topology ∈ {Linear, Ring}, num_links ∈ {1, 2}
- **Cores**: per chip, a 2D dual-multicast matmul rectangle (the compute grid minus one transport strip; 80-88 of the 99 rectangle cores on the QB2 perf shapes) plus 2·L port cores in the strip. Every chip runs its own program inside one `generic_op` dispatch.
- **Compute config**: caller fidelity; fp32 DEST + full sync always on for the matmul cores (fp32_dest_acc_en is a floor); every cross-device add in fp32 DEST, partials travel in bf16.
- **Golden baseline**: 411 supported_pass / 624 xfail_expected / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode, plus 6 registry-free regression tests passing (EVAL_CAPTURE_L1=0 run; `verifier_report.json`). The xfail bucket is exactly `dtype=bfloat8_b` ∪ `weight_dtype=bfloat4_b` (Refinement 1).
- **Accuracy**: rel-RMS 0.0047, PCC ≥ 0.99988, median 1 bf16 ULP on 4 shapes (HiFi2), see `verification_report.md`.
- **Perf (untraced, max over 4 chips)**: focus 142-144 µs (target 124.8), GLM 206-207 µs (target 207.3, met), MiMo 255-261 µs (target 220.6).

### [x] Refinement 1 — Numerical formats: bfloat8_b activations + bfloat4_b weights

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (activation) and `ttnn.bfloat4_b` to `SUPPORTED["weight_dtype"]`, closing `TARGET − SUPPORTED` on every axis. Per the Blocking Model these are knob-turns. They change the page format and tile size of `cb_act_k_window` (bf8b: 1088 B tiles) and `cb_weight_k_window` (bf4b: 576 B tiles), plus the `tA` / `tW` terms of `footprint()`, with no structure change. The output stays bfloat16 whatever the activation dtype (`check_output` enforces it). The golden bf8b-activation reference quantizes A like the device (`quantize_like_device`), so the cells score the op, not the input format. Keep `compute_kernel_config` behaviour unchanged: fidelity is the caller's, fp32 DEST stays forced on the matmul cores. Any cell that fails out of the box goes to `EXCLUSIONS` with its reason, not to a separate refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: it is the only generality gap, so it goes first. The three perf refinements below optimize bf16-activation / bf8b-weight configs that are already supported. It does not gate them, but land it first so they tune the final CB-format derivation. Things to check that the skill cannot see:
- The DRAM reads (`issue_a` / `issue_w`) and both multicast byte counts already use `tA` / `tW` from `buffer_aligned_page_size()`. Confirm bf4b's 576 B page satisfies the NoC / L1 alignment the window slot offsets assume (`a_win_bytes`, `w_part_off`).
- `choose_k_block` must see the smaller tiles. The bf4b-weight cases should keep their resident regime, and bf4b may let MiMo's W become resident, which overlaps Refinement 3.
- The matmul unpacks mixed formats (A bf8b × W bf4b). Check `matmul_block`'s `InitMode::Short` reconfig after the finish chains (bf16 arrivals / fp32 hold) for the new src formats.
- Tolerances for bf4b weights in the golden are `(pcc 0.99, rel_rms 0.08)`.

**Done when**: the golden suite (EVAL_CAPTURE_L1=0) shows the former `dtype=bfloat8_b` / `weight_dtype=bfloat4_b` xfail cells as `supported_pass` (or in `EXCLUSIONS` with a stated reason), 0 loud categories, and the acceptance suite, `test_regression.py`, `test_fabric_configs.py`, `test_ring_mock.py` and the precision baseline are still green.

### [x] Refinement 2 — Speed up the PERF FOCUS profile (Kimi K2.7 o_proj)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags the mandatory perf target: per device `A (1,1,640,2048) × W (2048,7168)`, `scatter_dim=-1`, activation bf16, weight bf8b, `topology=Ring`, `cluster_axis=1`, `num_links=2`, HiFi2, `fp32_dest_acc_en=False` (a floor), on the QB2 4-chip ring opened as 1x4 under `FABRIC_2D_TORUS_XY` with 14 KiB packets. Its perf goal is `target_us = 124.8` (roofline 93.6, unfused baseline 212.5). The soft precision gate is `rel_rms_threshold = 0.035`. Every knob of that contract is already in SUPPORTED, so optimize THIS config exactly, never a proxy. Measured today (untraced device-kernel ns, max over chips): 142-144 µs, with chips 1-3 at 132-135 µs and chip 0 at 142-144 µs (`verification_report.md` → Perf baseline). Take the lever from `ttnn/ttnn/operations/examples/master.md` and the design's Perf-lamps table. These are the measured facts that point at levers; the implementer chooses and measures:
- **Occupancy (knob-turn, `choose_grid`).** The row strip leaves an 11x9 rectangle, but the plan uses 10x8 = 80 matmul cores (`core_m=2, core_n=7, P=14`, subblock 1x7). The roofline assumes 98 cores, so 80 cores alone puts the matmul floor at ~85 µs, not 69. Measure grid / orientation / ragged-line alternatives (`PREFER_PORT_ROW_STRIP`, ragged `core_n`) against the port-to-Ethernet distance they cost.
- **Fused-increment cadence and chunking (knob-turn).** `P=14` is exactly 2 chunks of 7 tiles, so every second packet carries a fused atomic increment, which stalls the receiving router. The `INC_GROUP` variant was measured incorrect, so don't repeat it. A legitimate lever changes `P` (grid choice) so each core's share spans more packets.
- **K-block / buffer-depth co-tune (knob-turn).** `kb=16` (`KB_MAX`), `DEPTH_IN0/IN1 = 3`, `PREFETCH_STEPS = 1`. Use the master.md `double_buffer` / `compute_block_size` / `matmul_output_subblock` entries.
- **Chip-0 launch skew.** The implementer measured chips 1-3 at 132-134 µs and chip 0 at 161 µs. Measure traced (as the baseline was) before attributing it to the kernels. If it survives tracing, the ready fence or port placement on chip 0 is the region.

**Done when**: the measured device-ns of the focus case (max over the 4 chips, traced, production router config) improves and moves toward 124.8 µs. Its rel-RMS must stay ≤ 0.035, the golden suite must stay green (0 loud categories, EVAL_CAPTURE_L1=0), and there must be no regression across the config-spanning guard set. That set is one representative per distinct kernel path × placement: resident-A scatter -1, stream-W scatter -2, line placement (FABRIC_2D 2x2), ring placement (1x4 TORUS_XY), 1 and 2 links. `tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_perf.py --profile` is the measurement harness.

**Outcome**: Focus case, traced, max over 4 chips: 136.5-141 → 124.2-125.8 µs, at the 124.8 target. rel-RMS is 0.00454. GLM went 206-207 → 198.5 µs, and MiMo is unchanged within noise (244-262 vs 245-276). Untraced, chip 0 runs 8-25 µs longer than the others; under trace it matches them, so that skew was host launch skew, not kernel time.

The wins:
- Mirrored per-half entry orders: every entry sends half a block each way, so the transport tail drops from 131 to 115 µs and hides under the last matmul.
- FPU finish adds: 4.5 → 1.2 µs per relay/fin.
- Streamed fin write.
- Resident-regime fill levers: W reads on NoC1, 2 senders per line, look-ahead after the barrier, disjoint senders.

The case is now compute-bound. Of the 125 µs:
- 4 × 24.7 µs of matmul on 80 of the 99 rectangle cores, ~13% of it fp32 full-sync pack time;
- an ~11 µs DRAM fill in entry 0 (6.5 MB at ~175-200 GB/s aggregate);
- ~3 µs of finishes and a ~3 µs fin-write tail.

What I would try next, and why I did not:
- **More cores.** The column strip gives P=12 on 100 cores (24.7 → 21.3 µs per entry), but one port lands far from its Ethernet core and transport ends at ~150 µs. Ports would need to sit off-strip near each link.
- **A smaller first K-block** to shrink the fill. kb is one CB-window constant today, and kb 8 / 32 globally both measured worse.
- **16-bit half-sync DEST for the matmul**, so pack overlaps math. That changes the fp32-DEST precision policy, which is the design's call.
- **Refinement 4's sub-block sends.**

### [x] Refinement 3 — Speed up the rows-scattered MiMo-V2 profile (W residency L1 fit)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[2]` is the rows-scattered perf case: `A (1,1,2048,2048) × W (2048,4096)`, `scatter_dim=-2`, bf16 activation, bf8b weight, Ring, `cluster_axis=1`, 2 links, HiFi2, `fp32_dest_acc_en=True`, on the same 1x4 TORUS_XY placement. Perf goal `target_us = 220.6` (roofline 165.4, unfused baseline 400.7; measured today 255-261 µs untraced, max over chips); default precision gate. The case runs the **stream-xblock** regime: resident W needs ~1488 KiB per core against a ~1.33 MiB budget (`l1_ledger.md` worked example), so W crosses DRAM G = 4 times (~35.6 MB of weight traffic instead of 8.9 MB). The lever is to bring the resident-xblock regime inside the budget for this shape, then measure it against the tuned stream regime. Possible knobs: the stream windows `DEPTH_IN0` 3→2 on the non-resident operand, `DEPTH_HOLD` 2→1 when the entry walk does not lag, the per-core block `P=24` through the grid choice (88 of 99 rectangle cores), and `kb`. These are knob-turns on the planner's block surface, not a new topology. Use master.md `shared_input_reuse` / `double_buffer` as the measured references. **Folded L1-ledger findings** (no refinement of their own): the ledger's block-size-default departure (80-88 of 99 rectangle cores used) and the stream-regime DRAM delta both land here. Update `l1_ledger.md` (worked example, data-movement budget, regime rows) with whatever regime this case ends up in.

**Done when**: measured device-ns on the MiMo case improves toward 220.6 µs with the golden suite green, `_LAST_PLAN` reports the regime actually used, and there is no regression across the config-spanning guard set named in Refinement 2. The focus case must not regress either: it is already resident-A and must stay so.

**Outcome**: MiMo case, traced, max over 4 chips: 243-256 → **218.4-218.9 µs** (target 220.6). It now runs resident-xblock (`_LAST_PLAN`: resident, kb 8, A window depth 3, hold depth 1, grid 8x11), so W crosses DRAM once instead of 4 times (DRAM 48.2 → 21.5 MB). rel-RMS is 0.00487. Focus (124.1-125.8) and GLM (197.9-199.2) are unchanged, and both stay resident.

What made it fit (1488 → 1296 KiB against 1363 KiB):
- The second fp32 hold slot exists only for a lagged finish, and the mirrored resident walks never lag. `hold_depth()` now derives the depth from the schedule's lag decisions: −96 KiB.
- kb 8 instead of 16 for the streamed A window: −96 KiB.
- `RESIDENT_STREAM_DEPTHS = (3, 2)` is a live fallback knob: the resident regime tries depth 2 on the streamed operand before falling back to stream.

The three fitting plans (kb 8 / d3 / hold 1, kb 8 / d2 / hold 2, kb 16 / d2 / hold 1) measured the same within noise. The fill levers (1 sender, plain diagonal, look-ahead before the barrier, W on NoC0, prefetch 0 / 2) were all neutral or worse; unmirrored was +12 µs.

The case is now compute-bound. Zone timeline (chip 1):
- entry 0: 57.5 µs, the W fill (its first K-block alone is ~11 µs);
- 3 × 45.9 µs matmul entries on 88 of the 99 rectangle cores (block_m = 16 rows over 9 m-lines), with 3% ragged-n waste;
- ~10 µs of serial finish adds on the TRISCs;
- a ~7-13 µs fin-write tail (4 MB of output after the last finish).

With everything stubbed, transport alone is 113 µs, which hides under compute.

What I would try next, and why I did not:
- **A smaller first K-block** to cut the fill. kb is one CB-window constant; making it vary per step is a reader/compute schedule change.
- **Refinement 4's sub-block sends** to overlap the fill.
- **Using the 11 idle cores.** That needs a ragged `core_m` (2 / 1 rows) split, which is a grid-model change.

### [ ] Refinement 4 — Shrink the fill term: first send before a whole block is finished (overlap)

**Type**: perf

**Goal**: the fused roofline's `fill` term (one block of matmul before the first send = compute/G: 17.3 µs focus, 31.7 µs MiMo) is the part of the gap to target that no knob-turn removes. The prompt's design consequence says to "send at sub-block granularity … fine-grained sends shrink [fill] toward zero". The design's overlap lamp is `ENTRIES_PER_BLOCK = 2` (two waves along `n` per block). It is declared but asserted to 1 in `run()` because only one wave is built. This is a **schedule change**, not a knob-turn: entries double, and zone slots, arrival thresholds, port units and the lag decision are all re-derived from the schedule. That makes it a ⭐⭐⭐ T3 phase on its own. Target the PERF FOCUS shape (LOOSE_CASES[0]) first; MiMo is the second beneficiary.

**Verifier notes**: do this after Refinements 2 and 3. They settle the per-core block `P` and the regime, and the wave split divides `core_n` (or `core_m`), so it should split the tuned block, not today's. The deadlock-freedom argument (descending key per half) must be re-proved for the wave-split entry order. Write it in `op_design.md` next to the schedule tables before coding. `test_every_partial_exactly_once` and `test_regression.py` are the exactness guards for a schedule change.

**Done when**: measured device-ns on the focus case (and MiMo) improves over the post-Refinement-2/3 numbers with the golden suite green (0 loud categories), every exactness test green (acceptance `test_every_partial_exactly_once`, golden `test_regression.py`), and no regression across the config-spanning guard set.
