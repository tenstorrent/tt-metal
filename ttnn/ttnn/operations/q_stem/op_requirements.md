# Operation Requirements: q_stem

## Definition
- **Formula** (per group of G devices along `cluster_axis`; position p holds `q_b[p]`):
  `q[p] = x @ q_b[p]`; `heads[p] = q[p].view(B,S,Hl,Dn+Dr).permute(0,2,1,3)`; device q' receives row quarter q'
  of every p (output head `h = p*Hl + j`); `out[q'] = cat([nope[q'] @ wkv_b1[h], rot(rope[q'])], -1)` with
  `rot = identity` (NoPE) or `rope*cos + rotate_half_meta(rope)*sin` (RoPE). Output per device `(B, H, S/G, N+Dr)`.
- **PyTorch Reference**: `eval/golden_tests/q_stem/helpers.py::pytorch_q_stem`
  (acceptance-test twin: `tests/ttnn/unit_tests/operations/q_stem/test_q_stem.py::reference`).
- **Import Path**: `from ttnn.operations.q_stem import q_stem`
- **Function Signature**:
  `q_stem(input_tensor, q_b_proj_weight, wkv_b1_weight, *, cluster_axis, cos=None, sin=None,
  topology=Topology.Ring, num_links=2, output_layout=ttnn.TILE_LAYOUT, compute_kernel_config=None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Perf-1 anchor.** The PERF FOCUS (`feature_spec.LOOSE_CASES[0]`) is Kimi K3.1 as the model runs it:
> latent (1,1,640,1536) bf16 **in L1**, q_b_proj (1536,4608) **bfloat8_b**, wkv_b1 (96,128,512) bf16, output
> (1,96,160,576) **ROW_MAJOR** in DRAM, NoPE, **HiFi2, fp32_dest_acc_en=False**, 4-ring, 2 links, 7232 B packets;
> roofline 73.8 µs (DRAM), target 98.4 µs. Refinements 1 + 2 land every axis value that contract needs
> (q_b_proj_dtype=bfloat8_b, fp32_dest_acc_en=False, input_buffer=L1 → R1; output_layout=ROW_MAJOR → R2), so the
> first perf slot (Refinement 3) measures and optimizes exactly that config.

> **Measured Phase-0 breakdown** (verifier, Phase-0 Kimi case — bf16, HiFi4, fp32 DEST, TILE out, latent in DRAM —
> eval `ccl_perf` protocol, fastest chip): full op **190.5 µs** (slowest chip 207.3; not dispatch-bound; roofline
> 90.6, target 120.8). Stubbing the fabric payload (TRANSPORT): 193.1 (**no change — the exchange is fully hidden**);
> stubbing the output writes: 168.1 (−22.7); stubbing stage 3 + output + wkv reads: 137.7 (−53.1). Stage 1
> (x multicast + q_b slice + matmul) ends at ~138 µs and the stage-3 tail on the stage-1 cores follows it. The local
> part dominates with headroom against the roofline, so the queue's perf slots start with the OVERLAP tail and the
> local stage-1 levers; no TRANSPORT refinement is filed while the payload ablation stays inside noise
> (re-checked at every perf slot — see Refinement 6's Done-when).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]; **q_b_proj_dtype**: [bfloat16]; **wkv_b1_dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]; **output_layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **SUPPORTED op-specific axes**: fp32_dest_acc_en ∈ {True}, rope_mode ∈ {nope}, input_buffer ∈ {DRAM},
  memory_layout ∈ {INTERLEAVED}, cluster_axis ∈ {1}, topology ∈ {Ring}, num_links ∈ {2}
- **Cores**: per chip — 4 port cores + 1 x injector on row 0; stage 1 column-split over a W1×R1 sub-rect of the
  99-core compute rect (Kimi 9×8 = 72, 2 columns per core); stage 3 heads over all 99 compute cores (load-weighted).
- **Compute config**: `compute_kernel_config` exposed; default HiFi4 + fp32 DEST (`default_compute_kernel_config()`);
  only `fp32_dest_acc_en=True` accepted.
- **Golden baseline**: 15 supported_pass / 0 supported_fail / 1077 xfail_expected / 0 xpass_drift / 0 xfail_wrong_mode
  (1103 collected; 11 regression tests untagged: 8 pass, 3 RoPE xfail; `test_small_packet_size` fails only in the
  graded runner — an L1-profiling-plugin harness error, passes under `run_safe_pytest.sh`).

### [ ] Refinement 1 — Numerical configurability + L1 latent (weights bfloat8_b, 16-bit DEST, input_buffer L1)

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["q_b_proj_dtype"]` and `SUPPORTED["wkv_b1_dtype"]` (independently),
`False` to `SUPPORTED["fp32_dest_acc_en"]`, and `ttnn.BufferType.L1` to `SUPPORTED["input_buffer"]`. Weight CBs
(`cb_qb_slice`, `cb_wkv`) take the weight's page format (Bfp8_b, 1088 B pages — `Geometry.qb_tile_bytes` /
`wkv_tile_bytes` already derive from the dtype); intermediate formats follow `fp32_dest_acc_en`. The L1 latent is a
source change for the injector's `TensorAccessor` (interleaved over L1 banks). Cells that fail out of the box go to
`EXCLUSIONS`, not their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Anchor refinement for perf 1: three of the four model-config axis values the PERF FOCUS needs land here. Build it
  at the performance-conformance bar — same grid fill and batching as Phase 0, no single-core or per-tile stubs on
  the new paths.
- L1 ledger finding folded in: `cb_q_partial` is Float32 unconditionally (`q_stem_program_descriptor.py`). Under
  `fp32_dest_acc_en=False` that is a wide page behind 16-bit DEST — derive its format from the flag. `cb_q_slice`
  stays bf16 under both (q is the exchanged payload; the prompt requires it to arrive bit-exact as stored).
- DEST capacity is already derived (`_dest_limit`: 4 fp32 / 8 bf16 tiles). With 16-bit DEST the stage-3 subblock
  `sbw3` can reach 8; check the helper's subblock choice actually widens.
- Bfp8 q_b halves `cb_qb_slice`, so `S1_COLS_PER_CORE` / the stage-1 rect (lamp L7) may move. Don't re-tune it here
  (that is Refinement 6), but keep `check_fits_l1` honest. It already reads the L1 latent's bank share from the live
  allocator.
- `custom_program_hash` already includes the latent's buffer type (verifier fix). Extend it for anything else this
  refinement makes vary (it already covers the weight dtypes and `fp32_dest_acc_en`).
- 16-bit DEST across K = Q = 1536 at HiFi2 is graded at the `(bfloat16, False)` default tolerances
  (PCC ≥ 0.99, rel RMS ≤ 0.13); the model-config loose cases grade PCC at `LOWER_FIDELITY_PCC`.

**Done when**: every golden cell whose only non-Phase-0 axis values are among {q_b_proj_dtype=bfloat8_b,
wkv_b1_dtype=bfloat8_b, fp32_dest_acc_en=False, input_buffer=L1} passes (or sits in a justified `EXCLUSIONS` entry),
verifier categories clean, acceptance + regression green.

### [ ] Refinement 2 — Row-major query output (in-kernel untilize + stick writer)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["output_layout"]`. Stage 3 adds the `untilize_row_block` the
design names (`untilize<s3_block_n_tiles,…>` on `cb_prefix_out`, `untilize<Dr_t,…>` on `cb_rope_out`, into RM CBs).
The writer writes each output row's stick, `(N+Dr)*2` B, as two segments (prefix | rope) straight to the
ROW_MAJOR DRAM output. The output tensor is allocated ROW_MAJOR.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **No manipulation-op fallback is allowed here.** The prompt's Rules state "When output_layout is ROW_MAJOR: the
  op MUST write the row-major query from its own kernels. It MUST NOT produce a TILE result and convert it with
  ttnn.to_layout, ttnn.untilize…". There is no `[~]` wrapper escape hatch for this refinement: a wrapper would be a
  correctness violation, not a partial.
- Second anchor refinement: after it lands, every axis value of the PERF FOCUS contract is SUPPORTED. The RM
  writer is on the critical path: output writes are already −22.7 µs of exposed time at TILE (ablation above). Write
  one row block's 32 sticks per barrier, with NoC1 writes batched (dm shortlist: output write pass 41.7 µs at 96
  cores, `dm_estimates.json`), not a barrier per stick.
- Add `output_layout` to `custom_program_hash`.
- RoPE (Refinement 5) and m_non_aligned (Refinement 4) both extend this stage-3 → writer path. Keep the untilize as
  a separate block op after the prefix/rope producers, so a rotated rope tile slots in front of it unchanged.

**Done when**: every golden cell whose only non-Phase-0 axis values are among Refinement 1's values plus
`output_layout=ROW_MAJOR` passes. That includes the PERF FOCUS loose case
(`test_op_loose[1x1x640x1536-…-fp32_dest_acc_en=False-input_buffer=L1-…-output_layout=ROW_MAJOR-q_b_proj_dtype=BFLOAT8_B-rope_mode=nope-…]`).
Verifier categories clean.

### [ ] Refinement 3 — OVERLAP: hide the stage-3 tail of the PERF FOCUS behind stage 1

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` (PERF FOCUS, Kimi K3.1 model config — HiFi2, bfloat8_b q_b_proj, 16-bit
DEST, ROW_MAJOR out, L1 latent; roofline 73.8 µs, target 98.4 µs) is the target. Measure it with
`eval/golden_tests/q_stem/perf_measure.py` first, and re-take the ablation breakdown on this config (the
`QSTEM_ABLATE` hook: `SEND_PAYLOAD`, `OUT_WRITE`, `S3,OUT_WRITE,WKV_READ`). In Phase 0 the stage-3 work of the
heads placed on the stage-1 cores, plus their output writes, runs *after* stage 1 (~52 µs of tail on a 190 µs
call). This is the chain from the last landing to the output write (**OVERLAP**). Levers, from the design's lamps
and `ttnn/ttnn/operations/examples/master.md`:
- **Head placement / stage-1 core count** (core-assignment knobs `S1_COLS_PER_CORE`, `S3_COST_FACTOR`,
  `FREE_CORE_START`, lamps L7 / L8): move stage-3 rows off the cores that finish stage 1 last.
- **Interleave landed stage-3 rows into the stage-1 loop** on cores that own both, so their output writes overlap
  stage 1 (lamp L2).
- **Un-collapse `s3_block_m_tiles`**: it is pinned to 1 by a host assert because the landing row layout interleaves
  `[nope | rope]` per row. Lay a head's rows out nope-contiguous so stage 3 can run M > 1 blocks (amortizes the
  per-row matmul init and pack; larger output subblocks: master.md `matmul_output_subblock`, `compute_block_size`).
- The RM writer from Refinement 2: batch the sticks (master.md `noc_placement`, `double_buffer`).

No SUPPORTED change.

**Calculator**: the op model does not express stage 3 (a batched per-head matmul, per `op_design.md` Traffic
ranking), so this lever is ranked by measurement (the ablation above). Data-movement bound for the moved work:
`estimate.py` output-write group, rank 1 = 41.7 µs at 96 cores, NoC1, batch 9 (`dm_estimates.json`). Hiding it
behind stage 1 is the lever's ceiling.

**Done when**: measured fastest-chip device time on the PERF FOCUS improves (`python -m eval.ccl_perf compare`
verdict *win*: fastest lower beyond noise, slowest not higher). Its soft precision gate still holds, the golden
suite is green, and there is no regression across the config-spanning guard set: one representative per kernel
path × layout × placement — R1 and R2 wkv regimes, the K-blocked stage 1, TILE and ROW_MAJOR out, DRAM and L1
latent, plus the Phase-0 Kimi loose case.

### [ ] Refinement 4 — Rows per device that are not whole tiles (m_non_aligned)

**Goal**: add `"m_non_aligned"` to `SUPPORTED["alignment"]`: `S % 32 != 0` or `(S/G) % 32 != 0`. INPUTS: Kimi
S=160 (40 rows per device), S=96 (24 rows per device), S=48 (latent 1.5 tiles, 12 rows per device). This is the
design's regime R9, a scheme addition. Each source sends the (up to 2) tile rows that cover a quarter. The
destination realigns the rows with a 0/1 row-shift matrix applied by matmul (`out_tile = U_δ·tile_a + D_δ·tile_b`,
exact) to nope and rope **before** stage 3. That is legal because a left row permutation commutes with `@ wkv_b1`
and with the elementwise rope. The output's last quarter tile row is written with its padding rows (TILE) or only
the valid sticks (ROW_MAJOR).

**Verifier notes**:
- Stands alone: tricky non-tile alignment. The quarter boundary straddles tiles, so the send unit, the landing
  layout, the arrival thresholds and the output row mapping all change shape. It is not edge-tile padding. No
  inventory skill covers it (`/memory-layouts` covers RM/TILE and edge-tile masks, not a cross-device row shift).
- Ordered before RoPE (hardest first): it reshapes the exchange and the landing layout that RoPE's stage-3 block
  then reads. RoPE extends it once instead of being reworked.
- The shift must keep the exchanged q bit-exact. The 0/1 matmul at fp32 DEST, packed to bf16, is exact. With
  16-bit DEST (Refinement 1) the sum `U·a + D·b` has one non-zero term per element, so it is exact too; the
  golden check confirms it.
- Every chip sends `ceil`-covering tile rows: the busiest link carries up to +2 tile rows per quarter. Account for it
  in the fabric budget table of `op_design.md`.

**Done when**: every golden cell with `alignment=m_non_aligned` whose other axis values are SUPPORTED passes, verifier
categories clean, acceptance + regression green.

### [ ] Refinement 5 — Rotary embedding (rope_mode = rope)

**Goal**: add `"rope"` to `SUPPORTED["rope_mode"]`. `cos` / `sin` (per device `(1,1,S/G,Dr)`, bf16 TILE DRAM) are
read once per core that owns heads (the quarter's rows × `Dr_t`). `rope_block` becomes
`rope @ trans_tile` (matmul, `SrcOrder::Reverse`; the transformation tile is generated on device by the reader, so
no host tensor) followed by `* sin` and `+ rope * cos` (eltwise `mul` / `add`, no broadcast — `op_design.md` §
Broadcast Verification). New CBs `cb_cos`, `cb_sin`, `cb_trans`, `cb_rope_tmp` get ledger rows. RoPE math reference:
`rotary_embedding_llama.cpp` (read for the math, not called — the prompt forbids calling any rotary op).

**Verifier notes**:
- A compute scheme addition inside the existing stage-3 nest: a new block op, same loop nest, a new binary phase.
  No inventory skill covers it. It must work under every Refinement 1/2/4 value (16-bit DEST, ROW_MAJOR out,
  non-aligned rows: cos/sin rows follow the device's output rows).
- `test_regression.py::test_rope_pairing` and `test_rope_positions[*]` (currently xfail) must pass: the
  adjacent-pair `rotate_half_meta` with `[c0, c0, c1, c1, …]` tables.
- Unlocks the DeepSeek-V3.2 and GLM-5.3 model-config loose cases (`LOOSE_CASES[2:]`). They are guard-set members
  for later perf slots.
- Precision: the rope part is no longer a bit-exact copy. Grade it with the shared tolerances; add the rope part to
  the precision baseline (`test_q_stem_precision_baseline.py` reports nope / rope separately).

**Done when**: every golden cell with `rope_mode=rope` whose other axis values are SUPPORTED passes (including
`LOOSE_CASES[2]` and `[3]`), the RoPE regression tests pass, verifier categories clean.

### [ ] Refinement 6 — Local stage-1 throughput on the PERF FOCUS (block / buffer / injector co-tune)

**Type**: perf

**Goal**: on the PERF FOCUS (`LOOSE_CASES[0]`, Kimi K3.1 model config; target 98.4 µs, roofline 73.8 µs), speed up
stage 1. In Phase 0 it ends at ~138 µs, against the op model's 57.1 µs for the same one-band column split
(`op_design.md` Traffic ranking). The implementer recorded the 2-column cores as compute- and x-supply-bound.
These are knob-turns on the surface the planner exposed (master.md `compute_block_size`, `double_buffer`,
`mcast_topology`, `matmul_output_subblock`), co-tuned in one phase:
- `S1_BLOCK_M_TILES` (lamp L1): m = 1 pays one x round, one matmul re-init and one port hand-off per row tile and
  halves in0 reuse. Try `quarter_row_tiles` (5) with the same interleaved destination order. The chunk floor is
  whole tiles; coarser blocks amortize up to the point where the first send is delayed.
- `X_ROUND_DEPTH` / `X_ROUND_TILES_CAP` with the block size, and 2 rotating x injectors on different NoCs
  (lamp L3; `Mcast2DRotatingSenderConfig`). The 6 idle row-0 cores (lamp L9) are free for it.
- `S1_COLS_PER_CORE` / stage-1 rect (lamp L7), re-balanced after Refinement 1 halves the q_b slice.
- `QB_READ_BATCH` (lamp L10; the q_b prologue is ~31 µs in Phase 0).

No SUPPORTED change.

**Calculator**: `python3 .claude/tools/perf_calculator/op_model.py matmul 20 48 144 --fidelity HiFi2 --b-dtype bfp8_b
--designs --c-mem l1 --kb 4,8,16` (planner's run, `op_design.md`): current one-band column split 57.1 µs (73 cores)
vs. the K-split design 50.3 µs (110 cores). That is −12%, a **tie** inside the model's ~15% band, so the knob-turns
above come first and the K-split (regime R6, a scheme-change) is not this phase's lever. The x-multicast group in
`dm_estimates.json` ranks an unlinked 16 KiB multicast at 89.2 µs (path-reservation bound) above stage-1 compute,
which is why the injector count is in this phase.

**Done when**: measured fastest-chip device time on the PERF FOCUS improves (`ccl_perf compare` verdict *win*), the
soft precision gate holds, the golden suite is green, and there is no regression across the config-spanning guard
set (as in Refinement 3, plus the RoPE loose cases). Also re-run the `QSTEM_ABLATE=SEND_PAYLOAD` ablation on the
PERF FOCUS and record it. If the exchange is no longer hidden (stubbing the payload saves more than noise), file a
**TRANSPORT** follow-up `Refinement 6b`: link balance (R4 relay of the antipodal quarter), arrival-increment
interval (L4), port batch (L5), and Ethernet-probe port placement. Measure it against `fabric_ring_exchange` /
`fabric_all_gather`'s numbers.

### [ ] Refinement 7 — K-split stage 1 (regime R6) on the PERF FOCUS

**Type**: perf

**Goal**: if stage 1 still dominates the PERF FOCUS after Refinement 6 (check with the `QSTEM_ABLATE` breakdown),
split q_b_proj's K (= Q) across two core groups and combine the partial sums over the NoC. Add a combine block op
between `s1_matmul_block` and `s1_handoff_block`, as in `op_design.md` regime R6. This uses all 99 compute cores
(plus the idle row-0 cores) for stage 1 instead of a 72-core rect. It is a cross-core scheme-change (master.md T3
`shared_input_reuse` / `tensix_all_reduce`), so it is its own phase. No SUPPORTED change.

**Calculator**: same op-model run as Refinement 6: K-split 50.3 µs (110 cores) vs one-band 57.1 µs — rank 1 vs.
rank ~8, a tie within the model's band. Its cost is +11.8 MB of fp32 partial-sum NoC traffic (Traffic ranking).
Build it and keep it only on a measured device-ns win.

**Done when**: measured fastest-chip device time on the PERF FOCUS improves beyond noise with the slowest chip not
higher, precision gate holds, golden suite green, no regression on the guard set. Or, if the measured variant does
not win, the attempt is recorded with its numbers in `changelog.md` and the change is reverted (a measured null
result closes the phase).
