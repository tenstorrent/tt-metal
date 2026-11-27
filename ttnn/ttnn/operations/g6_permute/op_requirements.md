# Operation Requirements: g6_permute

## Definition
- **Formula**: `out.shape[k] = x.shape[dims[k]]`; `out[o_0, …, o_{r-1}] = x[i_0, …, i_{r-1}]` with `i_{dims[k]} = o_k`. A negative entry `d` stands for `d + r`. Values move and none is computed: bfloat16 is bit-identical; bfloat8_b is within one quantization step `2^(E−133)` of the host rendering, with each group's shared exponent equal to the largest of its permuted values.
- **PyTorch Reference**: `torch.permute(input_tensor, dims).contiguous()`
- **Import Path**: `from ttnn.operations.g6_permute import g6_permute`
- **Function Signature**: `g6_permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor`. `dims` is positional: a sequence of `r` ints forming a permutation of `range(r)`. The output takes the input's dtype, layout and memory placement. There is no `memory_config`, `dtype`, output tensor or `compute_kernel_config` parameter.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Cartesian rule (this op)**: SUPPORTED is per-axis, so each new axis value also admits cells that no regime serves (e.g. TILE × dims (0,1,4,2,5,3)). Each refinement lists the cell dicts it must put in `EXCLUSIONS` **in the same change**, because `validate()` may not accept a shape the kernels get wrong. No production case may ever go into EXCLUSIONS.
> **No wrapper fallback (this op)**: the spec forbids `ttnn.to_layout`, `tilize`, `untilize`, `pad`, `slice`, `typecast` at the entry point, and any second dispatch. No partial-tick escape through a manipulation op exists. Every regime is a native reader/writer pair (plus a compute kernel for Refinement 2), selected on the host, in one program.
> **One kernel set per regime (BB-04, BB-18)**: each regime gets its own reader/writer (and compute) sources, selected on the host. Do not add branches for another regime to `rows_tile_scatter`'s kernels.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED buffer**: ["dram"] (interleaved)
- **SUPPORTED shape-derived axes**: rank ∈ {4}; alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}. The verifier widened alignment from `h_non_aligned` after bit-exact tests with poisoned padding; the regime keeps W, so padding columns stay in padding.
- **SUPPORTED op-specific axes**: dims ∈ {(0, 2, 1, 3)}
- **Regime**: `rows_tile_scatter`. Each input tile's live prefix is read once and its 32 B half-rows go straight to their output pages. One CB, no compute kernel.
- **Cores**: full grid. `split_work_to_cores(grid, tensor_in_tiles, row_wise=True)` over input tiles; 2–32 active cores on the production shapes, the whole 110-core grid on large shapes.
- **Compute config**: none. There is no arithmetic and no compute kernel.
- **Golden baseline**: 20 / 20 registry cells inside SUPPORTED pass bit-exact. 12 cells are `xfail_expected` (FLUX ×4, Panoptic ×8). `test_regression.py` passes 4 / 7 under the graded runner; the 3 `test_malformed_dims_raise` failures are a harness L1-capture artifact (`verification_report.md`) and pass under `run_safe_pytest.sh`.
- **Device time**: 1.26–2.23 µs per Qwen cell, 1 program, 2048 B L1 per core.

### [x] Refinement 1 — ROW_MAJOR rank-6 block gather (FLUX), DRAM

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`, `6` to `SUPPORTED["rank"]` and `(0, 1, 4, 2, 5, 3)` to `SUPPORTED["dims"]`. Build regime `blocks_rm_gather` (`op_design.md` → Regimes) as a native ROW_MAJOR reader/writer pair:
- the work unit is `(n, c, h)` and a block is `block_h` units;
- per unit, `W·I` input pages of `J` values become `I·J` output pages of `W` values, through a local `[W, I·J] → [I·J, W]` halfword gather in L1.

This moves golden FLUX_73 `[1,32,64,32,2,2]`, FLUX_74 `[1,64,128,32,2,2]` and loose L0 and L1 from `xfail_expected` to passing, bit-exact. Put the cells admitted with no regime into `EXCLUSIONS`: `{layout: TILE_LAYOUT, dims: (0,1,4,2,5,3)}` and `{layout: ROW_MAJOR_LAYOUT, dims: (0,2,1,3)}`. Neither has a production case.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- **Anchor.** `LOOSE_CASES[0]` (FLUX_74) is the mandatory perf target. This refinement lands its whole contract: bf16, ROW_MAJOR, DRAM interleaved, rank 6, `w_non_aligned`, dims (0,1,4,2,5,3). No compute config exists to carry. Refinements 3 and 5 optimize the path built here, so build it to the performance-conformance bar, not correct-only:
  - spread the `(n, c, h)` units over the full grid (`row_wise=True`);
  - make `block_h` a block knob with one source of truth, defaulting to the coarsest block that fits the 1347776 B budget;
  - issue every read of a block before one barrier, and double-buffer the input CB when a core owns ≥ 2 blocks;
  - batch the writer the same way: one flush per block, not per page.
- **Record the baseline.** Put `device_kernel_ns` for L0 and FLUX_73 in the changelog. Refinement 3 gates on it.
- **Read alignment (risk R7).** Blackhole DRAM reads need local ≡ remote mod 64 (`NOC_DRAM_READ_ALIGNMENT_BYTES`), and each 4 B input page sits in its own 64 B DRAM slot. Land each page in its own 64 B-aligned L1 slot. The halfword gather is RISC-V work, never the compute engine: bf16 must stay bit-identical.
- **Every accepted shape.** The regime must be right for every RM rank-6 (0,1,4,2,5,3) shape `validate()` accepts, at every `alignment` tag, not only `W=32, I=J=2`. Add a few non-production acceptance shapes with other `W, I, J, H`, including an odd `J` and a `J·2` that is not a multiple of 16.
- **Host contract.** Extend `_cache_key` to cover every CT input of the new regime, `g6_permute_plan` to report it (BB-08), and the host refusal of BB-05 to its block.

**Done when**: FLUX_73 and FLUX_74 (golden and loose L0, L1) pass bit-exact; `verify_supported` loud categories are all 0; every Phase 0 cell still passes; L0's `device_kernel_ns` is recorded in the changelog.

### [x] Refinement 2 — bfloat8_b TILE NHWC→NCHW (Panoptic p3), plus L1 placement for every regime

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, `"l1"` to `SUPPORTED["buffer"]` and `(0, 3, 1, 2)` to `SUPPORTED["dims"]`. Build regime `nhwc_tile_bfp8_transpose` (`op_design.md` → Regimes). Per group `(n, ht, wt)`, 32 input tiles become 32 output tiles:
1. unpack bfloat8_b → `transpose_wh` → pack bfloat16 (exact);
2. the core-local row exchange;
3. unpack bfloat16 → pack bfloat8_b, with each group's shared exponent equal to the largest of its values.

Also carry `"l1"` to every built regime (`rows_tile_scatter`, `blocks_rm_gather`). That is a knob-turn: the accessors address L1 banks, and `block_cap` subtracts the output's own L1 pages per core. The new regime must also be right with buffer `"dram"`.

This moves golden PANOPTIC_53 `[1,128,256,32]` and loose L2, L15 (`w_scaled`), L16 (`bf8_odd_grid` + shared exponent) and L18 (`l1_free=1347776`) from `xfail_expected` to passing under the `bf8_step` gate. Put into `EXCLUSIONS`:
- `{dtype: bfloat8_b, dims: (0,2,1,3)}`
- `{dtype: bfloat8_b, dims: (0,1,4,2,5,3)}`
- `{dtype: bfloat16, layout: TILE_LAYOUT, dims: (0,3,1,2)}`
- `{layout: ROW_MAJOR_LAYOUT, dims: (0,3,1,2)}` (removed by Refinement 4)
- the non-aligned bfloat8_b cells `{dtype: bfloat8_b, alignment: h_non_aligned}` and `{dtype: bfloat8_b, alignment: w_non_aligned}`, unless the regime handles them. A padding value must never raise a group's exponent; no production case is non-aligned.

**Implementation skill**: /numeric-formats-metal, /interleaved-parallel

**Verifier notes**:
- **Order.** This comes before the ROW_MAJOR Panoptic regime (Refinement 4) because it is the most invasive generality change left: the op's first compute kernel, its first non-bf16 CB formats, and a third kernel role on the host. Refinement 4 then only extends Refinement 1's RM machinery and this refinement's L1 plumbing.
- **Skill scope.** `/numeric-formats-metal` covers the dtype-aware CB formats (bf8b in, bf16 intermediate, bf8b out) and the pack format. Its `compute_kernel_config` exposure does **not** apply: the spec has no such parameter, and `extras.math_fidelity` must be None. Keep any compute config internal. No bf16 value of another regime may route through the compute path.
- **L1 budget is a MUST** (spec rule, BB-05). The output's own L1 pages on a core (p3: 10 × 1088 B) plus every op buffer must stay ≤ 1347776 B; L18 checks this. The design's group working set (≈ 32 + 32 tiles) fits; the whole tensor (2.2 MB) does not. Record the new CBs in `l1_ledger.md`.
- **Denormal risk.** Groups with a tiny shared exponent unpack to bf16 denormals, which the unpacker or packer may flush. No production input hits this, but the "every shape `validate()` accepts" rule covers it, so check one adversarial group.
- **L1 for the other regimes.** Add one acceptance shape each for Qwen TILE in L1 and FLUX RM in L1.

**Done when**: PANOPTIC_53 (golden) and L2, L15, L16 and L18 pass; loud categories are all 0; every prior cell still passes; one Qwen shape and one FLUX shape pass with buffer `"l1"`.

### [x] Refinement 3 — Speed up the perf-flagged FLUX profile (gather tuning)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags FLUX_74 `[1, 64, 128, 32, 2, 2]` as the mandatory perf target: bf16 ROW_MAJOR, DRAM interleaved, dims (0,1,4,2,5,3), `gate: exact`, no fidelity. Its stated goal is "kernel time below stock 1056.20 µs and within 1.35 × roofline = 12.29 µs". Speed up Refinement 1's `blocks_rm_gather` on this exact shape and config with the relevant patterns in `ttnn/ttnn/operations/examples/master.md`. The read side issues ~4766 4-byte page reads per core against ~300 page writes, so it is likely RISC-issue-bound; confirm that with the zones first. Candidate levers:
- `double_buffer`: co-tune `block_h` and the input CB depth;
- `split_reader`: share the page reads between both data-movement RISCs;
- `trid_double_issue`: keep reads in flight across the block barrier;
- `bank_stagger`: per-core start offsets.

Walking each block's pages bank-major (`p, p+8, p+16, …` share a bank) also lets one `noc_async_read_one_packet_set_state` per bank run serve many `_with_state` reads. No SUPPORTED change.

**Verifier notes**: 12.29 µs is not reachable at this placement, so gate on a measured win instead. The figure counts the 2 MiB of values, but the input is 524288 ROW_MAJOR pages of 4 B, each in its own 64 B DRAM slot (32 MiB stored). Every page costs at least one DRAM burst, so the input alone is ≥ 16 MiB of DRAM traffic (≈ 33 µs at the 512 GB/s peak) before any NoC request overhead (`verification_report.md` → Recommendations). The whole-slot bank-coalesced read regime is Refinement 5, measured against this phase's tuned gather.

**Done when**:
- L0's `device_kernel_ns` improves by more than noise (> 3 %) over Refinement 1's recorded baseline and stays below stock;
- the bit-exact gate holds and the golden suite is green;
- nothing regresses across the config-spanning guard set: one representative per distinct kernel path × layout × placement, measured where the cell is > 32 tiles. Regimes whose kernels this phase does not touch only need to stay green.

**Outcome**: L0 device kernel time went 1022579 → 554900 ns (−45.7 %; R1 plan in the same session 1029265), FLUX_73 248450 → 144460 ns, FLUX in L1 −25 % / −21 %. All bit-exact. The win comes from reads split evenly over both NoCs on DRAM input, each NoC entering a bank through its own port, issued bank by bank in 8-round runs, with 256-page blocks. L0 is now bound by the DRAM ports' request rate: 524288 single-page reads over 16 ports at ~17 ns each. Ablation: writes stubbed −1.7 %, all payloads stubbed 49.8 µs; 55 cores are only +8 %, so the issue rate is not binding. Next is fewer requests (Refinement 5's bank-coalesced reads). `split_reader` and `trid_double_issue` were not built: the core-count, depth and block sweeps show the ports never starve. L1 placement is issue-bound (~28 ns per read), so a split reader or cheaper per-page addressing is its lever.

### [x] Refinement 4 — ROW_MAJOR NHWC→NCHW (Panoptic p4)

**Goal**: build regime `nhwc_rm_transpose` (`op_design.md` → Regimes) and remove `{layout: ROW_MAJOR_LAYOUT, dims: (0,3,1,2)}` from `EXCLUSIONS`. This moves golden PANOPTIC_44 `[1,128,256,32]` bf16 ROW_MAJOR L1 and loose L3 and L17 (`l1_free=1347776`) from `xfail_expected` to passing, bit-exact. The axis values (ROW_MAJOR, `"l1"`, (0,3,1,2)) are already in SUPPORTED from Refinements 1 and 2, so this refinement turns cells on rather than adding axis values. The regime must be right with buffer `"dram"` as well as `"l1"`.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- **Data path.** This extends Refinement 1's RM page machinery:
  - read input pages of `C` values (64 B) from other cores' L1 banks over the NoC;
  - transpose `[block_w, C] → [C, block_w]` locally with halfword copies;
  - write `C` partial output rows of `block_w` values at offset `w0·2`.
- **Write alignment.** Writes are 16 B-aligned, so `block_w` is a multiple of 8. Every accepted shape must still be right, including a `W` that is not a multiple of 8 (the tail block).
- **L1 budget is a MUST.** The output's own pages (p4: 38 × 512 B per core) plus the op's buffers must stay ≤ 1347776 B; L17 checks this.

**Done when**: PANOPTIC_44 (golden) and L3 and L17 pass bit-exact; loud categories are all 0; every prior cell still passes.

### [x] Refinement 5 — FLUX profile: bank-coalesced reads

**Type**: perf

**Goal**: FLUX_74 (`LOOSE_CASES[0]`), the only perf-flagged case, gets the hard lever as its own phase. Build and measure regime `blocks_rm_bank_coalesced` (`op_design.md` → Regimes, deferred "settled by one experiment against `blocks_rm_gather`"):
- Pages `p, p+8, p+16, …` of one DRAM bank are adjacent 64 B slots. Read `k` consecutive slots of one bank in one `k·64 B` transaction instead of `k` 4 B reads: 16× the bytes, 1/k the transactions.
- Give each core bank-contiguous page runs (`master.md` `read_coalescing` with its bank-affinity arm) and extract the 4 B payloads locally.

The host selects between gather and coalesced by measured result, and `g6_permute_plan` reports the selection (BB-08). No SUPPORTED change.

**Done when**:
- L0's `device_kernel_ns` improves by more than noise over Refinement 3's result;
- the bit-exact gate holds and the golden suite is green;
- nothing regresses across the config-spanning guard set (one representative per distinct kernel path × layout × placement).

**Outcome**: L0 / FLUX_74 DRAM 554900 → **83545 ns** (−84.9 %; R3's gather in the same session 545233), FLUX_73 DRAM 144460 → 25486, FLUX_74 L1 141290 → 67718, FLUX_73 L1 38410 → 25233; Panoptic p4 unchanged (59219 / 87228 against 59226 / 87339). All bit-exact; golden ROW_MAJOR slice 7 / 7. Shipped: coalesced reads for DRAM and L1 input (256 / 2048 pages per block), load-batched staging copy and gather, one staging stage, and on DRAM output the writer twin, bank-grouped writes of 64 units (90.5 → 83.4 µs). Bottleneck now: the DRAM read bytes. Reads alone take 71.2 µs (~508 GB/s for 32 MiB of 64 B slots holding 2 MiB of payload); writes add 5.4 µs, the copies 5.4 µs, sync floor 5.2 µs. Two stages of staging lose once the copies are fast (116 µs): reads that never pause congest the NoC links into the far columns (x = 13–15). Next: fuse the staging copy into the gather (read payloads straight from the bank-ordered slots) to drop one RISC pass, and pace the reads with a per-core in-flight cap so two stages stop congesting; not tried because each changes the reader–writer contract and the read floor leaves at most ~12 µs.
