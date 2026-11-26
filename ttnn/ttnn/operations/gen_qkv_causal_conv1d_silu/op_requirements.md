# Operation Requirements: gen_qkv_causal_conv1d_silu

## Definition
- **Formula**: `x[-3..-1] = history rows 0..2` (oldest first); `x[t] = input[0, t, input_column_offset + c]`;
  `y[t, c] = silu(tap0[c]·x[t-3, c] + tap1[c]·x[t-2, c] + tap2[c]·x[t-1, c] + tap3[c]·x[t, c])`,
  `silu(u) = u·sigmoid(u)` (precise); `q, k, v = y[..., 0:Q], y[..., Q:Q+K], y[..., Q+K:C]`, `C = Q + K + V`.
- **PyTorch Reference**:
  ```python
  def reference(conv_input, history, taps, widths):      # conv_input [1,T,C], history [1,3,C], taps 4×[1,1,C]
      seq = conv_input.shape[1]
      window = torch.cat((history, conv_input), dim=1).double()
      y = F.silu(sum(window[:, j:j + seq] * taps[j].double() for j in range(4)))
      return y.split(list(widths), dim=-1)                # (q, k, v)
  ```
  (= `F.conv1d` over `cat(history, x)` channels-first with `weight[c,0,j] = tap_j[c]`, `groups=C`, then SiLU and the split.)
- **Import Path**: `from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu, default_compute_kernel_config`
- **Function Signature**:
  ```python
  gen_qkv_causal_conv1d_silu(
      input: ttnn.Tensor, history: ttnn.Tensor,
      tap0: ttnn.Tensor, tap1: ttnn.Tensor, tap2: ttnn.Tensor, tap3: ttnn.Tensor,
      q_width: int, k_width: int, v_width: int, *,
      program_config: ttnn.QkvCausalConv1dSiluProgramConfig,
      actual_start: ttnn.Tensor, predecessor_carry: ttnn.Tensor,
      memory_config: Optional[ttnn.MemoryConfig] = None,
      compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
      sequence_parallel_axis: int = 0, input_column_offset: int = 0, page_channels: Optional[int] = None,
  ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]      # (q, k, v): bf16 TILE [1,T,Q], [1,T,K], [1,T,V]
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [ROW_MAJOR_LAYOUT, TILE_LAYOUT] (two regimes: `rm_staged_window`, `tile_untilize_staged_window`)
- **SUPPORTED placements**: in_buffer / history_buffer / taps_buffer ∈ {dram, l1}; output_memory ∈ {default, dram_interleaved, l1_interleaved}
- **SUPPORTED shape-derived axes**: history_layout = token_paged only; input_window = exact only; width_split, channel_count, seq_tiles, chunk_blocks = every value
- **SUPPORTED compute**: fp32_dest_acc_en ∈ {False, True}; caller's math_fidelity / dst_full_sync_en honoured; math_approx_mode=True and packer_l1_acc=True refused (ValueError)
- **Cores**: full grid from `device.compute_with_storage_grid_size()` — channel groups × token ranges (`select_plan`); 004: 110 cores, 001–003: 80 cores (uniform groups must divide C_t = 80)
- **Golden baseline**: 17 / 21 passing; 4 `xfail_expected` (conv1d-012, -013, -018, -019); 0 loud (per `verifier_report.json`)
- **Perf targets**: 001 8.7 µs, 002 10.4 µs, 003 13.2 µs, 004 128–131 µs — all beat native ("must") and "target"

### [x] Refinement 1 — Reader addressing generality: input_window column_window + history_layout channel_paged

**Goal**: add `"column_window"` to `SUPPORTED["input_window"]` and `"channel_paged"` to `SUPPORTED["history_layout"]`, for both input layouts (ROW_MAJOR and TILE) and every placement already supported, natively in the reader:
- **column_window**: read the convolution channels `[input_column_offset, input_column_offset + C)` of a wider input (`W > C` and/or `input_column_offset > 0`) where it lies.
  - RM: byte offset `2·(offset + c0)` inside the token-row page.
  - TILE: page `r·W_t + offset/32 + c`.
  - The halo / extra-tile-row paths for a core whose range starts at `t0 > 0` use the same addressing.
- **channel_paged**: read the three history rows from the `[1, C/P, 3P]` layout. Channel `c` of row `r` is page `c // P`, element `r·P + c mod P`.
  - A channel group whose slice crosses page boundaries issues one read per contiguous run per row. Pages straddling the op's channel groups must work (conv1d-012: P = 160 channels vs 8-tile groups).
  - Every run start is a multiple of 32 elements, i.e. 64 B aligned.

This moves conv1d-012, conv1d-013, conv1d-018 and conv1d-019 from `xfail_expected` to passing.

**Verifier notes**:
- **Both values are knob-turns** in the Blocking Model (design "TARGET mapping"): reader-only addressing.
  - No CB, block, schedule or L1-ledger change. The ledger's data-movement budget gains only the history run count (≤ 3 · (pages spanned) reads, on the `t0 = 0` core of each group).
  - They are bundled because each alone is far smaller than one focused pass, and conv1d-018/019 need both.
- **Order inside the refinement follows the op prompt:** column_window first, then channel_paged.
  - column_window is mostly plumbed already: the `input_col_offset_tiles` RT arg feeds both regimes' addressing, and `input_column_offset` is in the program-cache key. It is largely the SUPPORTED gate plus tests.
- **W not a multiple of 32** is a native corner outside the suite. It is refused today only because `W ≠ C` tags it `column_window`. Once column_window opens, refuse it explicitly (`UnsupportedAxisValue`) unless you support it. The TILE path's `input_width_tiles = W // 32` is wrong for a padded width.
- **`page_channels` is already in the program-cache key** (verifier fix): `[1,3,C]` and `[1,C/P,3P]` have the same shape when `C = 3P`.
  - Carry `P` to the reader as a CT arg (or a flag + RT), so the token-paged path stays branch-free.
- **No tensor-manipulation wrappers.** The native-kernel policy and the prompt's Layout rule forbid `ttnn.slice` / `reshape` / `to_layout` / `to_memory_config` on the input or the history, on the host or as a second program. There is no partial-tick wrapper fallback for this refinement: the in-kernel reads are the only acceptable path.
- **Hot path:** the reader change sits on the 001–004 critical path. Re-measure the four flagged targets.
- **Skill pointer:** none in the inventory covers reader addressing for this op.

**Done when**:
- conv1d-012, -013, -018 and -019 pass in the golden suite (018/019 also exercise TILE, L1 input/history/output, fp32 DEST, and 019's FP32_DEST_BOUND at HiFi4).
- `verify_supported` reports `xpass_drift = 0`, `xfail_wrong_mode = 0`, `supported_fail = 0`, and all 17 Phase 0 cells still pass.
- The acceptance, supplementary and precision-baseline tests pass.
- The device time of conv1d-001..004 does not regress.

### [ ] Refinement 2 — Speed up the latency-bound flagged profiles (conv1d-001, -002, -003)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags conv1d-001, -002 and -003 (RM DRAM input, token-paged history, default compute config, chunk 256, widths 512/512/1536) as mandatory perf targets.
- **Goals:** each already beats its *target* (17.0 / 18.5 / 22.5 µs, native's conv program alone). Their *stretch* is "near one program's launch cost"; the measured floor on this op is ~3 µs (launch + windowed tilize, no DRAM).
- **Measured (this verifier, BH 11×10):** 001 = 8.6 µs, of which dataflow alone is 7.0 µs and compute alone 4.4 µs. These calls are **dataflow-latency-bound**.
  - At 001 the plan picks `block_ch_tiles = 1` on 80 cores: ~3.4k DRAM requests of ≤ 64 B (35 stick reads + 8 tap face-row reads + history per core). DRAM is request-rate-bound at that size.
  - Forcing `block_ch_tiles = 2` (40 cores) measured **7.35 µs (−15%)**. `select_plan`'s tile-count makespan ignores per-request cost.
  - For 002 (w = 2: 10.06 vs w = 4: 10.32 µs) and 003 (w = 4: 13.3 vs w = 8: 16.7 µs) the current `w` is already the best. Their headroom is in the reader's request issue and latency, not the split.
- **Relevant levers** in `ttnn/ttnn/operations/examples/master.md`: request size / count (`double_buffer`, `read_coalescing`), `split_reader` and `trid_double_issue` (reader issue-bound at tiny `w`), `bank_stagger`, `noc_placement`. Plus a request-aware term in `select_plan` (a knob-turn of `block_ch_tiles`; keep BB-08: the plan stays reportable and production shapes get the same plan).
- No SUPPORTED change.

**Done when**:
- Measured device time improves on conv1d-001 and does not regress on conv1d-002 / -003.
- All three stay at or below their target_us.
- The golden suite is green.
- No regression across the config-spanning guard set: one representative per kernel path × layout × placement — RM DRAM (004), TILE (005), L1 input (006), L1 output (010), fp32 DEST (011), and the Refinement 1 paths (012 channel-paged, 013 column-window, 018/019 TILE + L1).

**Verifier notes**:
- This comes after Refinement 1 because the reader addressing it adds sits on this same critical path; optimize the final reader.
- Ablations must be **define-gated** (`KernelDescriptor.defines`). A source-edit ablation left stale kernel binaries in the JIT cache during verification (see `verification_report.md`).

### [ ] Refinement 3 — Speed up the DRAM-bound prefill profile (conv1d-004)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags conv1d-004 (T 4096, C 2560, RM DRAM input, default compute config: HiFi4, 16-bit DEST) as a mandatory perf target.
- **Goals:** 128–131 µs today. Target 164 µs is met; **stretch 117.13 µs** (70% of DRAM-floor bandwidth; floor 82 µs).
- **Measured (this verifier, define-gated ablations): 004 is compute-bound.**
  - compute-only 108.7 µs vs dataflow-only 99.1 µs (already ~81% of DRAM peak); full 130.4 µs.
  - Inside compute, the **precise SiLU on the packer thread costs ~68 µs** (compute without SiLU: 40.5 µs, ≈ 0.7 µs per output tile). The 4 HiFi4 MACs cost ~9 µs; the windowed tilize ~12 µs.
  - With SiLU removed the full op would be 108 µs, i.e. dataflow-bound.
- **Two regions, in order:**
  - (a) **cut the SFPU cost of the SiLU per output tile.** The precision contract must not weaken (x·sigmoid(x) at full precision; `math_approx_mode=True` stays refused).
  - (b) **recover the ~21 µs overlap loss** (full 130 vs max(compute 109, DM 99)): co-tune the `OUT_DEPTH` / `ROWS_DEPTH` buffer depths with `block_rows`. These are knob-turns; L1 is at 301 KB of 1 MiB.
  - Then DM-side levers (`bank_stagger`, `trid_double_issue`) once compute drops below DM.
- **Catalog:** see `ttnn/ttnn/operations/examples/master.md` (`compute_fusion`, `sfpu_tile_scope`, `double_buffer`, `bank_stagger`).
- No SUPPORTED change.

**Done when**:
- Measured device time on conv1d-004 improves, moving toward 117.13 µs.
- Golden is green, including conv1d-011's FP32_DEST_BOUND (fp32 DEST, cancelling taps).
- `test_gen_qkv_causal_conv1d_silu_precision_baseline.py` passes with ULP p99 and relative RMS no worse than the Phase 0 baseline in `verification_report.md`.
- No regression across the guard set named in Refinement 2.

**Verifier notes**:
- This is filed after Refinement 2 per the flagged-case order. It uses a different lever family, so it has its own phase.
- It is likely the harder phase: the SiLU lever is a compute-kernel change, T2–T3.
- If 004 is still above stretch after (a) and (b) with compute < DM, the next perf phase is the dataflow side. Note it in the changelog with the measured split.
