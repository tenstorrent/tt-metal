# Operation Design: gen_qkv_causal_conv1d_silu

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (data movement + tilize + FPU multiply-accumulate + SFPU activation), one device program |
| Goal | Four-tap depthwise causal convolution over the token axis of a channels-last `[1, T, W]` input (channel window `[offset, offset + C)`, `C = Q + K + V`), then SiLU, written as three bfloat16 TILE tensors q `[1,T,Q]`, k `[1,T,K]`, v `[1,T,V]`, in ONE program, reading a ROW_MAJOR input's pages directly (no tilize program). |
| Math | `x[-3..-1] = history rows 0..2`; `x[t] = input[0, t, offset + c]`; `y[t,c] = silu(Σ_{p=0..3} tap_p[c] · x[t-3+p, c])`; `q,k,v = y[..., 0:Q], y[..., Q:Q+K], y[..., Q+K:C]`; `silu(u) = u·sigmoid(u)` (precise). |
| Key algebraic fact used | `tap_p` is constant along tokens, so the token shift commutes with the tap multiply: `shift_k(x) ⊙ tap = shift_k(x ⊙ tap)`. The shift is therefore done **on raw bf16 data, by data movement** (exact, fidelity-independent), and the multiply-accumulate happens once in DEST. |
| References | `eval/golden_tests/gen_qkv_causal_conv1d_silu/{feature_spec.py,helpers.py,axes.py,native.md}`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md,ttnn-cb-memory-fundamentals.md,precision_convention.md}`; `ttnn/cpp/ttnn/kernel_lib/{tilize_helpers.hpp,untilize_helpers.hpp,eltwise/api/chain.hpp,sfpu_activation_helpers.hpp,local_copy_helpers_dataflow.hpp,tilize_helpers_dataflow.hpp}`; `tt_metal/hw/inc/api/compute/bcast.h`; `eval/op_template.py`; perf catalog `ttnn/ttnn/operations/examples/master.md` |

### Catalog entries that set knobs

| Catalog entry | Knob / decision it informed |
|---|---|
| `width_split` | Decode (T = 32: one tile-row) must split **channels** across cores, never tokens only; channel groups fill the grid. |
| `double_buffer` | Reader issues a whole tile-row of stick reads (32, or 35 with the halo) then ONE barrier; `ROWS_DEPTH = 2` slots so the reader fills block b+1 while compute consumes block b. |
| `read_coalescing` / `tile_reorder` | Each stick read is the whole contiguous `64·block_ch_tiles`-byte channel slice of the token row; never sub-slices. Tiebreak toward larger `block_ch_tiles` (bigger contiguous reads). |
| `noc_placement` | Cores enumerated row-wise over the grid (`x` fastest), reader on NoC0 / writer on NoC1 (descriptor defaults). |
| `compute_block_size` | `block_rows` is the coarsest that fits L1: one windowed-tilize pass and one conv pass per block amortize the tilize↔multiply↔SFPU init/reconfig switches. |
| `eltwise_l1_vs_dest_accumulate`, `compute_fusion` | The 4 tap products accumulate in ONE DEST slot (no L1 partials) and SiLU runs on that slot in DEST before the single pack (SFPU consumer → fuse). |
| `bank_stagger`, `split_reader`, `trid_double_issue` | Perf lamps only (see Perf lamps). |
| `shared_input_reuse`, `mcast_topology` | The deferred tap-multicast regime (taps are reuse-shared across the token split). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input` | ttnn.Tensor | yes | rank 3 `[1,T,W]`, bf16, TILE or ROW_MAJOR, interleaved DRAM/L1, `T>0`, `T%32==0`, `W ≥ offset + C` | — | buffer address RT; layout selects regime (CT) |
| `history` | ttnn.Tensor | yes | bf16 ROW_MAJOR interleaved; `[1,3,C]` (page_channels None) or `[1,C/P,3P]` | — | address RT |
| `tap0..tap3` | ttnn.Tensor | yes | bf16 TILE interleaved, `shape[-1]==C`, `volume==C` | — | addresses RT |
| `q_width, k_width, v_width` | int | yes | each > 0, multiple of 32 | — | tile counts CT (writer routing) |
| `program_config` | `ttnn.QkvCausalConv1dSiluProgramConfig` | yes | `channel_chunk_size`: >0, %32==0, ≤C, divides C | — | **validated only; never reaches the program** (the op chooses its own split; result is chunk-independent) |
| `actual_start` | ttnn.Tensor | yes | UINT32 ROW_MAJOR interleaved, volume 1 | — | never read (one sequence-parallel rank) |
| `predecessor_carry` | ttnn.Tensor | yes | `tensor_spec()` == `history.tensor_spec()` | — | never read (one rank) |
| `memory_config` | MemoryConfig \| None | no | interleaved DRAM or L1; sharded refused | None → `ttnn.DRAM_MEMORY_CONFIG` | output accessors CT; L1 share enters the L1 budget |
| `compute_kernel_config` | DeviceComputeKernelConfig \| None | no | `math_approx_mode` False, `packer_l1_acc` False | None → `default_compute_kernel_config()` | `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en` → ComputeConfigDescriptor |
| `sequence_parallel_axis` | int | no | mesh axis; that axis must have size 1 (else NotImplementedError) | 0 | host only |
| `input_column_offset` | int | no | multiple of 32; Phase 0 SUPPORTED: 0 with `W == C` | 0 | reader RT (byte/page offset) |
| `page_channels` | int \| None | no | None, or P>0, %32==0, divides C; Phase 0 SUPPORTED: None | None | reader (history fetch) |

`default_compute_kernel_config()` (exported factory, the single definition of "None"): `ttnn.init_device_compute_kernel_config(device_arch_or_default, math_fidelity=HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)` with `dst_full_sync_en=False` — i.e. a `WormholeComputeKernelConfig(math_fidelity=HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False, dst_full_sync_en=False)`. The entry point and `axes.py` both resolve None through it; no other default exists anywhere.

### Registry (op file; implementer writes the code)

| Item | Value |
|---|---|
| Axis names | exactly feature_spec TARGET: `dtype, layout, in_buffer, history_buffer, taps_buffer, output_memory, fp32_dest_acc_en, history_layout, input_window, width_split, channel_count, seq_tiles, chunk_blocks` |
| INPUT_TAGGERS | the six taggers verbatim from the prompt, in that order |
| Phase 0 SUPPORTED | dtype bf16; layout RM, TILE; the three buffers dram, l1; output_memory default/dram_interleaved/l1_interleaved; fp32_dest_acc_en False/True; history_layout token_paged; input_window exact; width_split, channel_count, seq_tiles, chunk_blocks: every value |
| EXCLUSIONS | `[]` |
| validate() order | ValueError checks (## Validation of the prompt) → build axes (pinned axes read off the call, taggers on `inputs = (tuple(input.shape), (Q,K,V), chunk, page_channels, offset)`) → SUPPORTED → EXCLUSIONS. Cached per call signature. INVALID never checked. |
| Plan reporting (BB-08) | module exports `get_last_plan()` → the `Plan` the most recent call ran (fields: `regime, block_ch_tiles, block_rows, num_ch_groups, cores_per_group, num_cores, core_seq_tiles_max`), and `select_plan(...)` (pure function, same result the call uses). |
| Plan pinning (regime-pinned tests) | module exports context manager `plan_limits(max_block_rows=None, max_cores_per_group=None)`; the caps enter `select_plan` and the program-cache key. |
| Program cache (BB-07) | Python-side cache keyed by: device grid, input (shape, layout, buffer type), history spec, taps spec, (Q,K,V), output memory config, resolved (math_fidelity, fp32_dest_acc_en, dst_full_sync_en), plan limits. A hit only rewrites buffer addresses in the cached descriptor's runtime args; channel_chunk_size is NOT in the key (it does not change the program). |

### Structural impossibilities

None to add to INVALID.

## Tensors

### Inputs

| Property | input | history (token-paged) | tap0..3 |
|----------|-------|---------|------|
| Shape | `[1, T, W]` | `[1, 3, C]` | logical volume C, last dim C (e.g. `[1,1,C]`) |
| Dtype | bfloat16 | bfloat16 | bfloat16 |
| Layout | ROW_MAJOR (page = token row, `2W` B) or TILE (page = tile; page id `r·W_t + col_t`) | ROW_MAJOR (page r = history row r, `2C` B) | TILE; page `ct` = tile `(0, ct)`; only tile row 0 is valid |
| Memory | interleaved DRAM or L1, read where it lies | interleaved DRAM or L1 | interleaved DRAM or L1 |

`actual_start`, `predecessor_carry`: validated, never read by any kernel. Nothing is ever written to an input.

### Output

| Property | q | k | v |
|----------|---|---|---|
| Shape | `[1, T, Q]` | `[1, T, K]` | `[1, T, V]` |
| Dtype | bfloat16 | bfloat16 | bfloat16 |
| Layout | TILE | TILE | TILE |
| Memory | `memory_config` or DRAM interleaved | same | same |

Allocated with `ttnn.allocate_tensor_on_device` (new tensors every call); `ttnn.generic_op([input, history, tap0..3, q, k, v], descriptor)` dispatches the single program (SPMD `ProgramDescriptor` on a MeshDevice: same program per device); the entry point returns `(q, k, v)`.

## Blocking Model

### Axes

`tensor_seq_tiles = T/32`, `tensor_ch_tiles = C/32` (exact: T and C are validated multiples of 32; every count below is a `ceil` per image, batch = 1).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| token (`tensor_seq_tiles` tile-rows) | **independent with a 3-row read-only halo** — `y[t]` reads `x[t-3..t]` only; no combine, a block needs the 3 rows before it (history for t=0) | `block_rows` (tile-rows per block) | coarsest that fits L1 within the core's assignment, balanced: `ceil(core_seq_tiles_max / ceil(core_seq_tiles_max / block_rows_fit))` | host `select_plan` → CT arg (sizes cb_rows / cb_window_tiles); per-core `core_seq_start`, `core_seq_tiles` RT | `cores_per_group` contiguous token ranges per channel group (Phase 0 spreads it) | knob-turn |
| channel (`tensor_ch_tiles` tiles) | **independent** — depthwise: no term couples two channels; the q/k/v split is a per-tile output routing of this axis | `block_ch_tiles` (= the core's channel-group width) | min-makespan divisor of `tensor_ch_tiles` (Work Distribution) | host `select_plan` → CT arg (tilize/untilize template width, stick page size, every CB) | `num_ch_groups = tensor_ch_tiles / block_ch_tiles` groups (Phase 0 spreads it) | knob-turn |
| tap (4) | **dependent** — the output sums over the 4 taps | `TAP_COUNT` (whole extent, 4 products into one DEST slot) | 4 (not split) | constant of the op's math (`TAP_COUNT = 4`, `HALO_ROWS = TAP_COUNT-1 = 3`) | not assigned across cores — a 4-term reduction fits one DEST slot; a cross-core combine would move 4× the partials | none (rejected regime) |
| batch | independent | — | 1 | contract (validate refuses batch ≠ 1) | not assigned (extent 1 by contract) | none |

Operand-reuse check (for the chosen split = channel groups × token ranges):

| Operand | Varies along the token split? | Varies along the channel split? | Consequence |
|---|---|---|---|
| input | yes | yes | read once (+3 halo rows per core-first block) |
| history | only the token-0 core of each group reads it | yes | read once per channel |
| taps | **no** | yes | **reuse-shared by construction**: every core of a channel group reads the same tap rows → `tap_multicast_per_group` regime row (deferred) |

Intermediates (re-running the table on what each stage produces): staging rows (token rows incl. halo × channels), window tiles (tap position × channel tiles × tile-rows) and output tiles all live on the core that computes them and are blocked by the same `block_rows × block_ch_tiles`. No core collects, coordinates or hands out work; no stage runs on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_rows` | `ROWS_DEPTH` (slots of `32·block_rows + 3` sticks) | 2 | reader fills block b+1 (DRAM latency) while compute consumes block b |
| `cb_output_tiles` | `OUT_DEPTH` (tile-rows of `block_ch_tiles` tiles) | 2 | writer drains tile-row m while compute packs m+1 |
| `cb_input_tiles` (TILE regime) | `IN_DEPTH` (tile-rows) | 2 | reader fetches tile-row j+1 while compute untilizes j |
| `cb_untilized_rows` (TILE regime) | `UNTILIZED_DEPTH` (tile-rows) | 2 | compute untilizes j+1 while reader copies j into the slot |
| `cb_tap_tiles` | — (resident) | 1 copy | n/a: loaded once, live for the whole kernel |
| `cb_window_tiles` | — (full block) | 1 block | n/a: sequential-phase intermediate, must hold the whole block |

All depths are host constants in one place in the program-descriptor module; every capacity is derived from them.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `tilize`/`untilize` helper width is a template parameter (`tilize_helpers.hpp:194-207`, `untilize_helpers.hpp:152-163`) | `block_ch_tiles` identical on every core | `block_ch_tiles` must divide `tensor_ch_tiles` (uniform channel groups) | a ragged last group would need a second compute kernel variant; with one kernel it would tilize the wrong width |
| fast tilize requires `block_width_tiles < 256` (`tilize_helpers.inl:65-69`) | `block_ch_tiles` | `block_ch_tiles ≤ 255` | silent fall to slow tilize (perf only) — clamp anyway |
| BH DRAM read alignment 64 B (`noc_parameters.h:379`, blackhole) | channel offsets / stick slice sizes | every slice starts at `2·(offset + c0)` with `c0, offset` multiples of 32 → 64 B multiple; size `64·block_ch_tiles` | misaligned DRAM reads return shifted data |
| Windowed tilize waits `32 + HALO_ROWS = 35` sticks from the slot start of each tile-row | `cb_rows` capacity | capacity = `ROWS_DEPTH·(32·block_rows + 3) ≥ 35` for any `block_rows ≥ 1` | `wait_front` larger than capacity deadlocks |
| `cb_rows` window reads must not straddle the ring end (tilize reads 32 contiguous sticks from the read pointer) | slot layout | every block occupies one whole slot of `32·block_rows + 3` sticks; per-block push = pop = `32·M_b + 3`; capacity is an exact multiple of the slot; only a core's LAST block may be shorter (no wrap after it) | a window straddling the wrap reads neighbouring L1 (wrong results, no hang) |
| Packed tap tile: 4 taps share one tile, tap p in row `2p` broadcast via `bcast_row_idx = 2p` (`bcast.h:192-203`, `llk_unpack_AB.h:294-318` BH / `:245-271` WH) | taps per channel tile = 4 | rows `0,2,4,6` (64 B-aligned face-row offsets `64p` and `512+64p`) | an odd row offset `32p` would break the 64 B DRAM-read alignment |
| DEST: one slot per output tile, 4 products accumulated + SiLU in place | tiles per acquire in `conv_silu_block` | ≤ `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`) — the realization's internal grouping, not a design knob | DEST overflow corrupts neighbouring tiles |
| L1 budget (BB-05): model leaves 1,048,576 B per core | `block_ch_tiles`, `block_rows` | `footprint(block_ch_tiles, block_rows) ≤ L1_BUDGET` (Work Distribution); `footprint(1,1)` = 18.8 KB (RM) / 26.4 KB (TILE) always fits, so no shape is refused for L1 | launch failure only in the model, not in an isolated test |

### Regimes

Named boundary for "minimum": **DRAM/L1-bank traffic of the interleaved tensors — each input element crosses once, each output element crosses once, taps and history once per channel.**

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `rm_staged_window` | **built** | `input.layout == ROW_MAJOR` | `(32·block_rows + 3) token rows × 32·block_ch_tiles channels` of row-major sticks → `block_rows × block_ch_tiles` output tiles | input: 1× + 3 halo sticks per core-first block with `t0 > 0` (later blocks copy their halo from the previous slot in L1); output 1×; history 1×; taps `cores_per_group`× (rows only, 64 B per tap per channel tile) | per block (intended once): tilize init/uninit, multiply init + SiLU init + format reconfigs (2 phase switches), slot halo copy, `wait`/`pop` of the window block; larger `block_ch_tiles` → longer contiguous stick reads (`64·block_ch_tiles` B) and fewer groups re-reading taps |
| `tile_untilize_staged_window` | **built** | `input.layout == TILE` | same block; body arrives as `block_rows × block_ch_tiles` input tiles | input 1× (tiles) + one extra tile-row (`block_ch_tiles` tiles) per core whose range starts at `t0 > 0` (its rows 29..31 are the halo); + core-local L1→L1 copy of every untilized tile-row into the slot (1× input bytes, NoC-local, not DRAM); rest as above | same as above plus one untilize init per block |
| `tap_multicast_per_group` | **deferred** — the redundant tap traffic is off the critical path by construction (tap reads are issued before the first block and compute only waits on taps at the first `conv_silu_block`, after a whole windowed tilize) and is ≤ 1.1% of the DRAM bytes of the one DRAM-bound target; a multicast adds a semaphore rendezvous to the START of every core of the group | taps reuse-shared, `cores_per_group > 1` | unchanged | taps 1× per channel instead of `cores_per_group`× (004: −420 KiB of 40.8 MiB) | — |
| `tilize_fused_mac` | **deferred** — needs an unpack mode that tilizes srcA from a row-major slot while row-broadcasting srcB; no compute API or helper provides it (the tilize helpers only pack tiles out; `bcast.h` unpacks tiled srcA) | — | same block | same DRAM; removes the `cb_window_tiles` round trip (4 packs + 4 unpacks per output tile, core-local) | — |
| `tile_matmul_shift` | **rejected** — superseded by `tile_untilize_staged_window`: shifting TILE data with constant shift matrices (`L_k·X_i + U_k·X_{i-1}`) is lossy below HiFi4 (data in srcA/srcB is truncated per fidelity phase) and costs 6 matmul tile-ops per output tile vs 1 untilize + exact data movement | — | — | same DRAM | — |
| `diag_matmul_conv` | **rejected** — superseded by the row-broadcast MAC: tap multiply as `S_k × diag(tap)` needs 4 diagonal tiles per channel tile built at start, 4 matmul tile-ops per output tile, and an in0 layout `[tile-row][tap]` the window producer does not emit | — | — | same | — |
| `rm_reread_windows` | **rejected (dead end)** — reading each stick 4× from DRAM into 4 aligned window buffers; superseded everywhere by the sliding pop on one slot (1× read) | — | — | input 4× | — |
| `token_halo_forwarding` | **rejected** — neighbour core sends its last 3 rows over the NoC; superseded by reading 3 sticks (DRAM, once per core) / copying them from the previous slot (L1); a semaphore handshake costs more than 3 small reads and serializes neighbouring token ranges | — | — | — | — |
| `tap_axis_split` | **rejected** — splitting the 4-term dependent axis across cores moves 4× the partials for a sum that fits one DEST slot | — | — | 4× output-sized partial traffic | — |

> **Verifier note (Phase 0 as built, 2026-11-26).** Three rows/lamps above changed status during
> implementation, each on a measured device-ns win (see `l1_ledger.md` implementer updates and
> `verification_report.md`): `tap_multicast_per_group` is **built** (the per-core tap fetch was
> DRAM-request-bound, not byte-bound: 004 185.9 → 152.9 µs); SiLU runs on the **packer thread**
> (`apply_activation_from_pack`, Perf lamp "SiLU on the math thread"); the default `block_rows` is
> capped at `OVERLAP_BLOCK_ROWS = 1` (Perf lamp "Overlap": 004 153 → 129 µs). Measured 004 profile
> after these: compute-only 109 µs, of which the precise SiLU is ~68 µs (compute without SiLU
> 40.5 µs); dataflow-only 99 µs; full 130 µs.

Regime selection function (exact, host): `regime = "rm_staged_window" if input.layout == ttnn.ROW_MAJOR_LAYOUT else "tile_untilize_staged_window"`. The intra-regime paths that only some grids/shapes trigger (multi-block cores → L1 halo copy; core-first block with `t0 > 0` → DRAM halo / extra untilized tile-row; ragged last block) are pinned by `plan_limits` tests in the acceptance test (both regimes).

### Traffic ranking

| Candidate split | DRAM crossings (004: T 4096, C 2560, 11×10 grid) | Cross-core | Occupancy (makespan, tiles/core) |
|---|---|---|---|
| channel groups only (`cores_per_group = 1`) — the minimum | input 1×, output 1×, history 1×, taps 1× | none | `block_ch_tiles = 1`, 80 cores × 128 tile-rows → 128 |
| **channel groups × token ranges (chosen; plan picks `cores_per_group`)** | input 1× + 315 KiB halo, output 1×, history 1×, taps 22× (440 KiB) → +1.8% | none | `block_ch_tiles = 16`, 5 × 22 = 110 cores → 96 (ideal 93.1) |
| token ranges only (`block_ch_tiles = tensor_ch_tiles`) | input 1× + halo, taps 110× | none | 110 cores, 2 vs 1 tile-rows → 160 |
| split the tap (dependent) axis + combine | +4× output-sized partials over the NoC | combine per tile | worse in both |

Both independent axes are split (no combine anywhere). The two leading candidates are within 1.8% of each other on bytes (comparable), so the tiebreak is occupancy-within-the-split: the plan picks the `cores_per_group` that minimizes makespan; `cores_per_group = 1` (the cheapest-traffic value of the SAME knob) is selected automatically whenever it is also min-makespan (decode 001).

### Block schedule

Per core (logical schedule; reader, compute and writer realize their parts asynchronously):

```cpp
load_taps();                                                  // once per core
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_rows_block(block_idx);        // reader: halo(3) + body(32·rows) sticks into one cb_rows slot
    untilize_rows_block(block_idx);    // compute, TILE regime only: body tiles -> row-major (reader copies into the slot)
    windowed_tilize_block(block_idx);  // compute: 4 shifted 32-row windows per tile-row -> tiles
    conv_silu_block(block_idx);        // compute: Σ_p window_p ⊙ row-bcast(tap_p) in DEST, SiLU in DEST, pack
    store_block(block_idx);            // writer: route each tile to q / k / v
}
```

`rows_b = block_rows` except the core's last block (`last_block_rows`); `M_b` below = `rows_b`.

| Block operation | Acts on (block shape) | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_taps` | `block_ch_tiles` packed tap tiles; tap p of channel tile c → row `2p` of tile c (two 32 B reads: face-0 row 0 → offset `64p`, face-1 row 0 → offset `512+64p`) | taps for the whole kernel | once per core; one barrier; issued before the first `load_rows_block` so it lands in the shadow of the first windowed tilize |
| `load_rows_block` | one `cb_rows` slot: sticks `[0,3)` = halo (tokens `t0-3..t0-1`), sticks `[3, 3+32·M_b)` = body; stick = `64·block_ch_tiles` B | the previous slot (source of the next halo) | per block: one slot reserve; halo source chosen once per block (table in Dataflow Strategy); RM: per tile-row 32 (first: 35) reads → one barrier → push 32 (first push 35); TILE: per tile-row one local L1→L1 copy of `2048·block_ch_tiles` B from `cb_untilized_rows` |
| `untilize_rows_block` (TILE) | `(M_b + extra) × block_ch_tiles` input tiles → row-major tile-rows; `extra = 1` only for a core's first block when `t0 > 0` (that row supplies the halo) | — | one untilize init/uninit per block |
| `windowed_tilize_block` | per tile-row m of the block: 4 tilizes of `block_ch_tiles` tiles from slot offsets `32m + p` (p = 0..3); window p = `x[t-3+p]` | `cb_rows` slot; produced windows accumulate in `cb_window_tiles` (`4·block_ch_tiles·M_b` tiles, layout `[m][p][c]`) | one tilize init (first call) / uninit (last call) per block; one `wait_front(35)` per tile-row; pops `1,1,1,29` per tile-row and `1,1,1,32` on the block's last tile-row (sums to `32·M_b + 3` = the slot) |
| `conv_silu_block` | `M_b × block_ch_tiles` output tiles; tile (m,c) = `silu(Σ_{p=0..3} win[m·4w + p·w + c] ⊙ rowbcast(tap_tile[c], row 2p))` (`w = block_ch_tiles`) accumulated in one DEST slot in fixed order p = 0,1,2,3 | taps (`cb_tap_tiles`), the window block | multiply init + SiLU init + format reconfig once per block; DEST acquire/commit per group of ≤ `DEST_AUTO_LIMIT` tiles; pushes `block_ch_tiles` per tile-row; one `pop(4·w·M_b)` of windows per block; taps waited once (first block), popped once (kernel end) |
| `store_block` | `M_b × block_ch_tiles` tiles; global channel tile `cg = g·w + c` → q if `cg < Q_t`, k if `< Q_t+K_t`, else v; page `r·X_t + (cg − base_X)` | — | per tile-row: wait `w`, `w` writes, one barrier, pop `w`; final `noc_async_write_barrier` before return (BB-03) |

Stall-shadow check: (1) compute's first wait is on `cb_rows` (DRAM latency) — nothing compute-side is independent of it except inits; (2) the tap wait is deliberately moved to the first `conv_silu_block`, so the tap fetch hides behind the first windowed tilize (legal: windowed tilize does not read taps); (3) TILE regime: the reader issues tile-row j+1's input reads before it waits on untilized tile-row j (independent work scheduled into the wait). No reorder changes floating-point order: the tap sum order is fixed p = 0..3 for every plan, so results are bit-identical across plans and chunk sizes.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block_rows at L1 max) | the writer idles during a whole block's windowed tilize, and the conv pass starts only after all `4·w·M` windows exist | `block_rows = 1` (tilize ↔ conv alternate per tile-row) on 004 |
| Grid synchronization / occupancy (uniform groups) | `w | tensor_ch_tiles` forces e.g. 003 (T 128) to 80 of 110 cores, makespan 4 tiles vs 3 achievable | two core groups with two compile-time `block_ch_tiles` values (two compute kernel variants) |
| SiLU on the math thread | SiLU (exp + reciprocal per element) dominates the math thread; the packer thread is mostly idle | run SiLU on the packer thread before pack (`apply_activation_from_pack`, `sfpu_activation_helpers.hpp:159-172`) so it overlaps the next group's 4 multiplies |
| Reader issue rate at tiny `block_ch_tiles` | decode (w = 1, 2) issues 35 reads of 64–128 B per tile-row; the reader may be issue-bound | split the stick reads across NCRISC + BRISC (`split_reader`) or trid double-issue |
| Bank alignment of token ranges | cores of different channel groups with the same slot start on the same DRAM bank | stagger each core's first tile-row (`bank_stagger`) |
| Packed tap tile (`bcast_row_idx`) | a non-zero `bcast_row_idx` adds an address offset per srcB unpack | 4 separate tap tiles per channel tile (`4·w` pages, row 0 each) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| taps → `cb_tap_tiles` | bf16, tile, row `2p` valid | reader: `TensorAccessor` per tap, 2 × 32 B reads per (tap, channel tile) | other rows don't-care: row-broadcast math uses srcB row `bcast_row_idx` only (`llk_unpack_AB.h:27`, BH) |
| RM input → `cb_rows` body | bf16 row-major sticks of `64·w` B | reader: `accessor.get_noc_addr(token, 2·(offset + c0))` (`tensor_accessor.h:165`), 32 reads + 1 barrier per tile-row | one DRAM crossing per stick |
| TILE input → `cb_input_tiles` → `cb_untilized_rows` → `cb_rows` body | bf16 tiles → row-major tile-row (`32` sticks of `64·w` B, contiguous) | reader reads `w` tiles per tile-row; compute `untilize`; reader one self-aimed L1→L1 read (`local_copy_helpers_dataflow.hpp:57`) into the slot | contiguous: an untilized tile-row of `w` tiles IS 32 sticks of the slot |
| halo → `cb_rows` sticks `[0,3)` | bf16 sticks | `t0 == 0`: history pages 0,1,2 at byte offset `2·c0`; not the core's first block: L1→L1 copy of the previous slot's sticks `[32·block_rows, 32·block_rows+3)`; core's first block with `t0 > 0`: RM → input token rows `t0-3..t0-1`; TILE → rows 29..31 of the untilized tile-row `t0/32 − 1` | the previous slot is intact: only the reader writes slots, and only after `reserve_back` |
| `cb_rows` → `cb_window_tiles` | row-major → tiles (exact copy, both DEST widths) | compute `tilize` (asymmetric stick pages, caller-managed wait, slide-by-pop) | window p of tile-row m starts at stick `32m + p` |
| windows ⊙ taps → DEST → SiLU → `cb_output_tiles` | bf16 operands → DEST (fp32 if `fp32_dest_acc_en`, else 16-bit) → bf16 | compute `conv_silu_block` | ELWMUL accumulates into DEST (`llk_math_eltwise_binary.h` BH: MOP acc bit 0 at 158-195, no ZEROACC in the standard path 291-320, "ELWMUL accumulates into DEST" 547) |
| `cb_output_tiles` → q/k/v | bf16 tiles | writer: 3 `TensorAccessor`s, per-tile routing | DRAM or L1 interleaved per `memory_config` |

TARGET mapping (placement and refinement classes):

| TARGET axis value | Phase 0 | Class of the unlock |
|---|---|---|
| layout ROW_MAJOR / TILE | both built (two regimes) | — |
| in_buffer / history_buffer / taps_buffer = l1 | built (TensorAccessor reads where the tensor lies) | — |
| output_memory l1_interleaved | built (writer accessor; per-core L1 share of q/k/v subtracted from the CB budget) | — |
| fp32_dest_acc_en True | built (compute define; DEST_AUTO_LIMIT halves) | — |
| input_window column_window (refinement 1) | refused | **knob-turn**: RM byte offset `2·(offset + c0)` inside the input page; TILE page `r·W_t + offset/32 + c`. No CB, block or schedule change. |
| history_layout channel_paged (refinement 2) | refused | **knob-turn**: reader-only — history (row r, channel c) lives in page `c / P` at element `r·P + c mod P`; a group whose channels straddle pages issues one read per contiguous 32-aligned run per row. |
| sharded placements | not in TARGET; refused by validation | — |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows` tile-rows × `block_ch_tiles` channel tiles of one channel group (+3-row halo) |
| Grid | `device.compute_with_storage_grid_size()` (never hardcoded); `N = grid.x · grid.y` |
| Per-core work | one channel group `g`, one contiguous token range `[core_seq_start, core_seq_start + core_seq_tiles)` tile-rows; `num_blocks_this_core = ceil(core_seq_tiles / block_rows)`, `last_block_rows = core_seq_tiles − (num_blocks_this_core − 1)·block_rows` |
| Remainder | token tile-rows spread `tensor_seq_tiles // U` (+1 for the first `tensor_seq_tiles % U` slots); idle cores beyond `G·U` get no kernels |

`select_plan` (exact; `w` = `block_ch_tiles`, `T_t = tensor_seq_tiles`, `C_t = tensor_ch_tiles`, page = 2048 B):

| Step | Rule |
|---|---|
| footprint | `footprint(w, M) = w·(ROWS_DEPTH·64·(32M + 3) + 2048·(TAP_COUNT·M + 1 + OUT_DEPTH) + tile_extra)`, `tile_extra = 2048·(IN_DEPTH + UNTILIZED_DEPTH)` in the TILE regime else 0. With depths 2: RM `w·(12288M + 6528)`, TILE `w·(12288M + 14720)` bytes |
| budget | `L1_BUDGET = 1,048,576 − out_l1_share`; `out_l1_share = Σ_{X∈q,k,v} ceil(T_t·X_t / num_l1_banks)·2048` when `memory_config` is L1, else 0 |
| candidates | every divisor `w` of `C_t` with `w ≤ 255`, `G = C_t / w ≤ N`, `footprint(w,1) ≤ L1_BUDGET` |
| per candidate | `U = min(N // G, T_t, max_cores_per_group)`, `core_seq_tiles_max = ceil(T_t / U)`, `makespan = core_seq_tiles_max · w` |
| choose `w` | minimum makespan; ties → larger `w`; then fewer cores |
| choose `block_rows` | `M_fit = max M ≤ min(core_seq_tiles_max, max_block_rows)` with `footprint(w, M) ≤ L1_BUDGET`; `nb = ceil(core_seq_tiles_max / M_fit)`; `block_rows = ceil(core_seq_tiles_max / nb)` |
| core of (g, s) | index `i = g·U + s`, logical `(i % grid.x, i // grid.x)` (row-wise) |

Plans on the 11×10 Blackhole grid (N = 110):

| Case | T_t × C_t | block_ch_tiles | G × U = cores | core rows | block_rows | footprint |
|---|---|---|---|---|---|---|
| 001 decode 1 user | 1 × 80 | 1 | 80 × 1 = 80 | 1 | 1 | 18.8 KB |
| 002 decode 16 users | 2 × 80 | 2 | 40 × 2 = 80 | 1 | 1 | 37.6 KB |
| 003 prefill 128 | 4 × 80 | 4 | 20 × 4 = 80 | 1 | 1 | 75.3 KB |
| 004 prefill 4096 | 128 × 80 | 16 | 5 × 22 = 110 | 6 / 5 | 3 (5-row cores: 3 + 2) | 694 KB |
| 020 (T 2048, C 2048) | 64 × 64 | 8 | 8 × 13 = 104 | 5 / 4 | 5 (4-row cores: one ragged block) | 544 KB |

## Circular Buffers

`w = block_ch_tiles`, `M = block_rows`, tile page = 2048 B, stick page = `64·w` B. All data formats Float16_b (bfloat16).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_rows` | 0 | `64·w` (one stick of the group) | `ROWS_DEPTH·(32·M + 3)` | spans token (`32M+3` rows incl. halo) × channel (page width `w`); `ROWS_DEPTH` slots; streams over tap (4 windows read it) | bf16 | reader | compute | whole kernel |
| `cb_tap_tiles` | 1 | 2048 | `w` | spans channel `w` and tap (4 rows packed per tile); constant along token → resident, not scaled by `M` | bf16 | reader | compute | whole kernel (loaded once) |
| `cb_window_tiles` | 2 | 2048 | `TAP_COUNT·w·M` | spans token `M`, channel `w`, tap 4 — sequential-phase intermediate (tilize finishes the block before the MAC consumes it) | bf16 (exact copies; accumulation never passes through it) | compute | compute | windowed_tilize → conv, per block |
| `cb_output_tiles` | 3 | 2048 | `OUT_DEPTH·w` | spans channel `w`; streams token (tile-row by tile-row); tap reduced in DEST | bf16 (output ABI) | compute | writer | conv → store |
| `cb_input_tiles` (TILE regime only) | 4 | 2048 | `IN_DEPTH·w` | spans channel `w`; streams token | bf16 | reader | compute | load → untilize |
| `cb_untilized_rows` (TILE regime only) | 5 | 2048 | `UNTILIZED_DEPTH·w` (one tile-row = `w` pages = 32 sticks) | spans channel `w`; streams token | bf16 | compute | reader | untilize → copy into `cb_rows` |

Sync table (push = wait for every CB):

| CB | Producer quanta | Consumer quanta | Ring closes because |
|---|---|---|---|
| `cb_rows` | per block: 35 then 32 per further tile-row (= `32·M_b + 3`) | `wait_front(35)` per tile-row; pops 1,1,1,29 (… last tile-row 1,1,1,32) = `32·M_b + 3` | every non-last block is exactly one slot; the last block of a core never wraps |
| `cb_tap_tiles` | `w` once | wait `w` once, pop `w` at kernel end | single cycle |
| `cb_window_tiles` | `w` per tilize call (4 per tile-row) | wait `4·w·M_b` per block, pop `4·w·M_b` | full block = capacity; last block shorter, no later wrap |
| `cb_output_tiles` | `w` per tile-row (reserve/push) | wait/pop `w` per tile-row | quantum `w` divides `OUT_DEPTH·w` |
| `cb_input_tiles` | `w` per tile-row | untilize waits/pops `w` per tile-row | quantum divides capacity |
| `cb_untilized_rows` | `w` per tile-row | reader waits/pops `w` per tile-row | quantum divides capacity |

Every CB has exactly one producer kernel and one consumer kernel. `cb_rows` is produced only by the reader in both regimes (that is why the TILE body takes the `cb_untilized_rows` → reader hop: the halo is written by the reader, and two producers on one CB are illegal).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `load_taps` | `w` packed tap tiles | dataflow raw NoC reads (TensorAccessor) | tap0..3 tensors | `cb_tap_tiles`, push `w` | resident until kernel end |
| 1 | `load_rows_block` | `32·M_b + 3` sticks | RM: raw stick reads; TILE: `local_addr` self-read | RM input / history / previous slot / `cb_untilized_rows` (wait+pop `w` per tile-row) | `cb_rows`, pushes 35, 32, … | slot full |
| 2 | `untilize_rows_block` (TILE) | `(M_b + extra) × w` tiles | `compute_kernel_lib::untilize<w, cb_input_tiles, cb_untilized_rows>` | `cb_input_tiles`, `w` per tile-row, popped | `cb_untilized_rows`, `w` per tile-row | — |
| 3 | `windowed_tilize_block` | `M_b` tile-rows × 4 windows × `w` tiles | `compute_kernel_lib::tilize<w, cb_rows, cb_window_tiles, …, WaitMode::NoWait>` with asymmetric pages; caller does `cb_wait_front(cb_rows, 35)` per tile-row | `cb_rows`: 35 waited per tile-row, popped 1,1,1,29/32 | `cb_window_tiles`: `w` per call | slot fully popped at block end |
| 4 | `conv_silu_block` | `M_b × w` output tiles | custom block op over the compute API (gap recorded in API Mapping) | `cb_window_tiles` (`4·w·M_b`, waited once, popped once), `cb_tap_tiles` (`w`, resident) | `cb_output_tiles`, `w` per tile-row | windows empty |
| 5 | `store_block` | `M_b × w` tiles | dataflow raw writes (TensorAccessor ×3) | `cb_output_tiles`, `w` per tile-row | q / k / v tensors | — |

Windowed tilize call pattern per tile-row (the "slide by pop" that realizes the shift; all calls `num_blocks = 1`): `total_input_pages = 1` for p = 0, 1, 2 (tilize 32 sticks from the read pointer, pop 1), then `total_input_pages = 29` for p = 3 (`32` on the block's last tile-row, which also releases the 3 trailing sticks). The asymmetric path pops `min(32, total_input_pages)` after reading 32 rows from the read pointer (`tilize_helpers.inl:212-239`); with `WaitMode::NoWait` it issues no wait (`:221-223`). Init once per block: first call `InitUninitMode::InitOnly`, middle `Neither`, last `UninitOnly` (`tilize_helpers.hpp:31-40`).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api (required) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | RM: `(cb_rows, cb_rows, cb_window_tiles)`; TILE: `(cb_input_tiles, cb_input_tiles, cb_untilized_rows)` — once, first statement | — | — | — |
| `untilize_rows_block` | helper | `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:152-163` | `<w, cb_input_tiles, cb_untilized_rows, InitAndUninit, WaitBlock>(in, out, M_b + extra)` | `cb_input_tiles` | `cb_untilized_rows` | `block_width_tiles = block_ch_tiles`; `num_blocks = M_b (+1)` |
| `windowed_tilize_block` | helper | `compute_kernel_lib::tilize` (asymmetric stick pages) | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` (asymmetric: `:133-141`) | `<w, cb_rows, cb_window_tiles, {InitOnly|Neither|UninitOnly}, WaitMode::NoWait>(rows, win, 1, {1|29|32})` | `cb_rows` | `cb_window_tiles` | `block_width_tiles = block_ch_tiles`; calls per block = `4·M_b` |
| `conv_silu_block` — reconfig | raw_api | `reconfig_data_format(icb0, icb1)`, `pack_reconfig_data_format(ocb)` | `tt_metal/hw/inc/api/compute/reconfig_data_format.h:162, 472` | `(cb_window_tiles, cb_tap_tiles)`, `(cb_output_tiles)` once per block | — | — | — |
| `conv_silu_block` — MAC init | raw_api | `mul_bcast_rows_init(icb0, icb1)` | `tt_metal/hw/inc/api/compute/bcast.h:452-457` | `(cb_window_tiles, cb_tap_tiles)` once per block | — | — | — |
| `conv_silu_block` — MAC | raw_api | `mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst, bcast_row_idx)` | `bcast.h:191-203` | ×4 per output tile into the SAME `idst`: `itile0 = m·4w + p·w + c`, `itile1 = c`, `bcast_row_idx = 2p`, p = 0..3 ascending | `cb_window_tiles`, `cb_tap_tiles` | DEST | — |
| `conv_silu_block` — SiLU | raw_api (via the chain's documented extension point if the implementer wraps it) | `silu_tile_init()`, `silu_tile<DST_ACCUM_MODE>(idst)` | `tt_metal/hw/inc/api/compute/compute_kernel_api.h:192-213` | init once per block; one call per output tile after its 4th MAC | DEST | DEST | — |
| `conv_silu_block` — DEST/pack | raw_api | `tile_regs_acquire/commit/wait/release`, `pack_tile` | `reg_api.h:50,96,63,106`; `pack.h:87` | per group of ≤ `DEST_AUTO_LIMIT` tiles | DEST | `cb_output_tiles` | group size ≤ `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`) |
| `load_taps`, `load_rows_block`, `store_block` | raw_api (dataflow) | `TensorAccessor::get_noc_addr(page, offset)`, `noc_async_read`, `noc_async_read_barrier`, `noc_async_write`, `noc_async_write_barrier` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `dataflow_api.h:547, 876, 1951` | — | tensors | CBs / tensors | read size `64·block_ch_tiles` B |
| TILE body copy / L1 halo copy | helper | `dataflow_kernel_lib::local_addr` (self-aimed read) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` | src = `cb_untilized_rows` read ptr or previous slot address | `cb_untilized_rows` / `cb_rows` | `cb_rows` | copy size `2048·block_ch_tiles` B / `3·64·block_ch_tiles` B |

**Helpers considered and rejected:**

| Raw entry | Candidate helper | File:line of the mismatch | Concrete reason / what would close the gap |
|---|---|---|---|
| `conv_silu_block` (MAC + SiLU) | `eltwise_chain` with 4 `BinaryFpu<Mul, …, BroadcastDim::Row, Dst::D0, DestAccumulation::PerRow>` + an SFPU SiLU element (custom `ckl::UnaryOp`, precedent `layernorm.cpp:44`) + `PackTile<…PerOuter…>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl:3397` (`SlotBase = dest_accumulation ? 1 : 0`), `:3335-3337` (lane width = transient lane width under accumulation), `:1903-1910` (DEST-only elements get a transient lane) | In a DEST-accumulating chain every DEST-only (SFPU) element runs on a transient lane at slot ≥ 1, never on the sticky accumulated D0, so "accumulate 4 products, then SiLU the sum" is not expressible in one chain. Also `BinaryFpu::exec` calls `mul_tiles_bcast<bt>` without `bcast_row_idx` (`chain.inl:1437-1470`), so the packed-tap tile is not addressable. Two chains with an intermediate `cb_preact` (Float32 under fp32 DEST) would add a buffer and a pack+unpack per tile — rejected by Rule 3 (fold into the accumulator). Gap closer: a chain option to apply post-accumulation DEST-only elements on the sticky slot, plus `bcast_row_idx` on `BinaryFpu`. |
| `conv_silu_block` | 4 non-accumulating `BinaryFpu` into D0..D3 + 3 SFPU `AddBinary` + SiLU (expressible in one chain) | — (expressible) | rejected on cost: 3 extra SFPU passes per output tile on the math thread that SiLU already saturates (~+30%); FPU accumulation in DEST is free |
| `load_rows_block` (RM) | `dataflow_kernel_lib::read_sticks_for_tilize<cb, TilizeGranularity::ROW>` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl:146-157` | barriers after EVERY stick (`:155`), and has no notion of the 3-row halo prefix from a different source (history / previous slot); the catalog `double_buffer` trap. Gap closer: a per-block batched-barrier mode with a caller-supplied prefix. |
| `load_taps` | `read_sticks_for_tilize` / any helper | — | the packed-tap layout (row `2p` of a tile) is op-specific; no helper writes tile face-rows |
| `store_block` | `write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp` | outputs are TILE pages routed to three tensors; the helper writes RM sticks to one tensor |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `conv_silu_block` | ELWMUL accumulate (×4) | `cb_window_tiles`: All (32 token rows × 32 channels, exact input data) | `cb_tap_tiles`: row `2p` (tap p for 32 channels; other rows don't-care) | Row (`BroadcastType::ROW`, `bcast_row_idx = 2p`): row `2p` replicated to all 32 output rows |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Window straddles the `cb_rows` ring end | tilize reads 32 contiguous sticks from the read pointer; a sliding window over a ring would eventually wrap | one block per whole slot; push = pop = `32·M_b + 3` per block; capacity is an exact multiple; only the last block is short |
| Pop count mismatch in the windowed tilize | pops are 1,1,1,29 per tile-row and 1,1,1,32 on the block's last row — an off-by-3 leaves the next block misaligned by 3 rows (silently wrong data) | the 3 trailing pops are bound to the LAST tile-row of each block; acceptance test pins multi-block cores (`plan_limits`) for both regimes |
| ELWMUL accumulation assumption | `mul_tiles_bcast_rows` ×4 into one slot relies on ELWMUL adding into DEST and on DEST being zero at acquire (released halves are cleared by the packer) — the same precondition the chain's DestAccumulation uses | fixed tap order p = 0..3; the fp32 cancelling-taps case (011) and every PCC case fail loudly if a product overwrites instead of accumulating |
| Halo source on non-first blocks | the halo must be the last 3 sticks of the PREVIOUS block (`[32·block_rows, 32·block_rows + 3)` of the previous slot), not of the current slot | the previous block is always full-height (only a core's last block is short) |
| TILE first-block halo with `t0 > 0` | rows 29..31 of tile-row `t0/32 − 1` must come from the input, not history | compute untilizes one extra tile-row on that block (`extra = 1`, RT flag); reader copies only its sticks 29..31 |
| Tap tile garbage rows | only rows 0, 2, 4, 6 of each `cb_tap_tiles` page are written | row-broadcast unpack/math use the selected row only (`llk_unpack_AB.h:27`); never use these tiles in a non-broadcast op |
| `bcast_row_idx` address offset | offset is computed from srcB's L1 format; tap row `2p` must be 16 B-aligned and inside faces 0/1 | rows `0,2,4,6` → offsets `0,64,128,192` within faces 0/1 |
| fp32 bound (case 011) | a 16-bit intermediate anywhere in the sum breaches `2^-12 Σ|x_j w_j|` | products formed on bf16 operands (exact), summed in fp32 DEST, SiLU on the fp32 sum, one bf16 rounding at pack; no CB carries a partial sum |
| SiLU precision | `math_approx_mode=True` is refused; `silu_tile` uses `APPROX` from the compute config | validation refuses approx; `silu_tile<DST_ACCUM_MODE>` |
| Reader/compute cycle in the TILE regime | compute → `cb_untilized_rows` → reader → `cb_rows` → compute | reader loop per tile-row: read tiles j+1, then wait/copy untilized j; compute untilizes the whole block before tilizing it; depth 2 on both hop CBs |
| Plan-dependent paths only on some grids | WH 8×8 and BH 11×10 produce different `block_ch_tiles`, `cores_per_group`, `block_rows` | `select_plan` uses the device grid; `plan_limits` tests pin the multi-block / ragged / L1-halo paths independent of the grid |
| Mesh | same program per device (SPMD descriptor); `sequence_parallel_axis` of size > 1 would need predecessor_carry | refuse (NotImplementedError) when the mesh axis has size > 1; `actual_start` is never read |
| L1 budget with L1 outputs | q/k/v interleaved in L1 consume L1 on the same cores | `out_l1_share` subtracted from the budget in `select_plan` |
| Kernel-config region (BB-04) | per-core RT args must stay small in every build flavour | RT args are a handful of scalars per kernel (addresses, group/range indices, halo kind); widths/knobs are CT |
