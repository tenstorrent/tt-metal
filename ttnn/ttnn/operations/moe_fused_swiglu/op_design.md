# Operation Design: moe_fused_swiglu

One MoE routed-expert block as ONE device program: three bfp4 matmuls with a fused SwiGLU
between them, DRAM in / DRAM out, `h` never materialised outside L1, token count read from a
device tensor at kernel runtime.

```
h   = SiLU(x @ W_gate) * (x @ W_up)      # [count, 2048]  INTERNAL, L1 only
out = h @ W_down                          # [capacity, emb]
```

---

## 1. Blocking Model

This section is the design. Everything below it is a realisation of the table.

### 1.1 Axes and characters

| Axis | Extent (tiles) | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|----------------|-----------|-------------------|---------------|-----------------|--------------|
| **M** — token rows | `M_t = ceil(count/32)`, **RUNTIME**, ≤ `input_m_tiles` | **independent** — the token axis is free; nothing reduces across tokens (feature_spec.py:25-27) | `M_BLOCK` | **16** (= 512 tokens) | **NOT split across cores.** Sequential outer loop `m_blocks = ceil(M_t/M_BLOCK)`. Runtime bound, so it *cannot* be a core-assignment axis (see §1.4) | scheme-change (needs a compile-time-fixed grid for a runtime extent) |
| **Kg** — emb, as gate/up contraction | `EMB_T = emb/32` ∈ {192, 224} | **dependent** — gate/up sum over emb | `KB1` (K tiles per matmul K-block) | **6** | **split across the 10 grid ROWS**: row `y` owns `Kr(y)` emb tiles, `split_work_to_cores(EMB_T,10)`. Cross-row partial sum = the phase-1 reduce (§4.3) | knob-turn (`KGROUPS`) |
| **Hn** — hidden 2048, as gate/up output | `HID_T = 64` | **independent** — each hidden column of `h` is computed in isolation | `HN_BLOCK` (hidden tiles per matmul out-block) | **2** | **split across the 13 grid COLUMNS**: column `x` owns `hn(x)` hidden tiles (`5` for x<12, `4` for x=12) | knob-turn |
| **Kh** — hidden 2048, as `down` contraction | `HID_T = 64` | **dependent** — `down` sums over hidden | `HN_PAD` (hidden tiles per phase-2 K-block) | **5** (13 K-blocks × 5 = 65 slots: 64 real + 1 zero) | **NOT split across cores** — kept sequential inside each core. This is the decision that avoids the `[count, emb]` cross-core reduction (§1.3) | scheme-change |
| **Ne** — emb, as `down` output | `EMB_T` ∈ {192, 224} | **independent** — each output column block is isolated | `EC` (emb tiles per core) | `ceil(EMB_T/130)` = **2** | **split across all 130 cores** (`split_work_to_cores(EMB_T,130)`) | knob-turn |
| **x** — activation, over Hn | — | **reuse-shared** — the same `x[:, Kr(y)]` feeds every hidden column, i.e. every one of the 13 cores in row `y` | `XSTAGE` (tile-rows staged per injector) | **1** | one rotating injector per tile-row per row; **multicast along the row** | realised in phase 1 (not deferred) |
| **h** — over Ne | — | **reuse-shared** — every core's `down` needs *all* 64 hidden columns | — | — | **grid-wide multicast**, 13 rounds, one per producing column | realised in phase 1 |

**Grid:** `13 × 10 = 130` logical worker cores — the full Blackhole p150 compute grid
(`tt_metal/core_descriptors/blackhole_140_arch.yaml:11`, `compute_with_storage_grid_range [0,0]..[12,9]`).
Phase 1 and phase 2 both run on **all 130 cores**. There is no single-core path and no idle
sub-grid.

**Buffer-depth knobs** (per streaming CB): `DEPTH_W = 2` (weight CBs), `DEPTH_PART = 2`
(reduce payload), `DEPTH_H = 3` (h all-gather / phase-2 in0 — 3 not 2 so the producer of a late
round is not flow-controlled by its own consumption), `DEPTH_OUT = 2`, `DEPTH_XSTICKS = 48` pages
(1.5 tile-rows of row-major sticks). Resident (depth-1) CBs: `cb_x_tiles`, `cb_gate_interm`,
`cb_up_interm`, `cb_out_interm`.

**Read-coalescing knob:** `WRUN` — the number of *bank-contiguous* weight tiles fetched per
`noc_async_read`. Phase-1 value: the maximal run implied by the ownership set (2–5 tiles for
W_gate/W_up, 2 tiles for W_down). See §1.5.

### 1.2 Why this split — bandwidth ranking of every candidate

`read_bytes` at emb 7168 is **87 % weights** (24.772 MB of bfp4, count-independent) and 13 %
activation (feature_spec.py:197-212). So the ranking criterion is: *which split reads each weight
byte exactly once, from the fewest, largest DRAM transactions, with the least NoC replication?*

| Candidate primary split | Weight bytes from DRAM | NoC bytes into each core | Verdict |
|---|---|---|---|
| **Hn across cores + Kg across cores (chosen)** | ×1 — every weight tile owned by exactly one core, no weight multicast at all | `\|x\|/KGROUPS` + `\|h\|` + one partial per reduce level | **chosen** |
| Hn only (no Kg split) | ×1 | `\|x\|` — 3.67 MB/core at count 256, ≈ 105 GB/s/core over the phase-1 budget | rejected: exceeds per-core NoC ingress (86 GB/s/NoC, `noc_parameters.h:12,287-290`) |
| M across cores (rows of tokens) | ×1 only if W is multicast down the M-split; DRAM ×1, but the injector must read its whole column | — | **rejected structurally**: `M_t` is a *runtime* value. A compile-time grid keyed on M idles `1 - M_t/(32·M_BLOCK·rows)` of the machine at the graded counts (M_t = 4 at count 128). |
| Kh across cores (split `down`'s contraction) | ×1 | a `[M_t, EMB_T]` partial per core — 16×224 = 3584 tiles ≈ 3.9 MB bfp8, reduced 10-deep | rejected: the reduction payload is **57× larger** than the `h` all-gather it replaces (§1.3) |
| Ne across cores for phase 2 (chosen) | ×1 | `\|h\|` = 240 tiles/K-block | **chosen** |
| Weight multicast (one reader, fan out) | ×1 DRAM but one core serialises the whole stream | — | rejected: `shared_input_reuse` measured 1.71× not 11× *because* "the single injector reads the whole stream serially" (`examples/shared_input_reuse/report.md`). With 24.8 MB there is no injector that can keep up. |

**The structurally hard part, resolved.** `W_down` contracts over the 2048 axis, which is the
*output* axis of gate/up, so any cross-core split of it makes every core hold a partial
`[count, emb]` output. Instead of splitting it, this design **rotates the axis between the two
phases** and pays for the rotation with an all-gather of `h`:

| | payload moved across cores | at count 256, M_BLOCK 16 |
|---|---|---|
| all-gather `h` (chosen) | `M_t × HID_T` tiles, once, bfp8, multicast | 8×64 = 512 tiles = 0.56 MB |
| reduce `[M_t, EMB_T]` partials 10-deep | `M_t × EMB_T` tiles per level | 8×224 = 1792 tiles = 1.95 MB **per level** |

`h` is narrow (2048) and the output is wide (emb): moving `h` is **≥ 3.5× cheaper per hop and
≥ 14× cheaper in total**, and it lands as a *broadcast* (one multicast per producer) rather than a
*reduction* (a tree with an add at every node). The `h` all-gather is additionally fused into
phase 2's K-block stream (§4.4), so it costs no extra L1 residency and overlaps compute.

**Cross-core reduction is still required** — for gate/up, over `Kg`. It is cheap *because* the
reduced object is `h`-shaped, not `out`-shaped: `M_BLOCK × HN_BLOCK` = 32 tiles = 35 KB per
sub-block per matrix. That is the same trade seen from the other side.

### 1.3 Operand-reuse check (mechanical, per (operand, chosen split) pair)

| Operand | Varies along the split it is read against? | Consequence |
|---|---|---|
| `W_gate`, `W_up` | vary along **both** Hn (columns) and Kg (rows) | fully disjoint per core → **no broadcast, no redundancy** |
| `W_down` | varies along Ne (the phase-2 split) | fully disjoint per core → **no broadcast** |
| `x` | does **not** vary along Hn (the column split) | **reuse-shared by construction** → multicast along the row (§4.2). Without this, `x` would be read 13× from DRAM (+47.7 MB at count 256, i.e. +168 % of the graded byte count). |
| `h` | does **not** vary along Ne (the phase-2 split) | **reuse-shared by construction** → grid-wide multicast (§4.4) |
| `counts`, `idx` | do not vary at all | one page each, read redundantly by every core's reader and writer — 260 × 64 B, below noise. Not worth a broadcast. |

### 1.4 Everything is a parameter; each has one source of truth

| Knob | Single source | Everything derived from it |
|---|---|---|
| `M_BLOCK` | host constant | `cb_x_tiles`/`cb_gate_interm`/`cb_up_interm`/`cb_h_local`/`cb_h`/`cb_out_interm`/`cb_out_tiles` page counts, `in0_num_subblocks`, the `m_blocks` runtime loop bound, the number of x-staging rounds |
| `KGROUPS` (=10) | host constant = grid height | `Kr(y)` per row, `cb_x_tiles` pages, the reduce depth |
| `HGROUPS` (=13) | host constant = grid width | `hn(x)` per column, `HN_PAD`, the number of h-mcast rounds |
| `HN_BLOCK`, `KB1` | host constants | `cb_w_gate`/`cb_w_up` pages, `cb_*_interm` pages, `num_k_blocks`, the reduce payload |
| `HN_PAD` | `ceil(HID_T/HGROUPS)` | phase-2 `in0_block_k`, `cb_h`/`cb_h_local` pages, `cb_w_down` pages |
| `EC` | `split_work_to_cores(EMB_T,130)` | `cb_out_interm`/`cb_out_tiles` pages, `cb_w_down` pages, `out_subblock_w` for phase 2 |
| `DEPTH_*` | host constants | the multiplier on the corresponding CB's page count |
| grid | `device.compute_with_storage_grid_size()` | `KGROUPS`, `HGROUPS`, every core-assignment formula |

No block factor, buffer depth, tile count, or core count is written twice or inlined. `emb`,
`capacity` and `hidden` are compile-time; **only `count` is runtime**, and it enters exclusively
as the trip count `m_blocks` and the per-block `m_tiles`, never as a CB size.

`out_subblock_h = 4`, `out_subblock_w = min(block_w, 2)` everywhere → DEST = 8 bf16 tiles =
`DEST_AUTO_LIMIT` at half-sync with `fp32_dest_acc_en=false`
(`dest_helpers.hpp:89-103`; `tensix_types.h:191-197`). `matmul_output_subblock` measured that the
win tracks subblock **size, not shape** (all four 8-tile shapes 1.46×, `2×2` 1.40×), so a skinny
`M_t` costs nothing here. `m_tiles` is rounded **up** to a multiple of `out_subblock_h` — legal
because rows past `count` are UNDEFINED (feature_spec.py:24-30).

### 1.5 Coalesced weight reads — the DRAM decision

Blackhole p150 has **8 DRAM banks** and interleaved page → bank is `page_id % 8`, with in-bank
slot `page_id / 8` at stride `aligned_page_size`
(`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:18-42, 287-300`). bfp4 tile = 576 B =
9 × 64 → already `DRAM_ALIGNMENT`-aligned (`tt_backend_api_types.hpp:125-126`,
`noc_parameters.h:391-394`), so `aligned_page_size == tile_size` and **pages
`p, p+8, p+16, …` are physically contiguous inside one bank.**

For all three weight tensors the tile-row stride is a multiple of 8 (`HID_T=64`,
`EMB_T ∈ {192,224}`), so `bank(kt·stride + nt) = nt % 8`: a column's tiles live entirely in one
bank, and a **stride-8 run of `L` columns at fixed `kt` is ONE `noc_async_read` of `L × 576` B**.

Ownership sets are therefore assigned as **maximal stride-8 runs**: hidden linear index
`i ∈ [0,64)` maps to `nt = (i / 8) + 8·(i % 8)`, and column `x` owns the contiguous linear range
`[start(x), start(x)+hn(x))`, which the host decomposes into 1–2 runs (mean run ≈ 2.8 tiles).
Same construction for `W_down`'s and the output's `EMB_T` axis.

Why this matters, from measurement (`examples/double_buffer/report.md:105-110`): a single core is
**transaction-rate-limited at ~8–9 M completed transactions/s (~115 ns each)**, so
`achieved GB/s ≈ rate × bytes-per-transaction`.

| bytes/transaction | modelled per-core read BW | 130-core transaction capacity | headroom over the 263 GB/s the count-256 target needs |
|---|---|---|---|
| 576 B (1 tile, the naive read) | ~5.0 GB/s | 650 GB/s | 2.5× |
| ~1670 B (mean run, W_gate/W_up) | ~14 GB/s | 1.8 TB/s | 7× |
| 1152 B (2 tiles, W_down / out) | ~10 GB/s | 1.3 TB/s | 5× |

Coalescing is not required to *reach* the target (1-tile reads already clear it on paper) — it is
taken because the 8.7 M/s figure is a Wormhole **single-core, no-contention** measurement and this
op runs 130 cores against 8 banks. `WRUN` is a knob: setting it to 1 reproduces the naive
per-tile read for ablation. The `/perf-measure` ablation should A/B exactly this.

Reads are issued from **both** data-movement RISC-Vs where a phase has ≥ 2 independent streams
(reader takes `W_gate`, writer takes `W_up`, on NoC0/NoC1 respectively) — the
`sparse_sdpa_gather` dual-issue split, measured at 1.2–1.7× when RISC-V-issue-bound
(`examples/split_reader/report.md`). Batching: ≥ 4 reads outstanding per barrier, plateau at 8
(`double_buffer/report.md`); `block=1 + barrier` is the measured worst cell and is forbidden.

### 1.6 One structure, not two — and why

The prompt allows two structures behind a predicate on the runtime count. **This design carries
one**, because the structure that serves the graded counts degrades *gracefully*, not
catastrophically, at `count = capacity`:

| count (emb 7168) | `m_blocks` at M_BLOCK 16 | real weight reads | real read bytes | tile-matmuls / core | compute floor* | DRAM floor @512 GB/s | graded target |
|---|---|---|---|---|---|---|---|
| 128 | 1 | ×1 | 26.60 MB | 1 419 | 11 µs | 52.0 µs | 91.80 µs |
| 256 | 1 | ×1 | 28.44 MB | 2 647 | 21 µs | 55.5 µs | 108.00 µs |
| 512 | 1 | ×1 | 32.11 MB | 5 293 | 42 µs | 62.7 µs | 161.82 µs |
| 5120 | 10 | ×10 | 296.4 MB | 52 923 | 423 µs | 579 µs | none |

\* 8 cycles per 32×32×32 tile-matmul at 4096 mul-adds/cycle LoFi
(`matmul_device_operation.cpp:2654-2661`, `GEMM_FLOPS.md:54-69`) × 1.35 GHz
(`blackhole_implementation.hpp:274`), scaled by the ~1.35 unpack:math ratio that bfp8 `in0` +
bfp4 `in1` gives (5 504 unpack bytes per 8 tile-matmuls at 64 B/cycle vs 64 math cycles).

Two readings of that table drive the whole design:

1. **All three graded counts fit in ONE M-block** at `M_BLOCK = 16`. The weights are therefore
   read **exactly once**, and the graded byte model (feature_spec.py:210-212) is not just a
   grading convention — it is what the op actually reads. This is the single most load-bearing
   knob value in the design.
2. At `count = capacity` the op is **compute-bound** (423 µs of tile-matmul vs 579 µs of DRAM at
   *peak*, i.e. the re-read hides only partially) — but a token-axis cross-core split would not
   reduce the tile-matmul count at all, so the "second structure" buys nothing on the axis that
   binds. What it would buy is removing the ×10 weight re-read; the cheap way to get that is to
   **raise `M_BLOCK`**, which is a knob, not a rewrite. `count = capacity` is reported, not
   targeted (feature_spec.py:344-347).

**`bfp8` for both `in0` operands is a decision, not an accident.** `x` and `h` are held as
`bfloat8_b` tiles (1088 B) rather than bf16 (2048 B) because `in0` unpack bytes are the
matmul's near-critical path: bf16 `in0` + bfp4 `in1` is ~3:1 unpack:math (unpack-bound, matching
the 37 cycles/tile-matmul that `matmul_output_subblock` measured with bf16 operands), while
bfp8 `in0` + bfp4 `in1` is ~1.3:1. It also halves the `x` and `h` multicast bytes and the
resident-`x` L1 footprint. Precision cost is negligible against a bfp4 weight (§8).

### 1.7 Lamps — scheme-changes phase 1 deliberately leaves reachable

| Lamp | What unlocks it | Why phase 1 does not foreclose it |
|---|---|---|
| **Token-axis cross-core split** (the second structure, for `count ≈ capacity`) | the compile-time grid becomes `(M-rows) × (Hn) × (Kg)` with `W_gate/W_up` multicast down the M-rows | `M_BLOCK` is already the outer loop; the grid factorisation is derived from one host expression; the weight readers already take a `(page_base, run)` list, which a multicast sender emits unchanged |
| **`Kh` (hidden) cross-core split for `down`** — the `[count, emb]` reduction the prompt names | an `EMB_T`-chunked cross-core reduce over the phase-2 K axis | phase 2 already accumulates through an `interm` CB with `packer_l1_acc`; a chunked cross-core add over `cb_out_interm` is an addition to, not a replacement of, that loop |
| **Weight multicast** (share `W_down` across an M-split) | `mcast_pipe` `SenderPipe` on the weight CB | the weight CB is already a plain streaming CB with one producer; swapping DRAM-read for mcast-receive changes only the reader |
| **Raise `M_BLOCK` to 32** (kills the ×10 re-read at capacity 5120) | `M_BLOCK` + trimming `DEPTH_H` to 2 | L1 is 1 317 KB of a 1 461 376 B budget (§5); nothing is sized by `capacity` |
| **`WRUN` sweep / DRAM-sharded weights** | `WRUN` knob; the run-list reader is layout-agnostic | reads are already expressed as (bank-contiguous run) rather than (tile) |
| **`HEIGHT_SHARDED` activation** (TARGET axis, not yet in the contract) | replace the x-staging read with `ttnn.cb_descriptor_from_sharded_tensor` — the resident shard *is* the per-core block, consumed in L1 with **no NoC read** | the x path already has a distinct "stage → tilize → mcast" stage that a sharded input simply skips; `M_BLOCK` would be pinned by the shard |

The prompt's contract fixes every tensor as DRAM interleaved, so the `*_SHARDED` values of
`TARGET["memory_layout"]`-equivalents do not appear in `feature_spec.py` for this op. The logical
shard is nevertheless what this design *is*: the row split is a logical height-shard of `x`
(knob-turn, no combine); the column split is a logical width-shard of `W_gate`/`W_up` whose
partials cross cores (scheme-change, already paid for in §4.3).

---

## 2. Overview

| Field | Value |
|-------|-------|
| Classification | fused (3 matmuls + SwiGLU + fused tilize + 2 collectives, one program) |
| Goal | Compute one routed expert's FFN block on device, honouring a device-resident token count, at maximum DRAM **read** bandwidth utilisation. |
| Math | `h = SiLU(x @ W_gate) * (x @ W_up)`; `out[0:count] = h @ W_down`; `out[count:capacity]` UNDEFINED |
| Mode | Hybrid (helper-first compute, hand-written coalesced dataflow) |
| References | `feature_spec.py`; `eval/op_template.py`; `.claude/references/generic_op_template/`; `references/cross_core_reduction_design.md`; `ttnn/cpp/ttnn/kernel_lib/`; `examples/master.md` entries `double_buffer`, `split_reader`, `noc_placement`, `matmul_output_subblock`, `shared_input_reuse`, `tensix_all_reduce`, `compute_block_size`, `compute_fusion`; `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/extract/kernels/dataflow/reader_extract.cpp` (device-side count read); `matmul_multicore_reuse_mcast_2d_program_factory.cpp` (mcast matmul CB set) |

### Parameters

| Name | Type | Required | Valid range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (`x`) | Tensor `(1,1,capacity,emb)` | yes | bf16 ROW_MAJOR **or** bfp8_b TILE, DRAM interleaved | — | — |
| `w_gate`, `w_up` | Tensor `(emb, 2048)` | yes | bfp4_b TILE, DRAM interleaved | — | — |
| `w_down` | Tensor `(2048, emb)` | yes | bfp4_b TILE, DRAM interleaved | — | — |
| `counts` | Tensor `(num_global_experts,)` | yes | uint32 ROW_MAJOR, DRAM interleaved, one page | — | RT (device) |
| `global_expert_idx_table` (`idx`) | Tensor `(num_local_experts,)` | yes | uint32 ROW_MAJOR, DRAM interleaved, one page | — | RT (device) |
| `local_expert_id` | int | yes | `[0, len(idx))` | — | **CT** |
| `input_m_tiles` | int | no | `[1, capacity/32]` | `capacity // 32` | **CT** (`M_T_MAX`) |
| `dtype` | DataType | no | `bfloat8_b` | `bfloat8_b` | CT |
| `memory_config` | MemoryConfig | no | DRAM interleaved | DRAM interleaved | host |
| `compute_kernel_config` | ComputeConfigDescriptor | no | — | `default_compute_kernel_config()` | CT |
| `emb` | derived `x.shape[-1]` | — | {6144, 7168} | — | **CT** |
| `capacity` | derived `x.shape[-2]` | — | {1024, 2048, 5120} | — | **CT** |
| `count` | `counts[idx[local_expert_id]]` | — | `[0, capacity]` | — | **RUNTIME, device-resident** |

`default_compute_kernel_config()` — exported from the op module, the single definition, `None`
resolves through it: `math_fidelity = MathFidelity.LoFi`, `math_approx_mode = True`,
`fp32_dest_acc_en = False`, `dst_full_sync_en = False`, `bfp8_pack_precise = True`,
`unpack_to_dest_mode` unset. LoFi because bfp4 carries ~4 mantissa bits and higher fidelity only
costs FPU passes (`fidelity_sweep/reduce_accumulate_fidelity_report.md`: LoFi ≈ HiFi2 within
0.5 %, HiFi4 +50–88 %). `fp32_dest_acc_en=False` keeps `DEST_AUTO_LIMIT = 8`.
`bfp8_pack_precise=True` for the bfp8 `h` and output.

### Tensors

**Input `x`** — `(1,1,capacity,emb)`; `bfloat16` ROW_MAJOR (page = one row = `emb*2` B) or
`bfloat8_b` TILE (page = 1088 B); DRAM interleaved. Rows `[count, capacity)` are arbitrary bytes.

**Weights** — `w_gate`/`w_up` `(emb,2048)`, `w_down` `(2048,emb)`, `bfloat4_b` TILE, page 576 B,
DRAM interleaved.

**Output** — `(1,1,capacity,emb)`, `bfloat8_b` TILE, page 1088 B, DRAM interleaved. Rows
`[0,count)` correct; `[count, ceil_tile(count))` written with tile-padding garbage;
`[ceil_tile(count), capacity)` **never touched** (not zeroed, not read).

---

## 3. Work Distribution

The Blocking Model's core-assignment, made concrete. Every formula is alignment-aware (`ceil`,
per-image) and every core count is derived from `compute_with_storage_grid_size()`.

| Field | Value |
|-------|-------|
| Work unit | one `(M-block, hidden-group)` gate/up block in phase 1; one `(M-block, emb-group)` output block in phase 2 |
| Grid | `13 × 10 = 130`, `CoreRange((0,0),(12,9))`; `HGROUPS = grid.x = 13`, `KGROUPS = grid.y = 10` |
| Outer loop | `m_blocks = ceil(M_t / M_BLOCK)`, **runtime**; `m_tiles = min(M_BLOCK, M_t - b*M_BLOCK)` rounded up to a multiple of `out_subblock_h` |
| Phase-1 per core `(x,y)` | rows: `Kr(y) = EMB_T/KGROUPS + (y < EMB_T % KGROUPS)` → 224: 4 rows × 23 + 6 × 22; 192: 2 × 20 + 8 × 19. columns: `hn(x) = HID_T/HGROUPS + (x < HID_T % HGROUPS)` → 64: 12 cols × 5 + 1 × 4 |
| Phase-1 sub-blocks | `ceil(hn(x) / HN_BLOCK)` calls per matrix, widths `2,2,1` (x<12) or `2,2` (x=12) |
| Phase-2 per core | `EC(i) = EMB_T/130 + (i < EMB_T % 130)` over the row-major core list, `i = y*13 + x` → 224: 94 cores × 2 + 36 × 1; 192: 62 × 2 + 68 × 1 |
| x-staging per core | injector for tile-row `t` iff `x == t % HGROUPS`; tile-rows `t ∈ [0, m_tiles)`; `ceil(M_BLOCK/HGROUPS)` = 2 tile-rows for `x < 3` |
| Remainder handling | every split above uses the explicit `base + (i < rem)` form; **no `floor`, no `//` on a tile count**. `M_t = ceil(count/32)` per image (leading dims are `(1,1)`). Cores with `EC = 1` finish phase 2 early and idle — 224/(130×2) = 86 % phase-2 grid efficiency, 192/(130×2) = 74 %. |
| `count == 0` | `M_t = 0` → `m_blocks = 0` → every kernel's outer loop body is skipped uniformly on all 130 cores. No CB push/wait, no multicast round, no semaphore. Cannot hang because the count is identical on every core and no collective is entered. Output tensor returned allocated and untouched. |
| Core placement | `row_wise = True` ordering; reader on NoC0 / writer on NoC1 (the default `ReaderConfigDescriptor`/`WriterConfigDescriptor` pairing that `examples/noc_placement/report.md` measured as 2.9× over the column default) |

**Compute regimes: exactly one.** There is no predicate on the runtime count that selects a
different loop nest, CB layout, or core assignment (§1.6). The only runtime variation is the trip
counts `m_blocks` / `m_tiles`. Therefore no regime-pinned tests are needed beyond the
count-pinned perf cases the golden suite already carries — but the acceptance test does pin
`count ∈ {0, tile-aligned, non-tile-aligned, capacity}` so the `m_tiles` tail is exercised.

**Device-side count read.** Every core's reader and writer independently perform, following
`reader_extract.cpp:104-117`:

```
idx_page   -> scratch;  g = idx_ptr[local_expert_id]      // local_expert_id is CT
counts_page-> scratch;  count = counts_ptr[g]             // NOT counts[0], NOT counts[local_expert_id]
M_t        = min( (count + 31) / 32, M_T_MAX )
m_blocks   = (M_t + M_BLOCK - 1) / M_BLOCK
```

Two one-page `noc_async_read`s into scratch CBs read through `get_write_ptr()` with nothing
pushed. The **compute** kernel cannot issue NoC reads, so the reader publishes `{count, M_t,
m_blocks}` as one 64 B page on `cb_token_meta`; compute `wait_front(1)`s, reads the words through
a `volatile tt_l1_ptr uint32_t*`, and pops at the very end of `kernel_main`. There is no host
readback and no host branch on the counts' contents.

---

## 4. Dataflow Strategy

```
                     DRAM                                     Tensix (core x,y)                       DRAM
 x  (bf16 RM sticks | bfp8 tiles) --NoC0--> cb_x_sticks --tilize--> [mcast along row y] --> cb_x_tiles
 W_gate slice (bank-run coalesced) --NoC0--> cb_w_gate  --matmul_block(K)--> cb_gate_interm
 W_up   slice (bank-run coalesced) --NoC1--> cb_w_up    --matmul_block(K)--> cb_up_interm
                                            cb_*_interm --pack--> cb_*_partial --[tree reduce down column x]--> cb_reduce_*_in
                                            root: add(+SiLU on packer) --> cb_gate_silu; mul(up) --> cb_h_local
                                            cb_h_local --[grid mcast, 13 rounds]--> cb_h
 W_down slice (bank-run coalesced) --NoC0--> cb_w_down  --matmul_block(Kh=13x5, packer_l1_acc)--> cb_out_interm
                                            cb_out_interm --pack bfp8--> cb_out_tiles --NoC1--> out
```

### 4.1 Format at each stage

| Stage | Format | Why |
|---|---|---|
| `x` in DRAM | bf16 RM sticks **or** bfp8 tiles | the contract's two production formats |
| `cb_x_sticks` | bf16, page = `Kr(y)*32*2` B (a *slice* of a stick) | one `noc_async_read` per stick per core, 1472 B at emb 7168 |
| `cb_x_tiles` (mcast payload, resident `in0`) | **bfp8_b** tiles | halves in0 unpack bytes and mcast bytes (§1.6); unifies both input formats after the read |
| gate/up K-accumulation | bf16 `interm` + `packer_l1_acc` | `packer_l1_acc` forces ≥ fp16_b for partials (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:107-109`) |
| reduce payload | bfp8_b | 2× less NoC than bf16, ~0.1 % error contribution (§8) |
| `h` | **bfp8_b** tiles | in0 of `down`; same unpack argument |
| `down` K-accumulation | bf16 `interm` + `packer_l1_acc` | same |
| `out` | bfp8_b tiles, coalesced 2-tile writes | contract |

### 4.2 Tensix→Tensix: the `x` row-multicast

| Property | Value |
|---|---|
| Group | one grid **row** `y` — `CoreRange((0,y),(12,y))`, 13 cores. All 13 need the identical `x[:, Kr(y)]`. |
| Sender | **rotating**: for round `t ∈ [0, m_tiles)`, sender = core `(t % 13, y)`. `McastArgs<CT,RT,SPAN=M_BLOCK>` (`mcast_pipe.hpp:327-395`), sender coords emitted host-side, one per round. |
| Payload per round | `Kr(y)` bfp8 tiles = tile-row `t` of `x`, restricted to row `y`'s emb range. 23 × 1088 = 25 KB at emb 7168. |
| Sender's work | read 32 sticks × `Kr(y)*64` B (bf16 path) or `Kr(y)` tiles as `ceil(Kr/8)` bank-runs (bfp8 path) → compute tilizes to bfp8 → writer multicasts |
| Sync | `DataReadySignal::Counter` + `consumer_ready` handshake, 2 semaphores. Receiver `receive(t)` (`mcast_pipe.hpp:274`); sender `send(src,dst,size)` (`:197`) with loopback into its own `cb_x_tiles`. |
| Ordering | round `t` lands at `cb_x_tiles` offset `t * Kr(y) * 1088`; `cb_x_tiles` is reserved once per M-block and pushed once after all `m_tiles` rounds → identical write pointer on every core, which `mcast_pipe` requires (`mcast_pipe.hpp:44-45`). |
| Fan-out correction | on Blackhole the mcast rect's ack count must drop 2 per row when it spans virtual columns 8/9; `McastRect::area()` already does this (`mcast_pipe.hpp:129-139`). A full 13-wide row spans them, so this is live. |
| Cost | 3.67 MB of `x` read from DRAM **once** total; `\|x\|/10 × 12/13` ≈ 180 KB received per core per M-block at count 256; `m_tiles` ≤ 16 handshakes ≈ 6 µs |

Rotating injectors are the fix for `shared_input_reuse`'s measured limitation ("the single
injector reads the whole stream serially"): each core injects ≤ 2 of the 16 tile-rows.

### 4.3 Tensix→Tensix: the gate/up partial reduce (the dependent-axis combine)

| Property | Value |
|---|---|
| Group | one grid **column** `x` — the 10 cores that split `Kg`. Not a multicast; unicast + counting semaphore. |
| Topology | binary tree, depth `ceil(log2 10) = 4`, root = core `(x, x % 10)` so the 13 roots spread over all 10 rows (SwiGLU + mcast-injection work is balanced). See `references/cross_core_reduction_design.md` for the topology comparison, the rectangular-group precondition and the silent-hang checklist; plain associative sum, **no Welford**. |
| Payload | per hidden sub-block, per matrix: `m_tiles × out_subblock_w` ≤ 32 tiles bfp8 = 35 KB. Every core sends **once** per level at most. |
| Levels | 2 semaphores (one per parity of level), payload lands in `cb_reduce_gate_in` / `cb_reduce_up_in`. |
| Add | non-root: `add<input(cb_*_partial), input(cb_reduce_*_in), output(cb_*_partial)>` (`eltwise_convenience.hpp:42-48`). Root, gate only: `add_bias_bcast_rows<Elementwise, SubblockMajor, NoPostBias, SiluActivation>` → SiLU rides the **packer thread** for free. |
| Pipelining | the hidden sub-blocks (3 per core) are reduced one at a time while the next is computed: transfer 35 KB (≈ 0.4 µs at 86 GB/s) vs compute per sub-block `m_tiles × Kr × HN_BLOCK` tile-matmuls (≈ 6 µs at m_tiles 16). Fully hidden; only the depth-4 fill (~1.6 µs) is exposed. |
| Ordering guarantee | level `l` semaphore is only signalled after `noc_async_write_barrier()`; each node waits for exactly its known child count. Payload address is a fixed `cb_reduce_*_in` slot per child parity, so no address negotiation. |

### 4.4 Tensix→Tensix: the `h` all-gather, fused into phase 2's K stream

| Property | Value |
|---|---|
| Group | the whole grid — `CoreRange((0,0),(12,9))`, 130 cores. |
| Rounds | `HGROUPS = 13`. Round `r`'s sender is column `r`'s reduce root, core `(r, r % 10)`. Rotating `McastArgs<…, SPAN=13>`, Counter signal + handshake, 2 semaphores. |
| Payload per round | exactly one phase-2 K-block: `m_tiles × HN_PAD` = 16 × 5 = **80 tiles bfp8 = 87 KB**, contiguous. |
| The padding trick | `HN_PAD = ceil(HID_T/HGROUPS) = 5`, so 13 rounds cover 65 hidden slots for 64 real columns. Column 12 (`hn = 4`) produces one **zero** tile-column locally to fill its round. Its matching `W_down` row is re-read (row 63 twice) and contributes nothing because `h`'s pad column is zero. This is what makes every round's payload uniform and **contiguous in the `in0` layout**, so each round is ONE `send()` instead of `m_tiles × runs` sends. |
| `in0` layout the sender must write | `cb_h` K-block order = `for sb in [0, m_tiles/4): for mr in [0,4): for k in [0,5): tile(m = 4*sb+mr, hidden = 5*r + k)` — i.e. exactly `matmul_block`'s `in0_subblock * (out_subblock_h*in0_block_k) + mr*in0_block_k + k`. The producer packs `cb_h_local` in that order directly out of the SwiGLU. |
| Streaming, not resident | `cb_h` is a depth-3 streaming CB (3 × 80 tiles = 131 KB), **not** a `m_tiles × 64` resident block (which would be 566 KB). Phase 2 consumes K-block `r` as round `r` lands, so the all-gather overlaps `down` compute and flow-controls itself through the CB. |
| Deadlock argument | every core runs the identical round loop `for r in 0..12 { reserve(80); if (r == my_column) {wait cb_h_local; send} else {receive(r)}; push(80) }`. Compute pushes `cb_h_local` **before** entering the phase-2 matmul, so a core that is producer for a late round never blocks its own consumption. Depth 3 > 1 guarantees a producer can be ≥ 1 round behind its consumers without stalling the ring. The sender self-excludes and loopbacks (`Mcast2D` sender-in-rect, `mcast_host.hpp:472-481`). |
| Cost | `m_tiles × 64` tiles bfp8 received per core per M-block = 0.56 MB at count 256 (≈ 18 GB/s over phase 2); 13 handshakes ≈ 5 µs. Multicast, not unicast: `tensix_all_reduce` measured 2.6–2.8× for mcast-all-gather over unicast at fan-out 8–16 (`examples/tensix_all_reduce/report.md`). |

---

## 5. Circular Buffers

Sizes at `M_BLOCK=16, Kr=23 (emb 7168), hn=5, HN_BLOCK=2, HN_PAD=5, KB1=6, EC=2`. **Every page
count is a function of the knobs — none is a function of `capacity`, `EMB_T`, or `count`.**

| Semantic name | Idx | Page size (B) | Num pages (formula) | pages | KB | Format | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|---|
| `cb_token_meta` | 0 | 64 | 1 | 1 | 0.1 | UInt32 | reader | compute | whole program |
| `cb_x_sticks` | 1 | `Kr*32*2` = 1472 | `DEPTH_XSTICKS` = 48 | 48 | 69 | Float16_b | reader | compute | per staging round |
| `cb_x_tiles` | 2 | 1088 | `M_BLOCK * Kr` | 368 | 391 | Bfloat8_b | reader (own tilize via loopback + remote mcast) | compute | resident per M-block |
| `cb_w_gate` | 3 | 576 | `DEPTH_W * KB1 * HN_BLOCK` | 24 | 14 | Bfloat4_b | reader | compute | streaming |
| `cb_w_up` | 4 | 576 | `DEPTH_W * KB1 * HN_BLOCK` | 24 | 14 | Bfloat4_b | writer (NoC1 dual-issue) | compute | streaming |
| `cb_w_down` | 5 | 576 | `DEPTH_W * HN_PAD * EC` | 20 | 11 | Bfloat4_b | reader | compute | streaming |
| `cb_gate_interm` | 24 | 2048 | `M_BLOCK * HN_BLOCK` | 32 | 64 | Float16_b | compute | compute | matmul K-accum |
| `cb_up_interm` | 25 | 2048 | `M_BLOCK * HN_BLOCK` | 32 | 64 | Float16_b | compute | compute | matmul K-accum |
| `cb_gate_partial` | 26 | 1088 | `DEPTH_PART * M_BLOCK * HN_BLOCK` | 64 | 70 | Bfloat8_b | compute | writer (reduce send) | per sub-block |
| `cb_up_partial` | 27 | 1088 | `DEPTH_PART * M_BLOCK * HN_BLOCK` | 64 | 70 | Bfloat8_b | compute | writer (reduce send) | per sub-block |
| `cb_reduce_gate_in` | 6 | 1088 | `M_BLOCK * HN_BLOCK` | 32 | 35 | Bfloat8_b | reader (remote child) | compute | per reduce level |
| `cb_reduce_up_in` | 7 | 1088 | `M_BLOCK * HN_BLOCK` | 32 | 35 | Bfloat8_b | reader (remote child) | compute | per reduce level |
| `cb_gate_silu` | 28 | 1088 | `M_BLOCK * HN_BLOCK` | 32 | 35 | Bfloat8_b | compute | compute | root only |
| `cb_h_local` | 29 | 1088 | `M_BLOCK * HN_PAD` | 80 | 85 | Bfloat8_b | compute | writer (mcast send) | per M-block |
| `cb_h` | 8 | 1088 | `DEPTH_H * M_BLOCK * HN_PAD` | 240 | 261 | Bfloat8_b | reader (remote mcast) | compute | streaming, 13 rounds |
| `cb_out_interm` | 30 | 2048 | `M_BLOCK * EC` | 32 | 64 | Float16_b | compute | compute | phase-2 K-accum |
| `cb_out_tiles` | 16 | 1088 | `DEPTH_OUT * M_BLOCK * EC` | 64 | 70 | Bfloat8_b | compute | writer | streaming |
| `cb_idx_scratch` | 9 | 64 | 1 | 1 | 0.1 | UInt32 | reader/writer (own, unpushed) | — | prologue |
| `cb_counts_scratch` | 10 | 1024 | 1 | 1 | 1 | UInt32 | reader/writer (own, unpushed) | — | prologue |
| **Total** | | | | | **1 353** | | | | budget **1 427 KB** (`bh_hal_tensix.cpp:91-92`, 1 461 376 B) |

74 KB slack. If the implementer needs more: `DEPTH_H = 2` frees 87 KB, `M_BLOCK = 8` frees
~430 KB (at the cost of 2 M-blocks at count 512 — see §1.6). 20 CBs of a 64 CB limit
(`circular_buffer_constants.h:37`).

**Ownership.** Every row above names exactly one producer and one consumer.
- `cb_w_up` is produced by the **writer** kernel, not the reader — that is the deliberate
  dual-NoC read split (§1.5), not an accident.
- `cb_gate_partial` / `cb_up_partial` are produced by compute and consumed **only** by the writer
  (which unicasts them to the reduce parent). Incoming partials land in the separate
  `cb_reduce_*_in`, produced by the reader (which pushes after the child's semaphore fires). Two
  CBs, not one, because that is two producers and two consumers.
- `cb_x_tiles` has one producer (the local **reader**: it pushes after the last multicast round,
  regardless of whether the bytes arrived by loopback or over the NoC) and one consumer (compute,
  which reads it for both the gate and the up matmul and pops it once at the end of the M-block).
- `cb_h_local` is produced by compute, consumed by the writer (mcast sender). `cb_h` is produced
  by the reader, consumed by compute. Separate CBs for the same logical object, because the local
  copy and the gathered stream have different producers.

**Sync.** For every CB, producer `push_back` count == consumer `wait_front` count per M-block:

| CB | pushes per M-block | waits per M-block |
|---|---|---|
| `cb_x_sticks` | `32 * my_stage_rows` | `32 * my_stage_rows` (tilize's asymmetric-page mode, `tilize_helpers.inl:150-151`) |
| `cb_x_tiles` | 1 × `M_BLOCK*Kr` | `matmul_block` waits `m_tiles*KB1` per K-block × `Kr/KB1` K-blocks × 2 matrices × sub-blocks, with `in0_policy = WaitAndRetainOnLastBlock`; compute issues one explicit `pop_front(M_BLOCK*Kr)` at the end |
| `cb_w_gate` / `cb_w_up` | `KB1*HN_BLOCK` per K-block per sub-block | same |
| `cb_*_interm` | `matmul_block`-internal, balanced by construction | same |
| `cb_*_partial` | 1 × `m_tiles*HN_BLOCK` per sub-block | same |
| `cb_reduce_*_in` | `num_children` × `m_tiles*HN_BLOCK` per sub-block | same |
| `cb_h_local` | 1 × `m_tiles*HN_PAD` | 1 |
| `cb_h` | 13 × `m_tiles*HN_PAD` | `matmul_block` waits `m_tiles*HN_PAD` per K-block × 13 |
| `cb_out_tiles` | `m_tiles*EC` | same |

At `count == 0`, `m_blocks == 0` and every one of these counts is 0 on every core.

---

## 6. API Mapping

| Phase | Type | Function | File:Line | Template params / args | Input CB | Output CB | Requirements |
|---|---|---|---|---|---|---|---|
| boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(in0,in1,out)` | `tt_metal/hw/inc/api/compute/matmul.h:393` (note `mm_block_init` is deprecated in favour of this) | once, first statement of `MAIN()` | — | — | mandatory before any helper |
| boot | helper | `ActivationInitHelper<KernelActivation::SILU>::init()` | `sfpu_activation_helpers.hpp:75-108`, SILU at `:86` | — | — | — | the helpers never issue this (`matmul_block_helpers.hpp:103`); forgetting it gives wrong values, not a hang |
| x tilize | helper | `tilize<Kr, Float16_b_dfb, Bfloat8_b_dfb, InitOnly/Neither, WaitBlock, …, RemapMode::AssumeConfigured>(num_blocks=1, total_input_pages=32)` | `tilize_helpers.hpp:187-197`; remap at `:73-78` / `.inl:182-199`; asymmetric page mode `.inl:150-151` | `block_width_tiles = Kr(y)` **is** the block knob; `AssumeConfigured` after one `RemapMode::Configure` call in the prologue, per the prompt's hot-loop guidance | `cb_x_sticks` | `cb_x_tiles` | fast path needs half-sync + bf16/fp32 input + non-fp32 output (`.inl:65-78`) — all hold. `dst_full_sync_en=False` is therefore load-bearing. |
| gate matmul | helper | `matmul_block<false, /*packer_l1_acc=*/true, LastBlockTarget::Interm, SubblockMajor, InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::WaitAndPopPerKBlock>` | `matmul_block_helpers.hpp:334-366`; shape `:136-167`; K-blocking `.inl:208-547`; L1-acc `.inl:401,488-518` | `MatmulBlockShape::of(in0_num_subblocks = m_tiles/4, in1_num_subblocks = 1, out_subblock_h = 4, out_subblock_w = min(HN_BLOCK, hn-off), in0_block_k = KB1, num_k_blocks = Kr/KB1)` — `in0_block_k` and `out_subblock_w` **are** the knobs | `cb_x_tiles`, `cb_w_gate` | `cb_gate_interm` | ASSERT `out_subblock_h*out_subblock_w <= DEST_AUTO_LIMIT` (`.inl:158`) → 4×2 = 8 ✓. `interm_buf` = `cb_gate_interm`; pass `out_buf` = `cb_gate_partial`. |
| up matmul | helper | same, second call | same | same shape, `cb_w_up` | `cb_x_tiles`, `cb_w_up` | `cb_up_interm` | reads `cb_x_tiles` a second time without re-reading DRAM/NoC — see the note below |
| pack partial | helper | `copy<input(cb_*_interm), output(cb_*_partial)>(EltwiseShape::tiles(m_tiles*HN_BLOCK))` | `eltwise_convenience.hpp:95-96` | — | `cb_*_interm` | `cb_*_partial` | only needed on non-root levels where the interm must survive; the last matmul K-block can pack straight to `cb_*_partial` via `LastBlockTarget::Out` |
| reduce add (non-root) | helper | `add<input(cb_*_partial), input(cb_reduce_*_in), output(cb_*_partial)>(EltwiseShape::tiles(n))` | `eltwise_convenience.hpp:42-48`; expansion `eltwise_convenience.inl:25-30` | `BroadcastDim::None` | `cb_*_partial`, `cb_reduce_*_in` | `cb_*_partial` (in-place) | in-place output CB == input CB is the documented pattern; FPU add, not SFPU |
| reduce add + SiLU (root, gate) | helper | `add_bias_bcast_rows<BiasBroadcast::Elementwise, SubblockMajor, NoPostBias, SiluActivation>` | `bias_add_helpers.hpp:156-168`; `Elementwise` `:28`; activation slot `.inl:163-171`; `SiluActivation` `sfpu_activation_helpers.hpp:44` | `BiasAddShape::of(m_tiles/4, 1, 4, HN_BLOCK)` | `cb_gate_partial`, `cb_reduce_gate_in` | `cb_gate_silu` | SiLU runs on the **PACKER thread** and replaces `tile_regs_wait()`, so it overlaps the MATH thread (`sfpu_activation_helpers.hpp:71-74`) — the SwiGLU activation is free. Bias CB lifecycle is caller-owned (`bias_add_helpers.inl:183`). |
| SwiGLU multiply (root) | helper | `mul<input(cb_gate_silu), input(cb_up_partial), output(cb_h_local)>(EltwiseShape::tiles(m_tiles*HN_BLOCK))` | `eltwise_convenience.hpp:58-64` → `BinaryFpu<…,Mul,None,D0,…>` `eltwise_chain.hpp:504-514` | `BroadcastDim::None` | `cb_gate_silu`, `cb_up_partial` | `cb_h_local` | **FPU multiply through L1, deliberately not SFPU and not DEST-reuse**: `examples/compute_fusion` measured SFPU-instead-of-FPU multiply at **0.58×** and DEST-reuse for an FPU consumer at **0.82×** (the L1 round-trip is 1.22× *faster*). |
| zero pad column (x=12 only) | helper | `unary<...>` is not needed — `FillScalar<Dst::D0>` in an `eltwise_chain` | `eltwise_fill.hpp:19-24` | fill 0.0 | — | `cb_h_local` tail | one `M_BLOCK`-tile column so round 12's payload is uniform (§4.4) |
| down matmul | helper | `matmul_block<false, /*packer_l1_acc=*/true, LastBlockTarget::Interm, SubblockMajor>` | as above | `MatmulBlockShape::of(m_tiles/4, 1, 4, EC, in0_block_k = HN_PAD, num_k_blocks = HGROUPS)` | `cb_h`, `cb_w_down` | `cb_out_interm` | `num_k_blocks = 13 > 2` so `packer_l1_acc` is worth enabling (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:100-104`) |
| pack output | helper | `copy<input(cb_out_interm), output(cb_out_tiles)>(EltwiseShape::tiles(m_tiles*EC))` | `eltwise_convenience.hpp:95-96` | reconfig bf16 → bfp8 | `cb_out_interm` | `cb_out_tiles` | the one genuine dtype boundary; reconfig **must stay on** here (`examples/compute_block_size/README.md:145-149`) |
| x mcast | helper | `McastArgs<CT,RT,SPAN=M_BLOCK>::sender(noc).send(src,dst,size)` / `.receiver(noc).receive(round)` | `mcast_pipe.hpp:327-395`, `:197`, `:274` | Counter signal, handshake on | — | `cb_x_tiles` | receivers must have identical `dst_l1` (`:44-45`) |
| h mcast | helper | same, `SPAN = HGROUPS` | same | Counter signal, handshake on | `cb_h_local` | `cb_h` | 13 rounds, one K-block each |
| host mcast wire | helper | `Mcast1D(PerRow, …, rotating)` / `Mcast2D(rect, sender, cfg)` + `owned_semaphores()` / `compile_time_args()` / `runtime_args(core)` | `host/mcast_host.hpp:156-415`, `:448-…` | `McastConfig{noc, handshake=true, data_ready=Counter, rotating_sender=true, base_sem_id}` | — | — | grid must be 0-anchored (`:174-178`) ✓. Rotating spans are `M_BLOCK` (x) and `HGROUPS` (h); where the helper's span does not match, emit the `4 + 2*SPAN` runtime words directly against `McastArgs::num_runtime_args()` (`mcast_pipe.hpp:346`). |
| weight read | **raw_api** | `noc_async_read(accessor.get_noc_addr(page_base), l1, run_len * 576)` + batched `noc_async_read_barrier()` | `dataflow_api_addrgen.h:18-42, 287-300` (bank/slot arithmetic); `tech_reports/tensor_accessor/tensor_accessor.md` | `run_len` = `WRUN`, ≥ 4 reads outstanding per barrier | — | `cb_w_*` | see the rejection note below |
| output write | **raw_api** | `noc_async_write(l1, accessor.get_noc_addr(page_base), run_len * 1088)` | same | 2-tile bank-runs | `cb_out_tiles` | — | same note |
| count read | raw_api | one-page `noc_async_read` into an unpushed scratch CB, then `volatile tt_l1_ptr uint32_t*` | pattern at `deepseek_prefill/extract/kernels/dataflow/reader_extract.cpp:83-117` | — | — | `cb_token_meta` | `idx` first, then `counts[idx[local_expert_id]]` |

### Helpers considered and rejected

| Raw API used | Helper considered | file:line of the mismatch | Concrete reason |
|---|---|---|---|
| bank-run `noc_async_read` for weights | `TensorAccessor::noc_async_read_page` / the per-tile read used by the production mcast matmul | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:386-414` | The page-granular helper issues **one transaction per tile**. tt-metal's own interleaved mcast matmul does exactly this and it is the pattern `examples/double_buffer/report.md:105-110` identifies as transaction-rate-limited: 576 B × ~8.7 M/s ≈ 5 GB/s per core. The bank-run read is the same `TensorAccessor` address computation with a longer length; there is no helper that expresses "read `L` bank-contiguous pages as one transaction" for an interleaved tensor — the only in-tree coalescing helper, `get_max_page_size_and_num_pages` (`matmul_utilities.cpp:346-367`), is **DRAM-sharded-only** and this op is granted interleaved weights (feature_spec.py:229-231). |
| bank-run `noc_async_write` for output | same | same | same, on the write side; the output's `EMB_T` stride is also a multiple of 8 so the identical run construction applies. |
| reduce tree transport (unicast + counting semaphore) | `mcast_pipe` `SenderPipe`/`ReceiverPipe` | `mcast_pipe.hpp:171-228` — `SenderPipe` is a **multicast** sender over a `McastRect`; `McastRect::area()` (`:130-139`) is a rectangle fan-out | A reduce tree's edges are point-to-point with a *different* destination per node and per level; expressing them as 1×1 multicast rects would allocate a semaphore pair per level per direction and lose the counting-semaphore fan-in that a tree node needs. `mcast_pipe` **is** used for both broadcasts (x, h), which is where its rectangle model fits. |
| `volatile tt_l1_ptr uint32_t*` read of `cb_token_meta` in compute | any CB helper | — | The compute kernel needs a scalar loop bound, not tiles. No helper reads a scalar out of a CB; this is the documented `deepseek_prefill` pattern. |
| `FillScalar` chain for the one zero pad column | `ttnn.zeros` / a host-side zero tensor | — | Would put a tensor in the signature and a DRAM read on the hot path for `M_BLOCK` tiles of zeros. |

**The `cb_x_tiles`-consumed-twice contract.** `matmul_block` waits and pops `in0` per K-block
(`.inl:262, 527-542`). Gate and up must both read the *same* resident `x` without a second
multicast. Wire it as `in0_policy = InputPolicy::WaitAndRetainOnLastBlock`
(`matmul_block_helpers.hpp:91`) plus an `In0SourceFn` (`:215-217`) supplying the per-K-block tile
base, with one explicit `cb_x_tiles.pop_front(M_BLOCK*Kr)` from compute after the second matmul.
If that combination proves unworkable, the documented fallback is a single
`copy<input(cb_x_tiles), output(cb_x_tiles_b)>` (`eltwise_convenience.hpp:95`) into a second CB —
costs one extra L1 pass over 368 tiles and 391 KB of L1, both of which the budget can absorb only
at `M_BLOCK = 8`. **Do not** solve it by multicasting `x` twice: that doubles the reuse-shared
traffic the whole row split exists to avoid.

---

## 7. Compute Phases

Per M-block `b ∈ [0, m_blocks)`. Core `(x,y)`; `off` iterates hidden sub-blocks.

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB state after |
|---|---|---|---|---|---|
| 0 | read `idx`, `counts`; publish `{count, M_t, m_blocks}` | raw | scratch (unpushed) | `cb_token_meta` (1) | `cb_token_meta` held for the whole program |
| 1 | prologue: `tilize` with `RemapMode::Configure` once | helper | — | — | DEST remap programmed; all later tilize calls use `AssumeConfigured` |
| 2 | stage x: read 32 sticks of tile-row `t` (`t % 13 == x`) | raw | — | `cb_x_sticks` (32/round) | drained by phase 3 |
| 3 | tilize bf16 sticks → bfp8 tiles | helper | `cb_x_sticks` (32 sticks) | `cb_x_tiles` local slot (`Kr`) | sender's slot filled |
| 4 | multicast round `t` along row `y`; receive the other rounds | helper | `cb_x_tiles` (send) | `cb_x_tiles` (recv) | after `m_tiles` rounds `cb_x_tiles` holds `m_tiles × Kr` tiles, **resident** |
| 5 | `matmul_block` gate, sub-block `off` | helper | `cb_x_tiles` (retained), `cb_w_gate` (`KB1*w` per K-block) | `cb_gate_interm` (`m_tiles*w`) | `cb_x_tiles` NOT popped |
| 6 | `matmul_block` up, sub-block `off` | helper | `cb_x_tiles` (retained), `cb_w_up` | `cb_up_interm` (`m_tiles*w`) | `cb_x_tiles` NOT popped |
| 7 | pack partials for transport | helper | `cb_*_interm` | `cb_*_partial` (`m_tiles*w`) | writer picks them up for the reduce send |
| 8 | reduce levels 1..4: `add` in-place | helper | `cb_*_partial`, `cb_reduce_*_in` | `cb_*_partial` | non-root cores are done with sub-block `off` after their send |
| 9 | root only: final gate add **with SiLU on the packer** | helper | `cb_gate_partial`, `cb_reduce_gate_in` | `cb_gate_silu` (`m_tiles*w`) | — |
| 10 | root only: SwiGLU multiply | helper | `cb_gate_silu`, `cb_up_partial` | `cb_h_local` slice (`m_tiles*w`) | after all `off`, plus the zero pad on x=12, `cb_h_local` = `m_tiles*HN_PAD` |
| 11 | push `cb_h_local` **before** phase 12 | — | — | `cb_h_local` (1 group) | writer can now send round `x` |
| 12 | `matmul_block` down over 13 K-blocks, `packer_l1_acc` | helper | `cb_h` (`m_tiles*HN_PAD` per round), `cb_w_down` | `cb_out_interm` (`m_tiles*EC`) | `cb_h` drained round by round → the all-gather overlaps this |
| 13 | pack bf16 → bfp8 | helper | `cb_out_interm` | `cb_out_tiles` (`m_tiles*EC`) | — |
| 14 | writer: coalesced 2-tile bank-run writes for rows `[b*M_BLOCK, b*M_BLOCK+m_tiles)` | raw | `cb_out_tiles` | — | rows past `M_t` never written |
| 15 | pop `cb_x_tiles`; next M-block | — | — | — | — |

Phases 5–11 iterate `off` over `ceil(hn(x)/HN_BLOCK)` sub-blocks; the reduce for sub-block `off`
overlaps the matmul for `off+1` (§4.3). Phase 12 overlaps the phase-4-style rounds of §4.4.

---

## 8. Broadcast Verification

| Phase | Op | CB_A valid region | CB_B valid region | BroadcastDim |
|---|---|---|---|---|
| 8 (reduce add) | `add` (FPU) | `cb_*_partial` 2D `[m_tiles, w]` → All | `cb_reduce_*_in` 2D `[m_tiles, w]` → All | `None` |
| 9 (root gate add) | `add_bias_bcast_rows<Elementwise>` | `cb_gate_partial` 2D → All | `cb_reduce_gate_in` 2D → All | `Elementwise` (not a broadcast — plain `add_tiles_init`, `bias_add_helpers.inl:51-58`) |
| 10 (SwiGLU) | `mul` (FPU) | `cb_gate_silu` 2D → All | `cb_up_partial` 2D → All | `None` |

No operand is a row/column vector anywhere, so no `BroadcastDim` other than `None` is legal here;
`eltwise_chain` requires it be passed explicitly and never infers it
(`eltwise_chain.hpp:461-466`).

---

## 9. Precision budget (PCC ≥ 0.98 vs an unquantized fp32 reference)

| Source | Modelled relative contribution | Note |
|---|---|---|
| bfp4 weights (given) | per-element ~6 %, but the contraction averages it: `ε/sqrt(N)` ≈ 6 %/84 ≈ **0.07 %** for gate/up (N = 7168), 6 %/45 ≈ 0.13 % for `down` | the format floor the prompt says to measure first |
| `x`, `h` in bfp8 | per-element ~0.4 % → `≈0.4 %/sqrt(N)` ≈ **0.005 %** / 0.01 % | the decision of §1.6 |
| bf16 `interm` with `packer_l1_acc` over `Kr/KB1 = 4` (gate/up) and 13 (`down`) L1-accumulate steps | ~2⁻⁹ per step, random walk → **≈0.4 %** / 0.7 % | `fp32` interm is the knob if PCC lands short; costs 2× on `cb_*_interm` (128 KB), which the budget can absorb |
| cross-core reduce payload in bfp8, 4 levels | **≈0.2 %** | |
| bfp8 output + `bfp8_pack_precise=True` | ~0.4 % | contract |

PCC 0.98 admits ≈ `sqrt(1-0.98²)` = 20 % residual RMS; the modelled total is under 1 %. If a cell
lands short, measure the pure format floor first (weights → bfp4 and `h` → bfp8 through
`from_torch`/`to_torch`, chain run in torch) before touching the kernel.

---

## 10. Registry contract (what the op file must declare)

| Item | Value |
|---|---|
| `INPUT_TAGGERS` | `{"emb": tag_emb, "capacity": tag_capacity, "fill": tag_fill}` over `(x_shape, w_gate_shape, w_up_shape, w_down_shape, count)` |
| `tag_emb` | `int(inputs[0][-1])` |
| `tag_capacity` | `int(inputs[0][-2])` |
| `tag_fill` | `count, capacity = inputs[4], inputs[0][-2]`; `"empty"` if `count == 0`; `"full"` if `count == capacity`; `"balanced"` if `count <= capacity // 16`; else `"partial"` — **verbatim** the rule at `feature_spec.py:135-143` |
| `SUPPORTED` | `input_format: [bf16_rm, bfp8_tile]`, `weight_dtype: [ttnn.bfloat4_b]`, `emb: [6144, 7168]`, `capacity: [1024, 2048, 5120]`, `fill: [balanced, partial, full, empty]` — **everything**, so every later refinement is a measured perf refinement |
| `EXCLUSIONS` | `[]` |
| `INVALID` | `[]` in the op file — it lives in `feature_spec.py` (`eval/op_template.py:24-32`) |
| `validate()` | first line of the entry point; raises `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract` (`_op_contract.py:23-31`). Gates on `input_format` / `weight_dtype` / `emb` / `capacity` only. **`fill` is observed-but-uncheckable** — it derives from a device-resident value and `validate()` is host-side and forbidden from reading `counts`. |
| `input_format` derivation | read off the tensor: `bf16_rm` iff `dtype == bfloat16 and layout == ROW_MAJOR_LAYOUT`; `bfp8_tile` iff `dtype == bfloat8_b and layout == TILE_LAYOUT` |
| Structural `ValueError`/`RuntimeError` | `x.rank != 4`; `x.shape[0:2] != (1,1)`; any weight `rank != 2`; `w_gate.shape != w_up.shape`; `x.shape[-1] != w_gate.shape[-2]`; `w_gate.shape[-1] != w_down.shape[-2]`; `counts`/`idx` not UINT32 ROW_MAJOR; `local_expert_id` out of range for `idx`. **Not** host-checkable and therefore not checked: `count <= capacity`, `idx[local_expert_id] < len(counts)` |
| Entry point | must **not** call `ttnn.to_layout` / `tilize` / `pad` / `slice`; the tilize is fused (phase 3) |

**Structural impossibilities** — nothing to add to `feature_spec.INVALID`. `input_format` already
collapses the dtype × layout cross to the two real combinations, every tensor is DRAM interleaved,
and every `(capacity, fill)` pair is realisable because `fill` is defined relative to `capacity`.

---

## 11. Expected performance, honestly

| emb | cap | count | graded read bytes | DRAM floor @512 GB/s | compute floor | **design floor** = max | target ns / util | best measured ns / util |
|---|---|---|---|---|---|---|---|---|
| 7168 | 5120 | 128 | 26.60 MB | 52.0 µs | 11 µs | **52 µs** (util 1.00 at peak) | 91 800 / 0.566 | 102 000 / 0.509 |
| 7168 | 5120 | 256 | 28.44 MB | 55.5 µs | 21 µs | **56 µs** | 108 000 / 0.514 | 120 000 / 0.463 |
| 7168 | 5120 | 512 | 32.11 MB | 62.7 µs | 42 µs | **63 µs** | 161 820 / 0.388 | 179 795 / 0.349 |
| 7168 | 1024/2048 | 256 | 28.44 MB | 55.5 µs | 21 µs | **56 µs** | 108 000 / 0.514 | — |
| 7168 | 5120 | 256 (bfp8_tile) | 27.09 MB | 52.9 µs | 21 µs | **53 µs** | (target derived for these bytes) | — |
| 6144 | 5120 | 256 | 24.38 MB | 47.6 µs | 18 µs | **48 µs** | none (report only) | — |
| 7168 | 5120 | 5120 | 98.19 MB | 192 µs (×10 real reads: 579 µs) | 423 µs | **579 µs** | none (report only) | — |

The graded targets sit at **1.7–2.6× the DRAM floor**, so the whole question is *achieved* DRAM
read efficiency, not bytes and not FLOPs — hence §1.5's transaction-size decision and the
requirement that all three graded counts fit one M-block. `count = capacity` is compute-bound and
carries no target; its ×10 weight re-read is a knob (`M_BLOCK`), not a structural defect. Report
every result as **utilisation AND device-kernel ns**, with `(emb, capacity, count)` and the
internal structure (`M_BLOCK`, `KGROUPS`, `HGROUPS`, `WRUN`, `KB1`) that produced it, and read a
result landing between `best_measured_util` and `expected_dram_read_util` as the real progress it
is. Do not benchmark against a default-config `ttnn.matmul`/`ttnn.linear` chain.

**First `/perf-measure` ablations, in priority order:** (1) `WRUN = 1` vs the bank-run read —
isolates the transaction-size hypothesis; (2) dual-NoC weight split on vs off; (3) `M_BLOCK` 8 vs
16 at count 512 — isolates the weight re-read; (4) `KB1` 4/6/8; (5) `DEPTH_H` 2 vs 3 — isolates
the all-gather/compute overlap; (6) `SKIP_COMPUTE` on `matmul_block` (`.inl:346-362`) to separate
the dataflow ceiling from the compute ceiling.

---

## 12. Key Risks and Gotchas

| Risk | Mitigation |
|---|---|
| **Bank-run read arithmetic** — the coalesced read assumes `num_dram_banks == 8`, `page_id % 8` bank mapping, and `aligned_page_size == tile_size` | all three are asserted host-side from `ttnn.get_dram_alignment()` / the buffer's `buffer_aligned_page_size()` and the device's DRAM bank count; `WRUN = 1` is the always-correct fallback. bfp4 576 B and bfp8 1088 B are both 64 B multiples ✓ |
| **`cb_x_tiles` consumed twice** by gate and up | §6's explicit contract; do not fix it by multicasting `x` twice |
| **`cb_h` must hold a full phase-2 K-block in the exact `in0` subblock layout** | §4.4 pins the tile order; the producer packs `cb_h_local` in that order out of the SwiGLU |
| **`HN_PAD` zero-pad column** on x = 12 | must be genuinely zero, not stale L1 — `FillScalar` it every M-block. The matching `W_down` row is re-read and harmless *only because* `h`'s pad is zero |
| Tile-padding sentinel leaking into a real row | impossible by construction: nothing reduces across the token axis, so garbage in `[count, ceil_tile(count))` of `x` stays in the same rows of `h` and `out`. Do **not** zero it (pointless) and do **not** write past `ceil_tile(count)` |
| **Rotating multicast senders and deadlock** | every core runs the identical round loop; `cb_h_local` is pushed before phase 12; `DEPTH_H = 3`; senders self-exclude and loopback. Run the `cross_core_reduction_design.md` silent-hang checklist. Use `scripts/run_safe_pytest.sh --dev`. |
| **Blackhole mcast fan-out** across virtual columns 8/9 | a 13-wide row rect spans them; `McastRect::area()` already subtracts them (`mcast_pipe.hpp:129-139`) — do not hand-compute the ack count |
| **`dst_full_sync_en` must stay False** | Blackhole fast tilize requires half-sync (`tilize_helpers.inl:65-78`), and full-sync would cost the math↔pack overlap (`llk_math_common.h:140-160`) |
| bfp weights on the mcast matmul path hang below `in0_tile.height < 16` (issue #42927, `matmul_device_operation.cpp:140-157`) | this op uses full 32×32 tiles everywhere; never introduce a tiny-tile geometry |
| **CBs that must hold a full block** | `cb_x_tiles` (resident `in0` for two matmuls), `cb_gate_interm` / `cb_up_interm` / `cb_out_interm` (matmul K-accumulation — `matmul_block` requires the whole out-block), `cb_h_local` (a whole mcast payload). These are the four CBs whose page count must not be reduced to a streaming depth. |
| Dtype reconfig | the only genuine format boundary is bf16 `interm` → bfp8 output (phase 13) and bf16 sticks → bfp8 tiles (phase 3). Keep reconfig **on** at both; disabling it across a real boundary is silent corruption (`examples/compute_block_size/README.md:145-149`). Everywhere else the format is constant across the boundary, so the compile-time fold elides it (`eltwise_chain.hpp:20-21`). |
| `count == 0` | uniform `m_blocks == 0` skip on all 130 cores; no collective entered; output returned allocated and untouched |
