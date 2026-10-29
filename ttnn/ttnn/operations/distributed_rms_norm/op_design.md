# Operation Design: distributed_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (CCL + compute): per-chip RMSNorm local work (the base op `rms_norm`) fused with a cross-device all-gather of one fp32 Σx² partial per row over TT-Fabric, as **one** `ttnn.generic_op` (one `MeshProgramDescriptor`) per call |
| Goal | Every device `p` of a collective group of `G` devices along `cluster_axis` holds `x[p]` of shape `(..., M, W_local)`; the group's hidden dim is `W = G·W_local`. Produce `out[p] = x[p] · rsqrt(Σ_g Σ_w x[g]² / W + ε) · γ[p]` on every device, with the exchange hidden behind the local work. |
| Math | `ss[n,h] = Σ_{g<G} Σ_{w<W_local} x[g][n,h,w]²` (logical columns only); `out[p][n,h,w] = x[p][n,h,w] · γ[p][w] · rsqrt(ss[n,h] · (1/W) + ε)` (γ ≡ 1 when absent). `W = G · W_local` is the logical global width — padding never enters the sum or the divisor. Equals `torch.nn.functional.rms_norm(concat_p x[p], (W,), concat_p γ[p], ε)` sliced back per device. |
| Mode | Hybrid — local half ported from the base op (`ttnn/ttnn/operations/base_ops/rms_norm/`: kernel-lib helpers for every compute phase + the base's raw-LLK perf passes); cross-device half ported from the fabric examples (`fabric_ring_exchange/kernels/port.cpp`, `fabric_all_gather` `plan()` placement). |
| References | Base op: `ttnn/ttnn/operations/base_ops/rms_norm/{op_design.md, changelog.md, rms_norm_program_descriptor.py, kernels/*}`. Transport: `ttnn/ttnn/operations/examples/{master.md, fabric_ring_exchange/, fabric_all_gather/, fabric_reduce_scatter/}`. Rules: `.claude/references/{blocking-model.md, l1-footprint-discipline.md, op-design-template.md, blackhole-fabric.md, precision_convention.md, ttnn-cb-memory-fundamentals.md}`. Golden suite: `eval/golden_tests/distributed_rms_norm/{feature_spec.py, roofline.py, helpers.py, axes.py, topology.py, test_ring_mock.py}`. Catalog entries used: `fabric_ring_exchange`, `fabric_all_gather`, `fabric_reduce_scatter`, `tensix_all_reduce`, `compute_block_size`, `double_buffer`, `noc_placement`, `sfpu_tile_scope`, `compute_fusion`, `shared_input_reuse`. |

### Work-kind labels used in this document

| Label | Meaning (prompt "Fused CCL op") | Where measured |
|-------|-------------------------------|----------------|
| **TRANSPORT** | how the stats bytes cross the fabric: relay vs multicast, directions, links, packets, the ready fence | taken from the examples' measurements (`fabric_ring_exchange/README.md`), never re-derived |
| **OVERLAP** | hiding the transport behind this op's local work: work split around the ports, wave pipeline, per-wave notify, launch-skew tolerance, last-landing → output chain | only in the op: exposed transport = op time − op time with the exchange stubbed (`DRMS_ABLATE=exchange`), slowest chip per call |
| **LOCAL** | the base op's single-chip work (read, Σx², combine on chip, scale, write) | base op `changelog.md`; re-checked here only where the fused grid changes it |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (mesh) | yes | per device rank ≥ 2, `(..., M, W_local)`, `Layout::TILE`, `dtype ∈ {float32, bfloat16, bfloat8_b}`, identical per-device shape / dtype / placement; `TensorMemoryLayout::INTERLEAVED` (DRAM or L1), `HEIGHT_SHARDED`, `WIDTH_SHARDED` (L1) | — | accessor args CT, base address RT |
| `cluster_axis` (kw-only) | `int` | yes | 0 or 1; mesh dim size ≥ 2 | — | host only (groups, ports) |
| `gamma` (kw-only) | `Optional[ttnn.Tensor]` (mesh) | no | `Layout::TILE` per device `(1, 1, 1, W_local)`, or `Layout::ROW_MAJOR` "sticks" per device `(1, 1, ceil(W_local/32), 32)` (tail zero-padded); `dtype ∈ {float32, bfloat16}`, independent of the input dtype; DRAM or L1 interleaved | `None` | presence + format CT, address RT |
| `epsilon` (kw-only) | `float` | no | finite, ≥ 0 | `1e-6` | RT (fp32 bit pattern) |
| `topology` (kw-only) | `ttnn.Topology` | no | `Linear`, `Ring` (import as `from ttnn.operations.ccl import Topology` at module scope — `ttnn.Topology` is not bound while `ttnn.operations` imports) | `Linear` | host only (exchange plan) |
| `num_links` (kw-only) | `Optional[int]` | no | `None` = every usable link discovered on every hop of the group (`ttnn.get_forwarding_link_indices`); explicit `1 ≤ n ≤ min_hops(discovered)` | `None` | host only (ports) |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` | no | any `math_fidelity` / `math_approx_mode`; `fp32_dest_acc_en ∈ {True, False}` (False refused for float32 input) | `default_compute_kernel_config()` | passed verbatim as the compute kernel `config=` (plus `unpack_to_dest_mode` for `cb_stats`, Key Risks) |
| `memory_config` (kw-only) | `Optional[ttnn.MemoryConfig]` | no | interleaved input: any interleaved config (DRAM/L1); sharded input: the input's own memory config | `ttnn.DRAM_MEMORY_CONFIG` | host only (output allocation) |

### Entry-point contract

| Item | Decision |
|------|----------|
| Default config | One exported factory `default_compute_kernel_config()` returning a **fresh** `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. Entry point: `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()`. `eval/golden_tests/distributed_rms_norm/axes.py` reads the same factory. Nothing else restates the default. |
| First line | `validate(input_tensor, cluster_axis=..., gamma=..., topology=..., num_links=..., compute_kernel_config=..., memory_config=...)`. |
| Hard validation (ValueError / RuntimeError) — inside `validate()`, **before** the support gate | (1) per-device rank < 2. (2) `cluster_axis ∉ {0,1}` or `mesh_shape[cluster_axis] == 1`. (3) γ shape wrong for its layout: TILE ≠ `(1,1,1,W_local)`, ROW_MAJOR ≠ `(1,1,ceil(W_local/32),32)`. (4) `ttnn.get_fabric_config()` is `DISABLED` (RuntimeError). (5) a hop between consecutive group members (and, for `Ring` with `G ≥ 3`, the wrap hop `G−1 → 0`) has no direct forwarding link (`len(ttnn.get_forwarding_link_indices(src, dst)) == 0`) — ValueError naming the hop, the topology and the fabric config. (6) explicit `num_links < 1` or `> min over the group's hops of the discovered link count`. The op never changes the fabric config. |
| `num_links` canonicalization | `None` → `min over every hop the plan uses of len(get_forwarding_link_indices)`. The canonical value is what `validate()` gates on and what the plan uses. |
| Index canonicalization | `cluster_axis` is 0 / 1 only (no negative form); `topology` as given. `Ring` with `G == 2` is planned as the line it physically is (one link pair; `topology_feasible` never emits that cell, but a caller may pass it). |
| Support gate | `validate()` builds the axes dict exactly as `eval/golden_tests/distributed_rms_norm/axes.py:classify_call` does (`dtype`, `fp32_dest_acc_en` from the resolved config, `layout`, `alignment` / `rank` from `INPUT_TAGGERS` on the per-device shape, `gamma_mode`, `gamma_dtype` / `gamma_layout` = `"none"` when γ is absent, `memory_layout`, `cluster_axis`, `topology`, canonical `num_links`), then checks `SUPPORTED` per axis (`UnsupportedAxisValue`) and `EXCLUSIONS` per cell (`ExcludedCell`) from `ttnn/ttnn/operations/_op_contract.py`. `"none"` is always accepted. INVALID is not declared or checked. |
| INPUT_TAGGERS | exactly two, reading the per-device shape `inputs[0]` with `H = shape[-2]`, `W_local = shape[-1]`: `alignment` → `"w_non_aligned"` if `W_local % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"`; `rank` → `len(inputs[0])`. (γ dtype / layout are classified from the γ tensor, not tagged — base `rms_norm.py:83-93`.) |
| EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` only (natively rejected, `precision_convention.md`). The base's `{bfloat8_b, w_non_aligned}` is **not** carried: this op zeroes the W-pad of a Bfp8_b tile by zeroing the pad datums' mantissa bytes (Dataflow Strategy → padding rules), which needs no exponent work. If the implementer measures that path failing, the exclusion is re-added with the measurement as its reason. |
| Shape-size refusal | `NotImplementedError` (not an axis) for: a sharded input whose shard grid leaves fewer than `2·L` free Tensix cores for the ports (regime `PORT-ON-SHARD`, deferred); a sharded input with `memory_config` ≠ the input's (regime `SHARD-IN-INTERLEAVED-OUT`, deferred); an interleaved input with a sharded `memory_config`. |
| Dispatch | Allocate the output (same per-device logical shape, dtype, TILE layout; placement per `memory_config`), fetch / create the persistent exchange resources (Work Distribution → persistent resources), build one `MeshProgramDescriptor` (one `ProgramDescriptor` per mesh coordinate), call `ttnn.generic_op([input, (gamma), landing, output], mesh_desc)` **once**. No host padding / slicing / synchronization on a warm call; no other ttnn op, no CCL op, no norm op, no example op is called. |

### Phase 0 support target (TARGET → Phase 0 / later)

Phase 0 SUPPORTED is the from-base list (TARGET ∩ base SUPPORTED) plus the prompt's values for the three cross-device axes.

| TARGET axis | Phase 0 SUPPORTED | Later / cost class |
|-------------|-------------------|--------------------|
| `dtype` | `float32`, `bfloat16`, `bfloat8_b` | — |
| `fp32_dest_acc_en` | `True`, `False` (EXCLUSION `{float32, False}`) | — |
| `layout` | `TILE` | — |
| `alignment` | `tile_aligned`, `w_non_aligned`, `h_non_aligned` | — |
| `rank` | 2, 3, 4 | — |
| `gamma_mode` | `gamma`, `no_gamma` | — |
| `gamma_dtype` | `float32`, `bfloat16`, `"none"` | — |
| `gamma_layout` | `TILE`, `ROW_MAJOR`, `"none"` | — |
| `memory_layout` | `INTERLEAVED`, `HEIGHT_SHARDED`, `WIDTH_SHARDED` | — . Placement is the physical form of the logical shards the work split already defines: `HEIGHT_SHARDED` cuts the independent row axis → **knob-turn** (regime D1-HS, zero-copy CBs); `WIDTH_SHARDED` cuts the dependent W_local axis → **scheme-change already built** (regime D3-WS: the on-chip combine D3 uses for interleaved, with the shard spec as the split). |
| `cluster_axis` | 0, 1 | — |
| `topology` | `Linear`, `Ring` | — |
| `num_links` | 1, 2 | — |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(..., M, W_local)`, rank ≥ 2, identical on every device; any `M`, `W_local ≥ 1` (non-multiples of 32 allowed) |
| Dtype | `float32`, `bfloat16`, `bfloat8_b` |
| Layout | `Layout::TILE` (each `(n, M, W_local)` image tile-padded independently) |
| Memory | `TensorMemoryLayout::INTERLEAVED` (DRAM or L1), `HEIGHT_SHARDED` / `WIDTH_SHARDED` (L1, one shard spec on every device) |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | TILE: `(1, 1, 1, W_local)`; ROW_MAJOR sticks: `(1, 1, ceil(W_local/32), 32)` — stick `t` = the 32 γ values of tile column `t` |
| Dtype | `float32` or `bfloat16`, independent of the input dtype |
| Layout | `Layout::TILE` or `Layout::ROW_MAJOR` |
| Memory | interleaved (DRAM or L1) |

### Output

| Property | Value |
|----------|-------|
| Shape | per device identical to the input |
| Dtype | input dtype |
| Layout | `Layout::TILE` |
| Memory | `memory_config` if given, else `ttnn.DRAM_MEMORY_CONFIG`; for a sharded input the input's shard spec (the golden runner passes `x.memory_config()`) |

## Blocking Model

The first design section. Everything below is downstream of these decisions. Semantics: `.claude/references/blocking-model.md`.

### Index space and symbols (all host-computed once, `ceil`, per image)

| Symbol | Definition |
|--------|-----------|
| `G` | group size `= mesh_shape[cluster_axis]`; group position `p` = the device's index along `cluster_axis` |
| `W` (global, logical) | `G · W_local` — the only divisor (`inv_W_bits = fp32(1/W)`, RT) |
| `Wt` (`tensor_width_tiles`) | `ceil(W_local / 32)` |
| `T` (`tensor_tile_rows`) | `prod(shape[:-2]) · ceil(shape[-2] / 32)` — each image tile-padded independently, never `floor(prod·M/32)` |
| `w_valid_in_last_tile` | `W_local − 32·(Wt − 1)` ∈ [1, 32] |
| `L` | links used `= min(num_links (canonical), publishers per wave)` |
| `N` | worker cores per chip `= |compute grid| − |port-core union|` (Work Distribution) |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (dims before `M`) | independent — the math never mixes leading indices | folded into `tile_rows` | folded | folding rule (`blocking-model.md` §1): identical treatment, contiguous in TILE page order | via `tile_rows` | — |
| `tile_rows` (32-row groups of `M`) | independent — each row's result depends only on its own row across devices | `block_rows` (R): tile-rows resident per block; `wave_rows` (q_w): tile-rows per exchange wave | R = `rows_this_core` (the whole assignment, one block) unless L1 forces a split; q_w = `ceil(rows_max / waves)` with `waves = min(rows_max, WAVES_MAX)` | host `fused_split()` → CT `block_rows`, `wave_rows`, `num_waves`; every R- / q_w-scaled CB capacity and trip count derived | contiguous runs over `G_h` row-groups (`divmod(T, G_h)`, base `_r3_interleaved_work`), spread over the grid left after the ports | knob-turn (R, q_w, WAVES_MAX, grid) |
| `width` (`W_local`, `Wt` tiles), reduction stages (`square_collapse`, `chip_sum`) | **dependent** — Σx² spans the whole local row, and the whole group's rows | `slice_tiles` (S): tiles of W_local one core owns | `fused_split()` (below): S = 4 at the 640×1792 focus case | host `fused_split()` → CT `S`; `G_w = ceil(Wt / S)` derived | split across `G_w` cores of a row-group (regime D3) with an on-chip combine; G_w = 1 is regime D1 | knob-turn within D3 (S); D1 ↔ D3 is the built regime pair |
| `width`, streaming stages (`scale_by_gamma`, `scale_by_inv_rms`, `store`) | independent — once inv_rms exists every output tile depends only on its own x, γ and row inv_rms | `w_chunk_tiles` (C) | `min(S, W_CHUNK_TILES)`, `W_CHUNK_TILES = 32` (base value) | one host constant `W_CHUNK_TILES`; `num_w_chunks = ceil(S / C)`, `Wp = num_w_chunks·C` derived | within the core (sub-blocks of the resident slice) | knob-turn |
| `devices` (group position `p`, `G` of them) | **dependent** — `ss` sums every device's partial; the caller fixed the split (each device holds one W_local slice) | `G` (not a knob: mesh geometry) | `mesh_shape[cluster_axis]` | the mesh | one slice per device (caller's); cross-device combine = all-gather of compact fp32 partials over the fabric, each device finalizes its own copy | TRANSPORT variants only (regime table X-*) |
| `gamma` (W_local elements) | **reuse-shared** — γ does not vary along `tile_rows`, the split axis | resident slice of `Wp` γ tiles per core | whole slice resident per core | derived from S | every worker reads its slice once from DRAM (stepping stone; G-MC multicast deferred) | refinement (G-MC) |
| `stats` (one fp32 per row per device) — the exchange's own data | independent along `tile_rows` | `wave_rows` (records per publisher per wave) | q_w | as `tile_rows` | each record published by the wave's owner of its row-group; each device all-gathers every peer's records | knob-turn (waves) |
| within-tile (32×32) | — (the tile is the unit) | — | — | — | — | partial tiles = padding rules, not blocking |

**Per-stage re-run of the table (every stage's own data has axes too):**

| Stage output | Its axes | Block it is produced in | Assignment | Note |
|--------------|----------|------------------------|------------|------|
| `sumsq_partial` / `partial_col` (1 tile per tile-row) | `tile_rows` | one wave (q_w tiles) | every slice core | collapsed column-0 slice partial — exactly the base R3 contributor partial |
| `chip_partial` (Σ over the G_w slices, 1 tile per tile-row) | `tile_rows` | one wave per row-group | the wave's **owner**, rotated `owner(g, w) = member (w mod G_w)` | rotation spreads the per-wave combine over members; with waves ≥ 2 no member owns two consecutive waves |
| compact record (32 fp32 per tile-row) | `tile_rows` | one wave | owner → its link's two port cores | the wire format: exactly `M·4 B` per device in total (+ nominal empty records of ragged groups) |
| `inv_rms` (1 column-0 tile per tile-row) | `tile_rows` | one wave | computed by the owner, all-gathered to the row-group | **every device computes its own** from the same G records summed in the same position order → bit-identical `ss` on every device; no device waits for another device's finalize |
| `x·γ`, `out` | `tile_rows × width` | R × Wp (γ pass), q_w × Wp per wave (inv pass) | every slice core | no stage computes on one core while others wait: the owner's extra work is one tile-row per wave and rotates |

The **port cores** are a communication role (one per (direction, link)), not a work assignment: they run no compute, own no rows, and are excluded from the work grid (`N` excludes them).

### Scheme committed for Phase 0

Per chip: the base op's W-split regime (R3 interleaved / sharded, here D3 / D3-WS) or row-parallel regime (R1 / R1-HS, here D1 / D1-HS), **re-split for the grid left after the ports**, with the per-core assignment cut into `num_waves` row **waves**. Each wave's per-row partials are combined on chip by the wave's owner, published as compact fp32 records into the ports, all-gathered around the group by store-and-forward relay, summed in group-position order and finalized on every device, and all-gathered back to the row-group. Waves pipeline: the exchange of wave k overlaps the local sums of waves > k, the γ pass of all waves and the reads of everything; only the last wave's exchange can stay exposed. **Every knob (S, R, q_w / WAVES_MAX, C, x_depth, stream_depth, L, the grid / core count, MIN_SLICE_TILES, MAX_GROUP_CORES) is a parameter with one source; none is an inlined literal.**

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_tiles` | `x_depth` | 1 when one block per core (the default: the whole assignment is resident); 2 when L1 forces `R < rows_this_core` | x_depth = 1: nothing to prefetch — the waves inside the block are the overlap; x_depth = 2: block k+1 reads during block k's scaling |
| `cb_output` (interleaved) | `stream_depth` | 2 | compute packs chunk j+1 while the writer drains chunk j |
| `cb_gather`, `cb_stats`, `cb_chip_partial`, `cb_rms_root`, `cb_partial_col`, `cb_sumsq_partial` | none (one wave quantum) | 1 wave | the next wave's quantum is produced only after this wave's is consumed by the same pair of threads in program order; the exchange-side pipelining lives in the **landing slots** (one per wave) |
| landing scratch (port cores) | `num_waves` slots per source chip | `num_waves` | every wave owns its own slot: a fast chip's wave k+1 never overwrites a slow chip's unread wave k |
| `cb_inv_rms`, `cb_x_gamma`, `cb_gamma_tiles` | none (whole block) | 1 block | resident across the block (Block schedule) |
| `cb_reduce_scaler` | none | 1 (2 with the sharded W-pad mask) | constant |

`stream_depth` is one host constant (2).

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| On-chip combine fan-in: `cb_gather` holds `G_w · q` partial pages on every worker (owners rotate) | `G_w` (interleaved) | `G_w ≤ MAX_GROUP_CORES = 24` in `fused_split()` (base `R3_MAX_GROUP_CORES`); WIDTH shards take `G_w` from the spec (base handles any G with reduce-scatter owners) and the footprint check | L1 overflow at program creation |
| Owners per wave: one owner per (row-group, wave) owns `q = ceil(q_w / owners)` rows, `owners = min(G_w, q_w)` | rows per owner | derived; nominal `q` quanta on every owner (ragged owners publish zero records) | ring drift on `cb_rms_root` / `cb_stats` (base changelog Refinement 6 "ragged-owner ring drift") |
| Port send: a segment is cut into packets of `ttnn.get_tt_fabric_max_payload_size_bytes()` (port.cpp `send_item`) | bytes per (wave, slot) | packetization, no extent cap | — |
| Packet-header pool: ≤ 12 headers per RISC (`port.cpp` static_assert) | distinct destination chips per port | relay schedule: 1 destination per port (+ the ready target, same chip) | compile error |
| Landing slot alignment 64 B (`SEG_ALIGN`, Blackhole DRAM/L1 NoC alignment) | `seg_stride` | `align_up(max_seg_bytes, 64)` | misaligned NoC writes → corrupted records |
| Record = 32 fp32 = 128 B per tile-row | record size | fixed; Float16_b partials are widened exactly (`bits << 16`) | — |
| Port cores must be outside the input's shard grid | free cores ≥ `2·L` (per chip union) | refuse (`NotImplementedError`, regime PORT-ON-SHARD) | a port core holding a shard would also have to run local work |
| DEST, `fp32_dest_acc_en=True`: 4 tiles half-sync (`dest_helpers.hpp:103`) | the cross-device fold keeps **2** tiles live (accumulator + next record page) | fold is sequential over `G` (any G) — never `G` live tiles | G > 4 would overflow DEST if the fold held all records at once |
| `num_waves ≤ WAVES_MAX = 8` | landing size, semaphore counts | clamp in `fused_split()` | landing L1 grows linearly |
| CB ring-wrap invariant (`blocking-model.md` §2) | every multi-page window | nominal quanta: `C` per tile-row on `cb_output`, `G_w·q` on `cb_gather`, `G·q` on `cb_stats`, `q` on `cb_rms_root` / `cb_chip_partial` | a window straddling `fifo_limit` → silent garbage |
| L1 budget (`ttnn.get_memory_view(device, L1).largest_contiguous_bytes_free_per_bank`, base `l1_budget()`) | S and R | `fused_split()` drops candidates whose footprint (ledger) exceeds the budget; then R halves (x_depth 2) | allocation clash |
| DRAM read alignment 64 B (Blackhole) | γ stick reads, W-pad / pad-tile writes | γ sticks start at 64 B-aligned page bases; the 16-element half move is L1→L1 (16 B alignment) | corrupted γ |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **D3** row-groups × W-slices + on-chip combine (base R3 interleaved, fused split) | **built** | `INTERLEAVED` and `fused_split(T, Wt, N).G_w > 1` | per core `R × S` resident (`R = rows_this_core`), waves of `q_w × S` | Minimum at DRAM: x read once, out written once. Above it by γ only: each row-group reads the γ slice once — `G_h · Wt` γ tiles' row-0 prefixes (≤ 9 % of x bytes at the focus case, ledger). On-chip NoC: per tile-row `G_w − 1` partial tiles (faces 0/2) to the owner + `G_w − 1` inv tiles (faces 0/2) back. Cross-device: `M·4 B` per device per group (the minimum wire). | per block: one set of chain / reduce inits; per wave: one combine round, one publish, one landed wait, one fold+finalize, one all-gather. Bigger q_w → fewer rounds but less overlap (lamp L2). |
| **D1** row-parallel, whole rows per core (base R1) | **built** | `INTERLEAVED` and `fused_split().G_w == 1` (Wt < MIN_SLICE_TILES, or the row count makes whole rows the cheapest split) | `R × Wt` resident, waves of `q_w × Wt` | as D3 with `G_w = 1`: no on-chip combine; γ read once per core | as D3 minus the combine |
| **D1-HS** HEIGHT_SHARDED (base R1-HS) | **built** | `HEIGHT_SHARDED` | the whole resident shard (zero-copy CB), waves of its tile-rows | 0 DRAM for x and out; γ once per core | one block per core |
| **D3-WS** WIDTH_SHARDED (base R3 sharded + its wave pipeline, Perf 1 `combine_pipeline`) | **built** | `WIDTH_SHARDED` | the whole resident shard; one row-group = all `T` tile-rows; waves of `q_w` rows; per wave `owners = min(G_w, q_w)` reduce-scatter owners | 0 DRAM for x and out; on-chip combine as D3; cross-device minimum | as D3 |
| **X-relay** store-and-forward ring / line all-gather of compact records — TRANSPORT | **built** (all regimes above) | always | per wave: one segment per (direction, link) | Ring: the busiest link carries `(G−1)/2` partial sets (the roofline's `links` term); Line: `G−1`. Records = 4 B per row = minimum. | per wave: one send per item + one arrival increment fused into the last packet; per call: one ready fence (~5 µs, fabric_ring_exchange `fence`) |
| **X-direct** multi-hop unicast to every peer — TRANSPORT | **rejected** — measured 1.1–1.6 µs slower than relay at every size (fabric_ring_exchange README §2: its fence waits for the 2-hop destination's ready) | — | — | same bytes | — |
| **X-padded-wire** ship each tile-row's partial as a tile (faces 0/2, 2 KiB fp32) — TRANSPORT | **rejected** — 16× the minimum wire (40 KiB/device at the focus case) moves the exchange from 5.5 µs (2.5 KiB) towards 7.6 µs (32 KiB) (fabric_ring_exchange table); superseded by X-relay's compact records | — | — | 16× minimum | — |
| **X-mcast** fabric multicast of the own segment — TRANSPORT | **deferred** — relay already sits 0.3 µs above the no-payload fence floor (5.38 vs 5.04 µs at 128 B), so multicast can save at most that; reachable: only the port's send items change (`exchange_plan()` and the port's item list), the landing layout and notify protocol stay | — | — | same | — |
| **X-root** reduce at one device + broadcast back — TRANSPORT | **rejected** — two exchange rounds (two fence-gated sends in series) instead of one; superseded by every device finalizing its own copy (redundant compute is one tile-row SFPU pass per wave) | — | — | 2 rounds | — |
| **COMPOSITE** pre-all-gather / all-gather / post-all-gather as three dispatches, stats through DRAM | **rejected** — violates one-dispatch; pays 3 launches + a DRAM round trip of the stats (prompt "Why it exists") | — | — | stats cross DRAM twice | — |
| **AG-X** all-gather x itself, each device normalizes full rows | **rejected** — dead end: moves `(G−1)·M·W_local·es` bytes over the links (≈ 7 MB/device at the focus case, ~70 µs at 2 links) to save a 2.5 KiB exchange | — | — | G× link bytes | — |
| **R2-stream** re-read x for the scaling pass | **rejected** — dead end (base R2): D3 keeps every slice resident at one DRAM crossing wherever a row is too wide | — | — | x crosses DRAM twice | — |
| **G-MC** γ slice read once per slice column + multicast to the `G_h` row-groups | **deferred** — positive reason: the Phase 0 core order (consecutive (y, x) runs of `G_w` per row-group, base `_group_work`) makes same-slice cores strided by `G_w` across the grid, and `ttnn.Mcast` builds only equal consecutive groups (base changelog Perf 1, helper friction); the γ term is ≤ 9 % of DRAM bytes at the focus case. Reachable: reorder cores column-major per slice so a slice's `G_h` cores are consecutive — a core-order change, the CB layout is unchanged (`cb_gamma_tiles` only changes who fills it). | `has_gamma` and D3/D1 | `cb_gamma_tiles` (Wp tiles) | removes `(G_h − 1)·Wt` γ prefix reads per chip; adds one multicast of S γ tiles per slice | — |
| **PORT-ON-SHARD** sharded input whose shard grid leaves < `2·L` free cores | **deferred** — positive reason: no `INPUTS` × TARGET shard geometry from `eval.sharding.auto_shard_config` on the 11×10 grid uses more than 86 cores (largest: WIDTH 32×8192 = 256 tiles → 86 cores), so ≥ 24 cores are free. Reachable: a helper worker reads the port core's shard over the NoC; the port protocol is unchanged | free cores < `2·L` | — | the moved shard crosses the NoC once | — |
| **SHARD-IN-INTERLEAVED-OUT** sharded input, interleaved output | **deferred** — positive reason: no golden / translated call reaches it (`helpers.py:run_distributed_rms_norm` passes the input's memory config for sharded inputs). Reachable: the D1-HS / D3-WS compute packs into a streaming `cb_output` instead of the shard-backed one; the interleaved writer path already exists | sharded input and `memory_config` interleaved | — | +1 DRAM write (the output) — same as the minimum for that placement | — |

**Regime-selection function (host, exact, `select_plan()` ported from base `rms_norm_program_descriptor.py:861`):**
`HEIGHT_SHARDED → D1-HS` (shard spec = split, base `_sharded_work` HEIGHT branch); `WIDTH_SHARDED → D3-WS` (shard spec = split); `INTERLEAVED → fused_split(T, Wt, N)`: `G_w == 1 → D1`, else `D3`. The base's `R3_OCCUPANCY_GAIN` / `R3_MIN_GROUP_CORES` gate is **replaced** by `fused_split()` (Work Distribution: the base rule falls off a cliff on the fused grid for 2048×1280 and gives no wave depth at 640×1792). Every branch is pinned by a regime-pinned test (acceptance test: D1 `(1,1,32,64)`, D3 `(1,1,640,1792)`, D1-HS / D3-WS sharded cases), and the selected regime + `(S, G_w, G_h, rows_max, num_waves)` is exposed by a host-only helper `plan_summary(input_tensor, ...)` for tests.

### Traffic ranking

Per device, bytes per tier at the focus case `640 × 1792` bf16 (T = 20, Wt = 56, x = 2.29 MB). Exact budget: `l1_ledger.md` → data-movement budget.

| Candidate split | DRAM crossings (x / out / γ) | Cross-core (on chip) | Cross-device | Rank | Decision |
|-----------------|------------------------------|----------------------|--------------|------|----------|
| D3: `G_h = 7` row-groups × `G_w = 14` slices (fused split) | 1 / 1 / 7 × slice prefixes (213 KB) | per tile-row 13 partial + 13 inv tiles (faces 0/2, 2 KiB fp32): 1.06 MB | 2.5 KiB own + relays | **1** | **built** |
| D3 with `G_h = 1` (all cores split W, all rows per core) | 1 / 1 / 1 × (30 KB) | per tile-row 55 partials + 55 inv: 4.5 MB; S = 1 | same | 2 | not chosen: −183 KB DRAM (γ) for +3.4 MB NoC and a fan-in of 56; covered by the G-MC row, which removes the same γ bytes without the fan-in |
| D1: whole rows, 20 cores | 1 / 1 / 20 × whole W (609 KB) | 0 | same | 3 | not chosen: more DRAM (γ), 20 of 106 cores, 1 wave (no overlap) |
| split `devices`? | — | — | — | — | not ours: the caller fixed the device split |
| all-gather x (AG-X) | 1 / 1 / … | 0 | ≈ 7 MB | 5 | rejected |
| R2-stream | 2 / 1 / … | 0 | same | 6 | rejected |

Occupancy is a separate question (`blocking-model.md` §4): D3 uses 98 of 106 cores at the focus case. It does not change the ranking.

### Block schedule

Logical schedule per worker core (reader, compute, writer are separate kernels; ports are separate cores). `block_idx` walks `num_blocks_this_core` blocks (1 in Phase 0); `w` walks the block's waves.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    // Phase A — needs no remote data
    for (uint32_t w = 0; w < num_waves_in_block; ++w) {
        load_wave(w);                    // reader: q_w x S tiles of x into the resident block (all waves issued up front)
        square_collapse_wave(w);         // compute: q_w raw Σx² tiles -> q_w column-0 slice partials
        contribute_wave(w);              // reader: partials (faces 0/2) -> owner(g, w)'s gather slots; G_w == 1: local
        if (owner(w)) chip_sum_wave(w);  // compute (owner): Σ over G_w partials -> chip partial (no finalize)
        if (owner(w)) publish_wave(w);   // reader (owner): compact fp32 records -> own slot (w, p) of both link ports
    }
    scale_by_gamma_block(block_idx);     // compute: x·γ for every wave (stall shadow; skipped without γ)
    // Phase B — needs the exchange
    for (uint32_t w = 0; w < num_waves_in_block; ++w) {
        if (owner(w)) collect_wave(w);              // reader (owner): wait landed(w); G records -> G column-0 pages
        if (owner(w)) cross_device_finalize_wave(w);// compute (owner): fold G pages in position order + rsqrt(·/W + ε)
        if (owner(w)) allgather_inv_wave(w);        // dataflow (owner): inv run -> every member's cb_inv_rms
        scale_by_inv_rms_wave(w);                   // compute: (x·γ)·inv for the wave's rows
        store_wave(w);                              // writer: the wave's output tiles -> DRAM / output shard
    }
    release_block(block_idx);
}
// port cores (separate kernel, concurrently from launch): ready_fence(); for w: send_wave(w) / relay / notify(w); rearm();
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `ready_fence` (port) — TRANSPORT | — | — | once per call, issued at launch, concurrent with Phase A (never on a worker's path) |
| `load_wave` | `q_w × Wp` tiles | earlier waves of the block | one read barrier per tile-row quantum (base `READ_BATCH_CHUNKS = None`: a whole slice row per barrier); reader issues every wave's reads without waiting on anything but CB space (x is resident) |
| `square_collapse_wave` | `grid(q_w, Wp)` + `q_w × 1` | `cb_x_tiles` (not popped) | one chain call + one reduce call per wave (one init each), cumulative wait so math starts on the first landed tile |
| `contribute_wave` | `q_w` partial tiles | — | one stateful write set-up per destination owner per wave, faces 0/2 only, rotated destination order (base `rms_norm_combine.hpp`); one semaphore increment per wave |
| `chip_sum_wave` (owner) | `G_w × q` tiles → `q` | — | one reduce call per wave (`AccumulateViaAdd`, `Skip`, no post-op) |
| `publish_wave` (owner) — OVERLAP | `q` records × 128 B → 2 ports | — | one 32-value extract per tile-row; one NoC write per port per wave; one `own_stats` increment per port per wave after the counted gate |
| `send_wave` (port) — TRANSPORT | segment of `n_pub(l) · q · 128 B` | landing slots of every wave | one item per destination per wave, last packet fused with the arrival increment |
| `notify` (port) — OVERLAP | — | — | per landed entry (notify quantum 1): one atomic increment per root of the link |
| `scale_by_gamma_block` | `grid(R, Wp)` (srcB-hold γ pass on Blackhole, chain carve-outs as base) | `cb_x_tiles`, `cb_gamma_tiles` | once per block (one init), base `SCALE_BLOCK_TILES` DEST windows |
| `collect_wave` (owner) — OVERLAP | `G × q` records → `G × q` column-0 pages | — | one landed wait per direction per wave; `G − 1` record reads (128 B each) from the port cores per tile-row; column-0 scatter of 32 values per record into pre-zeroed pages |
| `cross_device_finalize_wave` (owner) | `G × q` pages → `q` | — | per tile-row: `G` copies + `G − 1` SFPU adds (2 DEST slots) + one column-0 finalize pass (base `finalize_col0_body`) |
| `allgather_inv_wave` (owner) | `q` inv tiles → `G_w` members | — | one write set per member per wave (faces 0/2, rotated), one `SEM_INV` increment per member per wave |
| `scale_by_inv_rms_wave` | `grid(q_w, Wp)` | `cb_x_gamma` (or `cb_x_tiles`), `cb_inv_rms` | one chain / hold call per wave |
| `store_wave` | `q_w × Wp` tiles | — | one write barrier per C-tile quantum |
| `release_block` | — | — | pops once per block |

### Perf lamps

| Lamp | Kind | Why the default may be wrong here | Nearby alternative to measure |
|------|------|-----------------------------------|-------------------------------|
| L1 slice floor `MIN_SLICE_TILES = 4` (base value) | OVERLAP | at the focus case S = 4 gives 3 waves (G_w 14, G_h 7); S = 3 gives 4 waves at the same 12 tiles/core (G_w 19, G_h 5) — deeper overlap vs a larger fan-in | `MIN_SLICE_TILES = 3` on 640×1792 |
| L2 wave depth `WAVES_MAX = 8`, `q_w` | OVERLAP | more waves hide more but each pays a combine round, a publish, a landed notify and an all-gather; fewer waves expose `1/num_waves` of the writes plus the last exchange | `num_waves ∈ {1, 2, rows_max}` per perf case; `num_waves = 1` is the no-overlap control |
| L3 links used for the stats | TRANSPORT | fabric_ring_exchange: a second link is equal or slower at 128 B – 8 KiB; L = 2 reserves 2 more port cores (106 vs 108 workers) | carry every record on link 0 when `num_links = 2` (ports of link 1 run only the fence) — measure op time, not transport alone |
| L4 γ pass position | OVERLAP | the whole-block γ pass sits between Phase A and Phase B as the stall shadow; if wave 0's records land before it ends, the inv pass of wave 0 waits behind γ of later waves | γ per wave interleaved with the inv passes (base `R3_WAVE_GAMMA0_POS`) |
| L5 on-chip combine wire | OVERLAP | faces 0/2 tiles (2 KiB fp32) per partial put 26 KiB per tile-row on each owner, on the per-wave critical path before `publish_wave` | compact 128 B rows: contributor `j` writes its record into row `j` of the owner's page (2 × 64 B), owner `reduce<SUM, REDUCE_COL>`; the result row 0 *is* the wire record |
| L6 γ read mode | LOCAL | focus case is DRAM-bound and γ PREFIX reads 544 B per tile | base `GAMMA_DIRECT_READ = SEGMENTS` (64 B per tile; base measured 1 % better at 640 rows) |
| L7 owner placement | OVERLAP | rotation spreads combine work; owners far from the ports add NoC hops on the critical path | fixed owner = the member nearest the link's ports |
| L8 notify quantum | OVERLAP | per entry (default) costs `n_roots` atomics per entry; the owner needs all `G − 1` entries of a wave anyway | `notify_quantum = n_in` (once per wave per port) |
| L9 grid synchronization (decode) | OVERLAP | decode is fence-bound (~5 µs exchange vs 0.45 µs roofline); 14 slice cores × 4 tiles add a combine round to the exposed tail | `MIN_SLICE_TILES = 8` / `G_w ≤ 8` at `T = 1` |
| L10 NoC swap (base `NOC_SWAP_COLUMNS = 3`) | LOCAL | measured on the full 120-core p150b grid; the fused grid and the QB2 p300c layout differ, and port cores sit on the eth-row side | re-measure swap on/off on the focus case |
| L11 srcB-hold passes (base Perf 2) | LOCAL | carve-outs (`HOLD_GAMMA_MIN_ROWS`, `HOLD_INV_MIN_CW`) were tuned for the base's shapes; per-core blocks here are 3–5 tile-rows | chain vs hold on 640×1792 / 2048×1280 |
| L12 idle cores | LOCAL | `fused_split()` leaves 8 (focus) / 2 (2048×1280) workers idle | give the idle cores the ragged group's rows (uneven `G_h` assignment) |

**Stall-shadow check.** Two stages wait on something they do not produce:
(1) **the owner's `collect_wave(w)` waits on the fabric** (peers' records). Independent of it: the local sums of waves > w (Phase A runs them first), the reads of every wave (issued up front), and `scale_by_gamma` of every wave — legal because `x·γ·s = (x·γ)·s` (the two scalings commute), so γ is applied first exactly as the base op orders it; floating-point order is identical to the base. The cross-device sum is lifted *before* finalize by associativity of the sum, but its order is pinned (group position 0..G−1 on every device) so every device rounds identically.
(2) **every member's `scale_by_inv_rms_wave(w)` waits on the owner's all-gather.** Independent: the inv passes of earlier waves and the stores of earlier waves (Phase B is in wave order).
The **ports' ready fence** waits on peers launching; nothing on a worker waits on it except, transitively, `collect_wave`. A chip that launched early does all its Phase A and γ work during the skew (launch-skew tolerance).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_x_tiles` (INTERLEAVED) | input dtype tiles | reader: `TensorAccessor` page reads, page id `tile_row · Wt + w0 + t` for the core's slice `[w0, w0 + V)` (base R3 interleaved reader), one barrier per tile-row; nominal `Wp` tiles per tile-row pushed | pad tiles of the slice zeroed once at kernel start; W-pad columns zeroed per tile-row on the core holding tile `Wt − 1` (padding rules) |
| shard → `cb_x_tiles` (HEIGHT / WIDTH) | input dtype tiles | `ttnn.cb_descriptor_from_sharded_tensor(input)` — zero-copy; reader only pushes the shard's pages | never re-read over the NoC; W-pad handled by the compute-side mask (base L5 mechanism) |
| `cb_x_tiles` → `cb_sumsq_partial` → `cb_partial_col` | intermediate format (Float32 under fp32 DEST, Float16_b under 16-bit DEST) | compute: base `square_accumulate` chain (DEST-accumulated `x·x` per tile-row) then `reduce<SUM, REDUCE_ROW>` collapse (scaler 1.0, no post-op) | column 0 holds the slice's Σx² of each of the 32 rows |
| `cb_partial_col` → owner `cb_gather` | intermediate tiles, faces 0/2 | reader: base `rms_norm_combine.hpp:combine_write_tiles` + `rotated()`, contributor-major slots `j·q + r`, `SEM_GATHER` increment after the write barrier | owner's own partial copied locally into its slot |
| `cb_gather` → `cb_chip_partial` | intermediate | compute (owner): `reduce<SUM, REDUCE_COL, …, AccumulateViaAdd, …, ReduceWithinTile::Skip>` over `(G_w × q)` (base combine, no post-op) | the chip's local Σx² per row (unscaled) |
| `cb_chip_partial` → port landing (own slot) — OVERLAP | **compact record: 32 × fp32 (128 B) per tile-row**, little-endian, row order 0..31 | reader (owner): 32 loads from column 0 (face 0 rows 0–15, face 2 rows 0–15) of the page into `cb_record_scratch` (Float16_b widened `bits << 16`, exact), NoC write of the wave's `q` records to the same landing offset on **both** ports of its link, write barrier, then `own_stats += 1` on each port after the counted gate | landing offset = `((w · G + p) · seg_stride) + (k_l · q) · 128`, `k_l` = the publisher's index within link `l` |
| port ↔ port — TRANSPORT | landing slots, bytes as published | `fabric_ring_exchange/kernels/port.cpp` relay: ready fence once per call; per wave FWD sends own slot to `p+1`, then relays entries; BWD sends own to `p−1`; last packet of each item = fused write + arrival increment; re-arm counters at the end | Linear: the line schedule (fabric_all_gather `_schedule`): FWD carries `p, p−1, …, 0`, BWD `p, p+1, …, G−1`; a line end's missing direction has no connection |
| port → owners notify — OVERLAP | — | port `poll_notify`: per landed entry, `noc_semaphore_inc(root, SEM_LANDED_{FWD,BWD})` on every root of the link | owner of wave w waits `landed_d ≥ (w + 1) · n_in_d` for each direction d |
| port landing → `cb_stats` | intermediate pages, column 0 valid, other lanes zero | reader (owner): NoC read of the `G − 1` peer records (128 B each) from the two port cores, plus the own record from `cb_record_scratch`; scatter each into column 0 of its page (pages pre-zeroed once at kernel start; only column 0 is ever rewritten) | page order = group position `q = 0..G−1` (own at `q = p`) — identical on every device |
| `cb_stats` → `cb_rms_root` | intermediate | compute (owner): per tile-row `copy_tile(q=0) → D0`; for `q = 1..G−1`: `copy_tile(q) → D1`, `add_binary_tile(D0, D1, D0)` (SFPU, fp32 under fp32 DEST); then base `finalize_col0_body` (×(1/W), +ε, rsqrt in one column-0 SFPU pass); pack | position-order fold → bit-identical `ss` on every device |
| `cb_rms_root` → members' `cb_inv_rms` | intermediate, faces 0/2 | owner dataflow: base all-gather (`combine_write_tiles` + `rotated()`, `SEM_INV` per member), on the writer RISC-V when γ is resident or absent (base `R3_ALLGATHER_ON_WRITER`, increments handed to the reader) | members push `cb_inv_rms` when `SEM_INV` reaches the wave quota |
| γ DRAM → `cb_gamma_tiles` (TILE) | γ dtype tiles, row 0 valid | writer, once per kernel: base direct prefix read (`GAMMA_DIRECT_READ = PREFIX`, `gamma_row0_prefix_bytes`) of the slice's `V` tiles | `BroadcastDim::Row` reads tile row 0 only |
| γ DRAM → `cb_gamma_tiles` (ROW_MAJOR sticks) | γ dtype, row 0 valid | writer, once per kernel: for each slice tile `t`, one DRAM read of stick `w0 + t` (`32·es_g` B, 64 B-aligned page) into the page at offset 0 (fills face-0 rows 0–1), then an L1→L1 move of elements 16..31 (face-0 row 1) to face-1 row 0 (offset `256·es_g`) | new to this op (the base's RM γ was one W-long stick); no staging CB |
| `cb_x_tiles` × `cb_gamma_tiles` → `cb_x_gamma` | intermediate | compute: base γ pass (srcB hold `hold_mul_row` on Blackhole, chain carve-outs) | γ absent: skipped, the inv pass reads `cb_x_tiles` strided |
| (`cb_x_gamma` or `cb_x_tiles`) × `cb_inv_rms` → `cb_output` | output dtype | compute: base inv pass (`BroadcastDim::Col`, srcB hold / chain per carve-out) per wave | final rounding at this pack |
| `cb_output` → DRAM / output shard | tiles | writer: interleaved — the slice's valid tiles of each tile-row, one barrier per quantum; sharded — the CB is backed on the output shard (no write) | pad tiles never stored |

### Padding rules (op contract: input padding is arbitrary)

| Padding | Where | Treatment |
|---------|-------|-----------|
| W tile padding, interleaved (columns `w_valid_in_last_tile..31` of tile `Wt−1`) | enters Σx² | reader overwrites with zeros after the read lands, before the push (base). **Bfp8_b**: zero the pad datums' mantissa bytes (1 B each, sign+magnitude in the 1024 B mantissa section, face-major) — a zero magnitude is zero whatever the shared exponent; exponent bytes untouched |
| Pad tiles `V..Wp−1` of a slice | enter Σx² | zero-filled once at kernel start (fixed ring positions, nominal `Wp` per tile-row) — base |
| W padding, sharded | enters Σx² | compute-side mask on the core owning tile `Wt−1`: base `ReducePartialScaler::with_partial()` + `cb_sumsq_last` / `cb_sumsq_acc` |
| H padding (rows `M % 32..31` of the last tile-row of each image) | row-local | none — their records are exchanged and finalized like any row (finite or not, they never mix with valid rows) and land in output padding |
| γ positions ≥ W_local (TILE pad columns, RM stick tail) | multiply pad columns only | none |
| Record padding (ragged group / owner with no row in a wave) | exchanged | the owner publishes a zero record (nominal `q` records), so every port segment and `own_stats` count is uniform per wave |

### Tensix-to-Tensix and chip-to-chip contracts

| Contract | Decision |
|----------|----------|
| Worker → port (publish) | owner writes `q · 128 B` into `landing[w][p]` at its publisher offset on both ports of its link, `noc_async_write_barrier`, then (wave w ≥ 1 only after `SEM_OWN_COUNTED ≥ n_active_ports(l) · w`) `noc_semaphore_inc(port, SEM_OWN_STATS, 1)` on each port. Port sends wave w when `own_stats ≥ (w + 1) · n_pub(l)` and releases the counted gate once per wave (port.cpp `n_counted`). |
| Port → worker (notify) | port.cpp `poll_notify`, notify quantum 1, `SEM_LANDED_FWD` / `SEM_LANDED_BWD` (two program semaphore ids on every worker) |
| Worker ← port (collect) | owner NoC-reads records from the port cores' landing (port L1 is immutable until the next call's ready fence) |
| Chip ↔ chip (fence) | port.cpp ready fence: a port sends one ready to every port that writes into this chip through its direction, and waits for one ready per destination chip before its first send; counters re-armed to 0 at the end (trace-safe) |
| Landing addressing | one persistent L1 HEIGHT-sharded mesh tensor over the **union** of every chip's port cores (same address on every device); per port core: `num_waves · G · seg_stride` bytes; sends go to the destination chip's port of the same (direction, link) at its own coordinates (an RT word per item — placements may differ per chip) |
| On-chip combine (D3 / D3-WS) | base R3 contract (`rms_norm/op_design.md` Tensix-to-Tensix → R3, Refinement 6 reduce-scatter text, Perf 1 waves): contributor-major gather slots, faces 0/2, rotated order, `SEM_GATHER`, all-gather into `cb_inv_rms` with `SEM_INV`, `noc_async_atomic_barrier()` at reader kernel end |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile-row of one W-slice (`1 × S` tiles); per core a block of `R` tile-rows; per wave `q_w` tile-rows |
| Port cores (per chip) | `2 · L_ports` cores (`L_ports` = canonical `num_links`), one per (direction ∈ {FWD, BWD}, link), on **every** chip including line ends (receive-only ports keep the incoming direction). Placement ported from `fabric_all_gather` `plan()`: each port is the free core with the fewest NoC1 hops to its link's Ethernet core (`_worker_nearest_noc1`), receive-only / unconnected ports on any free core (`_worker_below`). **No probe program** (a probe is a second dispatch): the Ethernet channel is the first word `setup_fabric_connection()` returns (read by `WorkerToFabricEdmSender::build_from_args`, `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:113`), mapped to its physical core through the SoC descriptor tables `_eth_physical_lists` already reads; `setup_fabric_connection` is called on a throw-away `ProgramDescriptor` for the query. Sharded inputs: `allowed = grid − shard cores`. Cached per (mesh, fabric config, cluster_axis, topology, L, shard grid). |
| Worker grid | `grid − union over all chips of their port cores`, ordered (y, x) row-major (base order). Identical core **count** `N` on every chip, so `fused_split()` — a function of `(T, Wt, N)` only — gives every chip the same rows→waves→records mapping (the cross-chip record layout depends on it). QB2 1×4 ring, L = 2: `N = 110 − 4 = 106` when every chip's ports land on the same 4 cores; the rule re-runs on whatever `N` the union leaves. |
| `fused_split(T, Wt, N)` (interleaved; replaces base `_r3_interleaved_layout` + occupancy gate) | for `S` in `[min(Wt, MIN_SLICE_TILES) .. Wt]`: `G_w = ceil(Wt/S)`; skip if `G_w > MAX_GROUP_CORES` or `G_w > N` or the ledger footprint at `(S, R = ceil(T/G_h))` with `x_depth = 1` exceeds the L1 budget; `G_h = min(T, N // G_w)`; `rows_max = ceil(T / G_h)`; key = `(rows_max · S, −min(rows_max, WAVES_MAX), G_w)`; take the minimum key. Then `waves = min(rows_max, WAVES_MAX)`, `q_w = ceil(rows_max / waves)`, `num_waves = ceil(rows_max / q_w)`. Row ranges: `divmod(T, G_h)` (first `extra` groups one row more), cores: consecutive runs of `G_w` (base `_group_work`). If no candidate fits at `x_depth = 1`, retry with `R` halved and `x_depth = 2` (blocks, waves continue across blocks); if `R = 1` still does not fit, refuse. No safety fraction. |
| Sharded split | HEIGHT: shard i on the i-th core of `corerange_to_cores(shard grid, row_wise = orientation == ROW_MAJOR)`, G_w = 1, waves over its tile-rows. WIDTH: base `_sharded_work`: one row-group of all shard cores (G_w = shard count), waves over the `T` rows, `owners = min(G_w, q_w)` per wave. |
| Publishers → links | per wave, publishers (every (row-group, owner) pair, fixed order) are dealt round-robin over links: publisher `k` → link `k mod L`, offset index `k_l = k div L`; `L = min(L_ports, publishers per wave)` (links with no publisher run only the fence). `n_pub(l)` is constant across waves (nominal). |
| Remainder | rows: `divmod` groups differ by ≤ 1 tile-row; a group with fewer rows publishes zero records in its last wave; W: ragged last slice (`V < S`) pushes nominal `Wp` per tile-row, pad tiles zeroed; tile geometry always `ceil`, per image |
| Perf-case check on the fused grid (N = 106, L = 2) | **640×1792** (T 20, Wt 56): S 4, G_w 14, G_h 7 → 98 cores, rows 3/…/3/2, 3 waves (q_w 1), 12 tiles (24 KiB) per core; 7 publishers per wave (4 on link 0, 3 on link 1); segments 512 / 384 B. The base rule on this grid would give S 6, G_w 10, G_h 10, 2 rows/core (2 waves); on its full grid 11 groups with ragged 2/1 rows. **32×1792** (T 1): S 4, G_w 14, G_h 1 → 14 cores, 1 wave, 1 publisher → L = 1 used. **2048×1280** (T 64, Wt 40): S 5, G_w 8, G_h 13 → 104 cores, rows 5 (12 groups) / 4 (1), 5 waves, 25 tiles per core. The base rule **falls off a cliff** here: `G_w = 3 < R3_MIN_GROUP_CORES` sends it to R1 = 64 cores × 40 tiles, 1 wave (no overlap) — hence `fused_split()`. |
| Persistent resources | cached per key `(mesh id, cluster_axis, topology, L_ports, port-core union, num_waves, seg_stride, G)`: the landing tensor (`ttnn.from_torch(zeros, uint32, ROW_MAJOR, HEIGHT_SHARDED over the union, ShardTensorToMesh)`), global semaphores (`ttnn.create_global_semaphore(mesh, union, 0)`: READY + `G − 1` arrival counters). Created on the cold call only, with one `ttnn.synchronize_device(mesh)` after creation so every chip's counters read 0 before any chip's first program; **never** on a warm call. Program semaphores: workers `SEM_GATHER`, `SEM_INV` (or the base mcast ids), `SEM_LANDED_FWD`, `SEM_LANDED_BWD`, `SEM_OWN_COUNTED`; ports `SEM_OWN_STATS` — distinct ids, descriptors covering every core that addresses them. |
| Program cache | `custom_program_hash` per chip program from everything that changes a per-core RT layout or CT arg (regime, S, G_w, G_h, waves, q, L, topology, cluster_axis, G, p, port placement, γ format, dtypes, fidelity, placement) — `generic_op` keys only on the outer RT-grid size (fabric_ring_exchange README "Gotcha"). |
| Regime pinning | tests pin D1, D3, D1-HS, D3-WS and `num_waves ∈ {1, > 1}` (acceptance test + `plan_summary`) |

## Circular Buffers

Symbols: `R` block rows, `q_w` wave rows, `q` rows per owner per wave, `S` slice tiles, `C` chunk, `Wp = ceil(S/C)·C`, `G_w` slice cores per row-group, `G` devices, `T_in` / `T_out` / `T_g` tile bytes of input / output / γ dtype (Bfp8_b 1088 B), `T_i` intermediate tile (4096 B fp32 DEST, 2048 B 16-bit DEST). Owner-only CBs are allocated on every worker (owners rotate; same address everywhere).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_tiles` | 1 | `T_in` | interleaved `x_depth · R · Wp`; sharded = the shard (zero-copy) | spans R tile-rows × the slice: read once, used by square and both scaling passes | input dtype | reader | compute | block |
| `cb_reduce_scaler` | 2 | 2048 | 1 (2 with the sharded W-pad mask: full + partial) | constant | bfloat16 | reader | compute | kernel |
| `cb_sumsq_partial` | 3 | `T_i` | `q_w` | raw Σx² lanes of one wave's tile-rows | intermediate | compute | compute | wave (square → collapse) |
| `cb_inv_rms` | 4 | `T_i` | `R` | inv of every wave of the block, column 0 valid; waved in `q_w` quanta | intermediate | reader (pushes after `SEM_INV`) | compute | block |
| `cb_gamma_tiles` | 6 | `T_g` | `Wp` (γ only) | resident slice of row-0 γ tiles | γ dtype | writer | compute | kernel |
| `cb_x_gamma` | 7 | `T_i` | `R · Wp` (γ only) | x·γ of the whole block, consumed per wave by the inv pass | intermediate | compute | compute | block |
| `cb_partial_col` | 9 | `T_i` | `q_w` | collapsed slice partial of one wave | intermediate | compute | reader | wave |
| `cb_gather` | 10 | `T_i` | `G_w · q` | the owner's gather slots of one wave (remote writes land here) | intermediate | reader | compute | wave (owner) |
| `cb_rms_root` | 11 | `T_i` | `q` | finalized inv run of one wave before the all-gather | intermediate | compute | writer (all-gather on the writer RISC-V; reader when γ streams — base rule) | wave (owner) |
| `cb_sumsq_last` | 12 | `T_i` | `q_w` (sharded masked only) | x² of the partial last tile | intermediate | compute | compute | wave |
| `cb_sumsq_acc` | 13 | `T_i` | `q_w` (sharded masked only) | collapsed full-tile Σx² accumulator | intermediate | compute | compute | wave |
| `cb_chip_partial` | 14 | `T_i` | `q` | the owner's chip-local Σx² (unscaled) per owned row | intermediate | compute | reader | wave (owner) |
| `cb_stats` | 15 | `T_i` | `G · q` | the G devices' partials of each owned row, position order, column 0 | intermediate | reader | compute | wave (owner) |
| `cb_output` | 16 | `T_out` | interleaved `stream_depth · C`; sharded = the output shard (zero-copy) | one C-chunk of a tile-row in flight to the writer | output dtype | compute | writer | chunk |
| `cb_record_scratch` | 17 | 128 | `G · q` | compact records of one wave: the own record (built for publish) + `G − 1` peer records read from the ports; reader-internal (reserve once, never pushed) | raw fp32 | reader | reader (scratch, no consumer thread) | kernel |

`compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial)` is the compute kernel's first statement. Port cores have no CBs (landing is a tensor shard; headers from `PacketHeaderPool`). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | prologue | — | yes (scaler) | — | `cb_reduce_scaler` 1–2; `cb_gamma_tiles` Wp (writer); pad tiles of `cb_x_tiles` and `cb_stats` pages zeroed once | resident |
| 1 | `load_wave` | `q_w × Wp` | raw (TensorAccessor) | DRAM | `cb_x_tiles` Wp per tile-row | pushed, never popped until release |
| 2 | `square_collapse_wave` | `grid(q_w, Wp)` → `q_w × 1` | yes (chain + reduce) | `cb_x_tiles` cumulative, not popped | `cb_sumsq_partial` → `cb_partial_col` q_w | partials pushed |
| 3 | `contribute_wave` | `q_w` tiles | raw (base combine transport) | `cb_partial_col` q_w (popped) | owner's `cb_gather` slots | — |
| 4 | `chip_sum_wave` (owner) | `G_w × q` → `q` | yes (reduce, AccumulateViaAdd, Skip) | `cb_gather` G_w·q (BulkWaitBulkPop) | `cb_chip_partial` q | — (D1 / D1-HS, `G_w = 1`: no gather, no chip_sum — the collapse packs straight into `cb_chip_partial` and `contribute_wave` is skipped) |
| 5 | `publish_wave` (owner) | `q` records | raw | `cb_chip_partial` q (popped) | `cb_record_scratch` → both ports' landing | own_stats incremented |
| 6 | `scale_by_gamma_block` | `grid(R, Wp)` | yes (chain) / base srcB hold | `cb_x_tiles` strided, `cb_gamma_tiles` (held) | `cb_x_gamma` R·Wp | — |
| 7 | `collect_wave` (owner) | `G × q` records | raw | port landing, `cb_record_scratch` | `cb_stats` G·q | — |
| 8 | `cross_device_finalize_wave` (owner) | `G × q` → `q` | raw (custom fold) + base `finalize_col0_body` | `cb_stats` G·q (popped) | `cb_rms_root` q | — |
| 9 | `allgather_inv_wave` (owner) | `q` tiles → G_w members | raw (base combine transport) | `cb_rms_root` q (popped) | members' `cb_inv_rms` | `SEM_INV` incremented |
| 10 | `scale_by_inv_rms_wave` | `grid(q_w, Wp)` | yes (chain) / base srcB hold | `cb_x_gamma` (or `cb_x_tiles` strided), `cb_inv_rms` (caller-managed) | `cb_output` C per tile-row | — |
| 11 | `store_wave` | `q_w × Wp` | raw | `cb_output` C per quantum | DRAM | popped |
| 12 | `release_block` | — | raw | `cb_x_tiles` R·Wp, `cb_inv_rms` R, `cb_x_gamma` R·Wp | — | empty |
| P | port: fence / send / relay / notify / rearm | per wave segments | raw (example kernel) | landing | landing on peers, worker semaphores | counters 0 |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api (mandatory) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_x_tiles, cb_x_tiles, cb_sumsq_partial)` | — | — | — |
| prologue scaler | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()`; sharded mask: `calculate_and_prepare_partial_reduce_scalers` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:98`, `:153` | pool-type-aware, SUM → 1.0 | — | `cb_reduce_scaler` | — |
| `square_collapse_wave` (square) | helper | `compute_kernel_lib::eltwise_chain(IterationShape::grid(q_w, Wp), BinaryFpu<Mul, x, x, Dst::D0, DestAccumulation::PerRow>{}, PackTile<cb_sumsq_partial, PerOuter, PerOuter>{})` — base call unchanged | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:121` (IterationShape), `:208` (Cumulative), `:280-285` (PerRow); base `kernels/rms_norm_compute.cpp` square stage | `WaitPolicy::Cumulative`, `PopPolicy::None` on `cb_x_tiles` | `cb_x_tiles` | `cb_sumsq_partial` | shape `(q_w, Wp)` |
| `square_collapse_wave` (collapse) | helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, cb_sumsq_partial, cb_reduce_scaler, cb_partial_col, BulkWaitBulkPop>(ReduceInputBlockShape::of(q_w, 1))` (no post-op); masked sharded: + `ReducePartialScaler::with_partial()` / `Accumulate::at(cb_sumsq_acc, 1)` (base) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:616-637`, `:290-299` | `ReduceAlgorithm::Auto` | `cb_sumsq_partial` | `cb_partial_col` | shape `(q_w, 1)` |
| `chip_sum_wave` | helper | `reduce<SUM, REDUCE_COL, cb_gather, cb_reduce_scaler, cb_chip_partial, BulkWaitBulkPop, …, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(G_w, q), …)` (base R3 owner combine, without the finalize post-op) | `reduce_helpers_compute.hpp:156` (AccumulateViaAdd), `:191` (Skip), `:616-637` | row pitch `q` (contributor-major) | `cb_gather` | `cb_chip_partial` | `(G_w, q)` |
| `scale_by_gamma_block` | helper / raw (base Perf 2) | Blackhole: base `hold_mul_row` (srcB hold); carve-outs: `eltwise_chain(grid(R, Wp), BinaryFpu<Mul, x strided, γ BroadcastDim::Row>, PackTile<cb_x_gamma>)` | base `kernels/rms_norm_compute.cpp:118` (`hold_mul_row`), `kernels/rms_norm_srcb_hold.hpp`; `chain.hpp:261` (Strided) | base `SCALE_BLOCK_TILES = 8` | `cb_x_tiles`, `cb_gamma_tiles` | `cb_x_gamma` | `(R, Wp)` |
| `scale_by_inv_rms_wave` | helper / raw (base Perf 2) | base `hold_mul_col` or `eltwise_chain(grid(q_w, Wp), BinaryFpu<Mul, A, inv BroadcastDim::Col>, PackTile<cb_output>)` | base `kernels/rms_norm_compute.cpp:185`; `chain.hpp:208-266` | caller waits `cb_inv_rms` per wave | `cb_x_gamma` / `cb_x_tiles`, `cb_inv_rms` | `cb_output` | `(q_w, Wp)` |
| `cross_device_finalize_wave` (fold) | raw_api | `copy_tile(cb_stats, q, D)`; `add_binary_tile(D0, D1, D0)` (SFPU, with `add_binary_tile_init`) | `tt_metal/hw/inc/api/compute/tile_move_copy.h:126`; `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h:102` | `G` fixed by the mesh; 2 DEST slots | `cb_stats` | DEST | `q` |
| `cross_device_finalize_wave` (finalize) | raw_api (base) | base `finalize_col0_body(inv_W_bits, eps_bits)` via `VectorMode::C` | base `kernels/rms_norm_compute.cpp:80-91`, `:326-331` | `inv_W_bits = fp32(1/(G·W_local))` | DEST | `cb_rms_root` | — |
| `contribute_wave`, `allgather_inv_wave` | raw_api (base) | `rms_norm_combine::combine_write_tiles<T_i>`, `rotated()`, `noc_semaphore_inc`, `noc_async_atomic_barrier` | base `kernels/rms_norm_combine.hpp:28-47`; base reader `scatter_block` / `pipe_block` (`kernels/rms_norm_reader.cpp:277`, `:411`) | faces 0/2 | `cb_partial_col` / `cb_rms_root` | remote `cb_gather` / `cb_inv_rms` | q, q_w |
| `load_wave`, `store_wave`, W-pad / pad-tile zeroing, γ reads | raw_api (base) | `TensorAccessor`, `noc_async_read/write`, barriers, MEM_ZEROS reads | `tech_reports/tensor_accessor/tensor_accessor.md`; base reader / writer | per tile-row quanta | DRAM | `cb_x_tiles` / `cb_gamma_tiles` / DRAM | Wp, C |
| `publish_wave`, `collect_wave` | raw_api | L1 loads/stores (column-0 extract / scatter), `noc_async_write`, `noc_async_read`, `noc_semaphore_inc`, `noc_semaphore_wait_min`-style non-blocking polls in the base wave poll loop | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | `q` records | `cb_chip_partial`, landing | landing, `cb_stats` | q |
| port: connection, fence, send, relay, notify, rearm | raw_api (example) | `fabric_ring_exchange/kernels/port.cpp` (copied; additions: per-item destination port coords, per-port `n_pub`, landed semaphore id per direction) — `WorkerToFabricEdmSender::build_from_args`, `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route`, `to_noc_fused_unicast_write_atomic_inc`, `send_current_slot_non_blocking` | `ttnn/ttnn/operations/examples/fabric_ring_exchange/kernels/port.cpp:88-250`; `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:91` | `G`, `num_waves`, payload (CT) | landing | peer landing | num_waves, seg bytes |
| host: placement | host | `ttnn.setup_fabric_connection`, `ttnn.get_forwarding_link_indices`, `ttnn.get_eth_forwarding_direction`, `fabric_all_gather` `plan()` helpers (`build_groups`, `_schedule`, `_needs_ready`, `_worker_nearest_noc1`, `_eth_physical_lists`) copied | `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:371-412, 514-640, 717-835` | — | — | — | — |

**Helpers considered and rejected (raw_api rows):**

| Raw use | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|---------|------------------|----------------------|---------------------------------|
| cross-device fold | `reduce<SUM, …, AccumulateViaAdd>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:129` ("pairwise FPU add_tiles(acc_to_dest)") | FPU adds read the records through SrcA/SrcB (tf32): the prompt requires fp32 when summing the exchanged partials; the SFPU fold keeps fp32 in DEST |
| cross-device fold | `eltwise_chain(…, CopyTile, AddBinary)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp:20` (`AddBinary<In0, In1, Out>` is one DEST-DEST op per chain element); `chain.hpp:121` (IterationShape walks tiles of one shape) | a chain applies a fixed element list per tile of its shape; a G-way fold over G pages of one CB into one accumulator is not a per-tile chain, and G ≥ 5 would need G live DEST slots in one window (`dest_helpers.hpp:103`: 4 under fp32 DEST) |
| column-0 finalize | `reduce` post-op lambda | `reduce_helpers_compute.hpp:637` | base Refinement 6 measured the stock full-tile post-op at ~1.8 k cycles/tile-row; `finalize_col0_body` is the base's justified raw sfpi (base `rms_norm_compute.cpp:80`) |
| port egress | `dataflow_kernel_lib::ccl::FabricStreamSender` / `DirectConn` | `ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp:207` (`FabricConnectionManager conn_`), `:299-305` (`unicast_route`: 1-D hop routing) | `DirectConn` builds from the `FabricConnectionManager` RT-arg block, while the generic_op host path emits the `WorkerToFabricEdmSender` block of `ttnn.setup_fabric_connection` (`fabric.cpp:151`) that the measured port kernel consumes; the measured transport (fabric_ring_exchange) is that kernel. Migration is an ergonomics refinement once the host can emit the manager block. |
| on-chip combine transport | `mcast_pipe` `SenderPipe` / `ReceiverPipe` | base changelog Perf 1 "Helper bypasses" (`SenderPipe::send()` one region + one signal per call; no face-strided subset write) | faces 0/2 subset writes to many destinations — base bypass, kept |
| `publish` / `collect` | `local_copy_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp` (page / region copies) | the transform is a 32-element strided gather (column 0 of two faces) into a packed record and back — not a region copy |
| γ RM stick half-move | `tilize<…>` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (consumes 32 rows per tile-row) | one row of one tile: a 16-element L1 move vs a tilize of 31 garbage rows |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| square | Mul (DEST-accumulate) | `cb_x_tiles`: All (pad columns / pad tiles zeroed) | `cb_x_tiles` (same tile): All | None |
| collapse | reduce SUM REDUCE_ROW | `cb_sumsq_partial`: All | `cb_reduce_scaler`: Row0 | — (output Col0) |
| chip_sum | reduce SUM AccumulateViaAdd Skip | `cb_gather`: faces 0/2 (Col0 meaningful) | `cb_reduce_scaler` | — (lane-wise) |
| cross-device fold | SFPU add | `cb_stats` page q: Col0 (other lanes zero) | `cb_stats` page q′: Col0 | — (lane-wise) |
| scale_by_gamma | Mul | `cb_x_tiles`: All | `cb_gamma_tiles`: Row0 | Row |
| scale_by_inv_rms | Mul | `cb_x_gamma` / `cb_x_tiles`: All | `cb_inv_rms`: Col0 (faces 0/2 written) | Col |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Devices disagree on the rows → waves → records mapping | the record offset of a row must be the same on every device, or a device sums another row's partial | `fused_split()` is a function of `(T, Wt, N)` only and `N` is computed from the port-core **union**, identical on every chip; sharded splits come from the one shard spec |
| Wrong divisor | the mean divides by `W = G·W_local`, never `W_local`, never padded width | `inv_W_bits = fp32(1 / (G · W_local))`; the record never carries a divided value (raw Σx²) |
| Device-to-device rounding differences | each device finalizes its own copy | records are exact fp32 bits; fold order is group position `0..G−1` on every device; own record goes through the same compact path |
| fp32 lost in `copy_tile` | an Float32 page unpacked through SrcA is truncated to tf32 | set `unpack_to_dest_mode[cb_stats] = UnpackToDestFp32` on the op's copy of the compute config (`ComputeConfigDescriptor.unpack_to_dest_mode`, `program_descriptors.cpp:679`) under fp32 DEST |
| Wave w+1 publish inflating wave w's count | `own_stats` is one counter per port | the counted gate (port.cpp `n_counted` / `SEM_OWN_COUNTED`): a wave-(w+1) owner may write its records early but increments only after both of its ports counted wave w |
| Back-to-back calls overwrite unread landing | traced calls run with no host sync | port ready fence each call; counters re-armed to 0 at the end; owners finish reading before their program ends |
| Probe program = a second dispatch | `fabric_all_gather` places ports with a probe program | host-only placement from `setup_fabric_connection`'s eth-channel word + SoC descriptor; fallback (unknown map): first free cores of the grid row nearest the Ethernet row |
| generic_op program-cache aliasing | the cache keys only on the outer RT-grid size | `custom_program_hash` per chip program (Work Distribution) |
| Ragged owners / groups | uneven pushes break CB ring alignment (base Refinement 6 bug) | nominal `q` per owner per wave, zero records for absent rows, `cb_rms_root` popped one page at a time at a ring end (base fix) |
| Writer-side NoC atomics on BRISC trip `--dev` assert | base Perf 1 | arrival increments of the writer-side all-gather handed to the reader (NCRISC), as in the base |
| Reader exits with semaphore increments in flight | base Perf 1 latent race | unconditional `noc_async_atomic_barrier()` at reader kernel end |
| Ports colliding with shards | auto shard grids start at (0,0) row-major | placement `allowed = grid − shard cores`; refuse below `2·L` free cores |
| Bfp8_b W-pad zeroing | new mechanism (base excluded it) | zero mantissa bytes only; tests include bf8b `w_non_aligned` with poisoned padding; on failure re-add the EXCLUSION with the measurement |
| Line ends | Linear: chip 0 has no FWD peer, chip G−1 no BWD peer | receive-only ports (no connection), ready sent through the opposite connection (`_needs_ready`), schedule from `_schedule(p, G, ring=False)` |
| Measuring overlap | the prompt judges the exposed transport, not the transport | measurement-only `DRMS_ABLATE=exchange` (ports skip fence/sends, owners treat every landed wait as satisfied and fold the own record only) and base-style device zones (`prt_*` from port.cpp, `rd_*`/`cmp_*`/`wr_*` from the base) — never on by default |

### Structural impossibilities

None missing from `feature_spec.py` INVALID.

## Implementation deviations (Phase 0, implementer)

Recorded against the binding / advisory items above; each has a reason, and where it is a perf lever the knob stays live.

| Design item | As built | Why |
|-------------|----------|-----|
| Port placement nearest each link's Ethernet core (`_worker_nearest_noc1`, eth-channel word) | fabric_ring_exchange's placement: `2·L` free cores nearest the middle of logical row 0 (next to the Ethernet row), identical on every chip; shard cores excluded | measured transport of the example uses it; identical coordinates on every chip make the "same landing address at the same port coords" contract trivial. Nearest-eth placement is a TRANSPORT refinement |
| Ports for every `L_ports` link | only `L = min(num_links, publishers per wave)` links get ports; `2·num_links` cores are still reserved so `N` (and the split) does not depend on the split | a link with no publisher would only run the fence |
| Port send order strict `(wave, item)` + counted gate (`SEM_OWN_COUNTED`) | per-wave `own_stats[w]` semaphores (no gate) and each send item advances its own wave counter | OVERLAP: the gate cost a port round trip per publish, strict order one hop latency per wave in series (measured: 2048×1280 last-launched chip 42.4 → 33.5 µs, exchange-stubbed 32.4) |
| Reader issues every wave's reads up front | wave-ordered reads with transaction ids, `READ_LOOKAHEAD_WAVES = 1` waves in flight ahead (knob) | all-up-front reads were served out of wave order across 104 cores, so some groups' wave 0 landed last and gated every port's first send |
| Reader Phase B in wave order (collect → all-gather → wait inv per wave) | non-blocking poll over collect / all-gather / announce / inv-push; compute folds every owned wave before scaling | an owner's collect of wave w+1 waited behind wave w's all-gather (~4 µs per wave in series) |
| `SEM_INV` / `SEM_GATHER` single cumulative counters | announces (`SEM_INV` increments) of wave w gated on the owner's own count of waves < w; `SEM_GATHER` one bit field per owned wave | cumulative counts are only correct when increments arrive in wave order (bug found by the golden ring-mock suite: ragged last row-group read stale inv rows) |
| inv all-gather on the writer RISC-V when γ is resident | on the reader (NCRISC) for every case | keeps every NoC atomic on NCRISC (base Perf 1 BRISC-atomic `--dev` assert); Phase B on the reader is a light poll loop |
| `cb_gather` one wave quantum | one region per owned wave (`max_owned · G_w · q`) | remote contributors may run a wave ahead of the owner's compute (WAR) |
| `x_depth = 2` blocks when the whole block does not fit | not built; refused with `NotImplementedError` | affects 128×8192 HEIGHT_SHARDED, 4096×1024 WIDTH_SHARDED and prefill ≥ 8192 rows × 3584 (golden / translated); the per-wave γ fallback (lamp L4) is built and covers WIDTH_SHARDED 2048×2048 with γ |
| SHARD-IN-INTERLEAVED-OUT deferred | built (compute streams per tile-row, writer stores with the shard's chunk widths) | cheap once output placement is its own flag; a sharded output must still carry the input's shard spec |
| srcB-hold γ / inv passes (base Perf 2) | kernel-lib chains everywhere | lamp L11; per-core blocks here are 3–5 tile-rows |
| γ multicast | per-core γ reads (as designed: G-MC deferred) | — |

## Refinement 1 — L1 fallback ladder, passes, resharding (implementer)

**Ladder** (`blocking_options` / `first_fit`, predicate-guarded first fit; option 0 is the Phase 0 plan, so every shape
that fit keeps it bit-for-bit — `plan_summary` of the perf cases is unchanged):
1. per-wave `x·γ` (`GAMMA_WAVE`, Phase 0's fallback);
2. per-chunk `x·γ` (`GAMMA_CHUNK`: `cb_x_gamma = q_w·C`, the γ pass fused per chunk ahead of the inv pass), γ resident,
   then γ streamed (`gamma_stream`: `cb_gamma_tiles = STREAM_DEPTH·C`, the writer pushes one chunk per (wave with rows,
   chunk), one item ahead of its stores — base `GAMMA_RESIDENT = False`);
3. chunk-cap halving (`cap` 32 → 1, base `_search_blocking`);
4. **passes** (multi-block rows): the core's rows run in `ceil(NW / PW)` pass blocks of `PW` waves; `cb_x_tiles`
   (interleaved: `x_depth·R_pass·WS`, `x_depth` 2 then 1), `cb_inv_rms` (`PW·q_w`), `cb_gather` (`regions` =
   owned waves per pass) and `cb_x_gamma` (block mode) are sized for one pass. Waves stay numbered across passes, so the
   exchange (landing slots, own_stats, landed notifies, records) is untouched. Reuse safety: before a pass writes into
   other members' gather regions / inv slots, a row-group barrier (`SEM_PASS`: each reader waits until its compute has
   popped every inv page of the previous pass, clears its `SEM_GATHER` fields, then increments every member's
   `SEM_PASS`); gather regions are addressed by owned-wave ordinal mod `regions`, which follows the owner's CB ring;
5. lean: the whole ladder again with `cb_partial_col` single-buffered and one tile-row per square→collapse round
   (`SQ_ROWS`, raw Σx² CBs of one tile-row);
6. the pass rungs again at up to `WAVES_MAX_PASSES = 12` smaller waves (`rewave`): worker semaphores (GATHER, INV, PASS,
   LANDED) live on worker cores only and the ports' per-wave own_stats ids start at 0 on the port cores (ids are per
   core), so the wave count is bounded by `NUM_SEMAPHORES` minus the fabric connection's appended ids;
7. scratch-backed CBs (last resort, only where the op would otherwise refuse): when the caller's tensors sit at the
   bottom of L1 the static CB region is tiny; every CB is then carved out of one per-worker scratch L1 tensor placed in
   the largest free hole (`cb_descriptor_from_sharded_tensor` with offsets, dynamic CB addresses).

**Budget** = the gap below the lowest allocated L1 block (CBs grow up from the bank base), not the largest free block;
re-measured after the plan's exchange scratch exists (re-plan if it grew); cached exchange resources are evicted and
the budget re-measured once before refusing.

**Exchange resources** are cached per (mesh, port cores, G) and grown / shrunk on demand (Phase 0 cached one landing +
semaphore set per plan, which piled up in L1 over a long run), and are released explicitly when dropped (a reference
held by a failed call's traceback otherwise pins them low in L1).

**Resharding**: the output CB is shard-backed only when the output carries the input's own shard placement; any other
output (interleaved, a different shard spec, an interleaved input with a sharded output) streams through `cb_output`
to the writer's output `TensorAccessor` (non-local writes to the output's shard cores). One dispatch.

**Not built**: the 131,072-row prefill (translated `test_ds_distributed_norm`): the landing scratch holds every row's
records for every wave (~650 KB per bank at 12 waves) and one pass block of x needs ~800 KB — Refinement 1a.

## Refinement 1a — x re-read rung (implementer)

Long prefill (131072 × 3584 per device) leaves no ladder rung that fits: the landing scratch holds every row's records
(≈ `T/L · G · 128 B` ≈ 540 KB per bank, independent of the wave count), and resident x scales with the wave (`T·Wt/N`
tiles per core / 12 waves at most). The new last rung (`reread_options`, interleaved input only, tried after every
other rung of every wave count, so all earlier plans keep their rank and plan) makes the x footprint independent of the
wave: Phase A — the reader streams x one tile-row at a time into `cb_x_tiles` (`x_depth·SQ·WS`, pads zeroed per row),
compute squares + collapses each round and pops it; Phase B — the writer re-reads x one (wave, chunk, tile-row) item
ahead of the matching store into `cb_x_reread` (`STREAM_DEPTH·C`), compute does x·γ (per row, `cb_x_gamma` = `C`) then
× inv_rms → `cb_output`. x is read twice from DRAM on this rung only. The exchange, waves, passes and landing layout are
unchanged. Deadlock note: with x streamed, the reader pushes wave w+1 before servicing wave w, so `cb_partial_col`
(and `cb_chip_partial`) must stay two waves deep on this rung. Option (b) of the queue (a credit-managed landing ring)
was not needed and was not built.

## Refinement 2 — LOCAL perf on the fused grid (implementer)

Measured breakdown first (640×1792, 1×4 ring, traced slowest chip, exchange stubbed; `DRMS_ABLATE` payload ablations
keep every CB / semaphore handshake): stub 19.9 µs; reads + writes + γ stubbed together 11.6 µs (the on-chip chain:
square → combine → publish → collect → fold → inv all-gather → scale — larger than the 9.0 µs DRAM roofline on its
own); reads alone −2.5, writes alone −3.2, γ −0.25 µs. Zones: the x reads landed 3–8 µs (top grid rows last), the first
output store came at 10.4 µs and the stores ran until ~20 µs; in the real op the stores ran 17→25.5 µs (≈ 286 GB/s,
east columns last), after the exchange.

What changed (all knobs single-source in the descriptor, each with a measurement-only env override):
- **reader prologue ahead of the x reads** (unconditional): on Blackhole `write_zeros_l1_barrier()` is a full
  `noc_async_read_barrier`, so the reduce-scaler fill behind the reads waited ~3 µs for every read in flight before the
  next wave's reads could issue.
- **`READ_LOOKAHEAD_WAVES` 1 → 0**: each wave's reads go out once the previous wave landed, so DRAM serves the grid wave
  by wave and wave 0 can publish early (lamp L2-adjacent; 1: op +2.5 µs, 2: +2.0 µs at 640×1792).
- **`COMBINE_LAG` (new knob) 1 → 0**: the reader contributes / gathers / publishes wave w right after wave w landed
  (Phase 0 waited for wave w+1's reads), and compute's chip sum follows its square directly. No new deadlock edge: a
  member's contribution depends only on its own read + square, the owner's gather only on members' contributions,
  publish only on the owner's own chip sum.
- **NoC swap, `NOC_SWAP_COLUMNS = 3`** (base Perf 2 `dram_noc_balance`, lamp L10, re-measured here): the 3 westmost
  logical worker columns (next to DRAM column 0) run the reader on NoC1 and the writer on NoC0 (second reader / writer
  descriptor pair, same sources, `_split_noc_swapped`); interleaved inputs only. 1 / 2 / 3 / 4 / 5 columns: stub 19.1 /
  18.3 / 17.7 / 17.8 / 18.7 µs.

Measured nulls (kept out of the tree, with the measurement that bounds them):
- **γ multicast (G-MC)**: the γ ablation is flat at the new defaults (17.75 vs 17.80 µs): the γ prefixes (≤ 213 KB) land
  in the first ~1.2 µs, before x saturates DRAM. A multicast cannot beat removing the reads entirely; the G-MC deferral
  stands, now with a measured reason.
- **srcB-hold passes (lamp L11)**: every eltwise chain stubbed (`DRMS_ABLATE=chains`) moves the stub 17.8 → 16.2 µs but
  the op is flat (21.5 → 21.7; 2048×1280 33.5 → 33.1): the op waits on the exchange, not on compute. The base's own
  carve-outs also select the chain here (3 tile-rows < `HOLD_GAMMA_MIN_ROWS` = 4 under fp32 DEST; inv hold measured no
  win on the per-row-pushed interleaved path).
- **idle cores / wider slice (lamps L1 / L12)**: the 8 idle workers cannot lower `rows_max = 3` at `G_w = 14`; the
  nearest split that uses them, S = 6 (`G_w` 10 × `G_h` 10, 2 rows, 2 waves, 100 cores), improves the stub (17.2 µs)
  but loses on the op (26.1 / 24.8 vs 23.2 / 23.9 µs, 16 traced calls): one wave fewer to hide the exchange. S = 4 kept.
- **per-wave γ** (`GAMMA_WAVE` forced) and **1 / 2 waves**: flat on the stub before the other levers (the stub was then
  "all reads → chain → all writes" whatever the schedule).

What binds now (640×1792): op 21.4 µs, stub 17.8 µs, all payloads stubbed 12.6 µs. The op is bound by the exchange
chain (Refinement 3): collect (1.35 µs per owned wave), fold (0.7), inv all-gather (1.0) after the records land; the
local DRAM traffic now adds ~5 µs on top of the 12.6 µs chain floor instead of ~8.
The x re-read rung's deadlock note (Refinement 1a) assumed the lagged order; with `COMBINE_LAG = 0` the reader no longer
pushes wave w+1 before servicing wave w, and the rung keeps `COMBINE_DEPTH` anyway (harmless).
