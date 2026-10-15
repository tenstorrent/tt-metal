# Operation Design: matmul_reduce_scatter

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + CCL reduce-scatter over Fabric) |
| Goal | One device program per call that computes every device's partial `A[g] @ W[g]` block by block in transport order. The cross-device sum of each block is reduced along the shorter way to its owner while later blocks are still being computed. Partials land directly in the L1 of the matmul core that owns the same output tiles on the next chip. That core does the add and hands the sum to a port core, which only forwards. No partial and no relay scratch ever touches DRAM. |
| Math | `P = Σ_g A[g] @ W[g]` over the G devices of the group. `scatter_dim=-2`: `out[p] = P[..., p·M/G:(p+1)·M/G, :]`. `scatter_dim=-1`: `out[p] = P[..., :, p·N/G:(p+1)·N/G]` |
| Mode | Hybrid. The compute half is a 2D dual-multicast matmul built on `matmul_block`, as in `eval/prompts/matmul.txt`. The transport half adapts `fabric_reduce_scatter` / `fabric_all_gather`: schedules, port cores, arrival counters, the ready fence and one-hop routing under every fabric config. |
| References | `ttnn/ttnn/operations/examples/fabric_reduce_scatter/program_descriptor_with_inline_kernels.py` (`_entries` :597, `_PORT_SENDER` :362, ready fence :484-494, re-arm :568-573, `route_one_hop` :376); `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py` (`build_groups` :371, `_needs_ready` :407, `allowed_cores` :704, `plan` :717); `.claude/references/blackhole-fabric.md`; `.claude/references/blackhole-ccl-kernels.md`; `ttnn/ttnn/operations/examples/mcast_topology/mcast_topology.py:364-418`; `eval/golden_tests/matmul_reduce_scatter/{feature_spec.py,roofline.py,helpers.py}` |

### Catalog entries that set knobs (`ttnn/ttnn/operations/examples/master.md`)

| Entry | Knob it informs | What it says here |
|---|---|---|
| `mcast_topology` | core-assignment of the two operands | A 2D split needs **two 1D mcast families**: `Mcast1D` PerRow plus PerColumn, at disjoint semaphore bases. This replaces every core re-reading its own slices (1.91×, 8× fewer DRAM reads). |
| `shared_input_reuse` | the cross-block operand | Read a shared operand once and mcast it; never re-read it per consumer. |
| `matmul_output_subblock` | subblock area | Subblock area tracks DEST capacity: 8 tiles gives 1.46×, 4 gives 1.40×. This is why the matmul cores run fp32 DEST in **full-sync** (8 fp32 tiles) instead of half-sync (4). |
| `eltwise_l1_vs_dest_accumulate` | K accumulation | Packer L1-accumulate avoids the spill/reload round trip (5×). This is the `packer_l1_acc` + `Interm` path. |
| `compute_block_size` | per-entry helper call | Each helper is called once per whole per-core block (one entry), never per tile-row (1.65×). |
| `double_buffer` | buffer depths | Depth 2 on every streaming window. Issue a batch of reads, then one barrier. |
| `fabric_all_gather`, `fabric_reduce_scatter` | transport schedule, batch `group`, increments | Ring beats line. Balanced halves on an even ring. A batch of 4 chunks with a 2×-batch CB. Fused increments stall the receiving router, so put them on few packets. One ordered sender per counter. The ready fence between calls. Chunks of `max_payload / tile_bytes` tiles. |
| `tensix_all_reduce_compute` | fp32 multi-contributor sum | Seed DEST once, then accumulate in fp32 DEST, never through a bf16 round trip. |
| `noc_placement`, `split_reader` | lamps only | DRAM-reader line orientation; splitting reader issue across RISCs only if the reader is shown to be issue-bound. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor (mesh) | yes | per device `(..., M, K)`, rank 2-4, all leading dims 1, TILE, DRAM interleaved | — | addresses RT |
| `weight` | ttnn.Tensor (mesh) | yes | per device `(K, N)`, TILE, DRAM interleaved | — | address RT |
| `cluster_axis` | int | yes (kw) | 0 or 1; `G = mesh_shape[cluster_axis] ≥ 2` | — | host (schedule) |
| `scatter_dim` | int | no (kw) | -2 / -1. A non-negative index is canonicalized to its negative equivalent (`d - rank`) **before** the support check. | -2 | host (schedule) |
| `topology` | `Topology` (`from ttnn.operations.ccl import Topology`) | no (kw) | Linear, Ring | Linear | host (schedule) |
| `num_links` | int or None | no (kw) | explicit value: `1 ≤ L ≤ min over the group's hops of len(get_forwarding_link_indices)`, else ValueError. None means all usable links. | None | host (ports) |
| `compute_kernel_config` | ComputeConfigDescriptor | no (kw) | any fidelity; `fp32_dest_acc_en` is a floor | `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=True | CT (fidelity) |
| `memory_config` | MemoryConfig or None | no (kw) | interleaved DRAM or L1. A sharded memory_config raises NotImplementedError: TARGET has no placement axis. | DRAM interleaved | output alloc |

### Registry contract (op file)

| Item | Value |
|---|---|
| `INPUT_TAGGERS` | exactly `{"alignment": tag_alignment}`. `inputs[0]` is A's shape, `inputs[1]` is W's shape. Returns `"tile_aligned"` iff M, K, N are all `% 32 == 0`, else `"non_aligned"`. |
| `SUPPORTED` (Phase 0) | `dtype: [bfloat16]`, `weight_dtype: [bfloat16, bfloat8_b]`, `layout: [TILE]`, `alignment: ["tile_aligned"]`, `cluster_axis: [0, 1]`, `scatter_dim: [-1, -2]`, `topology: [Linear, Ring]`, `num_links: [1, 2]` |
| `EXCLUSIONS` | `[]` |
| `validate()` | First line of the entry point. It builds axes from the call (dtype = activation dtype, weight_dtype, layout, alignment via the tagger, cluster_axis, canonical scatter_dim, topology, num_links). It raises `UnsupportedAxisValue` for an axis value outside SUPPORTED, then `ExcludedCell`. `num_links=None` is not a registry value: it is resolved after validate to all usable links, and the code is generic in L. |
| Caller errors → `ValueError` | rank outside 2-4 or a leading dim ≠ 1; K mismatch; inputs not DRAM interleaved / not TILE; `G < 2`; scattered extent not `% (32·G)`; explicit `num_links` ≤ 0 or above the discovered links on any hop; `topology=Ring` with `G ≥ 3` when the active fabric config does not wrap `cluster_axis` (wrapping configs: TORUS_X → axis 1, TORUS_Y → axis 0, TORUS_XY → both, 1D_RING → its routed line), or when `get_forwarding_link_indices(last, first)` is empty |
| Structural impossibilities | none (feature_spec `INVALID = []` is correct) |
| Default compute config | exported `default_compute_kernel_config()`; `None` resolves through it |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | A: `(..., M, K)`, rank 2-4, all leading dims 1. W: `(K, N)`. Identical on every device. |
| Dtype | A: bfloat16 (Phase 0); TARGET adds bfloat8_b. W: bfloat16, bfloat8_b (Phase 0); TARGET adds bfloat4_b. |
| Layout | TILE |
| Memory | DRAM interleaved (both) |

### Output

| Property | Value |
|----------|-------|
| Shape | scatter -2: `(..., M/G, N)`; scatter -1: `(..., M, N/G)` (A's leading dims kept) |
| Dtype | bfloat16 always |
| Layout | TILE |
| Memory | `memory_config`, default DRAM interleaved; allocated on every device of the mesh |

## Blocking Model

Notation (tile counts, all `ceil`):

| Symbol | Meaning |
|---|---|
| `Mt = ceil(M/32)`, `Kt = ceil(K/32)`, `Nt = ceil(N/32)` | tensor extents in tiles |
| block `j` | the region of `P` owned by group position `j`. scatter -2: rows `[j·Mt/G, (j+1)·Mt/G)` × all columns. scatter -1: all rows × columns `[j·Nt/G, (j+1)·Nt/G)`. |
| `block_m_tiles`, `block_n_tiles` | per-block extents: scatter -2 gives `(Mt/G, Nt)`, scatter -1 gives `(Mt, Nt/G)` |
| `core_m`, `core_n` | this core's share of one block, i.e. the per-core block: `ceil(block_m_tiles/grid_m)`, `ceil(block_n_tiles/grid_n)` |
| `P_nom = core_m·core_n` | nominal tiles of one per-core sub-block, which is the unit every handoff moves |
| entry `e` | the e-th block in this chip's transport order (Schedule tables below) |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `g` (device in group, G) | **dependent**: the result sums over devices | `partials_per_device` | 1. Each device holds exactly one K-slice by contract, so there is nothing to choose. | the op contract | one device per position. The combine is the fabric reduce-scatter (built). | — |
| `j` (scatter block / entry, G) | **independent** for the matmul, but the **transport-order axis**: each block has a different owner and a different chain | `blocks_per_entry` | 1. An entry is one whole block. This is the coarsest unit that still lets block j+1's matmul hide block j's transport. | host schedule builder (`entries_per_block = 1`) | **not split across cores**: every matmul core walks all G entries in time. Splitting j over disjoint core sets makes all blocks finish together and kills the overlap the op exists for (regime `block-parallel`, rejected). | knob-turn: `entries_per_block = 2` splits each block into two waves along `n` (overlap lamp) |
| `m` (block rows, `block_m_tiles`) | **independent** (output rows), and makes W **reuse-shared**: W does not vary with m | `core_m` (= block_extent[m]) | `ceil(block_m_tiles / grid_m)`: the whole per-core assignment | `choose_grid()` (host) → CT arg | split over the grid's m-lines. W is mcast along each n-line (PerRow/PerColumn per orientation). | knob-turn (grid choice) |
| `n` (block cols, `block_n_tiles`) | **independent** (output cols), and makes A **reuse-shared**: A does not vary with n | `core_n` | `ceil(block_n_tiles / grid_n)`: the whole per-core assignment | `choose_grid()` → CT arg | split over the grid's n-lines. A is mcast along each m-line. | knob-turn |
| `k` (Kt) | **dependent**: the contraction | `kb` (K-block, `in0_block_k`) | the largest divisor of Kt that fits L1 (`choose_k_block()`). Focus case: 32 or 64, depending on the device's allocator base (`l1_ledger.md` worked example). | `choose_k_block()` → CT arg | **not split across cores**. Within a core it is a sequential packer-L1 accumulation into the fp32 hold slot. The cross-core K split is regime `k-split` (deferred). | scheme-change (k-split) |
| leading dims | independent, all extent 1 by contract | — | folded into m | — | — | — |
| transport chunk (flattened m·n of one sub-block, **stage data**) | independent: each tile of a finished sub-block travels on its own | `chunk_tiles` | `min(floor(max_payload / 2048), P_nom)` = 7 at 14 KiB | host from `ttnn.get_tt_fabric_max_payload_size_bytes()` → CT arg | each sub-block's chunks go to exactly one port (link `k mod L`, direction from its role). The sub-block's add runs on the core that owns it, so **no stage computes on a designated core**. Ports are a communication role only: gather and send. | — (mechanism cap) |

Re-running the table on the intermediate data (the per-entry partial sums) gives the same independent `m`, `n`. Their combine is assigned to the **same core that produced the partial**, so all `grid_m·grid_n` cores share the relay and final adds. No core does the adds while the others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_act_k_window` (stream regime) | `depth_in0` | 2 | mcast of K-block t+1 overlaps the matmul of K-block t |
| `cb_weight_k_window` (stream regime) | `depth_in1` | 2 | same, for W |
| `cb_partial_hold` | `depth_hold` | 2 (= lag 1 + 1) | entry e+1's matmul can pack while entry e (a relay or fin) still waits for its arrival |
| `cb_finished` | `depth_handoff` | 2 | compute finishes entry e+1 while a port is still gathering entry e |
| `cb_port_chunks` | `port_group` (chunks per batch). Capacity is 2·group. | 4 | a reader batch of reads overlaps a sender batch of packets. Measured best or within 10% at every size (fabric_* reports). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric max payload (14 KiB + 64 B header) | `chunk_tiles` | `min(floor(get_tt_fabric_max_payload_size_bytes() / 2048), P_nom)`. A chunk never spans two cores' sub-blocks. | The packet exceeds the payload: router assert or hang |
| DEST (fp32 + `dst_full_sync_en`) | matmul subblock `h·w` | `h | core_m`, `w | core_n`, `h·w ≤ 8` (the largest such area is picked) | ASSERT at `matmul_block_helpers.inl:158`, or DEST overflow |
| `caller_owns_pack_target` contract | matmul output layout | requires TileRowMajor + `packer_l1_acc` + `Interm`: static_assert at `matmul_block_helpers.inl:139` | compile error. With any other combination: orphaned `wait_front` and deadlock |
| Packet header pool: 12 per RISC | port sender header ring | `port_group ≤ 8` | `PACKET HEADER POOL EXHAUSTION` hang |
| Semaphore ids: 16 per core | per-core semaphores | matmul core: 4 (two Mcast1D families) + 2 (`consumed_fwd/bwd`). Port: fabric connection (2), declared after the op's own. | program build failure "Semaphore id N already in use" |
| CB ring-wrap invariant | `kb` | `kb | Kt`. The resident regime re-credits exactly `Kt/kb` pushes per entry, which must equal the capacity. | the pointer misses `fifo_limit`, the ring desyncs and reads wrong tiles |
| Landing-zone slots | arrivals per core | `slots_fwd + slots_bwd ≤ G + 1` (line interior `G+1`, balanced ring ≤ `G`, unbalanced ring `G-1`). Each direction is sized from the host schedule as the maximum over the two core halves. | the fabric writes past the zone into foreign L1 (silent corruption) |
| Transport strip length | ports | `2·L ≤` strip length (10 or 11) | no core for a port: ValueError at plan time |
| Halves (balanced ring) | matmul cores | `grid_m·grid_n ≥ 2` | an empty half: a direction carries nothing and the counters never complete |
| L1 budget | `kb`, residency | the regime predicates below | CB / L1-buffer clash at program build |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (min: A, W cross DRAM once; output written once; each partial crosses each fabric hop of its chain once; zero DRAM for partials) | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **resident-xblock** (the cross-block operand is kept in L1 for all G entries: A for scatter -1, W for scatter -2) | built | `F(kb, resident=True) ≤ L1_budget` for some `kb | Kt` with `kb ≥ min(Kt, KB_MIN_RESIDENT=8)` (`choose_k_block`) | per core per entry `core_m × core_n × Kt`, with K-blocks of `kb` | **the minimum.** A and W each cross DRAM once. After entry 0, the resident operand is re-credited locally with no NoC traffic. Output written once. Partials go L1→fabric→L1. | Bigger `kb`: fewer mcast rounds (`Kt/kb` per entry, each one sender handshake plus a valid flag) and fewer L1-acc passes per entry. Bigger per-core block (`core_m·core_n`): the per-entry `matmul_block` init + reconfig, the per-entry finish-op init, 2 handoff signals per core-entry and 1 fused increment per core-entry are spread over more tiles. Intended frequency: inits once per entry, mcast once per K-block, signals once per (core, entry). |
| **stream-xblock** (the cross-block operand is re-read from DRAM and re-mcast for every entry) | built | `¬` the predicate above, and `F(kb, resident=False) ≤ L1_budget` for some `kb | Kt` | same | **Above the minimum by (G-1)× the cross-block operand.** scatter -1: A crosses DRAM G times. scatter -2: W crosses G times. This is the stepping stone the resident regime layers on: same loop nest, only the sender re-fetches and the CB capacity differs. | same terms, plus: bigger `kb` amortizes the per-K-block DRAM read barrier of the re-fetch |
| **ring-balanced** schedule | built | `topology == Ring and G even and G ≥ 4` | entries = G blocks, opposite block split in core halves | busiest link direction carries `(G-1)/2` blocks, which is the ring minimum | — (schedule, not extent) |
| **ring-unbalanced** schedule | built | `topology == Ring and G odd and G ≥ 3` | entries = G blocks | busiest direction carries `ceil((G-1)/2)` blocks; odd G has no opposite block to split | — |
| **line** schedule | built | `topology == Linear`, or `G == 2` | entries = G blocks | busiest link carries `G-1` blocks, the line minimum | — |
| **L1-landing** (arrivals land in the receiving matmul core's L1 zone) | built | `F(...) ≤ L1_budget` above, which includes the zones | per-core zone slot = `P_nom` tiles | partials: **0 DRAM bytes** (the roofline's relay-scratch term `2(G-1)B` is removed) | — |
| **DRAM-landing** (arrivals land in a DRAM scratch page, as in `fabric_reduce_scatter`) | deferred: no TARGET/INPUTS shape needs it. The largest zone is `(G+1)·P_nom·2 KiB` ≤ 256 KiB for every INPUTS shape at G ≤ 8 on a ≥ 80-core grid. The host raises NotImplementedError naming this regime if `F(kb=1)` ever exceeds the budget. It stays reachable: the sender's destination is a host-computed `(core, offset)` pair that becomes a `(DRAM page)` pair, and the reader's zone credit becomes a DRAM read. | `F(1, False) > L1_budget` | same | +`2(G-1)·B` DRAM bytes per chip | — |
| **k-split** (split Kt across 2+ core groups inside a chip, then an intra-chip partial combine) | deferred: it only helps when `grid_m·grid_n` is far below the core count (block extent < ~20 tiles: GLM indexer `wk`/`weights_proj`, N ≤ 128). Those calls are bound by the program floor plus the transport fixed cost (~15 µs) against < 2 µs of matmul, so the occupancy gain cannot show. Reachable: an extra combine phase before `finish`, in the same core-local schedule. | `block_m_tiles·block_n_tiles < cores/2` | `core_m × core_n × Kt/s` | +cross-core partial traffic `(s-1)·P` per entry | — |
| **block-parallel** (G disjoint sub-grids compute the G blocks concurrently) | rejected: superseded by the built sequential-entry walk. Every block finishes at the end, so the transport cannot overlap the matmul (violates the overlap rule). | — | — | same DRAM | — |
| **dedicated final/adder cores** (the `fabric_reduce_scatter` roles: port cores add relays, F final cores add own + arrivals and write the output) | rejected: superseded by in-grid combine. The own block is computed last, so 2-4 final cores reading `3·B` and writing `B` serialize a ~35-45 µs drain after the matmul. Port relays also read two streams per chunk through one core. The in-grid design spreads both adds over all matmul cores (drain ≈ one sub-block finish + write). | — | — | +`2(G-1)·B` DRAM (relay scratch) + final-core reads | — |
| **matmul-to-DRAM partial** (write each block's partial to DRAM, then reduce-scatter it from there) | rejected (dead end): it adds `2·M·N·2` bytes of DRAM per chip (36 µs on the focus case, more than the compute roof). Nothing better is built on top of it. | — | — | +2·M·N·2 B DRAM | — |
| **per-core DRAM reads, no mcast** | rejected: superseded by the built 2D mcast. A is read `grid_n`× and W `grid_m`× (≈ 8-10× DRAM traffic, ~300 µs of DRAM on the focus case). | — | — | +(grid_n-1)·A + (grid_m-1)·W | — |
| **port-L1 landing with credits** | rejected: superseded by L1-landing in matmul cores. A port-side ring buffer needs a credit round trip per slot. Landing on the consumer core needs no credits because every slot is written once per call. | — | — | — | — |
| **unfused** (matmul op, then reduce-scatter op) | rejected: violates the one-dispatch and overlap rules | — | — | — | — |

**Regime selection (host, single source `choose_regime()`)**, evaluated per call from shapes, dtypes, `G`, `L` and the device:
1. `(strip, orientation, grid_m, grid_n, core_m, core_n, subblock) = choose_grid(...)` (Work Distribution).
2. Pick the transport schedule: Ring with `G ≥ 3` is ring-balanced if G is even, else ring-unbalanced. Otherwise line.
3. `(resident, kb) = choose_k_block(...)`. The first fit wins: resident with the largest `kb | Kt` and `kb ≥ min(Kt, 8)`. Otherwise stream with the largest `kb | Kt` that fits. Otherwise NotImplementedError (DRAM-landing deferred).
4. Test hooks live on the **package** module `ttnn.operations.matmul_reduce_scatter` and are read from it at call time, so `monkeypatch.setattr(package, ...)` takes effect:
   - `_FORCE_XBLOCK_RESIDENCY ∈ {None, True, False}`. When not None it replaces the residency decision; a forced `True` that does not fit raises ValueError.
   - `_LAST_PLAN` is a dict the op overwrites on every call: `{"xblock": "resident"|"stream", "schedule": "line"|"ring_balanced"|"ring_unbalanced", "kb": int, "grid": (grid_m, grid_n), "strip": "row"|"col"}`.

   The acceptance test pins both residency regimes through these hooks, so regime coverage does not depend on the box's grid size.

`F(kb, resident)` is the per-core footprint, defined in `l1_ledger.md`. `L1_budget = L1_bank_size − L1_allocator_base − L1_HEADROOM (64 KiB, named constant)`. The staging tensor is part of `F`, not of the headroom.

### Traffic ranking

Per chip, for the matmul half (the cross-device split is fixed by the contract: every device holds one K-slice). Counts are crossings of the whole tensor.

| Candidate split | DRAM crossings (A / W / out / partials) | Cross-core / fabric traffic | Verdict |
|---|---|---|---|
| 2D (m,n) per block, entries in time, cross-block operand resident, L1 landing | 1 / 1 / 1 / 0 | A mcast once per m-line, W mcast once per entry per n-line. Partials: on-chip NoC reads by ports (`(G-1)` blocks per chip in total), fabric `(G-1)/2·B` (ring) or `(G-1)·B` (line) per busiest link direction. Zone writes land straight in L1. | **cheapest; built** (resident-xblock) |
| same, cross-block operand re-streamed | G / 1 (s-1) or 1 / G (s-2) / 1 / 0 | + `(G-1)` re-mcasts of that operand | built as the fallback when residency does not fit |
| k-split across cores | 1 / 1 / 1 / 0 | + intra-chip partial combine `(s-1)·P` per entry | deferred (see regime) |
| block-parallel sub-grids | 1 / 1 / 1 / 0 | same | rejected: no overlap |
| no-mcast | grid_n / grid_m / 1 / 0 | none | rejected |
| DRAM landing / DRAM partials | 1 / 1 / 1 / +2(G-1)B or +2·M·N·2 | same | deferred / rejected |

Chosen: **2D (m,n) + time-walk over j + resident cross-block operand + L1 landing.** It is also the cheapest-traffic split. The stream regime is selected only by the L1 predicate. Occupancy (core count) is decided separately in `choose_grid`. It never overrides this ranking.

### Schedule tables (host, single source `build_schedule(p, G, topology)`)

For chip p and block `b ≠ p`, with `δf = (b - p) mod G` (ring) or `b - p` (line), `δb = G - δf` (ring) or `p - b` (line). Each (block, half h ∈ {0,1}) gets a role for every core whose `half(k) = h`.

| Topology | (b, h) → direction, role at chip p |
|---|---|
| line | `b > p`: fwd, `start` iff `p == 0` else `relay`. `b < p`: bwd, `start` iff `p == G-1` else `relay`. Halves identical. |
| ring-unbalanced (`n_f = G//2`, `n_b = G-1-n_f`) | `δf ≤ n_f`: fwd, `start` iff `δf == n_f` else `relay`. Otherwise bwd, `start` iff `δb == n_b` else `relay`. Halves identical. |
| ring-balanced (`H = G/2`) | h=0: `δf ≤ H` → fwd, start iff `δf == H`; else bwd, start iff `δb == H-1`. h=1: `δf ≤ H-1` → fwd, start iff `δf == H-1`; else bwd, start iff `δb == H`. Outside these cases, relay. |
| all | `b == p`: `fin`. Arrivals come from fwd (sent by p-1) iff p has a fwd upstream for (p, h), and from bwd likewise. Line ends have one; ring cores have both. |

- **Entry order** (identical rule on every chip): blocks `b ≠ p` sorted by `key(b) = max_h distance-to-owner along that half's direction`, descending. Ties: fwd before bwd, then lower b. The own block p comes last.
- **Deadlock freedom.** A relay of (b, h) at distance δ depends only on the upstream's (b, h) at distance δ+1. Every block with a relay half sends both halves the same way, so the upstream's key is strictly larger. Every chip processes keys in descending order. Ports never wait on arrivals, and zone slots never need credits. So every wait chain ends at a `start`.
- **G = 4 balanced, chip p** (the perf placement): entries `[p+2, p+1, p-1, p]`. Half 0: `start(fwd), relay(fwd), start(bwd), fin`. Half 1: `start(bwd), start(fwd), relay(bwd), fin`.
- **Per-core arrival list per direction d** (zone layout): the blocks this chip's core-half receives from d, in this chip's entry order. These are the relays in direction d plus `fin` when an upstream exists. A block's slot index in that list is its **zone slot**. Its position in the *upstream's* send order to this core-half is its **arrival threshold** (cumulative increments).
- **Port units**: port `(d, l)` of chip p forwards, in entry order, the units `(b, h)` whose role is `start(d)` or `relay(d)`, restricted to cores with `link(k) = l`.

### Block schedule

Matmul cores (all `grid_m·grid_n` cores of a chip, in mcast lockstep):

```cpp
pending = NONE;
for (uint32_t entry_idx = 0; entry_idx < num_entries; ++entry_idx) {      // G entries, transport order
    load_operand_block(entry_idx);        // reader: Kt/kb K-block windows of A and W (mcast, or re-credit of the resident operand)
    matmul_block_entry(entry_idx);        // compute: core_m x core_n x Kt -> one fp32 slot of cb_partial_hold
    if (pending != NONE) {
        land_arrivals_block(pending);     // reader: wait arrival counter(s), credit zone slot(s) of `pending`
        finish_block(pending);            // compute: add2 (relay) or add3 (fin) -> cb_finished
        handoff_or_store_block(pending);  // writer
        pending = NONE;
    }
    if (needs_arrival(entry_idx) && entry_idx + 1 < num_entries) {
        pending = entry_idx;              // stall shadow: the add waits behind the next matmul
    } else {
        land_arrivals_block(entry_idx);   // only the last (fin) entry, or no-op for start
        finish_block(entry_idx);          // copy (start) | add2/add3 (last entry)
        handoff_or_store_block(entry_idx);
    }
}
```

Port cores (per chip, per direction d, per link l):

```cpp
fence_ready();                             // send ready to the peer port, wait own ready (once per call)
for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
    gather_unit(unit_idx);                 // reader: per core, wait mailbox, NoC-read its sub-block from cb_finished, ack consumed
    send_unit(unit_idx);                   // sender: chunks -> downstream core's zone slot; fused inc on each core's last chunk
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_operand_block` | `Kt/kb` windows: `core_m×kb` (A), `kb×core_n` (W) | the resident operand (all `core_m×Kt` or `Kt×core_n`) for all G entries | DRAM reads + mcast: once per K-block for the streamed operand, and once per call for the resident one (entry 0 only, then a local re-credit per K-block). Sender handshake once per K-block per family. |
| `matmul_block_entry` | `core_m × core_n × Kt`, K-blocks of `kb`, subblock `h×w` (largest area ≤ 8) | one `cb_partial_hold` slot (caller-owned reserve, L1-accumulated across K-blocks) | `matmul_block_init` + reconfig: once per entry. Hold reserve/push: once per entry. L1-acc toggle: once per K-block. |
| `land_arrivals_block` | `P_nom` pages per arriving direction | zone slots (written by the fabric, never moved) | one `noc_semaphore_wait_min` + one reserve/push per (entry, direction) |
| `finish_block` | `P_nom` tiles: `copy` (start), `add2` = hold + arrival (relay, or fin at a line end), `add3` = hold + arr_fwd + arr_bwd (fin) | — | one chain init per entry |
| `handoff_or_store_block` | `P_nom` tiles | `cb_finished` slot until the port acks | start/relay: 1 mailbox increment to port `(d, k mod L)`, then a pop on that port's consumed ack. fin: valid tiles written to the output with one write barrier. |
| `fence_ready` | — | — | once per call per port: 1 header-only packet + 1 wait |
| `gather_unit` | the unit's cores × `P_nom` tiles, in `chunk_tiles` chunks, batches of `port_group` | `cb_port_chunks` | 1 mailbox wait + 1 consumed ack per (core, unit). One read barrier per batch. |
| `send_unit` | chunks of ≤ `chunk_tiles` | header ring (`port_group` headers) | 1 packet per chunk. 1 fused write+inc per (core, unit), on that core's last chunk. Flush once per batch. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap: `entries_per_block = 1` | The first send waits for one whole block of matmul (T/G). In link-bound small-K cases (shared-expert down: links 35.5 µs vs compute 17 µs) that wait is on the critical path. | `entries_per_block = 2`: two waves along `n` per block (`core_n/2`). Halves time-to-first-send. Costs one extra set of per-entry inits and mcast rounds. |
| Fused increment on every core's last chunk | `P_nom ≤ 2·chunk_tiles` on the focus case (12 tiles → 2 packets), so every second packet stalls the receiving router (rule: every packet halves the link rate). | Fold the increment into every `port_group`-th core share, with each core waiting on a group counter. Or size `P_nom` to more chunks through the grid choice. |
| fp32 DEST on the matmul cores, even when the caller asked for 16-bit | Required, because the adds run on these cores and must accumulate in fp32. `dst_full_sync_en` restores 8-tile subblocks, but pack no longer overlaps math. | Half-sync fp32 (4-tile subblocks, overlapped pack). The perf pass measures which wins at `Kt ≥ 16`. |
| `choose_grid` strip and orientation | The transport strip (row vs column) changes per-core tiles by ~15% on the focus case (12 vs 14) and moves ports relative to the Ethernet cores. | The other strip; ports placed nearer their Ethernet columns. |
| Grid synchronization: full grid for small blocks | When `block_m_tiles·block_n_tiles` is near the core count, per-core blocks of 1-4 tiles make mcast handshakes and port signals dominate. | `choose_grid` capped at `cores ≤ block_tiles/4`: fewer cores with larger blocks. |
| `kb` and `KB_MIN_RESIDENT` | Residency trades a smaller `kb` (more mcast rounds per entry) for (G-1)× fewer reads of the cross-block operand. | Stream at the largest `kb` vs resident at a smaller `kb`, on the focus and GLM cases. |
| Reader issue load | One NCRISC issues both mcast families plus the zone credits. Core (0,0) is a sender for both families during entry 0. | Move the W family to BRISC (`split_reader`), but only if the reader is shown to be issue-bound. |
| `depth_handoff = 2` | A port gathering ~50 cores per unit may ack late cores after the next entry's finish. | `depth_handoff = 3`. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| A K-block → in0 senders | bf16 tiles | TensorAccessor DRAM reads (NoC0), batch then one barrier | senders = the first core of each m-line |
| A → m-line | bf16 | `Mcast1D` family "a" (`SenderPipe::send` / `ReceiverPipe::receive`) | resident regime (scatter -1): entry 0 only. Later entries re-credit the same pages: reserve + push with no write. The ring wraps exactly because the capacity is `core_m·Kt`. |
| W K-block → in1 senders → n-line | bf16 / bfp8_b | DRAM reads + `Mcast1D` family "b" | resident regime (scatter -2): entry 0 only, then re-credit |
| matmul | DEST fp32 (full sync) → `cb_partial_hold` Float32 | `matmul_block`, packer L1 accumulation into the caller-reserved slot | user fidelity. fp32 DEST is always on (floor exceeded). |
| finish | hold Float32 + arrivals bf16 → `cb_finished` bf16 | eltwise chain: copy / SFPU fp32 add / FPU add + SFPU fp32 add | every cross-device add happens in fp32 DEST |
| handoff → port | bf16 | port reader NoC0 reads from the matmul core's `cb_finished` slot (fixed address from the staging tensor) into `cb_port_chunks`. Writer → port mailbox increment, port → `consumed_d` increment. | intra-chip |
| port → next chip | bf16 packets of ≤ `chunk_tiles` | fabric unicast write to `get_noc_addr(core_k, zone_d_base + slot·P_nom·2048 + chunk_off)` on the downstream chip. A fused atomic inc of that core's `arr_d` global semaphore rides the core's last chunk. Route: `route_one_hop` (2D: by node; 1D: 1 hop). | Same logical core and same staging address on every chip (mesh lockstep allocation). Sender on NoC1. |
| arrival → compute | bf16 | the reader waits `arr_d ≥ threshold`, then reserve + push of the zone slot (data already landed) | no copy |
| fin → output | bf16 | writer: TensorAccessor page writes of the valid tiles of the sub-block (NoC1), one barrier | output page = `local_row·out_n_tiles + local_col` of block p |

Tensix-to-Tensix contracts (all in the built regime):

| Contract | Producer → consumer | Signal | Re-arm |
|---|---|---|---|
| handoff ready | matmul writer `k` → port `(d, k mod L)` | `noc_semaphore_inc(port_mailbox[k])`, +1 per (core, unit) | the port decrements each slot by its final count at call end. Safe because the next call's signals come only after the next launch. |
| handoff consumed | port reader → matmul core `k` | program-local `consumed_d` +1 per (core, unit), after the read barrier | program-local: reset at every launch |
| arrival | upstream port sender → matmul core `k` (remote chip) | global `arr_fwd` / `arr_bwd`, one fused inc per (core, unit) | reader: wait for the total, then `noc_semaphore_inc(…, 0u - total)`, then atomic barrier |
| ready fence | port `(d,l)` of chip p ↔ port `(-d,l)` of chip p+d | global `ready` +1, header-only packet | consumer: wait ≥ 1, then decrement 1 |

**TARGET axes as regime rows.** `dtype` bfloat8_b (activation) and `weight_dtype` bfloat4_b are **knob-turns**: CB page formats and tile sizes in `F`, with no structure change. TARGET has no placement (`memory_layout`) axis, so no physical shard regime is needed. Every other TARGET value is in Phase 0.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one entry's per-core block `core_m × core_n × Kt` (one block, one pipeline pass) |
| Grid | Matmul rectangle = the compute grid minus a **transport strip** (logical row 0, or logical column `gx-1`). Within it, `grid_m × grid_n` cores in orientation `m_on_x` or `m_on_y`, chosen by `choose_grid`. Ports: `2·L` cores in the strip, the cells nearest the Ethernet row (row strip: spread across columns; column strip: rows 0..2L-1). No probe dispatch: the one-dispatch rule forbids the reference's probe program, and probe placement is a lamp. |
| Per-core work | G entries × `core_m·core_n·Kt` tile-MACs, plus the G finish ops of `P_nom` tiles |
| Remainder | Ragged last m-/n-line: the runtime extents `core_m' = block_m_tiles − (grid_m−1)·core_m` and `core_n'`. CB push/pop counts stay **nominal**. DRAM reads skip invalid tiles. `matmul_block` gets `in1_per_core_w = core_n` and `last_in1_subblock_w_valid` for the narrowed last subblock. Handoff and fabric carry the nominal `P_nom` tiles. Output writes cover only valid tiles. All geometry uses `ceil`. |

`choose_grid(gx, gy, block_m_tiles, block_n_tiles, L)` (host, single source):
```
for strip in (row0, col_last) with 2L ≤ strip length:
  rect = (gx, gy-1) or (gx-1, gy)
  for orient in (m_on_x, m_on_y):
    ext_m, ext_n = rect in that orientation
    core_m = ceil(block_m_tiles / min(ext_m, block_m_tiles)); grid_m = ceil(block_m_tiles / core_m)
    core_n = ceil(block_n_tiles / min(ext_n, block_n_tiles)); grid_n = ceil(block_n_tiles / core_n)
    sb = max{h·w : h | core_m, w | core_n, h·w ≤ 8}
    key = (core_m·core_n, -sb, ragged tiles, grid_m·grid_n, strip != row0)
pick the minimum key
```
Focus case on QB2 (11×10 grid, scatter -1, G=4): column strip, `m_on_x`, `grid_m=10, core_m=2`, `grid_n=10, core_n=6`, subblock 2×3. That is 100 matmul cores and 12 tiles per core per entry. The row strip would give 14 tiles.

**Core roles:** `k` = row-major index over the used `(gm, gn)` cores. `half(k) = 0 if k < ceil(n_cores/2) else 1`. `link(k) = k mod L`.

**Regime-pinned tests (required):** resident and stream are both exercised through `_FORCE_XBLOCK_RESIDENCY` on a small shape where both fit, plus the natural selection on the focus shape (resident) and the MiMo shape with bf16 W (stream). Line and ring schedules are pinned by topology.

## Circular Buffers

Matmul cores (`grid_m·grid_n`):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_act_k_window` | 0 | A tile (2048) | resident (scatter -1): `core_m·Kt`; else `depth_in0·core_m·kb` | spans m, k (resident) / streams k (window) | A dtype (Float16_b) | reader | compute | whole call |
| `cb_weight_k_window` | 1 | W tile (2048 / 1088) | resident (scatter -2): `Kt·core_n`; else `depth_in1·kb·core_n` | spans k, n / streams k | W dtype | reader | compute | whole call |
| `cb_partial_hold` | 2 | 4096 | `depth_hold·P_nom` | spans m, n; streams j (two entries live under the lag) | Float32 (DEST is fp32) | compute (matmul, caller-owned reserve/push) | compute (finish) | per entry, lag ≤ 1 |
| `cb_finished` | 3 | 2048 | `depth_handoff·P_nom` | spans m, n; streams j. **Backed by the staging tensor** at offset 0, so ports know the address. | Float16_b (the partial travels in bf16, which the spec allows; fin = output dtype) | compute | writer | per entry until acked or written |
| `cb_arrivals_fwd` | 4 | 2048 | `slots_fwd·P_nom` | one slot per forward-travelling arrival of this core-half. Staging tensor at offset `depth_handoff·P_nom·2048`. | Float16_b | reader (credit only; the fabric writes the bytes) | compute | whole call (each slot written once) |
| `cb_arrivals_bwd` | 5 | 2048 | `slots_bwd·P_nom` | same, backward. Staging offset after fwd. | Float16_b | reader | compute | whole call |

Port cores (`2·L` per chip):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_port_chunks` | 0 | 2048 | `2·port_group·chunk_tiles` | a batch window. A batch ends at a multiple of `port_group` chunks and at the end of each unit (reference §6.2), so a reservation never wraps. | Float16_b | reader (port) | writer (sender) | whole call |

Non-CB L1 state: `port_mailbox` (cached L1 tensor on the strip, one uint32 per matmul core, zero at creation and restored to zero by each port at call end). Global semaphores `arr_fwd`, `arr_bwd`, `ready` over the whole compute grid (cached per mesh; `synchronize_device` **once at creation only**). Staging tensor (cached per mesh, L1 HEIGHT_SHARDED over the matmul rectangle, grow-on-demand; it backs `cb_finished`, `cb_arrivals_fwd`, `cb_arrivals_bwd` through `cb_descriptor_from_sharded_tensor(idx, staging, address_offset, total_size, core_ranges)`). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_operand_block` | `Kt/kb` × (`core_m×kb`, `kb×core_n`) | `mcast_pipe` + `Mcast1D` | DRAM A/W | `cb_act_k_window` (`core_m·kb` per push), `cb_weight_k_window` (`kb·core_n` per push) | resident operand: data stays and pages are re-credited per entry |
| 2 | `matmul_block_entry` | `core_m×core_n×Kt` | `matmul_block` (caller-owned pack target) | `cb_act_k_window`, `cb_weight_k_window`: waited and popped per K-block | `cb_partial_hold`: compute reserves `P_nom`, the helper L1-accumulates, compute pushes `P_nom` | one fp32 slot pushed |
| 3 | `land_arrivals_block` | `P_nom` per arriving direction | raw (semaphore wait + CB credit) | global `arr_d` | `cb_arrivals_d`: reserve/push `P_nom`, no write | slot visible to compute |
| 4a | `finish_block` (start) | `P_nom` | `copy` | `cb_partial_hold` (`P_nom`, popped) | `cb_finished` (`P_nom`) | — |
| 4b | `finish_block` (relay; fin with one upstream) | `P_nom` | `binary_sfpu<AddBinary<>>` | `cb_partial_hold`, `cb_arrivals_d` (`P_nom` each, popped) | `cb_finished` | — |
| 4c | `finish_block` (fin with two upstreams) | `P_nom` | `eltwise_chain(CopyTile, BinaryFpu Add→D1, AddBinary, PackTile)` | `cb_partial_hold`, `cb_arrivals_fwd`, `cb_arrivals_bwd` | `cb_finished` | — |
| 5 | `handoff_or_store_block` | `P_nom` | raw dataflow | `cb_finished`: waited and popped in FIFO order. Signalling never waits on consumption (use `cb_pages_available_at_front`). | port mailbox / output DRAM | popped after the ack (start/relay) or after the write barrier (fin) |
| 6 | `fence_ready` | — | raw fabric (reference :484-494) | — | — | ready counter back to 0 |
| 7 | `gather_unit` | unit cores × `P_nom` | raw dataflow | remote `cb_finished` slot `e mod depth_handoff` | `cb_port_chunks` (batches of `port_group` chunks) | `consumed_d` acked per core |
| 8 | `send_unit` | chunks | raw fabric (reference `_PORT_SENDER`) | `cb_port_chunks` | remote zones | header ring flushed per batch |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | helper (LLK) | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold)` | contract: `matmul_block_helpers.hpp:100-106` | once, first statement of `kernel_main`. Chains use Chain-owned per-element init. | — | — | — |
| `matmul_block_entry` | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:358` (template list :339-357); caller-owns contract `matmul_block_helpers.inl:139`; L1-acc pack `inl:413`, `inl:434` | `<transpose=false, packer_l1_acc=true, LastBlockTarget::Interm (:62), OutputCBLayout::TileRowMajor (:47), InitMode::Short, WaitAndPopPerKBlock, WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock, 0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, caller_owns_pack_target=true>`. Args `(in0, in1, out=cb_partial_hold, interm=cb_partial_hold, MatmulBlockShape::of(core_m/h, core_n/w, h, w, kb, Kt/kb) (:160), {}, {}, in1_per_core_w=core_n, out_row_width=core_n)`; `last_in1_subblock_w_valid` (:158) on ragged n-lines | `cb_act_k_window`, `cb_weight_k_window` | `cb_partial_hold` | `kb` (`in0_block_k`), `core_m`, `core_n`. The subblock `h, w` is derived (largest area ≤ `DEST_AUTO_LIMIT`, `dest_helpers.hpp:103`, = 8 under fp32 + full sync, `:89-100`). |
| `finish_block` start | helper | `compute_kernel_lib::copy<input(cb_partial_hold), output(cb_finished)>(IterationShape::tiles(P_nom))` | `eltwise/api/convenience.hpp:87`; `IterationShape` `chain.hpp:121` | default policies | `cb_partial_hold` | `cb_finished` | `P_nom` |
| `finish_block` relay | helper | `compute_kernel_lib::binary_sfpu<AddBinary<>, input(cb_partial_hold), input(cb_arrivals_d), output(cb_finished)>` | `convenience.hpp:80`; `AddBinary` `eltwise/binary/sfpu/basic.hpp:20`, `sfpu.inl:38` | CopyTile ×2 → SFPU fp32 add → pack | hold, arrivals | `cb_finished` | `P_nom` |
| `finish_block` fin (2 upstreams) | helper | `eltwise_chain(IterationShape::tiles(P_nom), CopyTile<input(cb_partial_hold)>, BinaryFpu<Add, input(cb_arrivals_fwd), input(cb_arrivals_bwd), Dst::D1>, AddBinary<D0,D1,D0>, PackTile<output(cb_finished)>)` | `chain.hpp:533`, `CopyTile :483`, `BinaryFpu :491`, `PackTile :504`; the same chain is exercised in `tests/eltwise/chain/reconfig/scenarios.cpp:74-80` | D1 = fwd+bwd (FPU into fp32 DEST), D0 += D1 (SFPU fp32) | 3 CBs | `cb_finished` | `P_nom` |
| `load_operand_block` mcast | helper | `dataflow_kernel_lib::SenderPipe::send`, `ReceiverPipe::receive`, `McastArgs<CT,RT>` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_pipe.hpp:55,74,126,138`; `mcast/kernel/mcast_args.hpp:239` | PRE_HANDSHAKE on. One family per operand. | DRAM | operand windows | window = `kb` |
| mcast host | helper | `ttnn.Mcast1D(device, matmul_crs, PerRow/PerColumn, Mcast1DFixedSenderConfig(), McastConfig(handshake=True))` + `.attach(desc, "a"/"b", kernels)` | `mcast/host/mcast_host.hpp:303` (Shape `:275`, Config `:78`); Python usage `examples/mcast_topology/mcast_topology.py:364-373, 413-414` | attach **before** any fabric setup on the same program | — | — | — |
| `land_arrivals_block` | raw_api | `noc_semaphore_wait_min`, `cb_reserve_back` / `cb_push_back`, re-arm with `noc_semaphore_inc(…, 0u - total)` | `blackhole-ccl-kernels.md` §5.2 | — | global `arr_d` | `cb_arrivals_d` | — |
| `handoff_or_store_block` | raw_api | `cb_pages_available_at_front`, `noc_semaphore_inc`, `noc_semaphore_wait_min`, `TensorAccessor` writes | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:443` | — | `cb_finished` | mailbox / output | — |
| `fence_ready`, `send_unit` | raw_api | `WorkerToFabricEdmSender::build_from_args`, `to_noc_unicast_write`, `to_noc_fused_unicast_write_atomic_inc`, `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `route_one_hop` | `examples/fabric_reduce_scatter/program_descriptor_with_inline_kernels.py:362-575` (copy and adapt) | header ring `port_group` | `cb_port_chunks` | remote L1 | `chunk_tiles`, `port_group` |
| `gather_unit` | raw_api | `noc_async_read(get_noc_addr(core_x, core_y, finished_addr + slot_off), …)`, `noc_async_read_barrier` | `dataflow_api.h` | — | remote `cb_finished` | `cb_port_chunks` | `port_group` |
| host plumbing | raw_api | `build_groups`, `_needs_ready`, `allowed_cores`; `ttnn.setup_fabric_connection` (appended last); `ttnn.get_forwarding_link_indices`; `ttnn.create_global_semaphore`; `ttnn.cb_descriptor_from_sharded_tensor` | `examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:371,407,704`; `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:518` | copy the helpers. **Do not** call `plan()` (it dispatches the probe program). | — | — | — |

**Helpers considered and rejected (for every raw_api row):**
- `land_arrivals_block`: no helper credits a CB whose bytes a remote chip wrote. `mcast_pipe`'s `ReceiverPipe` (`mcast_pipe.hpp:126`) waits on a same-chip sender's flag or counter, and its round model is single-sender per round. The arrival here is a fabric fused-inc counter with host thresholds. A wait plus a CB credit is the whole block op.
- `handoff_or_store_block`, `gather_unit`: `mcast_pipe` (`mcast_pipe.hpp:55`) is one-to-many. The handoff is many-to-one (≈ n_cores/L producers into one port), with per-producer acks and non-FIFO ack arrival across directions. No helper expresses a pull with per-producer consumed acks.
- `send_unit`, `fence_ready`: the CCL helper library (`kernel_lib/ccl/`, `blackhole-ccl-kernels.md` §12) is marked [unmeasured]. It routes by 1D hop count, and its schedule header encodes the classic reduce-scatter step schedule, not per-core zone destinations. The measured raw sender of `fabric_reduce_scatter` is the reference to copy.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| finish relay | SFPU add (D0 + D1) | `cb_partial_hold`: All | `cb_arrivals_d`: All | None |
| finish fin | FPU add → D1 | `cb_arrivals_fwd`: All | `cb_arrivals_bwd`: All | None |
| finish fin | SFPU add (D0 + D1) | `cb_partial_hold` (D0): All | D1: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| One-dispatch rule vs the reference's placement probe | `fabric_all_gather.plan()` runs a probe `generic_op` on first use, which would be a second dispatch | Host-only port placement in the strip. Copy `build_groups` / `_needs_ready` / link discovery, never `plan()`. |
| Zone address on the peer chip | The fabric writes `get_noc_addr(core_k, zone_addr)` on the *other* chip. A per-program CB address is not guaranteed to match across chips. | Zones and handoff are CBs **backed by a cached mesh-allocated L1 tensor**, which has the same address on every chip. Logical→virtual core coordinates come from `worker_core_from_logical_core`, which the reference also uses chip-agnostically. |
| Counters credited out of order | Arrival thresholds assume one ordered sender per counter | `arr_d` on core k is incremented only by upstream port `(d, k mod L)`. Each fused inc has `flush=true`, so it covers all earlier writes of that sender. Thresholds are counted in the **upstream's** send order. |
| Handoff ack confusion across directions | A core's entries go to fwd and bwd ports, which ack independently | Separate `consumed_fwd` / `consumed_bwd`. FIFO pops wait on the counter of the popped entry's direction. |
| Port reads a core's slot before it is ready | Units mix cores of different halves, so a total count can be satisfied by the wrong core | Per-core mailbox slot on the port, counted per (core, unit). The port waits on `mailbox[k]` for each core. |
| Global semaphores and the mailbox must return to zero | A trace or a repeated call reuses them. A stale count makes the next call read early or hang. | The reader decrements `arr_d` by its total. Ports decrement `ready` by 1 and each mailbox slot by its final count. One-time `synchronize_device` only at creation. |
| Semaphore id collisions | `Mcast1D.attach` and `setup_fabric_connection` both allocate ids | matmul cores: Mcast attach plus the op's own semaphores. Ports: the op's own semaphores first, then fabric setup. They are different core sets. |
| fp32 DEST forced on the matmul cores | The cross-device add must be fp32, and DEST mode is kernel-wide (`chain.hpp:44-46`) | `fp32_dest_acc_en=True, dst_full_sync_en=True` on the matmul compute kernel, which keeps 8-tile subblocks. Fidelity is the caller's. Lamp recorded. |
| HiFi4 + fp32 DEST with bf16 inputs | `matmul_block_helpers.hpp` warns of silent K-accumulator corruption (WH B0, #38306) | Blackhole is the target. The acceptance test runs a HiFi4 cell and checks PCC. |
| Chain after matmul in one kernel | Boot is `SrcOrder::Reverse` for matmul; the eltwise chain re-inits its own unpack config | Chains keep the default Chain-owned init + reconfig, and `matmul_block` re-inits every entry (`InitMode::Short`, `INPUT_AND_OUTPUT`). Bring up finish-after-matmul on one core first. |
| fp32 hold → CopyTile precision | Unpacking a Float32 CB to DEST goes through srcA (TF32) unless UnpackToDestFp32 is set | Acceptable: the result is packed to bf16 and travels in bf16. Optional: `unpack_to_dest_mode[cb_partial_hold] = UnpackToDestFp32`. |
| Ragged per-core blocks | push counts must stay nominal (ring-wrap invariant) | nominal pushes; valid-only DRAM reads/writes; `last_in1_subblock_w_valid`; garbage only in tiles that are never stored |
| Arrivals late on link-bound shapes | Line topology and small K: the compute stalls on `land_arrivals_block` | Correct by the deadlock argument (descending key). Performance there is bounded by the links anyway (roofline). |
| Line ends and the fence | A line end has no link in one direction but must still send `ready` to its neighbour | Port exists iff the link exists. `send_ready` follows `_needs_ready` semantics (every port whose peer sends to this chip). |
| L1 clash with model-resident L1 tensors | The staging tensor and the CBs need `F` bytes per core | `L1_budget` subtracts the allocator base and a headroom constant. The staging tensor is cached grow-on-demand per mesh, so different shapes never stack. |
| `num_links=None` on a 4-link box | Not a registry value | Resolved to all usable links after `validate()`. The code is generic in L, with `2L ≤` strip length. |
