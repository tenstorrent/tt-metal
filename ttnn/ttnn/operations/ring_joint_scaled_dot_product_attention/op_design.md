# Operation Design: ring_joint_scaled_dot_product_attention

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (CCL ring all-gather of K/V over Fabric + flash attention), one `ttnn.generic_op` over a `MeshProgramDescriptor` |
| Goal | Sequence-parallel ring attention: every device keeps its queries; its queries attend to all K/V of its group (plus optional rear joint keys), with the K/V transport hidden behind local attention work. |
| Math | for each group, device p: `out[p] = softmax(s·Q[p]·keysᵀ + M) · values`, keys/values = every member's K/V at its global positions, then the joint keys; `M = -inf` on keys at global position ≥ `logical_n` (and on joint rows ≥ the valid joint length); `joint_out = softmax(s·JQ·keysᵀ + M) · values` (non-causal). `s = scale or 1/sqrt(dh)` |
| Mode | Derivative (new fused op; transport adapted from `examples/fabric_all_gather`, compute adapted from the single-chip SDPA kernels) |
| References | `ttnn/ttnn/operations/examples/fabric_all_gather/` (README.md, report.md, `program_descriptor_with_inline_kernels.py`); `ttnn/ttnn/operations/examples/fabric_ring_exchange/` (README.md, `program_descriptor.py:229` custom program hash); `references/blackhole-fabric.md` §6 rules 1–10; `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`, `.../dataflow/dataflow_common.hpp`, `.../compute/joint_sdpa.cpp`, `.../dataflow/joint_reader.cpp`, `.../dataflow/joint_writer.cpp`; `ttnn/cpp/ttnn/kernel_lib/` helpers; `eval/golden_tests/ring_joint_scaled_dot_product_attention/` (feature_spec.py, helpers.py, roofline.py) |

**Kind labels used throughout.** Every cross-device item is tagged **TRANSPORT** (how bytes cross the fabric: links, directions, packets, fences — taken from `fabric_all_gather`, not re-derived) or **OVERLAP** (hiding the transport behind the attention: ordering, landing notification, launch-skew tolerance, the tail chain — the fusion itself).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_q` | mesh Tensor | yes | per device (b, hq, n_q, dh); TILE, DRAM interleaved | — | RT (address) |
| `input_tensor_k` | mesh Tensor | yes | per device (b, hk, n_kv, dh) | — | RT (address) |
| `input_tensor_v` | mesh Tensor | yes | per device (b, hv, n_kv, dv) | — | RT (address) |
| `joint_tensor_q/k/v` | mesh Tensor or None | all-or-none | (b, hq/hk/hv, L or L_dev, dh/dh/dv) | None | RT (address) |
| `logical_n` | int | yes (kw) | `1 ≤ logical_n ≤ G·n_kv` (else ValueError) | — | RT (compute, reader, writer) |
| `logical_l` | int | no | 0 (replicated) or `0 < logical_l ≤ G·L_dev` (sharded; else ValueError) | 0 | RT |
| `cluster_axis` | int | yes | 0, 1 | — | host (groups) |
| `topology` | `Topology` | no | Linear, Ring (Ring needs a wrap on `cluster_axis` under the active fabric config and a usable wrap hop; else ValueError). A 2-device Ring runs the Linear schedule. | Linear | host (schedule) |
| `num_links` | int or None | no | `1 ≤ num_links ≤ min over the group's hops of len(get_forwarding_link_indices)` (else ValueError); None = that minimum | None | host (port count) |
| `is_causal` | bool | no | Phase 0: False | False | CT |
| `is_balanced` | bool | no | Phase 0: False | False | CT |
| `scale` | float or None | no | > 0 | `dh**-0.5` | CT (fp32 bits) |
| `compute_kernel_config` | `ComputeConfigDescriptor` or None | no | fidelity ≥ requested is honored; `fp32_dest_acc_en` gated by SUPPORTED | `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=True | CT |
| `memory_config` | MemoryConfig or None | no | DRAM interleaved (Phase 0) | DRAM interleaved | host |

Entry-point order: `validate()` first line (support refusals: `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`), then caller-error checks raising `ValueError` (num_links above discovery, Ring without wrap / on a non-wrapping config, `logical_n > G·n_kv`, `logical_l > G·L_dev`, joint tensors partly given, shape/dtype mismatch across tensors, non-TILE / non-DRAM-interleaved inputs, `hq % hk`, `hq % hv`). `compute_kernel_config=None` resolves through the exported `default_compute_kernel_config()` factory. Topology enum at import time: `from ttnn.operations.ccl import Topology`. The op reads the active fabric config (`ttnn.get_fabric_config()`) and mesh shape and never changes them.

Registry axes (names exactly as in `feature_spec.py` TARGET): call-derived `dtype` (Q), `kv_dtype` (K), `fp32_dest_acc_en`, `layout`, `joint_mode` ("none" / "replicated" if `logical_l == 0` / "sharded"), `mask_mode`, `is_balanced`, `seq_padding` ("exact" iff `logical_n == G·n_kv`), `cluster_axis`, `topology`; tagger-derived `alignment`, `attention_kind`, `kv_heads_mode`, `joint_len` (the four `INPUT_TAGGERS`, reading per-device Q/K/V shapes and the optional LOGICAL joint Q shape).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q (b, hq, n_q, dh); K (b, hk, n_kv, dh); V (b, hv, n_kv, dv); identical on every device; rank 4; n_q, n_kv, dh, dv % 32 == 0; Phase 0: n_q == n_kv, hq == hk == hv |
| Dtype | Q/JQ: bfloat16 (Phase 0) — TARGET adds bfloat8_b; K/V/JK/JV: bfloat16 (Phase 0) — TARGET adds bfloat8_b, may differ from Q |
| Layout | TILE |
| Memory | DRAM interleaved |
| Joint | replicated: (b, h*, L, d*) with L any length (Phase 0: L % 32 == 0); sharded: L_dev = roundup32(ceil(L/G)) rows per device |

### Output

| Property | Value |
|----------|-------|
| Shape | out (b, hq, n_q, dv); joint_out (b, hq, L, dv) replicated / (b, hq, L_dev, dv) sharded / None |
| Dtype | bfloat16 always |
| Layout | TILE |
| Memory | `memory_config` or DRAM interleaved |
| Rows | output rows at padded query positions and joint rows ≥ valid joint length are unspecified (written, never checked) |

### Persistent per-program resources (cached, never per call)

| Resource | Shape / placement | Cache key | Purpose |
|----------|-------------------|-----------|---------|
| `kv_landing_k` | DRAM interleaved, (G−1)·(b, hk, n_kv, dh) tile pages, K dtype | (mesh id, cluster_axis, G, b, hk, hv, n_kv, dh, dv, kv dtype, joint layout) | remote K head-slices land here (TRANSPORT) |
| `kv_landing_v` | DRAM interleaved, (G−1)·(b, hv, n_kv, dv) | same | remote V head-slices land here |
| `sem_arrival` | global semaphore over every used core | (mesh id, port+compute core set) | per-port landing counter (TRANSPORT + OVERLAP) |
| `sem_ready` | global semaphore | same | per-call fence between chips (TRANSPORT) |
| `sem_consumers_done` | global semaphore | same | port re-arm gate: every compute reader has stopped polling (OVERLAP) |
| `sem_poll` | global semaphore (never a counter) | same | the compute reader's private NoC-read target for a port's arrival count (verifier fix: polling into `sem_arrival` left stale counts on cores re-placed as ports in a later call) |
| port plan | per device: port cores per (direction, link), links per hop, Ethernet-adjacent placement | (mesh id, cluster_axis, topology, num_links, fabric config) | `fabric_all_gather.plan()`-equivalent, probed once |

Landing slot of source s on device p: `slot(p, s) = (s − p − 1) mod G ∈ [0, G−2]`; landing page = `slot·tensor_pages + source_page`. Semaphore creation runs once per cache key, followed by one `ttnn.synchronize_device` (every counter must be zero on every chip before any chip's first increment) — exactly `fabric_all_gather.py:1095-1102`; later calls allocate nothing and never synchronize. A second call with new tensor addresses only patches runtime args; the program uses a `custom_program_hash` covering every structural input (`fabric_ring_exchange/program_descriptor.py:229`: generic_op's cache ignores the per-core RT-arg layout).

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Two kinds of block exist: the **attention unit** (compute cores) and the **transport entry** (port cores). Both are fixed here; everything below realizes them.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (b) | independent — no result spans batches | `batches_per_unit` | 1 — each batch has its own K/V stream, so a unit spanning two batches shares no operand and doubles L1 | host constant `BATCHES_PER_UNIT = 1` | units round-robin over the compute grid (see Work Distribution) | — (no gain: nothing is shared across b) |
| `q_head` (hq) | independent for the result; **reuse-shared on K/V under GQA/MQA** (hq/hk Q heads read one K head) | `q_heads_per_unit` | 1 — Phase 0 is MHA (hq == hk == hv): no two Q heads share a K/V stream | host constant `Q_HEADS_PER_UNIT = 1` | round-robin units | knob-turn: GQA/MQA units of `hq/hk` heads share one K/V block stream (K/V DRAM reads ÷ hq/hk) |
| `q_rows` (n_q tiles = `tensor_q_tiles = ceil(n_q/32)`) | independent — rows of the softmax never interact | `q_block_tiles` (Sq) | `choose_block_extents()` (below): coarsest that fits L1 after the grid is filled; FOCUS (fp32 off) 16, Phase 0 dh 128 fp32 8 | host, CT arg to reader/compute/writer | round-robin units | knob-turn |
| `joint_q_rows` (L tiles = `ceil(L/32)`) | independent | same `q_block_tiles` | same | same | joint units interleaved after each (b, h)'s main units | knob-turn |
| `keys` (N_k tiles = G·`tensor_kv_tiles` + joint tiles) | **dependent** — softmax normalizer and output span every key | `k_block_tiles` (Sk) | `choose_block_extents()`: FOCUS 8, Phase 0 dh 128 fp32 8 | host, CT arg | NOT split across cores: every unit walks all key blocks sequentially (online softmax accumulates in place); split across DEVICES by the caller, resolved by the ring all-gather | scheme-change (key-split + cross-core combine: `rejected`, see Regimes) |
| `key_source` (segment of `keys`: local, joint, each remote member) | dependent (part of `keys`) — order is free because the online-softmax combine is commutative/associative | one source at a time inside the walk | walk order fixed: local → joint → remote by hop distance (upstream first on ties) | host computes order, RT args | per unit | — |
| `head_dim` (dh tiles = `DHt`) | dependent — contraction of Q·Kᵀ | `head_dim_tiles` | whole extent (DHt): one matmul K-block; splitting it only adds partial-sum spills | derived from Q shape | not split | — |
| `v_head_dim` (dv tiles = `DVt`) | independent output columns, but every column needs the same P and row statistics | `v_head_dim_tiles` | whole extent (DVt): a split recomputes QKᵀ/softmax per split | derived from V shape | not split | — |
| `transport_entry` (b, kv head j, j < max(hk, hv)) — the landing stage's data | independent — each entry lands and is consumed on its own | one entry = K head j (if j < hk) + V head j (if j < hv) of one batch | 1 entry | host schedule | ports: each (direction d, link l) carries its link's DRAM-bank share of every entry it sends; compute: none (communication role) | knob-turn (entry = several heads) |
| `transport_chunk` (pages within an entry) | independent | `run_pages` | `max(1, get_tt_fabric_max_payload_size_bytes() // page_bytes)` = 7 bf16 tiles at the production payload | host, CT arg | per port, bank round-robin within the port's bank set | — (fabric-wide setting) |
| `device` (group position p) | dependent (`keys` split by the caller) | — (caller's shard, fixed) | n_kv rows per device | caller | one program per device | — |

**Stage-by-stage occupancy.** (1) Attention units: spread over every compute core. (2) Landing (TRANSPORT): port cores only move bytes — a communication role, no attention computed on them (`blackhole-fabric.md` rule 8: keep other work off the sending core). (3) Landing notification: no compute stage; consumers poll counters (see Dataflow). (4) Finalize (1/rowsum, normalize): on the unit's own core, inside the unit — nothing to spread. (5) Output write: the unit's own writer. No stage computes on one core while others wait.

### `choose_block_extents()` — the single source of `q_block_tiles` / `k_block_tiles` (host, pinned)

```
T_q, T_kv  = tile bytes of Q, K/V dtypes;  T_acc = 4096 if fp32_dest_acc_en else 2048;  T_bf = 2048
footprint(Sq, Sk) = q_depth·Sq·DHt·T_q + kv_depth·Sk·(DHt + DVt)·T_kv + Sq·Sk·T_acc
                  + 5·Sq·T_acc + 2·Sq·DVt·T_acc + out_depth·Sq·DVt·T_bf + (2 + mask_tiles)·T_bf
total_q_tiles = b·hq·(tensor_q_tiles + joint_q_tiles)
Sq_occ = largest power of two ≤ max(1, total_q_tiles // C)          # step 1: fill the grid
for Sq in powers of two, descending, Sq ≤ min(Sq_occ, pow2_ceil(tensor_q_tiles)):    # step 2: coarsest Sq
    Sk = largest power of two ≤ pow2_ceil(tensor_kv_tiles) with footprint(Sq, Sk) ≤ L1_CB_BUDGET
    if Sk exists and Sk ≥ min(4, pow2_ceil(tensor_kv_tiles)): return Sq, Sk
return 1, largest Sk with footprint(1, Sk) ≤ L1_CB_BUDGET   (ValueError if none)
```

Sq is maximized first because it sets the split-ranking cost (K/V DRAM re-reads = Q blocks per head); Sk only amortizes per-block fixed cost (extent ranking). Inventory was minimized before this solve (`l1_ledger.md`). Values (L1_CB_BUDGET ≈ 1.3 MB): dh=dv=128 fp32 DEST → (8, 8), 1062 KB; dh=dv=128 bf16 DEST (FOCUS) → (16, 8), 1184 KB.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_k_block`, `cb_v_block` | `kv_block_depth` | 2 | reader fetches key block i+1 while compute works on block i (the reader is DRAM-latency bound, compute bound overall) |
| `cb_q_block` | `q_block_depth` | 1 | Q is read once per unit and stays resident for its whole key walk; depth 2 would prefetch the next unit's Q (lamp) |
| `cb_out_tiles` | `out_block_depth` | 1 | the writer drains a finished unit during the next unit's whole key walk |
| `cb_port_chunks` (port cores) | `port_batch_chunks` (`group`) | 4 chunks per batch, CB = 2·group chunks (112 KiB) | `fabric_all_gather` README: batch 4 best or within 5% everywhere; 2 is 11–18% slower at size |
| stats / accumulators | — | ping-pong pairs (prev/cur) | concurrent lifetimes, not depth (see ledger) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric packet payload | `run_pages` | `max(1, payload // page_bytes)`; chunk = pages consecutive in ONE DRAM bank | a chunk larger than the payload is truncated by the router; pages from two banks in one packet land at the wrong addresses |
| Interleaved bank walk | chunk page set | a chunk is pages `b + (m+t)·B` of one bank b (`for_each_chunk`, `fabric_all_gather.py:40-87`) generalized to an absolute page range `[entry_base, entry_base + entry_pages)` with bank = page mod B | a walk relative to the slice base mis-assigns banks for slices not starting at a multiple of B → wrong pages |
| Arrival counter (32-bit) per port | entries per call | one increment per (port, entry); counts ≤ (G−1)·b·max(hk,hv) per call | n/a (bounded) |
| Landing index packing | entry index / slot in RT args | 32-bit fields (the example's `shard | part<<16` packing is replaced by separate fields) | a 16-bit slot/entry field silently wraps for large b·heads |
| matmul_block in0 retention | `q_block_tiles·DHt` | Q CB holds the whole unit's Q (one in0 block, `WaitAndRetainOnLastBlock`) | Q re-fetched per key block |
| Online softmax requires a finite row max | first processed key block of every unit | fully padded key tiles are skipped (never read, never computed); every processed block has ≥ 1 valid key column; the local source is processed first and always contains ≥1 valid key when `logical_n > p·n_kv`, otherwise the first non-empty source is used | `exp(-inf − (-inf)) = NaN` poisons the row |
| DEST capacity (subblocks) | none on the block — the matmul helper's subblock factors | derived from block size + `fp32_dest_acc_en` (4 tiles fp32 / 8 tiles bf16, half-sync) | n/a (helper-owned) |
| `mcast` not used in Phase 0 | — | — | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `qstationary_dram_kv`** — Q-stationary flash attention; each unit streams all key blocks from DRAM (local K/V from the inputs, remote from the landing buffers); ring all-gather of K/V into DRAM landing buffers, head-major, gated per entry | **built** | `not is_causal` and joint_mode ∈ {none, replicated} and MHA (Phase 0 SUPPORTED); is the base every other row layers onto | unit = (1 batch, 1 Q head, `q_block_tiles` rows) × key blocks of `k_block_tiles` × whole DHt / DVt | minimum (named boundary DRAM): Q, local K/V, joint read once; out written once; remote K/V lands once. Actual: K/V are re-read once per Q block of the same head, `Uq = ceil(n_q_t/Sq) + ceil(L_t/Sq)` times (FOCUS: 16× on 167.8 MB local + 503 MB landed ≈ 10.7 GB/call); own K/V read once per sending direction (2× on a ring interior); relayed slices read once; remote K/V written once to landing | Sq: K/V re-reads ÷ Sq (split ranking). Sk: per key block — matmul/reduce/eltwise inits and data-format reconfigs (~6 phase switches), CB handshakes on K, V, scores, stats, one landing check; intended once per key block. Unit: Q read + stats init + finalize once per unit |
| **R2 `qstationary_kv_row_mcast`** — R1 plus K/V broadcast: the cores of one grid row hold `row_len` Q blocks of the SAME (b, head) per wave; one injector per row (or `injectors_per_row` split by key block) reads each K/V block once and multicasts it along the row (`mcast_pipe` SenderPipe/ReceiverPipe) | **deferred** — the multicast layers onto R1's per-core streaming (stepping stone, `blocking-model.md` §5); R1's unit order is already head-major and row-contiguous so the structure survives; this is refinement #1 in the queue | non-causal, every case where a head has ≥ 2 Q blocks per wave (all perf cases) | same unit; plus the row group shares one key-block stream | K/V DRAM re-reads ÷ ~row_len (FOCUS: ~10.7 GB → ~1.1 GB per call); adds the same bytes once per row on NoC multicast + one handshake per key block | per key block: one mcast + one semaphore round trip per row, amortized over Sk·(DHt+DVt) tiles |
| **R3 `qstationary_gqa_shared_kv`** — a unit = `hq/hk` Q heads of one KV group sharing one key-block stream | **deferred** — Phase 0 SUPPORTED is MHA (`kv_heads_mode: mha`); a knob-turn of `q_heads_per_unit` once `kv_heads_mode` gqa/mqa/mla are unlocked | `hk < hq` | unit = (1 b, hq/hk heads, Sq rows) | K/V DRAM reads ÷ hq/hk vs. per-head units | same as R1 per head-group |
| **R4 `causal_contiguous` / `causal_balanced`** — R1 with a per-(unit, key block) class {skip, full, diagonal-masked}; zig-zag positions for balanced | **deferred** — `mask_mode: causal` is outside Phase 0 SUPPORTED; R1's key-block plan already classifies blocks (skip / full / masked for padding), so causal only adds the diagonal class and the position formula (`helpers.global_positions`) | `is_causal` | same | skipped blocks are neither read nor computed: K/V reads ∝ keys_seen | same |
| **R5 `sharded_joint`** — joint K/V slices travel the ring as extra transport entries (joint slice of source s lands in a joint landing slot), joint Q stays local | **deferred** — `joint_mode: sharded` outside Phase 0 SUPPORTED; entries are already per (source, b, head) so a joint entry is one more entry kind | `logical_l > 0` | same | adds (G−1)·L_dev joint rows landed once | same |
| **R6 `joint_q_split_across_group`** — replicated joint queries split 1/G per device, then a joint_out all-gather | **deferred** — the replicated-joint perf rows (Flux1) are link-near; this adds a second transport phase whose payload exists only after the last landing (serial tail, `G−1` slices of `L·hq·dv·2/G` B); the structure keeps it reachable because joint units are separate units with their own writer target | joint_mode replicated | same | saves (G−1)/G of the joint attention FLOPs (Flux1: ≈ 1/3 of total compute ·¾); adds `L·hq·dv·2·(G−1)/G` B per device over the links after compute | — |
| **R7 `kstationary_key_split_combine`** — split `keys` across cores: each core keeps a K/V slice resident and streams all Q blocks of its head through it, emitting (O, m, l) partials combined across cores | **rejected** — superseded by R2: at FOCUS it moves ≈ 280 MiB/head (Q re-read ×28 + fp32 partials written and re-read) vs R2's ≈ 47 MiB/head, and adds a cross-core combine | — | — | Q crosses DRAM N_k/Sk_res times + partials 2× | — |
| **R8 `ring_move_q`** — Q and fp32 partial outputs travel the ring, K/V stay | **rejected** — moves Q (n·hq·dh) out and fp32 O + stats (n·hq·dv·4) back per hop: ≥ K/V bytes for MHA and hq/hk× more for GQA/MQA/MLA; needs a cross-device combine | — | — | links: ≥ 1.5–16× R1's K/V bytes | — |
| **R9 `gather_then_attend`** — finish the all-gather, then run attention | **rejected** — the dead end: same bytes as R1 but the whole transport (3.46 ms at FOCUS link roofline) is exposed; R1 is the same code with a landing gate | — | — | same as R1 | — |
| **R10 `l1_landing`** — remote K/V lands directly in compute L1 | **rejected** — unreachable for these shapes: each remote head slice (FOCUS 4 MiB) is consumed by `Uq` units across many waves, so it must persist beyond any core's L1; DRAM landing is required for the re-reads | — | — | — | — |

Regime selection: exactly one built regime → `select_regime()` returns R1 for every SUPPORTED cell; refinements add the predicate column above as the selector and require regime-pinned tests.

### Traffic ranking

Per device, per call, FOCUS (n = 8192, H = 40, dh = dv = 128, G = 4 ring, bf16, Sq = 16): K/V block B = 167.8 MB, Q 83.9 MB, out 83.9 MB.

| Candidate split | DRAM crossings | Cross-core (NoC) | Cross-device (links) | Verdict |
|-----------------|----------------|------------------|----------------------|---------|
| (b, head, q block) units, K/V streamed per unit from DRAM (R1) | Q 1×, out 1×, K/V: Uq·G blocks read = 64 × 167.8 MB ≈ 10.7 GB + landing writes 3 × 167.8 MB + transport reads ≈ 3 × 167.8 MB | per-unit DRAM reads only | all-gather (G²/8)·B per busiest direction | **built** (stepping stone) |
| same + K/V row multicast (R2) | K/V ≈ 1.1 GB | each K/V byte multicast once per row | same | cheapest; deferred (layers on R1) |
| same + GQA head-group units (R3) | K/V ÷ hq/hk | — | — | knob-turn after GQA unlock |
| key split + combine (R7) | Q ×28, partials 2× ≈ 280 MiB/head | combine | same | rejected |
| move Q around the ring (R8) | Q 1×, partials | — | ≥ K/V bytes | rejected |

Occupancy is not in this table (settled in Work Distribution). The chosen built split fills the grid; its excess is the K/V re-read, which R2 removes — recorded as the deferred row, not hidden.

### Stall-shadow (OVERLAP)

| Waiting stage | What it waits for | Work independent of it, scheduled into the stall | Permission |
|---------------|-------------------|--------------------------------------------------|------------|
| a unit's remote key blocks | remote entry (s, b, head) landed in `kv_landing_*` | the unit's local key blocks and joint key blocks (replicated) — always processed first | the online-softmax combine of (O, m, l) over key blocks is commutative and associative: any fixed block order gives the same softmax up to FP rounding; order is fixed per unit (deterministic run to run) |
| the whole first wave | the first remote entries | waves are head-major and the transport is head-major, so wave w needs heads ≈ [w·C/Uq, (w+1)·C/Uq) — delivered at ~9 heads/ms vs consumed at ~1.3 heads/ms (FOCUS); only the very first entries' latency is exposed, behind the first unit's local + joint blocks (≈ 1/G of a unit) | head-major page order of (b, h, n, d) tensors |
| port relay reads | upstream entry landed | none on the port reader (its own next-head read is queued after the relay in head-major order; the sender drains its CB independently) | — (nothing independent; stated) |

FP-order note: the key-block order differs from the reference's global order; gated on the golden PCC / rel-RMS baseline (precision is a cap on error; online softmax in any order is within it).

### Block schedule

Logical schedule per compute core (reader, compute, writer realize their parts asynchronously):

```cpp
startup_block();                                   // writer: scalers + key mask tile(s); compute: hw startup
for (uint32_t unit_idx = 0; unit_idx < num_units_this_core; ++unit_idx) {
    load_q_block(unit_idx);                          // reader: Sq x DHt Q tiles (main or joint Q)
    for (uint32_t kb = 0; kb < num_key_blocks(unit_idx); ++kb) {   // local -> joint -> remote by hop
        await_landing(unit_idx, kb);                  // reader, remote blocks only: per entry, not per block
        load_kv_block(unit_idx, kb);                  // reader: Sk x DHt K (transposed order), Sk x DVt V
        score_block(kb);                              // S = Q . K^T            (Sq x Sk)
        mask_block(kb);                               // only the block holding the last valid key tile
        row_max_block(kb);                            // m_cur = max(m_prev, rowmax(S))
        exp_block(kb);                                // P = exp(s.(S - m_cur)) in place; l_cur = row partial sums
        pv_block(kb);                                 // O_cur = P . V          (Sq x DVt)
        rescale_block(kb);                            // kb>0: a = exp(s.(m_prev-m_cur)); l_cur += a.l_prev; O_cur += a.O_prev
    }
    finalize_block(unit_idx);                         // l = rowsum_in_tile(l); out = O . (1/l)  -> bf16
    store_out_block(unit_idx);                        // writer: valid tiles to out or joint_out
}
release_ports();                                      // reader: +1 consumers_done on every port of this device
```

Per port core (TRANSPORT):

```cpp
fence_block();                                       // sender: ready to upstream peer port, wait own ready
for (uint32_t e = 0; e < num_entries; ++e) {         // e over (b, kv head), head-major
    send_own_entry(e);                               // reader: own K/V slice e (this link's banks) -> CB; sender: chunks -> downstream landing slot
    for (each source r relayed on this port) {
        await_arrival(e, r);                         // reader: arrival >= index(e, r) + 1
        relay_entry(e, r);                           // reader: landing slot -> CB; sender -> downstream
    }
}
drain_and_rearm();                                   // sender: wait arrival >= expect_in and consumers_done >= C; decrement both
```

| Block operation | Block shape | Resident across it | Intended frequency of inits / sync |
|-----------------|-------------|--------------------|-------------------------------------|
| `startup_block` | 1 + 1 scaler tiles, `mask_tiles` mask tiles | scalers, masks for the whole program | once per program |
| `load_q_block` | Sq × DHt | — | once per unit; one reserve/push of the whole block |
| `await_landing` | one entry (s, b, kv head) | landed-count cache per port | at most once per (unit, remote source); polls only when the cached count is short |
| `load_kv_block` | Sk_valid × DHt (K) + Sk_valid × DVt (V) | — | once per key block; one reserve/push per CB per block |
| `score_block` | (Sq × DHt) · (DHt × Sk) | Q (retained) | one matmul init per key block |
| `mask_block` | Sq × 1 tile column (the last valid key tile of the block) | mask tile (never popped) | only on the ≤ 1 boundary block per source segment (main tail; joint tail later) |
| `row_max_block` | Sq × Sk_valid → Sq | m_prev | one reduce init per key block |
| `exp_block` | Sq × Sk_valid in place + Sq row partial sums | m_cur | one chain init per key block |
| `pv_block` | (Sq × Sk) · (Sk × DVt) | — | one matmul init per key block |
| `rescale_block` | Sq stats + Sq × DVt | O_cur, l_cur | per key block except the first of the unit |
| `finalize_block` | Sq → 1/l; Sq × DVt normalize | — | once per unit |
| `store_out_block` | Sq_valid × DVt | — | once per unit; one wait/pop of the block, writes barrier once per unit |
| `fence_block` | — | — | once per call per port |
| `send_own_entry` / `relay_entry` | one entry share in `run_pages` chunks, `group` chunks per batch | — | one fused write+inc per (port, entry), one flush per batch |
| `drain_and_rearm` | — | — | once per call per port |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap (Sk) | Sk = 8 at Sq = 16 keeps the reader one key block ahead; a larger Sk could serialize K/V fetch against compute at unit start | Sk 4 vs 8 vs 16 at fixed Sq (FOCUS) |
| grid synchronization / tail | round-robin units leave a partial last wave (FOCUS fp32-off: 1280 units / 106 cores → 12.08 waves, last wave 8 units) | Sq 16 vs 8 (2560 units: 24.2 waves) vs uneven last-wave split |
| Q prefetch | `q_block_depth` = 1 stalls compute at each unit start for one Q block read | `q_block_depth` = 2 |
| scale placement | scale folded into the exp SFPU pass (fp32) costs one SFPU multiply per score tile | pre-scale Q once per unit (Sq·DHt tiles instead of Sq·Sk per key block), gated on precision |
| exp approximation | exact exp is the SFPU's largest cost per score tile (the roofline does not model it) | `Approx::Fast` exp, gated on the golden PCC / rel-RMS |
| sum via L1-acc pack | the row partial sums are produced by a second L1-accumulating pack of every P tile | `reduce<SUM, REDUCE_ROW>` over P after exp (one FPU pass) |
| single O accumulator | O ping-pong costs 2·Sq·DVt·T_acc of L1 that could raise Sq | rescale O in place, then pv_block packs with L1 accumulation onto it |
| landing increment interval (TRANSPORT) | one increment per (port, entry) delays a relay by one entry share (FOCUS ≈ 55 µs/hop) | `inc_every` = 8 chunks within an entry (example default) |
| landing poll (OVERLAP) | consumer readers poll a remote port counter | port reader multicasts landed counts to compute cores |
| ring balance (TRANSPORT) | balanced split of the far source by bank halves (`fabric_all_gather` `balance=True`, 1.26–1.28× at size) | `balance=False` on short sequences (< 64 tiles per entry share) |

Catalog entries that informed the knobs: `fabric_all_gather` (port placement, run_pages from payload, batch 4 chunks / 112 KiB, ready fence, balance, ring over line, 2 links 1.9×), `fabric_ring_exchange` (custom program hash; direct multi-hop sends are slower than relay → relay), `blackhole-fabric.md` §6 (full packets, NoC1 senders, one producer per (link, direction), bank round-robin), `shared_input_reuse` + `mcast_topology` (R2's injector multicast, 1.7–2.3×), `compute_block_size` (per-phase fixed cost ≈ 320 ns → coarse Sk), `matmul_output_subblock` (subblocks left to the helper), `double_buffer` (kv depth 2; when compute-bound the depth lever saturates), `sfpu_tile_scope` (column-0 stats ops).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| own K/V → fabric (TRANSPORT) | K/V dtype tiles | port reader (NCRISC, NoC0): `for_each_chunk` over the entry's absolute page range restricted to this link's banks → `cb_port_chunks`; port sender (BRISC, NoC1): `to_noc_unicast_write` to the downstream device's `kv_landing_*` slot page, last chunk of each entry `to_noc_fused_unicast_write_atomic_inc` on the downstream port's `sem_arrival` | copy of `fabric_all_gather.py:89-305` with entries = (b, kv head) slices of K then V; Ring: toward p+1 sources (p, p−1, …), toward p−1 (p, p+1, …), balanced far source split by bank halves (`_schedule`, `:384`); Line: toward each end |
| relay (TRANSPORT) | same | port reader waits `sem_arrival ≥ index + 1`, reads the landed slice from its own `kv_landing_*`, sender forwards | store-and-forward one hop at a time (relay beats direct multi-hop: `fabric_ring_exchange`) |
| per-call fence (TRANSPORT / launch skew) | — | sender: `to_noc_unicast_atomic_inc` of the peer port's `sem_ready`, then `noc_semaphore_wait_min(ready, 1)`, decrement (`fabric_all_gather.py:240-255`) | no chip writes a neighbour's landing buffer before that neighbour started this call (i.e. finished the previous call's reads); local work needs no fence |
| landing notification (OVERLAP) | — | compute reader computes, per remote source s and entry e, the required count `e·E_port + rank_port(s) + 1` on every port carrying a share of s, and polls those ports' `sem_arrival` by NoC read (4 B) with a cached last-seen value and a back-off; `E_port` and `rank_port(s)` are host-computed RT args | increments only at entry ends ⇒ the count is an entry index, no chunk arithmetic on device (`blackhole-fabric.md` rule 5) |
| port re-arm (OVERLAP) | — | each compute reader, after its last poll, `noc_semaphore_inc(port, sem_consumers_done, 1)` on every port of its device; each port sender at the end waits `arrival ≥ expect_in` and `consumers_done ≥ C_device`, then decrements both | a port must not reset its counter while a consumer may still poll it |
| DRAM → compute (Q) | Q dtype tiles | compute reader, TensorAccessor over Q (main) or JQ (joint unit), tile-row-major into `cb_q_block` | once per unit |
| DRAM → compute (K) | K dtype tiles | compute reader: K tile (key t, d) placed at CB index `d·Sk + t` (transposed block order, as `dataflow_common.hpp:145-185`); source = K input (local), JK (joint), or `kv_landing_k` slot (remote) | the matmul helper transposes within the tile (`transpose=true`) |
| DRAM → compute (V) | V dtype tiles | same sources; natural row-major (key t, d) at `t·DVt + d` | — |
| key padding mask | bf16 tile | writer fills once per program: column c of the tile = 0 if `c < logical_n mod 32` else −inf (all rows identical); bf16 −inf = 0xFF80 (`dataflow_common.hpp:332` `fill_neginf_tile`, `:372` `fill_vertical_tile_bf16` as the pattern) | only generated when `logical_n mod 32 ≠ 0`; fully padded key tiles are skipped by both reader and compute (shared `key_block_plan`) |
| compute intra-core | T_acc (fp32 when `fp32_dest_acc_en`) for scores, stats, O; bf16 for `cb_out_tiles` | CBs (ledger) | — |
| compute → DRAM | bf16 tiles | writer: TensorAccessor over `out` / `joint_out`, valid rows only (`tensor_q_tiles` / `ceil(L/32)` rows) | output rows at padded query positions are written (unspecified) |

**Key-block plan** (one header function compiled into reader and compute, so push/wait counts agree by construction): `key_block_plan(unit)` yields, in order, segments {local s = p, joint (if any), remote s by hop distance (upstream first on ties)}; per segment `valid_tiles(s) = clamp(ceil((logical_n − s·n_kv)/32), 0, tensor_kv_tiles)` (main; contiguous layout) or `ceil(L_valid/32)` (joint); blocks of `k_block_tiles` over the valid tiles, last block ragged (runtime extent `Sk_valid`); a block is `masked` iff it contains the last valid tile of a segment whose valid count is not a multiple of 32. A segment with 0 valid tiles is skipped (no landing wait). Each block's push/pop counts are `Sk_valid·DHt` / `Sk_valid·DVt` — the same value on both sides from the same function (ragged extent narrows the transfer and the compute shape; it is never padded with stale tiles). Causal and balanced (R4) extend only this function.

**Placement axis (`memory_layout`).** Not in TARGET: inputs are DRAM interleaved only. The op's only physical shard is the caller's sequence split across devices (the `device` axis), which is dependent and resolved by the ring all-gather (a scheme the op always runs). The `cluster_axis` / `topology` axes change only the host schedule (groups, directions, wrap) — knob-turns over the same kernels.

**Deferred R2 contract (Tensix-to-Tensix).** Row group = compute cores of one logical grid row (minus ports); the row's units in a wave share (b, head). Injector = the row's first core (or `injectors_per_row` cores, each owning every k-th key block); SenderPipe/ReceiverPipe (`mcast/kernel/mcast_pipe.hpp:55`, `:126`) with a consumer-ready semaphore per key block; receivers keep `cb_k_block` / `cb_v_block` as the landing CB (their reader becomes a ReceiverPipe instead of a DRAM reader); `await_landing` moves to the injector only.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one attention unit: (b, q head, q block of `q_block_tiles` rows) of main Q, or of joint Q |
| Units | `U = b·hq·(ceil(tensor_q_tiles/Sq) + ceil(joint_q_tiles/Sq))`, ordered `for b, for h: [main q blocks], [joint q blocks]` (head-major, matching the head-major transport) |
| Grid | per device: every allowed worker core (`compute_with_storage_grid_size()`) minus that device's port cores (`2·num_links`: one per (direction, link), placed by the Ethernet probe next to its link's Ethernet core, `fabric_all_gather.py:436`, `:581`); compute cores ordered row-major (y, then x); `C` = their count (QuietBox 2, 2 links: 110 − 4 = 106) |
| Per-core work | core c (0 ≤ c < C) owns units `u = c, c + C, c + 2C, …` (round-robin ⇒ wave w = units [w·C, (w+1)·C), heads advancing in lockstep with the transport; consecutive cores of a grid row hold consecutive q blocks of one head — the R2 row group) |
| Remainder | `num_units_this_core = ceil((U − c) / C)`; last q block of a head has `last_q_block_tiles = tensor_q_tiles − (ceil(tensor_q_tiles/Sq) − 1)·Sq` rows (tile geometry `ceil`, per (b, h)); joint `joint_q_tiles = ceil(L/32)`; ragged blocks keep nominal CB capacity, narrow the transfer/compute extent; cores with 0 units still run startup + `release_ports` |
| Ports | per device, per (direction d ∈ {fwd, bwd}, link l): one port core (reader NCRISC NoC0, sender BRISC NoC1, one fabric connection = one producer per routing plane); line ends keep receive-only / send-only ports |
| Groups | `build_groups(mesh_shape, cluster_axis)` (`fabric_all_gather.py:371`); groups along the other axis run concurrently with identical programs per position |

Regime selection: single built regime (R1). Refinements that add R2–R5 must state the predicate from the Regimes table and add regime-pinned tests.

## Circular Buffers

Compute cores (sizes: `T_q`, `T_kv` per dtype tile; `T_acc` = 4096 B if `fp32_dest_acc_en` else 2048 B; `T_bf` = 2048 B):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_q_block` | 0 | T_q | `q_block_depth·Sq·DHt` | spans q_rows, head_dim | Q dtype | reader | compute | per unit (retained across its key walk) |
| `cb_k_block` | 1 | T_kv | `kv_block_depth·Sk·DHt` | spans keys (block), head_dim; transposed order | K dtype | reader | compute | per key block |
| `cb_v_block` | 2 | T_kv | `kv_block_depth·Sk·DVt` | spans keys (block), v_head_dim | V dtype | reader | compute | per key block |
| `cb_key_mask` | 3 | T_bf | `mask_tiles` = 1 (Phase 0: main tail; +1 joint tail when non-aligned joint is unlocked) | constant | Float16_b | writer | compute | whole program (never popped) |
| `cb_max_scaler` | 4 | T_bf | 1 | constant | Float16_b | writer | compute | whole program |
| `cb_sum_scaler` | 5 | T_bf | 1 | constant | Float16_b | writer | compute | whole program |
| `cb_scores` | 6 | T_acc | `Sq·Sk` | spans q_rows, keys (block) | acc | compute | compute | per key block (S → P in place) |
| `cb_max_a` / `cb_max_b` | 7 / 8 | T_acc | `Sq` each | spans q_rows | acc | compute | compute | ping-pong across key blocks |
| `cb_sum_a` / `cb_sum_b` | 9 / 10 | T_acc | `Sq` each | spans q_rows | acc | compute | compute | ping-pong across key blocks |
| `cb_row_factor` | 11 | T_acc | `Sq` | spans q_rows; holds α = exp(s(m_prev−m_cur)) during the walk and 1/l at finalize | acc | compute | compute | per key block, then finalize |
| `cb_out_acc_a` / `cb_out_acc_b` | 12 / 13 | T_acc | `Sq·DVt` each | spans q_rows, v_head_dim | acc | compute | compute | ping-pong across key blocks |
| `cb_out_tiles` | 16 | T_bf | `out_block_depth·Sq·DVt` | spans q_rows, v_head_dim | Float16_b | compute | writer | per unit |

Port cores:

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_port_chunks` | 0 | page_bytes (K/V dtype tile) | `2·port_batch_chunks·run_pages` | two batches of `group` chunks, a reservation never wraps (`fabric_all_gather.py` reader comment) | K/V dtype | port reader | port sender | whole call |

Sync: every compute CB is pushed and waited in the same quantum, from `key_block_plan` (`Sk_valid·DHt` / `Sk_valid·DVt` per key block, `Sq·DHt` per unit for Q, `Sq·DVt` per unit for output); `cb_port_chunks` follows the example's batch rule (push at `group` chunks, at entry ends, and at every multiple of `group` in the CB).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `startup_block` | 3 tiles | helper (scalers) + raw fill (mask) | — | `cb_max_scaler` 1, `cb_sum_scaler` 1, `cb_key_mask` mask_tiles | resident for the program |
| 2 | `load_q_block` | Sq × DHt | raw dataflow | DRAM Q / JQ | `cb_q_block` Sq·DHt | retained by compute until finalize |
| 3 | `await_landing` | entry | raw dataflow | port `sem_arrival` (remote) | — | cache updated |
| 4 | `load_kv_block` | Sk_valid × (DHt, DVt) | raw dataflow | DRAM K/V / JK/JV / landing | `cb_k_block` Sk_valid·DHt, `cb_v_block` Sk_valid·DVt | — |
| 5 | `score_block` | Sq × Sk_valid (K = DHt) | helper | `cb_q_block` Sq·DHt (retain), `cb_k_block` Sk_valid·DHt (pop) | `cb_scores` Sq·Sk_valid | Q still fronted |
| 6 | `mask_block` | Sq × 1 (last tile column) | helper | `cb_scores` (in place, real credit: pop + push), `cb_key_mask` (no pop) | `cb_scores` | masked scores |
| 7 | `row_max_block` | Sq × Sk_valid → Sq | helper (+ post-op) | `cb_scores` (no pop), `cb_max_prev` Sq (no pop) | `cb_max_cur` Sq | m_prev and m_cur both fronted |
| 8 | `exp_block` | Sq × Sk_valid | helper chain (custom element) | `cb_scores` (in place), `cb_max_cur` (bcast col, no pop) | `cb_scores` Sq·Sk_valid (P), `cb_sum_cur` Sq (row partials, L1-acc) | P fronted |
| 9 | `pv_block` | Sq × DVt (K = Sk_valid) | helper | `cb_scores` (pop), `cb_v_block` (pop) | `cb_out_acc_cur` Sq·DVt | — |
| 10 | `rescale_block` | Sq; Sq × DVt | helper (or SDPA fallback) | `cb_max_prev` (pop), `cb_max_cur`, `cb_sum_prev` (pop), `cb_out_acc_prev` (pop) | `cb_row_factor` Sq (α, popped here), `cb_sum_cur` += α·l_prev, `cb_out_acc_cur` += α·O_prev | prev ⇄ cur swap |
| 11 | `finalize_block` | Sq; Sq × DVt | helper | `cb_sum_cur` (pop), `cb_out_acc_cur` (pop), `cb_max_cur` (pop), `cb_q_block` (pop Sq·DHt) | `cb_row_factor` Sq (1/l), then `cb_out_tiles` Sq·DVt | unit state released |
| 12 | `store_out_block` | Sq_valid × DVt | raw dataflow | `cb_out_tiles` Sq·DVt (pop) | DRAM out / joint_out | — |
| 13 | port blocks | entry, chunks | raw dataflow (copied from the example) | DRAM input / landing | `cb_port_chunks` → fabric | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_block, cb_k_block, cb_scores)` | `matmul_block_helpers.hpp:100-104, 236-237` (boot contract) | once, first statement | — | — | — |
| `score_block` | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-372` | `transpose=true`, `packer_l1_acc=false`, `LastBlockTarget::Out`, `OutputCBLayout::TileRowMajor`, `in0_policy=WaitAndRetainOnLastBlock`, `in1_policy=WaitAndPopPerKBlock`; `MatmulBlockShape::of(.., in0_block_k=DHt, num_k_blocks=1)` (`:139`); interm = out | `cb_q_block`, `cb_k_block` | `cb_scores` | block = Sq × Sk_valid; subblocks from the helper/auto-tuner given `fp32_dest_acc_en` |
| `mask_block` | helper | `compute_kernel_lib::eltwise_chain` with `BinaryFpu<BinaryFpuOp::Add, input(cb_scores, Strided), input(cb_key_mask, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::None)>` + `PackTile` back to `cb_scores` | `eltwise/api/chain.hpp:533`, `:491`, `:504`, `:447` | `IterationShape::grid(Sq, 1)` over tile column `Sk_valid−1` | `cb_scores`, `cb_key_mask` | `cb_scores` | Sq |
| `row_max_block` | helper | `compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores, cb_max_scaler, cb_max_cur, ReduceInputPolicy::WaitUpfrontNoPop>` with `post_reduce_op` = max with the `cb_max_prev` tile (binary max in DEST), skipped on the unit's first key block | `reduce_helpers_compute.hpp:616-637`; `ReduceInputBlockShape::of(Sq, Sk_valid)` `:290` | — | `cb_scores`, `cb_max_prev` | `cb_max_cur` | Sq, Sk_valid |
| `exp_block` | helper | `eltwise_chain(grid(Sq, Sk_valid), BinaryFpu<Sub, input(cb_scores), input(cb_max_cur, BroadcastDim::Col, Upfront, None)>, ScaledExp, PackTile<output(cb_scores)>, PackTile<output(cb_sum_cur, …, L1Accumulation::Enabled)>)` where `ScaledExp` is a custom `UnaryOp` element issuing `exp_tile_init<approx, scale_fp32>` (scale fused as in `compute_common.hpp:312`) | `eltwise/api/chain.hpp:533`, `:272` (L1Accumulation), `:305` (BroadcastDim); `eltwise/unary/math.hpp:22` (`Exp` has no scale parameter) | scale as CT fp32 bits | `cb_scores`, `cb_max_cur` | `cb_scores`, `cb_sum_cur` | Sq, Sk_valid |
| `pv_block` | helper | `compute_kernel_lib::matmul_block` | `matmul_block_helpers.hpp:339-372` | `transpose=false`, `TileRowMajor`, in0/in1 `WaitAndPopPerKBlock`; `MatmulBlockShape::of(.., in0_block_k=Sk_valid, num_k_blocks=1)` | `cb_scores`, `cb_v_block` | `cb_out_acc_cur` | Sq × DVt block, K = Sk_valid |
| `rescale_block` α | helper | `eltwise_chain(tiles(Sq), BinaryFpu<Sub, input(cb_max_prev), input(cb_max_cur, …, PopPolicy::None)>, ScaledExp, PackTile<output(cb_row_factor)>)` | `eltwise/api/chain.hpp:533` | — | `cb_max_prev`, `cb_max_cur` | `cb_row_factor` | Sq |
| `rescale_block` l, O | helper | `mul<input(cb_sum_prev), input(cb_row_factor, BroadcastDim::Col, …, PopPolicy::None), output(cb_sum_cur, …, L1Accumulation::AddToExisting)>` and the same for O over `grid(Sq, DVt)` | `eltwise/api/convenience.hpp:51`, `chain.hpp:272` | — | `cb_sum_prev`, `cb_out_acc_prev`, `cb_row_factor` | `cb_sum_cur`, `cb_out_acc_cur` | Sq, DVt |
| `finalize_block` 1/l | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sum_cur, cb_sum_scaler, cb_row_factor, ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(Sq, 1), …, NoAccumulation{}, [](uint32_t d){ recip_tile_init(); recip_tile(d); })` | `reduce_helpers_compute.hpp:616-637` (documented post-reduce recip example `:592-603`) | — | `cb_sum_cur` | `cb_row_factor` | Sq |
| `finalize_block` normalize | helper | `mul<input(cb_out_acc_cur), input(cb_row_factor, BroadcastDim::Col, …, PopPolicy::AtEnd), output(cb_out_tiles)>(IterationShape::grid(Sq, DVt))` | `eltwise/api/convenience.hpp:51` | — | `cb_out_acc_cur`, `cb_row_factor` | `cb_out_tiles` | Sq, DVt |
| `startup_block` scalers | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_max_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>()` and `<cb_sum_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `reduce_helpers_dataflow.hpp:97-99` | bf16 packed, pool-type-aware | — | scalers | — |
| `startup_block` mask | raw_api | bf16 tile fill: 0 / −inf columns | pattern `transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:332` (`fill_neginf_tile`), `:372` (`fill_vertical_tile_bf16`) | `logical_n mod 32` RT | — | `cb_key_mask` | — |
| `load_*`, `store_out_block` | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_write` + barriers once per block | `tech_reports/tensor_accessor/tensor_accessor.md` | — | — | — | Sq, Sk_valid |
| `await_landing`, `release_ports` | raw_api | `noc_async_read` of the port's `sem_arrival` (4 B) into an L1 scratch word; `noc_semaphore_inc` on `sem_consumers_done` | `fabric_all_gather.py:299-303` (counter semantics) | — | — | — | — |
| port reader / sender / fence | raw_api | `for_each_chunk`, `WorkerToFabricEdmSender::build_from_args`, `route_one_hop`, `to_noc_unicast_write`, `to_noc_fused_unicast_write_atomic_inc`, `send_current_slot_non_blocking`, `to_noc_unicast_atomic_inc`, `noc_semaphore_wait_min` | `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:40-87, 89-170, 184-196, 235, 246-255, 278-284, 299` | `run_pages`, `group`, num_banks CT | DRAM | fabric | `run_pages`, `port_batch_chunks` |
| host transport plan | raw_api | `build_groups`, `_schedule`, `plan` (links via `ttnn.get_forwarding_link_indices`, payload via `ttnn.get_tt_fabric_max_payload_size_bytes`), `probe_ethernet_cores`, `_worker_nearest_noc1`, `ttnn.setup_fabric_connection`, `ttnn.create_global_semaphore` | `fabric_all_gather.py:371, 384, 717-834, 436, 581, 942, 1099` | — | — | — | — |

**Helpers considered and rejected (raw_api rows):**

| Raw entry | Helper considered | File:line of the mismatch | Reason |
|-----------|-------------------|---------------------------|--------|
| port reader/sender | `ccl_helpers_dataflow.hpp` `FabricStreamSender` / `arm_fused_write_inc` | `ccl/ccl_helpers_dataflow.hpp:297-302, 451, 486` | Usable for the egress, but the measured transport (batching ring of pre-routed headers, `send_current_slot_non_blocking` without per-packet flush, bank-run chunks — `blackhole-fabric.md` rules 1, 3, 9) is the example's raw code; copying it keeps the measured rate. The implementer MAY substitute the helper if it reproduces rules 1/3 (no per-packet flush); record the choice. |
| mask fill | `eltwise/generators/fill.hpp` (compute-side fill) | `eltwise/generators/fill.hpp` | The mask is produced once by the writer (dataflow) so compute never spends SFPU time on it; a compute fill would cost a compute phase per program for 1 tile. |
| landing poll | `mcast/kernel/mcast_semaphore.hpp` | — | No multicast in Phase 0; the poll is a 4-byte remote read. |
| `exp_block` (if the chain cannot express the second L1-accumulating pack per row) | `eltwise_chain` | `eltwise/core/chain.inl:3015` (two PackTiles allowed if they do not collide); `chain.inl:2978` | If the per-row L1-accumulated sum target cannot be addressed by row from a `grid(Sq, Sk)` walk, realize `exp_block` as the SDPA block function `sub_exp_block_bcast_cols_inplace` (`compute/compute_common.hpp:294`) — same contract (P in place + row partial sums with L1 acc). Record the gap. |
| `rescale_block` (if `AddToExisting` onto a fronted CB is not expressible) | `mul<>` with `L1Accumulation::AddToExisting` | `eltwise/api/chain.hpp:272-276` | Fallback: `mul_block_bcast_cols` with pack-accumulate (`compute_common.hpp:410`) and `mul_tiles_bcast_cols_inplace` + `add_block_inplace` (`:564`, `:535`). Same contract. |
| `row_max_block` (if `post_reduce_op` cannot read `cb_max_prev`) | `reduce` with `Accumulate` | `reduce_helpers_compute.inl:715-716` (the accumulator reload POPS the accumulator CB) | `Accumulate` would consume m_prev, which the rescale still needs. Fallback: `reduce_c` (`compute_common.hpp:202`, reduce + `binary_max_tile` against prev without popping it). |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| mask_block | add | `cb_scores` All | `cb_key_mask` All (rows identical) | None |
| exp_block | sub | `cb_scores` All | `cb_max_cur` Col0 (REDUCE_ROW out) | Col |
| rescale α | sub | `cb_max_prev` Col0 | `cb_max_cur` Col0 | None (Col0 result valid) |
| rescale l | mul | `cb_sum_prev` All (32-wide row partials) | `cb_row_factor` Col0 | Col |
| rescale O | mul | `cb_out_acc_prev` All | `cb_row_factor` Col0 | Col |
| finalize normalize | mul | `cb_out_acc_cur` All | `cb_row_factor` Col0 (REDUCE_ROW out + recip) | Col |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| NaN from a fully masked first block | online softmax with `m = -inf` | fully padded key tiles are skipped by `key_block_plan`; every processed block has ≥1 valid column |
| reader / compute disagree on ragged key blocks → hang | ragged `Sk_valid` and skipped segments vary per source | one shared `key_block_plan` header used by reader and compute |
| port re-arm while a consumer still polls | compute readers poll the port counter until their last unit | `sem_consumers_done` gate before the port decrements |
| landing buffer overwritten by the next call | upstream writes while this device still reads the previous call | per-call ready fence (`fabric_all_gather.py:240-255`) — a port sends ready only when its program starts |
| relay deadlock on an unpushed tail | own-entry chunks stuck below `group` keep the sender idle around the ring | push partial batches at every entry end (example reader rule) |
| bank walk over a head slice | slice bases are not multiples of the bank count | walk absolute page indices (bank = page mod B), not slice-relative |
| in-place phase ordering on `cb_scores` | mask/exp write back the CB the next phase reads | every in-place phase moves real CB credit (pop + push), `l1-footprint-discipline.md` Rule 3.2 |
| program cache hit with a different RT layout | generic_op ignores per-core RT-arg layout | `custom_program_hash` over all structural inputs (G, topology, num_links, port/compute cores, Sq, Sk, dtypes, shapes, fp32_dest, fidelity) |
| first-call semaphore creation syncs the mesh | trace capture must not contain it | created once per cache key; models warm up before tracing |
| Ring along an axis the fabric does not wrap | the wrap hop has no links | ValueError from link discovery (`get_forwarding_link_indices` empty on the wrap hop) or a non-wrapping config — never a hang |
| 1D fabric configs route by hop count | FABRIC_1D / 1D_RING | `route_one_hop` switches on `ROUTING_MODE` (`fabric_all_gather.py:184-196`) |
| fp32 stats with bf16 output | out is bf16 always | only `cb_out_tiles` is bf16; every intermediate follows `fp32_dest_acc_en` |
| a dynamic landing order breaks determinism | dynamic "whichever landed first" order changes FP sums run to run | fixed block order per unit; waits are blocking, never reordering |

### Structural impossibilities

None found beyond `feature_spec.py`'s INVALID list.

### Refinement queue (verifier input, from this design)

| # | Kind | Refinement | Class |
|---|------|------------|-------|
| 1 | local compute / reuse (enabler for FOCUS) | `fp32_dest_acc_en: False` (bf16 T_acc, Sq 16) — required for the FOCUS cell to run at all | knob-turn |
| 2 | OVERLAP (measure exposed transport per chip on FOCUS first) | landing poll vs mcast notify; increment interval; first-wave ordering | knob-turn |
| 3 | local compute (if the breakdown shows DRAM-bound) | R2 K/V row multicast | scheme-change (deferred row) |
| 4 | features | GQA/MQA/MLA (R3), bfloat8_b, causal / balanced (R4), sharded joint (R5), non-aligned replicated joint (mask tile 2), cross-attention | knob-turns / deferred rows |
