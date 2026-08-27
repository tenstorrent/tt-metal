# Port Plan — nlp_create_qkv_heads_decode

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode`, ported from the
`ProgramDescriptor` API (`ProgramDescriptorFactoryConcept`, `create_descriptor`) to Metal 2.0
(`ProgramSpecFactoryConcept`, `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Context worth knowing:** a v1 port of this op (2026-07-22) capitulated the Sharded and Subcoregrid factories on a
framework bug — a borrowed-memory DFB's device base address was corrupted whenever a node's present-DFB-id set had an
interior hole (the multi-work-unit non-overlap configs hit it). That bug was fixed on `main` by
`3f173de1a13` (*"[Bug fix]: #51409 on splitting dfb id to be program unique and adding a device facing id that is
unique within core group"*), which is why this v2 port targets all three factories. The non-overlap
(2-work-unit, borrowed-output-DFB) configs are the regression canary — watch them specifically in verification.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (all three factories; each defines `create_descriptor` returning
  `tt::tt_metal::ProgramDescriptor`, inside a `program_factory_t` variant — no direct-descriptor shape).
- Variants: three factories on one DeviceOperation, runtime-selected by `select_program_factory`
  (`device/nlp_create_qkv_heads_decode_device_operation.cpp:12-25`):
  - `NLPCreateQKVHeadsDecodeInterleavedProgramFactory` — input not sharded
  - `NLPCreateQKVHeadsDecodeShardedProgramFactory` — width-sharded input, full coregrid
  - `NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory` — width-sharded input, subcoregrids
- Custom `compute_program_hash`: none — default reflection-based hash (no `attribute_values`/`to_hash` backdoor).
- Each factory instantiates ONE kernel source 2× (Reader-config phase-1 + Writer-config phase-2) — or 4× in the
  sharded/subcoregrid non-overlap configs (q pair on `q_cores`, k pair on disjoint `k_cores`). Pure data movement;
  no compute kernels anywhere in the op.

### Variant: Interleaved factory (`nlp_create_qkv_heads_decode_interleaved_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per core) | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | `q_cores` | [0]=element_size, [1]=sub_tile_line_bytes, [2]=c_16, [3]=c_17, [4]=c_18, [5]=head_size, [6]=num_q_heads, [7]=num_kv_heads, [8]=head_tiles, [9]=1 (phase), [10]=use_aligned_path, [11]=dram_alignment, [12]=c_0, [13..]=TensorAccessorArgs(in_buffer) | [0]=in_tile_offset_by_batch, [1]=in_buffer (`Buffer*`) | none | O2 (unset) | ReaderConfigDescriptor{} |
| writer | same source | `q_cores` | same, with [9]=2, [12]=c_1 | same | none | O2 (unset) | WriterConfigDescriptor{} |

No CRTAs, no named CTAs anywhere in the op.

#### CBs
| index | total_size | core_ranges | data_format | page_size | buffer | condition |
|---|---|---|---|---|---|---|
| c_16 (q_out) | q_num_tiles * tile_size | `q_cores` | input dtype | tile_size | borrowed `output[0].buffer()` | always |
| c_17 (k_out) | k_num_tiles * tile_size | `k_cores` (== q grid by construction) | input dtype | tile_size | borrowed `output[1].buffer()` | always |
| c_18 (v_out) | v_num_tiles * tile_size | `v_cores` = **q** grid (factory `:74`) | input dtype | tile_size | borrowed `output[2].buffer()` | always |
| c_0 (reader scratch) | (head_tiles+1) * dram_alignment | `q_cores` | Float16_b (placeholder) | dram_alignment | plain | only `use_aligned_path` (DRAM input && sub_tile_line_bytes < dram_alignment, factory `:98-100`) |
| c_1 (writer scratch) | same | `q_cores` | Float16_b (placeholder) | dram_alignment | plain | only `use_aligned_path` |

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | delivery |
|---|---|---|
| `TensorAccessorArgs(in_buffer).append_to(reader_cta)` (factory `:152`), kernel `TensorAccessor(qkv_args, q_start_addr)` (kernel `:59`) | `input_tensor` | base as `Buffer*` RTA slot 1; **Case 1** (all reads via accessor `.page_id`/`.offset_bytes`) |

#### Work split
n/a — no `split_work_to_cores`; per-core RTA `in_tile_offset_by_batch` computed in a `grid_to_cores(q_cores)` loop
(`i < 16 ? i*sub_tile_line_bytes : (i-16)*sub_tile_line_bytes + 512*element_size`, factory `:181-188`).

### Variant: Sharded factory (`nlp_create_qkv_heads_decode_sharded_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per core) | opt_level | config |
|---|---|---|---|---|---|---|
| q_reader | `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | `q_cores` | [0]=element_size, [1]=sub_tile_line_bytes, [2]=c_16, [3]=c_17, [4]=c_18, [5]=head_size, [6]=num_q_heads, [7]=num_kv_heads, [8]=head_tiles, [9]=1, [10]=num_x, [11]=num_y, [12]=process_qv, [13]=process_k, [14]=use_batch_offset, [15]=index_stick_size, [16]=c_15, [17..]=TensorAccessorArgs(batch_offset_buffer or nullptr) | [0]=in_buffer (`Buffer*`), [1]=batch_offset_buffer (`Buffer*`) or literal 0, [2]=i (core index), [3..3+num_x)=noc_x table, [..+num_y)=noc_y table | O2 (unset) | Reader |
| q_writer | same | `q_cores` | same, [9]=2 | same | O2 | Writer |
| k_reader (only `!overlap_qk_coregrid`) | same | `k_cores` (disjoint from q_cores) | same as q_reader, [12]=0, [13]=1 | same shape (k-core loop) | O2 | Reader |
| k_writer (only `!overlap`) | same | `k_cores` | same as k_reader, [9]=2 | same | O2 | Writer |

process flags: overlap → all instances (2) have process_qv=1, process_k=1. Non-overlap → q pair (1,0), k pair (0,1).

#### CBs
| index | total_size | core_ranges | data_format | page_size | buffer | condition |
|---|---|---|---|---|---|---|
| c_15 (batch-offset, reader idx) | batch-offset tile size | `qk_cores` (q ∪ k) | batch_offset dtype | 1 | plain | only `batch_offset.has_value()` |
| c_14 (batch-offset, writer idx) | same | `qk_cores` | same | 1 | plain | allocated when `batch_offset.has_value()` — **DEAD: no kernel CTA in this factory ever carries c_14** (writer CTA copies override only [9], factory `:200-201`; k copies `:218-220,230-231`) |
| c_16 (q_out) | q_num_tiles * tile_size | `q_cores` | input dtype | tile_size | borrowed `output[0]` | always |
| c_17 (k_out) | k_num_tiles * tile_size | `k_cores` | input dtype | tile_size | borrowed `output[1]` | always |
| c_18 (v_out) | v_num_tiles * tile_size | `v_cores` = output[2] grid (== q grid) | input dtype | tile_size | borrowed `output[2]` | always |

#### Tensor accessors
| host site | originating Tensor | delivery |
|---|---|---|
| `TensorAccessorArgs(batch_offset_buffer).append_to(...)` (factory `:189`), kernel `TensorAccessor(index_args, batch_offset_tensor_addr)` (kernel `:48`) | `batch_offset` (optional) | base as `Buffer*` RTA slot 1 (literal 0 when absent); **Case 1**, conditional on `use_batch_offset` |
| (no accessor) kernel raw walk `qkv_read_addr = q_start_addr + in_tile_offset_by_batch`, remote reads via `UnicastEndpoint {noc_x, noc_y, addr}` (kernel `:66-116`) | `input_tensor` | base as `Buffer*` RTA slot 0; **Case 2** (raw base pointer) |

#### Work split
n/a — per-core RTA `i` (index_in_cores); NoC-coordinate tables identical on every core.

### Variant: Subcoregrid factory (`nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp`)

Same structure as the Sharded factory with these deltas:
- Kernel source: `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp`.
- CTAs: [10]=in_num_cores (single count, replaces num_x/num_y), [11]=process_qv, [12]=process_k,
  [13]=use_batch_offset, [14]=index_stick_size, [15]=batch-offset CB index, [16..]=TensorAccessorArgs.
  **The writer copies override [15]=c_14** (factory `:198,229`) — so here c_14 is live: readers use c_15, writers c_14.
- RTAs: [0]=in_buffer, [1]=batch_offset|0, [2]=i, [3..3+in_num_cores)=noc_x per input core,
  [..+in_num_cores)=noc_y per input core.
- CB census: c_15 touched only by reader instances (1 locked producer per node), c_14 only by writer instances —
  both **self-loops**, conditional on `batch_offset.has_value()`.
- **Anomaly preserved as-is (do not fix):** the factory sizes the V output CB from Q's shard spec —
  `v_shard_spec = output[0].shard_spec()` and `v_cores = q_shard_spec.grid` (factory `:114-116`) while backing it
  with `output[2].buffer()`. Masked today because q/kv head counts both pad to 32. The port reproduces the same
  numeric size (computed from `output[0]`'s shard spec), backed by the V tensor.

### Shared kernels
none — all three kernel sources are op-owned and bound by exactly one factory each; census grep over
`ttnn/cpp/ttnn/operations/` shows no external binders; no `_metal2` forks exist; all three factories convert in this
one port, so no fork rungs apply (audit confirms).

### Flags
- Sharded factory `c_14` is allocated-but-dead (see CBs table) — dropped in the port; the suspected missing writer
  CTA override is the ops team's call, recorded in the port report, NOT fixed here.
- Kernel loops advance the NoC-coordinate cursor after the final tile and re-index the coordinate table one past the
  end (`reader_tm_...cpp:182-191`, `..._on_subcoregrids.cpp:226-231`). Harmless today (fetched value never used).
  With `get_vararg(i)` the same one-past-the-end index is read — `get_vararg` is a read of the vararg L1 block, so
  behavior is unchanged (reads garbage that is never used, exactly as legacy). Preserved, not fixed.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — all three factories
  (no `override_runtime_arguments` anywhere; `Is able to port? == yes` on all three sheet rows; relaxations `none`).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: the three factories port together (one DeviceOperation, one PR). Factory structs keep
  their names; `create_descriptor` is replaced by `create_program_artifacts` in each. No pybind touches — the
  nanobind file binds only the composite user-facing function.

## Planned Spec Shape

Typed name constants are declared **function-locally** in each factory's `create_program_artifacts` (not in an
anonymous namespace) — the three factory `.cpp`s share a unity-build TU and the DFB/tensor names repeat across them.

### Variant: Interleaved

- KernelSpecs: 2 — `reader` / `writer`, same source, differing in CTAs (`PHASES_TO_READ`=1 vs 2), scratch DFB binding
  (READER_SCRATCH vs WRITER_SCRATCH, same accessor name `aligned_scratch`), and hw_config (reader vs writer default).
- DataflowBufferSpecs:
  - `q_out` / `k_out` / `v_out`: entry_size=tile_size, num_entries=<x>_num_tiles, data_format=input format,
    `borrowed_from` the Q/K/V output TensorParameters. Endpoint: reader PRODUCER + writer CONSUMER (**1P+1C**,
    role-free raw-write work split; labels cosmetic on Gen1).
  - `reader_scratch` / `writer_scratch` (conditional on `use_aligned_path`): entry_size=dram_alignment,
    num_entries=head_tiles+1, data_format=Float16_b (placeholder, as legacy). Endpoint: owning kernel bound
    PRODUCER **and** CONSUMER (**self-loop**; sync-free single-toucher scratch).
- SemaphoreSpecs: none.
- TensorParameters: `qkv_in` (input; bound by both kernels, accessor `qkv_in`), `q_out_tensor` / `k_out_tensor` /
  `v_out_tensor` (borrow-only; no kernel TensorBinding — legal, borrowed_from counts as use).
- WorkUnitSpecs: 1 — {reader, writer} on `q_cores`.

### Variant: Sharded

- KernelSpecs: 2 (overlap) or 4 (non-overlap) — q_reader/q_writer on `q_cores`, k_reader/k_writer on `k_cores`.
  Per-instance CTAs (`PHASES_TO_READ`) and defines (`PROCESS_QV` / `PROCESS_K` / `USE_BATCH_OFFSET`).
- DataflowBufferSpecs:
  - `q_out` / `v_out`: borrowed from Q/V outputs; bound by the q pair only (q_reader PRODUCER, q_writer CONSUMER).
  - `k_out`: borrowed from K output; bound by the pair that processes K — overlap: q pair; non-overlap: k pair.
    (1P+1C in every config.)
  - `batch_offset` (conditional on `batch_offset.has_value()`): entry_size=1, num_entries=batch-offset tile size,
    data_format=batch-offset format. **Multi-binding**: every instance (2 or 4) is a locked FIFO producer
    (`reserve_back(1)`/`push_back(1)` on the same per-node instance) with no consumer anywhere →
    `advanced_options.allow_instance_multi_binding = true`, and each instance binds PRODUCER **and** CONSUMER.
    The P+C-per-instance shape is forced by the validator's self-loop set-equality rule
    (`program_spec.cpp:1503`: once any kernel binds both roles, producer set must equal consumer set) combined with
    the per-node census under the flag (≥1 producer AND ≥1 consumer per node, `program_spec.cpp:1426` — two locked
    producers alone leave the consumer side empty). Faithful: each instance genuinely fills and reads back its page.
    NOT a hidden co-fill/consumer — see the brief's Watch-for; do not give the writer its own DFB (ops-team change).
  - Sharded `c_14`: **dropped** (dead — zero endpoints in every config of this factory).
- SemaphoreSpecs: none.
- TensorParameters: `qkv_in` (Case 2 — bound by every instance; kernel pulls the base via
  `TensorAccessor(tensor::qkv_in).get_bank_base_address()` and keeps the raw shard-walk),
  `batch_offset_tensor` (conditional; bound by every instance when present), Q/K/V outputs (borrow-only).
- WorkUnitSpecs: 1 (overlap: {q_reader, q_writer} on q_cores) or 2 (non-overlap: + {k_reader, k_writer} on k_cores).

### Variant: Subcoregrid

Same as Sharded, except:
- Kernel source is the subcoregrid kernel; CTA `in_num_cores` replaces `num_x`/`num_y`.
- Batch-offset DFBs (conditional): `batch_offset_reader` self-looped by reader instances (accessor `batch_offset`),
  `batch_offset_writer` self-looped by writer instances (same accessor name `batch_offset`, mirroring the legacy
  CTA[15] override). In non-overlap, q_reader and k_reader both self-loop `batch_offset_reader` over disjoint node
  sets (producer set == consumer set — legal), likewise the writers. No multi-binding flag needed (one toucher/node).
- V output DFB sized from `output[0]`'s shard spec (legacy anomaly, preserved), borrowed from the V tensor.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| Interleaved reader+writer (1 source × 2 configs, both on q_cores) | reader, writer | wu_main (q_cores) | q_out/k_out/v_out: reader=PRODUCER, writer=CONSUMER; scratch: each instance self-loops its own DFB |
| Sharded q_reader+q_writer (q_cores) | q_reader, q_writer | wu_q (q_cores) | q_out/v_out (+k_out in overlap): reader=P, writer=C; batch_offset: both P+C (flag) |
| Sharded k_reader+k_writer (k_cores, non-overlap only) | k_reader, k_writer | wu_k (k_cores) | k_out: k_reader=P, k_writer=C; batch_offset: both P+C (flag) |
| Subcoregrid — same scheme as Sharded | q/k reader/writer | wu_q, wu_k | outputs as above; batch_offset_reader self-looped by readers, batch_offset_writer by writers |

Both instances of each pair cover **every** node of their grid (dual-instance work-split by tile phase, not disjoint
node sets) — this is the two-toucher 1P+1C pattern, NOT the demoting-per-group-CTA case and NOT (for the output CBs)
the multi-binding flag.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| interleaved factory `:186-187` RTA slot 1 | `in_buffer` (`Buffer*`) | `TensorBinding{qkv_in}` (Case 1) |
| interleaved factory `:152` | `TensorAccessorArgs(in_buffer).append_to(cta)` + kernel `TensorAccessorArgs<13>()` | binding mechanism end-to-end; kernel `TensorAccessor(tensor::qkv_in)` |
| interleaved CTA[2..4] | c_16/c_17/c_18 magic CB indices | `DFBBinding` q_out/k_out/v_out |
| interleaved CTA[10] | `use_aligned_path` flag CTA | `compiler_options.defines` `USE_ALIGNED_PATH` + kernel `#ifdef` |
| interleaved CTA[12] | scratch CB index (c_0 / c_1 per instance) | per-instance `DFBBinding` (READER_SCRATCH / WRITER_SCRATCH, one accessor name) |
| sharded/sub factory RTA slot 0 | `in_buffer` (`Buffer*`) | `TensorBinding{qkv_in}` (**Case 2**: kernel `TensorAccessor(tensor::qkv_in).get_bank_base_address()`, raw walk unchanged) |
| sharded/sub factory RTA slot 1 | `batch_offset_buffer` (`Buffer*`) or literal 0 (`push_batch_offset` lambda) | conditional `TensorBinding{batch_offset_tensor}`; absent path binds nothing |
| sharded factory `:189` / sub `:185` | `TensorAccessorArgs(batch_offset_buffer or nullptr).append_to(cta)` + kernel `TensorAccessorArgs<17|16>()` | binding mechanism; kernel `TensorAccessor(tensor::batch_offset_tensor)` under `#ifdef USE_BATCH_OFFSET` |
| sharded CTA[12]/[13], sub CTA[11]/[12] | `process_qv` / `process_k` flag CTAs | defines `PROCESS_QV` / `PROCESS_K` + kernel `#ifdef` (they gate DFB/name bindings) |
| sharded CTA[14], sub CTA[13] | `use_batch_offset` flag CTA | define `USE_BATCH_OFFSET` + kernel `#ifdef` |
| sharded CTA[16], sub CTA[15] | batch-offset CB index | `DFBBinding` (accessor `batch_offset`) |
| sharded factory `:79-88` | dead `c_14` CBDescriptor | dropped (no spec; no CTA carried it) |
| all positional CTAs | positional list | named CTAs: `ELEMENT_SIZE`, `SUBTILE_LINE_BYTES`, `head_size`, `num_q_heads`, `num_kv_heads`, `head_size_num_tiles`, `PHASES_TO_READ`, `DRAM_ALIGN_BYTES` (interleaved), `num_x`/`num_y` (sharded), `in_num_cores` (sub), `index_stick_size` (names match the kernel variables they land in) |
| all positional RTAs | positional list | named RTAs `in_tile_offset_by_batch` (interleaved), `index_in_cores` (sharded/sub) + **runtime varargs** for the NoC-coordinate tables (below) |

**Retained varargs (genuine indexed collections):** the sharded/subcoregrid NoC-coordinate tables — CTA-bounded
variable-count blocks read with runtime indices (`in0_mcast_noc_x[qkv_x]` etc., counts `num_x`+`num_y` /
`2*in_num_cores`). Kernel reads become `get_vararg(idx)`. The three leading scalars do NOT ride the varargs:
`q_start_addr` and `batch_offset_tensor_addr` dissolve into tensor bindings; `index_in_cores` is a named RTA.
Interleaved kernel: no varargs.

## Applied Patterns

- Two-toucher DFB → assign 1P+1C (dual-instance work-split), per the patterns catalog: all q/k/v output DFBs,
  every factory.
- Sync-free / single-ended CB → self-loop DFB: interleaved scratch DFBs; subcoregrid batch-offset DFBs.
- Multi-binding advanced option: sharded `batch_offset` DFB only (2 locked producers per node; each instance bound
  P+C to satisfy the census + self-loop set-equality; `allow_instance_multi_binding = true`).
- Conditional / optional DFB & tensor bindings via defines: `USE_ALIGNED_PATH`, `USE_BATCH_OFFSET`,
  `PROCESS_QV`/`PROCESS_K` (per-instance output-DFB gating in non-overlap mode).
- Multi-variant factories: unchanged three-factory variant, each converts in place.
- Unity-build hygiene: spec-name constants declared function-locally per factory.
- Caution — avoid varargs: applied; only the two genuine coordinate tables remain varargs.

## Deferred / Flagged

- **Sharded `c_15` shape**: the brief prescribes "the flag"; the validator additionally forces each producer instance
  to also bind CONSUMER (census requires ≥1 consumer per node even under the flag; the self-loop set-equality check
  `program_spec.cpp:1503` then requires producer set == consumer set). Planned shape: every instance P+C + flag.
  Recorded for the report (brief/recipe don't spell out the census interaction).
- **Borrowed-DFB multi-WU regression watch**: non-overlap sharded/subcoregrid configs re-exercise the v1-port
  framework bug fixed by `3f173de1a13`. Verification must include `overlap_coregrid=False` tests explicitly.
- The one-past-the-end vararg read (Flags above) — preserved behavior, noted for the ops team.
