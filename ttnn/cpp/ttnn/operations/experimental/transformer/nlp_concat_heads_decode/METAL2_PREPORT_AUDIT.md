# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`

- **`NLPConcatHeadsDecodeDeviceOperation`** (`device/nlp_concat_heads_decode_device_operation.hpp`) — single DeviceOperation, two factories selected by `operation_attributes_t::on_subcoregrids`:
  - `NLPConcatHeadsDecodeProgramFactory` (`device/nlp_concat_heads_decode_program_factory.cpp`) — default full-grid path
  - `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp`) — sub-core-grid path

Kernels (both op-owned, both referenced; no unreferenced kernel files in the directory):

- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp` — used only by `NLPConcatHeadsDecodeProgramFactory`, **instantiated twice** per program (Reader config + Writer config, same source, CTA index 6 selects phase 1 vs phase 2)
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` — used only by `NLPConcatHeadsDecodeSubcoregridsProgramFactory`, same dual-instantiation pattern

No other op instantiates either kernel file (repo-wide grep, `experimental/quasar/**` excluded per recipe); no `_metal2` fork exists beside either. There is no quasar copy of this op.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** *not pinnable* — `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing in this checkout (the `metal_2.0/` docs tree is untracked here), so the doc version cannot be pinned to a commit.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `NLPConcatHeadsDecodeDeviceOperation` → `NLPConcatHeadsDecodeProgramFactory`, `NLPConcatHeadsDecodeSubcoregridsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — both kernels fully Device 2.0 (`Noc`, `CircularBuffer` wrapper, `UnicastEndpoint`, `CoreLocalMem`); zero holdovers |
| *Prereqs* — Cross-op escapes | **Ok** — kernel includes are all `tt_metal/*` (class 1); no cross-op donors, no borrowed kernel files |
| *Feature Support* — overall | **GREEN** — all Appendix A rows N/A |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at fixed constexpr indices |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`; sheet `Execution Model` = `SPMD`) |
| *TTNN Readiness* — Custom hash | No (sheet `no`/`no` incl. backdoor; grep confirms no `compute_program_hash` / `attribute_values` / `to_hash`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; grep of device-op confirms) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `no`; grep confirms) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `nlp_concat_heads_decode_nanobind.cpp` binds only the public op function |
| *TTNN Readiness* — Op-owned tensors | No (sheet blank; `create_descriptor` returns `ProgramDescriptor`, no `buffers` vector) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (both factories; matches sheet `Porting Target`) |
| *Port work* — Offset base pointer | **none** — clean base + separate scalar offset RTA, added kernel-side (already-split shape) |
| *Port work* — Tensor bindings (per binding) | `input` → **Case 2** (raw pointer; `Buffer*`-binding delivery) · `output` → **clean** (borrowed-memory CB → `borrowed_from`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (both rows, verbatim) — clears |
| *Port work* — TensorAccessor 3rd arg | none — no `TensorAccessor` exists anywhere in the op, so no accessor passes a 3rd arg |
| *Port work* — CB endpoints | **1P+1C** on the single CB (`c_16`) in both factories (two role-free raw-writer touchers — dual-instance work-split) |

**CB endpoints** are dispositions, not gates (see `audit/metal2_audit.md` → CB endpoints): the one CB here takes a **1P+1C assignment** — two touchers, both sync-free raw writers, so bind one instance PRODUCER and the other CONSUMER (cosmetic on Gen1). No self-loops, no multi-binding flag, no dead CBs.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, same directory). All five gate-bearing subjects clear for both factories: Device 2.0 ✓, Feature compatibility ✓ (all N/A), TTNN factory concept ✓ (`Is able to port? == yes`, both rows), Offset base pointers ✓ (already-split shape), TensorAccessor 3rd arg ✓ (no sites). Port targets `ProgramSpecFactoryConcept` for both factories.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet fetched fresh this session (Diego's *"Operations analysis"*); two rows for `experimental/transformer/nlp_concat_heads_decode`, one per factory. Both rows: `Is able to port?` = `yes`, `Concept` = `descriptor`, `Op Classification` = `PD Op (pointer-patching)`, `Known op issues` empty, `Smuggled pointer` = `no`, `TensorParameter relaxation` = `none`, `Pointer patching perf issue?` = `OK`. Cross-check against code (all clean, sheet agrees on every checked column):
  - `Concept`: both factories define `static ProgramDescriptor create_descriptor(...)` (`device/nlp_concat_heads_decode_program_factory.cpp:17`, `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp:18`) → `descriptor` ✓
  - `Custom hash`: no `compute_program_hash` / `attribute_values` / `to_hash` anywhere in the op directory ✓
  - `get_dynamic_runtime_args`: absent from the device-op ✓
  - `override_runtime_arguments`: absent ✓
  - `Pybind descriptor`: `nlp_concat_heads_decode_nanobind.cpp` binds only the public `ttnn::experimental::nlp_concat_heads_decode` function; no `create_descriptor` binding, no `nb::class_` of internals ✓
  - Factory-set match: 2 sheet rows ↔ 2 factories in `program_factory_t` variant (`device/nlp_concat_heads_decode_device_operation.hpp:20-21`); no phantom or missing rows ✓
  - Cross-column invariants: none violated ✓
  - Note: the sheet's `Smuggled pointer` = `no` is consistent with the code — the factories push the **`Buffer*` object** (not `->address()`) into `KernelDescriptor::RTArgList` (`nlp_concat_heads_decode_program_factory.cpp:130`, `..._subcoregrids_program_factory.cpp:137`), the annotated `BufferBinding` form the framework patches on cache hits. Matches `Op Classification` = `PD Op (pointer-patching)`.
- **Device 2.0 (every kernel used):** GREEN — no violations table needed. Both kernels are structurally Device 2.0 throughout: `Noc noc` + `noc.async_read(src_ep, CoreLocalMem<uint32_t>(...), size, {.noc_x, .noc_y, .addr}, {})` + `noc.async_read_barrier()`; `CircularBuffer cb_q_out(cb_id_q_out)` wrapper with the `cb_q_out.get_write_ptr()` **method** (not the free function); `UnicastEndpoint`. No `InterleavedAddrGen*`/`ShardedAddrGen`, no raw `noc_async_read` free functions, no CB-index-keyed free-function holdovers of any kind (the sanctioned `get_tile_size(cb_id)` / `get_local_cb_interface(cb_id)` do not appear either). `get_arg_val` / `get_arg_addr` / `get_compile_time_arg_val` are arg-reading APIs, not CB-index-keyed free functions — not violations.
- **Feature compatibility:** all Appendix A entries scanned against host + kernel code in both factories:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no type reference, no `global_circular_buffer` field on either `CBDescriptor`, no remote-CB idioms |
  | CBDescriptor `address_offset` (non-zero) | N/A | field never set (default 0) on the single `CBDescriptor` in each factory |
  | GlobalSemaphore | N/A | no reference; op uses no semaphores at all |

- **CB endpoints (GATE-free):** one CB per factory — `c_16` (`q_output`), borrowed-memory (`.buffer = output.buffer()` @ `nlp_concat_heads_decode_program_factory.cpp:54`, `..._subcoregrids_program_factory.cpp:64`), allocated over `q_cores` (the output shard grid). Census per node, per factory: **two distinct touchers** — the Reader-config and Writer-config instances of the same kernel source, both over the full `q_cores` range (dual-instance work-split, face (c) of the multi-binding hunt). Both touchers are **role-free**: each only raw-writes via `cb_q_out.get_write_ptr()` + offset as the NoC-read destination (kernel line 49 resp. 48); grep confirms **zero FIFO ops** (`reserve_back`/`push_back`/`wait_front`/`pop_front`) and no `evil_set_*` cursor drivers in either kernel. Nothing drains the CB — the output is resident (borrowed from the output buffer). Disposition: **1P+1C** — bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1). Same census in every config of both factories (the phase CTA changes *which bytes* each instance writes, never the toucher set). No dead CBs, no self-loops, no multi-binding flag needed. Hidden-second-writer hunt (face (a)): the second writer here is fully visible (the dual instance) and there is no semaphore-gated co-fill — no semaphores exist in the op.
- **Offset base pointers:** GREEN — the already-split shape. The only address-bearing arg is the input buffer, delivered as a **`Buffer*` binding** (clean base; the framework injects `->address()`). The head offset is a **separate scalar RTA** (`in_tile_offset_by_batch`, computed host-side @ `nlp_concat_heads_decode_program_factory.cpp:119-124` / `..._subcoregrids_program_factory.cpp:127-131`) that the kernel adds itself (`qkv_read_addr = q_start_addr + in_tile_offset_by_head`, kernel line 45 resp. 44). No host arithmetic is ever folded into the base. Output side is a borrowed-memory CB at base (no `address_offset`). Op is **not** in the offset-base-pointer triage doc's tables (`2026-07-19_offset_base_pointers.md`) — reconciliation outcome: *no fold, op not in the tables → clean*. No Type 3 (`address_offset` N/A above); no Type 4 (`ttnn::narrow` not used).
- **TensorAccessor 3rd argument:** N/A — **no `TensorAccessor` is constructed anywhere in this op** (host or kernel), so no accessor passes a 3rd argument and the subject never fires. (This is the *no sites* finding, not "sites classified redundant".) Op is not in the 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`) — consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, identical classification in both factories):
  - `input` — **Case 2** (raw pointer). Delivery today: `Buffer* in_buffer` pushed into `RTArgList` (`nlp_concat_heads_decode_program_factory.cpp:57,130`; `..._subcoregrids_program_factory.cpp:67,137`) → framework `BufferBinding` (correct-on-cache-hit today, so routine port work, not a live hazard). Kernel consumption: `q_start_addr` (arg 1) used **raw** — hand-rolled NoC address assembly `{.noc_x, .noc_y, .addr = qkv_read_addr}` against remote input-shard cores; no `TensorAccessor` ever constructed. Port: bind as `TensorParameter`/`TensorBinding`; kernel pulls the base via the sanctioned `TensorAccessor::get_bank_base_address` bridge and keeps the existing raw walk unchanged.
  - `output` — **clean** (borrowed-memory DFB). `CBDescriptor{.buffer = output.buffer()}` → `DataflowBufferSpec::borrowed_from` the output `TensorParameter`; kernel keeps its `get_write_ptr()`-based access. Legality resolved under CB endpoints (1P+1C).
- **TensorParameter relaxation:** `none` (both sheet rows, verbatim).
- **TensorAccessor 3rd arg:** none — no sites.
- **CB endpoints:** 1P+1C assign `(c_16 q_output, both factories, all configs)` — two role-free raw-writer touchers (Reader-config + Writer-config instance of the same kernel source); bind one PRODUCER, one CONSUMER. No other CBs exist.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-toucher shape to watch):** `c_16` is the dual-instance work-split (face (c)) — same `kernel_source` in two `KernelDescriptor`s differing only by `ReaderConfigDescriptor`/`WriterConfigDescriptor` and CTA index 6 (phase 1 vs 2), both over `q_cores`. Resolves to **1P+1C**, *not* the multi-binding flag (only two touchers; no third kernel, no FIFO-role doubling).
- **Cross-op / shared kernels:** none — both kernel files are op-owned and single-consumer (repo-wide grep); no `_metal2` fork exists; this port would create nothing shared.
- **RTA varargs (genuine, both kernels):** the NoC-coordinate blocks are a variable-count vararg block — args `2 .. 2+num_x+num_y` (default kernel) / `2 .. 2+2*in_num_cores` (subcoregrid kernel), read via a raw L1 pointer walk `(tt_l1_ptr uint32_t*)(get_arg_addr(2))` with CTA-driven counts that vary across instantiations (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp:31-32`, `..._subcoregrid.cpp:31-32`). No per-element names exist → port via the RTA vararg mechanism; the `get_arg_addr` pointer walk must become `get_vararg`-style indexing. The two leading scalars are **nameable** and must not ride the varargs: arg 0 `in_tile_offset_by_head` (named RTA, per-core value) and arg 1 `q_start_addr` (disappears into the input tensor binding). Note: the coord blocks are *identical across cores* (host builds them once, `nlp_concat_heads_decode_program_factory.cpp:69-78`), so they are CRTA-vararg candidates, while arg 0 is genuinely per-core.
- **CTA plumbing that dissolves in the port:** CTA 2 is the CB index (`q_output_cb_index`) — replaced by the `dfb::` token binding; CTA 6 is the phase selector, set positionally on a copied vector (`writer_compile_time_args[6] = 2` @ `nlp_concat_heads_decode_program_factory.cpp:103` / `..._subcoregrids_program_factory.cpp:110`) — becomes a named CTA per instance.
- **Reader and writer instances receive byte-identical RTA lists** (the factories build one `rt_args` per core and emplace it into both descriptors) — the port can keep that symmetry; only the phase CTA differs.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. Kernel includes (identical in both kernels): `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h` — all bucket 1 (`tt_metal/*`), no concern. No function-call escapes into other ops' helpers; no shared-pool or cross-family donor functions. Borrowed kernel files: none — the factory instantiates only the op's own two kernels; neither is instantiated by any other op; no `_metal2` fork exists beside either. Per-call detail omitted (all ✓).
- **Relaxation candidates:** none — the op has no custom hash to mine.
- **TTNN factory analysis:** op-owned tensors: none (plain `ProgramDescriptor` return). MeshWorkload need: none. Pybind: public op function only (`nlp_concat_heads_decode_nanobind.cpp:18-31`); no descriptor/internal pybind. Custom hash: none. `get_dynamic_runtime_args`: none. `override_runtime_arguments`: none. Target concept: `ProgramSpecFactoryConcept`, both factories. `preallocated_output` is an ordinary optional output tensor arg (`device/nlp_concat_heads_decode_device_operation_types.hpp:19`), returned pass-through by `create_output_tensors` — not an op-owned tensor.

## Misc anomalies  *(team-only, non-gating)*

- **Dead shadowed local:** `uint32_t q_write_addr = 0;` at `reader_tm_tile_layout_nlp_concat_heads_decode.cpp:47` and `..._subcoregrid.cpp:46` is immediately shadowed by the loop-local `uint32_t q_write_addr` (line 53 resp. 52); the outer variable is dead (and the inner `q_write_addr += tile_size` at the loop tail updates the shadowing copy that is re-derived next batch iteration — intentional, but the dead outer declaration invites misreading).
- **`memory_config` accepted and ignored end-to-end:** the nanobind arg (`nlp_concat_heads_decode_nanobind.cpp:28`) is plumbed through `ttnn::experimental::nlp_concat_heads_decode` into `ttnn::prim::nlp_concat_heads_decode`, where it is discarded (`device/nlp_concat_heads_decode_device_operation.cpp:128`, parameter commented out); the output memory config is always derived in `compute_output_specs`. Silent no-op for callers who pass it.
- **Tile-geometry generality split:** the default factory/kernel hardcode 32×32-tile face geometry (`16`, `512 * element_size`, `256 * ELEMENT_SIZE` — `nlp_concat_heads_decode_program_factory.cpp:121-123`, kernel lines 52/69/71), while the subcoregrids pair derives everything from the tensor's tile/face shape (`..._subcoregrids_program_factory.cpp:34-48`). Consistent behavior for the enforced TILE layout, but the two paths would diverge for non-32 tile shapes.
- **Default kernel assumes a dense row-major input grid:** the default factory passes per-axis NoC coordinate vectors built from the input grid's bounding box (`nlp_concat_heads_decode_program_factory.cpp:66-78`), which assumes the input shard grid fills its bounding box row-major from (0,0) — exactly the limitation the subcoregrids factory exists to lift (the host-side `on_subcoregrids` selection at `device/nlp_concat_heads_decode_device_operation.cpp:134-140` guards it).

## Questions for the user

*(none)*

## Recipe notes

- **Provenance line unavailable:** `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing — the `metal_2.0` docs tree is untracked (`??` in git status) in this checkout, so the audit records that instead of a hash, per the recipe's fallback instruction.
