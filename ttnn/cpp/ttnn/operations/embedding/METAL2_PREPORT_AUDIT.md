# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/embedding`

One device operation, three program factories:

- **`ttnn::prim::EmbeddingsDeviceOperation`** (`device/embedding_device_operation.hpp:19`)
  - `EmbeddingsFusedProgramFactory` (`device/embeddings_fused_program_factory.cpp`) — **BLOCKED**
  - `EmbeddingsRMProgramFactory` (`device/embeddings_rm_program_factory.cpp`) — clear
  - `EmbeddingsTilizedIndicesProgramFactory` (`device/embeddings_tilized_indices_program_factory.cpp`) — clear

Factory selection (`device/embedding_device_operation.cpp:17-26`): TILE-layout indices → `EmbeddingsTilizedIndicesProgramFactory`; else `tilized` attribute → `EmbeddingsFusedProgramFactory`; else `EmbeddingsRMProgramFactory`.

Every kernel file in `device/kernels/` is referenced by a factory — there is no dead kernel code in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `0efcf9f88ae 2026-08-17 docs(metal_2.0): CTA varargs are in, and five columns read present-tense`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/embedding` |
| **Overall** | **RED at op level; subset `EmbeddingsRMProgramFactory` + `EmbeddingsTilizedIndicesProgramFactory` is clear** |
| **DOps / Factories** | `EmbeddingsDeviceOperation` → `EmbeddingsFusedProgramFactory`, `EmbeddingsRMProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all ten kernels/headers the op exercises are structurally Device 2.0; no holdovers |
| *Prereqs* — Cross-op escapes | Ok — no function-call escape outside `tt_metal/*`; two borrowed kernel files (detail below) |
| *Feature Support* — overall | GREEN — every Appendix A entry is `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no kernel reads a compile-time arg at a varying index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, on all three factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three rows) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No — sheet `no`; no `compute_program_hash` / `attribute_values` / `to_hash` anywhere in the op directory |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no`; no such hook on the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — sheet `no`; no such method on any factory |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `embedding_nanobind.cpp` binds only `ttnn::embedding` and the `EmbeddingsType` enum |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (all three factories) |
| *Port work* — Offset base pointer | **GATE** → ops team + framework/Audrey, flag early — **Type 2 (accessor-fed offset)** in `EmbeddingsFusedProgramFactory` |
| *Port work* — Tensor bindings (per binding) | Case 1 on `input` / `weights` / `output` in every factory, except `output` in the RM factory's height-sharded config, which is clean (borrowed-memory DFB) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) on all three rows |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) — one site, `embeddings_rm_writer_chunked.cpp:26` |
| *Port work* — CB endpoints | legal / self-loop / conditional DFB — no dead CB, no multi-binding |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution — a **self-loop** (one toucher), a **1P+1C assignment** (two touchers), the **multi-binding advanced-option flag** (a census the roles can't fit), or a **dead-CB drop** (zero endpoints). Recorded per `(CB, config)` below.

## Result

**RED at op level; subset `EmbeddingsRMProgramFactory` + `EmbeddingsTilizedIndicesProgramFactory` is clear.**

One blocker, and it is confined to a single factory. `EmbeddingsFusedProgramFactory`'s reader kernel builds its weights `TensorAccessor` on a base that is **not** the weights tensor's base — it is `weights_base + weight_offset`, where `weight_offset` is a per-core runtime argument the host computes from the output shard width. Metal 2.0 builds the accessor straight from the `tensor::weights` binding token, and that token delivers the base address only; there is no place to inject an offset base. This is the **Type 2 (accessor-fed offset)** offset-base-pointer wall. It routes to the **ops team**, and — because the resolution is a design question rather than a mechanical arg split — should be **flagged early to framework/Audrey**.

Everything else clears. Device 2.0 is complete across all ten kernels and headers the op exercises. No Appendix A feature is in use. All three factory rows on the readiness sheet read `Is able to port? = yes`, with `Concept = descriptor`, `TensorParameter relaxation = none`, and no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound `create_descriptor` — every one of which I confirmed against the code.

So the two remaining factories can be ported now, and `METAL2_PORT_BRIEF.md` is issued scoped to them. That covers the row-major output path and the TILE-layout-indices path; the fused tilize-in-the-op path waits on the ops team.

**Path forward for the RED.** This is an op-readiness prerequisite, not a missing Metal 2.0 feature. Once the ops team resolves how `EmbeddingsFusedProgramFactory` addresses a column slice of the weights table without pre-offsetting the accessor base, the factory comes back for a cheap re-audit. Two candidate remedies are sketched under *Gate detail* below — both are auditor observations for the owning teams to evaluate, not instructions, and neither is porter work.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All three factory rows read `yes`. Cross-check against the code came back clean on every primary column:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | each factory defines `static ProgramDescriptor create_descriptor(...)` — `embeddings_fused_program_factory.hpp`, `embeddings_rm_program_factory.hpp:13`, `embeddings_tilized_indices_program_factory.hpp` |
  | `Custom hash` | `no` | no `compute_program_hash` in the op directory |
  | `Backdoor custom hash` | `no` | no `attribute_values` / `to_hash` in the op directory |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on `EmbeddingsDeviceOperation` (`device/embedding_device_operation.hpp:30-37` declares only `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`) |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` on any factory |
  | `Pybind descriptor` | `no` | `embedding_nanobind.cpp:40-52` binds `ttnn::embedding` only |
  | `Smuggled pointer` | `no` | consistent — the factories push `Buffer*` objects into `RTArgList`, never `->address()` (see *Tensor bindings*) |
  | Factory-set match | 3 rows | 3 factories in `program_factory_t` (`device/embedding_device_operation.hpp:24-28`) — one-to-one, no phantom or missing row |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on `descriptor` rows, and `Op-owned tensors?` is empty on `descriptor` rows.

- **Device 2.0 (every kernel used):** **GREEN.** All ten files are structurally Device 2.0 — `Noc`, `CircularBuffer` / `DataflowBuffer`, `CoreLocalMem<T>`, `UnicastEndpoint`, and the object-method `noc.async_read` / `noc.async_write` forms. A scan for legacy idioms (`noc_async_read`, `noc_async_write`, `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, `get_noc_addr_from_bank_id`, raw semaphore addresses) returned **zero hits** across all ten. There are no CB-index free-function holdovers: every `get_write_ptr` / `get_read_ptr` in the op is a wrapper **method** call on a `CircularBuffer` object.

  | File | Owner | Device 2.0 |
  |---|---|---|
  | `device/kernels/dataflow/embeddings.cpp` | embedding (RM reader) | ✓ |
  | `device/kernels/dataflow/embeddings_tilize.cpp` | embedding (fused reader) | ✓ |
  | `device/kernels/dataflow/embedding_ind_tilized.cpp` | embedding (TILE-indices reader) | ✓ |
  | `device/kernels/dataflow/embeddings_rm_writer_chunked.cpp` | embedding (RM chunked writer) | ✓ |
  | `device/kernels/dataflow/embeddings_common.hpp` | embedding (shared header, all three readers) | ✓ |
  | `device/kernels/compute/tilize_chunked.cpp` | embedding (fused chunked compute) | ✓ |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared pool `ttnn/cpp/ttnn/kernel/` | ✓ |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` (fused writer) | ✓ — already on `DataflowBuffer` |
  | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | `data_movement/tilize` (fused non-chunked compute) | ✓ |
  | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | official shared kernel library | ✓ |

  Two call shapes I checked and am **not** flagging:
  - `writer_unary_interleaved_start_id.cpp:24` — `get_local_cb_interface(cb_id_out).fifo_page_size`. This is one of the two **sanctioned** CB-index free functions; sanctioned means sanctioned even though the kernel holds a `DataflowBuffer` that exposes its own metadata accessors. A port moves the lookup onto the object; that is a port-stage change, not a Device 2.0 gap.
  - `embeddings_common.hpp:68-74` and `:85-92` — `my_x[noc_id]` / `my_y[noc_id]` used to build a `UnicastEndpoint` pointing at the core's own L1. The Device 2.0 migration guide uses exactly this shape in its own migrated examples (`device_api_migration_guide.md:521`), so it is the Device 2.0 idiom for self-addressing, not a holdover.

- **Feature compatibility:** every Appendix A entry is `N/A` — the op uses none of them. Greps across the whole op directory for `GlobalCircularBuffer`, `global_circular_buffer`, `CreateGlobalCircularBuffer`, `remote_index`, `GlobalSemaphore`, `global_semaphore`, `CreateGlobalSemaphore`, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, and `cb_descriptor_from_sharded_tensor` all returned zero hits. The op declares **no semaphores at all** — plain or global.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `CBDescriptor` in the op sets `.global_circular_buffer`; the two Buffer-backed CBs use the plain `.buffer` field (the ordinary borrowed-memory pattern) |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `CBDescriptor` in the op sets `.address_offset` at all, so it defaults to zero |
  | GlobalSemaphore | N/A | the op declares no semaphores of any kind |

- **CB endpoints (GATE-free):** every CB is legal or carries a straightforward disposition. **No dead CB, no multi-binding, and no hidden second writer** — the latter is structurally ruled out here, because a raw semaphore-gated co-fill needs semaphores and this op has none. Census below is per `(CB, config)` per node; run for the clean subset, as the Red scoping rule directs.

  **`EmbeddingsRMProgramFactory`** — reader is `embeddings.cpp` on every core; a writer exists only when the output is interleaved.

  | CB | Config | Touchers on a node | Verdict | Resolution |
  |---|---|---|---|---|
  | `c_0` (output staging) | interleaved, non-chunked | reader locked producer (`embeddings.cpp:71,77`) + `writer_unary_stick_layout_interleaved_start_id.cpp` locked consumer (`:32,35`) | plain 1:1 | legal — no action |
  | `c_0` (output staging) | interleaved, chunked | reader locked producer + `embeddings_rm_writer_chunked.cpp` locked consumer (`:33,43`); its `get_read_ptr()` at `:34` is a peek on its own consumer binding, not a third endpoint | plain 1:1 | legal — no action |
  | `c_0` (output, borrowed) | height-sharded output | reader only — no writer kernel is created (`embeddings_rm_program_factory.cpp:211`) | 1 toucher | **self-loop**, and the DFB is **`borrowed_from`** the output `TensorParameter` (`embeddings_rm_program_factory.cpp:133-135` sets `out_cb_desc.buffer = out_buffer`) |
  | `c_1` (index scratch) | all | reader only (`embeddings.cpp:47,48,96`) | 1 toucher | **self-loop** |
  | `c_2` (local weight cache) | `PADDED`, `BINARY` | reader only (`embeddings_common.hpp:38-41`, `:44-51`) | 1 toucher | **self-loop** |
  | `c_2` (local weight cache) | `GENERIC` | not allocated — no `CBDescriptor` is pushed (`embeddings_rm_program_factory.cpp:151-173`) | n/a | **conditional DFB** — the spec exists under `PADDED`/`BINARY` and not under `GENERIC`. The legacy factory already gates the allocation, so this is a translation, not new structure |

  **`EmbeddingsTilizedIndicesProgramFactory`** — reader is `embedding_ind_tilized.cpp`, writer is always `writer_unary_stick_layout_interleaved_start_id.cpp` (`embeddings_tilized_indices_program_factory.cpp:174`); this factory has no sharded-output special case.

  | CB | Config | Touchers on a node | Verdict | Resolution |
  |---|---|---|---|---|
  | `c_0` (weight stage → output) | all | reader locked producer (`embedding_ind_tilized.cpp:54,59`) + shared writer locked consumer | plain 1:1 | legal — no action. Note `output_cb_index = src0_cb_index` (`embeddings_tilized_indices_program_factory.cpp:132`): one CB serves as the reader's staging buffer *and* the writer's output CB, which is what makes it a genuine 1P+1C |
  | `c_1` (index scratch) | all | reader only (`embedding_ind_tilized.cpp:47,48,128`) | 1 toucher | **self-loop** |
  | `c_2` (local weight cache) | `PADDED`, `BINARY` | reader only (`embeddings_common.hpp:38-41`, `:44-51`) | 1 toucher | **self-loop** |
  | `c_2` (local weight cache) | `GENERIC` | not allocated (`embeddings_tilized_indices_program_factory.cpp:108-130`) | n/a | **conditional DFB**, as above |

  On the `c_1` and `c_2` self-loops: both are reserve-only scratchpads. `c_2` is reserved and written but never committed — `prepare_local_cache` calls `reserve_back` + `get_write_ptr` and no `push_back` — which is intentional, since nothing downstream drains it. `c_1` is reserved once at the top of each reader and committed at the very end purely to leave the CB balanced. Neither has a second toucher in any config.

- **Offset base pointers:** **RED — Type 2 (accessor-fed offset), scoped to `EmbeddingsFusedProgramFactory`.**

  **The site.** `device/kernels/dataflow/embeddings_tilize.cpp:36`:

  ```cpp
  auto weights = TensorAccessor(weights_args, weight_buffer_src_addr + weight_offset);
  ```

  - `weight_buffer_src_addr` is RTA slot 0's sibling at slot 1 (`embeddings_tilize.cpp:16`), fed the `weights_buffer` pointer at `embeddings_fused_program_factory.cpp:322`.
  - `weight_offset` is RTA slot 4 (`embeddings_tilize.cpp:19`), fed the host's per-core `weight_offset` at `embeddings_fused_program_factory.cpp:325`.
  - The host advances it at `embeddings_fused_program_factory.cpp:340-344`: `weight_offset += weight_block_size`, wrapping to 0 when it reaches `weight_page_size`. `weight_block_size` is the output shard width in bytes (`:203`), so consecutive cores get 0, `shard_width·elem_size`, `2·shard_width·elem_size`, … — the factory splits the embedding row across cores by column, and each core's accessor is pre-shifted onto its own column slice.
  - Every read through that accessor inherits the shift: `read_token_async` (`embeddings_common.hpp:78-83`, `:101-106`, `:108-113`) and `prepare_local_cache` (`:41`, `:47`, `:50`) all address by `page_id`, so the offset baked into the bank base is what selects the column.

  **Why it gates.** Metal 2.0's `TensorAccessor(tensor::weights)` constructor takes the token alone and reads the base from the binding's CRTA slot (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:416-421` for the interleaved specialization, `:97-102` for the sharded one). There is no parameter for an offset base, and the token carries only `args` plus the address CRTA offset (`tt_metal/hw/inc/api/tensor/tensor_binding_token.h:42-47`). Once the weights tensor is a `TensorParameter`, the accessor's base is the framework's to supply — the `weight_offset` has nowhere to go. This is exactly the wall the offset-base-pointer analysis describes for Type 2: the offset **is** the accessor's base, not a relocatable trailing `+`.

  Note the fold happens **in the kernel**, not on the host — the factory passes a clean `Buffer*` and a separate scalar. That is the shape the Type-1 remedy produces, and for a Type-1 site it would be a clean pass. It does not clear a Type-2 site: relocating the addition into the kernel only helps when the sum is consumed as a raw NoC address. Here the sum is consumed as the accessor's base, so the wall is unchanged. I judged it **Type 2** on that basis.

  **Not in the triage tables.** `embedding` does not appear in `2026-07-19_offset_base_pointers.md`. Per that document's contract — a dated prior, not an authority — this is a fold the analysis never saw, so I classified it from the recognition model rather than waving it through for being unlisted.

  **Config reach.** `weight_offset` is identically 0 on the factory's interleaved path (the increment lives only in the `output_sharded` branch), so only the sharded, column-split configuration exercises a non-zero offset. That does **not** narrow the gate below factory granularity: `embeddings_tilize.cpp:36` is unconditional, one program-spec factory covers both configs, and a port that dropped the offset would silently mis-address every sharded run. The whole factory is blocked.

  **Routing.** Ops team, **and** framework/Audrey — flag early. Record for that discussion:

  | Op | Factory | Argument | Offset expression |
  |---|---|---|---|
  | `embedding` | `EmbeddingsFusedProgramFactory` | reader RTA — `weight_offset` (slot 4), summed into the weights accessor base | `weights_base + Σ shard_width·elem_size` (wrapping at the full weight row), `embeddings_fused_program_factory.cpp:340-344` |

  **Two candidate remedies — auditor observations, for the owning teams to evaluate.** Both are flagged because the general Type-2 case is harder than this one looks, and it would be a shame for embedding to wait on a design that a narrower fix already covers. Neither is porter work, and neither is validated:

  1. **Move the shift into the per-read options.** `noc.async_read`'s options struct already carries `offset_bytes`, and the op's own RM sibling uses it for exactly this purpose (`embeddings.cpp:63`, `embeddings_tilize.cpp:61`). Threading `weight_offset` through `read_token_async` and `prepare_local_cache` as an added `offset_bytes` term, and leaving the accessor on the clean base, would express the same addressing without a pre-offset base. This would touch `embeddings_common.hpp`, which all three readers share, so it is not a one-file change — and it is a semantic change either way, hence ops-team work rather than port work.
  2. **The token's `args` are reachable.** `TensorBindingToken` exposes `static constexpr args_t args` (`tensor_binding_token.h:45`), which the porting docs already sanction passing to donor functions expecting a `TensorAccessorArgs<N>`. In principle a kernel could construct an accessor from `tensor::weights.args` plus a hand-computed base. I am **not** recommending it: the token's `args` default their CRTA offset, while the token constructor deliberately advances it past the address slot (`tensor_accessor.h:417-419`), so the two are not interchangeable once any dynamic accessor field is in play. Recorded so the framework discussion can rule it in or out deliberately rather than by omission.

- **TensorAccessor 3rd argument:** **GREEN — one site, Class 2 (redundant / inert).**

  `device/kernels/dataflow/embeddings_rm_writer_chunked.cpp:26` — `TensorAccessor(dst0_args, dst_addr, output_page_size)`.

  1. **Sharded or interleaved?** Interleaved. This kernel is instantiated only under `!output_sharded` (`embeddings_rm_program_factory.cpp:211,213-228`), and its accessor args come from `TensorAccessorArgs(*output.buffer())` on that interleaved buffer (`:219`).
  2. **Correct or wrong magnitude?** Correct. `output_page_size = output.padded_shape()[-1] * output.element_size()` (`embeddings_rm_program_factory.cpp:53`) — the true row-major logical page, i.e. the same value as `buffer->page_size()`. On an interleaved accessor the value is realigned to the allocator alignment regardless, so even the unaligned case would be inert; here it is also the right value on the nose.

  → **Class 2.** Port action: drop the argument; Metal 2.0 supplies the aligned page size implicitly. The 3rd-arg triage doc lists `embeddings_rm_writer_chunked` as `2 — Redundant`, which matches my own read.

  No other accessor in the op passes a 3rd argument. The other five accessor constructions — `embeddings.cpp:39,40`, `embeddings_tilize.cpp:35`, `embedding_ind_tilized.cpp:35,36`, `writer_unary_stick_layout_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id.cpp:36` — are all two-argument.

## Port-work summary  *(mirrors the brief; scoped to the clean subset)*

- **Tensor bindings** (per binding, per factory):

  | Factory | Binding | Config | Case | Evidence |
  |---|---|---|---|---|
  | RM | `input_tensor_arg` | all | **Case 1** | `Buffer*` pushed at `embeddings_rm_program_factory.cpp:264`; kernel feeds it to `TensorAccessor` at `embeddings.cpp:39` |
  | RM | `weight_arg` | all | **Case 1** | `Buffer*` at `:265`; `TensorAccessor` at `embeddings.cpp:40` |
  | RM | `output` | interleaved (both writers) | **Case 1** | `Buffer*` at `:280` / `:283`; `TensorAccessor` at `embeddings_rm_writer_chunked.cpp:26` / `writer_unary_stick_layout_interleaved_start_id.cpp:20` |
  | RM | `output` | height-sharded | **clean** | no writer kernel; the tensor is reached through the borrowed-memory CB at `:133-135` → ports via `DataflowBufferSpec::borrowed_from` |
  | TilizedIndices | `input_tensor_arg` | all | **Case 1** | `Buffer*` at `embeddings_tilized_indices_program_factory.cpp:209`; `TensorAccessor` at `embedding_ind_tilized.cpp:35` |
  | TilizedIndices | `weight_arg` | all | **Case 1** | `Buffer*` at `:210`; `TensorAccessor` at `embedding_ind_tilized.cpp:36` |
  | TilizedIndices | `output` | all | **Case 1** | `Buffer*` at `:224`; `TensorAccessor` at `writer_unary_stick_layout_interleaved_start_id.cpp:20` |

  All pointer deliveries use the **`Buffer*`-binding form** — the factories push the `Buffer*` object into `KernelDescriptor::RTArgList`, never `->address()`. The framework auto-registers these as `BufferBinding`s and patches them on cache hits, so none of them is the silent-stale-address hazard; they are routine port work, and the sheet's `Smuggled pointer = no` agrees. The kernel still receives a raw `uint32_t` base and feeds it to a `TensorAccessor` in every case, which is what makes them Case 1.

- **TensorParameter relaxation:** `none` on all three factory rows — nothing to apply.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg @ `embeddings_rm_writer_chunked.cpp:26` (Class 2 — no `dynamic_tensor_shape` needed).
- **CB endpoints:** self-loop `c_1` (all configs, both factories) and `c_2` (`PADDED` / `BINARY`, both factories); self-loop + `borrowed_from` on `c_0` in the RM factory's height-sharded config; conditional DFB for `c_2` (live under `PADDED`/`BINARY`, absent under `GENERIC`, both factories); every other `(CB, config)` is a legal 1:1.

## Heads-ups  *(mirrors the brief; scoped to the clean subset)*

- **CB endpoints (multi-binding shapes to watch):** none. The op declares no semaphores, so the hidden-second-writer face cannot occur; no CB in either clean factory has more than two touchers in any config.
- **Cross-op / shared kernels:** both clean factories instantiate `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` from the shared pool. **No `_metal2` fork exists beside it** — this port creates the first one. Other factories binding the same file, as a **sunset list**: `data_movement/concat` (`concat_program_factory.cpp:234`, row-major path) and `data_movement/copy` (`copy_same_memory_config_program_factory.cpp:39`, row-major interleaved path). Note that `data_movement/slice`'s row-major writer is a **separate file** with a similar name (`slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp`) and is not a co-borrower.
- **RTA varargs:** none. Every runtime arg in both clean factories is read at a constant index as a distinct field, and every compile-time arg is read at a constexpr index — there is no counted loop over args and no data-selected index. The one argument that *looks* variable is `pad_token`: `prepare_local_cache`'s `pad_token_arg_idx` parameter (`embeddings_common.hpp:35,37`) is a compile-time-constant default supplied at each call site, so the read is at a fixed slot and the argument gets a name. (That slot is wrong in one factory — see *Misc anomalies*.)
- **Kernel includes will need revisiting, not just swapping.** All three readers, the chunked writer and `embeddings_common.hpp` include `api/dataflow/circular_buffer.h` and use the `CircularBuffer` wrapper. The port moves them to `DataflowBuffer` / `api/dataflow/dataflow_buffer.h`; the include set is part of that change, not incidental to it. `writer_unary_interleaved_start_id.cpp` (fused factory only) shows the target shape.
- **`embeddings_common.hpp` is shared by all three readers, including the blocked one.** Any change the port makes to it lands on `embeddings_tilize.cpp` too, which is not being ported. The header's four file-scope globals (`pad_token`, `pad_local_addr`, `zero_local_addr`, `one_local_addr` — `:24-27`) and its two function templates are the shared surface to be careful with.
- **`prepare_local_cache` reserves without committing.** `embeddings_common.hpp:38-41` and `:44-51` call `reserve_back` + `get_write_ptr` and never `push_back`. That is deliberate — the CB is a local scratch cache with no consumer — but it means the `c_2` DFB's producer binding is the only binding, and the self-loop is what makes it expressible.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No function-call escape leaves `tt_metal/*` in either clean factory. Two kernel files are borrowed by file path; both are Device 2.0 compliant, so neither creates a scheduling blocker.

**Summary table — includes (function-call escape).**

| Op kernel | Include | Resolved donor | Class | Status |
|---|---|---|---|---|
| `embeddings.cpp`, `embeddings_tilize.cpp`, `embedding_ind_tilized.cpp`, `embeddings_rm_writer_chunked.cpp`, `embeddings_common.hpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | `tt_metal/*` — LLK / HAL | 1 | ✓ no concern |
| `embedding_ind_tilized.cpp:11` | `api/debug/dprint.h` | `tt_metal/*` | 1 | ✓ no concern — but unused (see *Misc anomalies*) |
| `embeddings.cpp`, `embeddings_tilize.cpp`, `embedding_ind_tilized.cpp` | `embeddings_common.hpp` | own directory | — | in-op, not an escape |
| `tilize_chunked.cpp:8` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | official shared kernel library | 2 | ✓ lib team handles internally *(fused factory only — blocked)* |

Per-call detail is omitted: every roll is ✓.

**Borrowed kernel files (file-path instantiation).**

| Kernel file | Owning pool / family | Also instantiated by | `_metal2` fork beside it? |
|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared pool `ttnn/cpp/ttnn/kernel/` | `data_movement/concat` (RM path), `data_movement/copy` (RM interleaved path) | **No** — this port creates the first fork |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly shared — ~30 factories across `reduction`, `data_movement`, `matmul`, `kv_cache`, `experimental/transformer`, and others | **Yes** — `writer_unary_interleaved_start_id_metal2.cpp` already exists beside it; a second copy also exists under `copy/typecast/device/kernels/dataflow/` *(fused factory only — blocked)* |
| `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | `data_movement/tilize` | no other factory instantiates this exact path — `data_movement/tilize`'s own factories bind `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` instead | **No** *(fused factory only — blocked)* |

The last row is worth a second look by whoever owns `data_movement/tilize`: `EmbeddingsFusedProgramFactory` is the only remaining consumer of `tilize/device/kernels/compute/tilize.cpp`, while every `tilize` factory has moved to the shared-pool copy at `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`. That looks like a file the tilize migration left behind, and it belongs to a blocked factory anyway.

### Relaxation candidates

None. The op has no custom hash, so there is no hash to mine for the properties it actually depends on.

### TTNN factory analysis

Sheet-derived facts, each confirmed against the code (evidence in *Gate detail*):

- **Current concept:** `descriptor` on all three factories — each defines `create_descriptor` returning a `ProgramDescriptor`.
- **Op-owned tensors:** none. The `descriptor` concept cannot carry them, and the sheet's column is empty.
- **MeshWorkload need:** none. `Execution Model` reads `SPMD` and no factory returns a `WorkloadDescriptor`.
- **Target concept:** `ProgramSpecFactoryConcept` for all three, which matches the sheet's own `Porting Target` column. `Override runtime args method?` is `no`, so the framework refreshes tensor bindings on a cache hit and each factory writes one method.
- **Custom hash:** absent — the default hash applies. Not a gate either way.
- **`override_runtime_arguments`:** absent.
- **Pybound `create_descriptor`:** absent. `embedding_nanobind.cpp` exposes only the user-facing `ttnn::embedding` and the `EmbeddingsType` enum, so the port removes no user-visible binding.
- **`get_dynamic_runtime_args`:** absent — the deprecated hook is not present on the device-op.
- **Gate conjuncts, all clear:** `TensorParameter relaxation = none`, no `get_dynamic_runtime_args`, no genuine multi-program.

## Misc anomalies  *(team-only, non-gating)*

These are latent issues noticed while reading the op. They route to the ops team; the port does not act on them.

- **`EmbeddingsTilizedIndicesProgramFactory` reads the pad token from the wrong runtime-arg slot.** The reader kernel takes its pad token from slot 6 (`embedding_ind_tilized.cpp:42` passes `pad_token_arg_idx = 6`; `embeddings_common.hpp:37` reads it), but this factory puts `col_offset % FACE_HEIGHT` in slot 6 (`embeddings_tilized_indices_program_factory.cpp:215`) and the real pad token in slot **7** (`:217`). So under `EmbeddingsType::PADDED` with TILE-layout indices, the kernel treats a small per-core face-column index (0-15) as the pad token: indices that happen to equal it get substituted with the wrong weight row, and the actual padding index is looked up normally. Slot 7 is never read.

  The other two readers are correct — `embeddings.cpp` and `embeddings_tilize.cpp` both put `pad_token` at slot 6 and pass `pad_token_arg_idx = 6` — so this is a single-factory off-by-one, presumably from the extra `starting_index` argument this factory carries.

  It appears untested: no test in `tests/ttnn/unit_tests/operations/data_movement/test_embedding.py` passes `padding_idx`, and the sweep that does (`tests/sweep_framework/sweeps/data_movement/embedding/embedding_pytorch2.py`) draws indices with `torch.randint` over vocabularies of 500-250k rows, where hitting a 0-15 value is unlikely and hitting the true padding index is unlikely too. My guess is that is why it has gone unnoticed.

  **This one has a porter consequence, so it is also in the brief.** The port converts positional args to named ones, and naming a `pad_token` argument would *fix* the mismatch as a side effect — a functional change the port's zero-change contract does not permit. Whoever owns the fix should land it before or alongside the port so the porter is not the one silently changing behavior.

- **Dead compile-time arg — `EmbeddingsTilizedIndicesProgramFactory`.** `embeddings_tilized_indices_program_factory.cpp:140` bakes `input_page_size` into CTA slot 3, and `embedding_ind_tilized.cpp` never reads slot 3 (it jumps from slot 2 at `:27` to slot 4 at `:29`). The kernel derives the value at runtime instead (`:51`, `input.get_aligned_page_size()`). The slot cannot simply be removed without shifting `TensorAccessorArgs<7>` at `:33`.

- **Dead and misnamed compile-time arg — `EmbeddingsTilizedIndicesProgramFactory`.** `embedding_ind_tilized.cpp:31` declares `constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(6)` and never uses it. The host puts `FACE_HEIGHT` (16) in that slot (`embeddings_tilized_indices_program_factory.cpp:143`), so the name describes neither a block size nor a byte count.

- **Dead compile-time arg — `EmbeddingsFusedProgramFactory`.** `embeddings_fused_program_factory.cpp:217` bakes `weight_page_size` into CTA slot 4, which `embeddings_tilize.cpp` never reads (slot 3 at `:26`, then slot 5 at `:27`).

- **Dead compile-time arg in the shared stick writer — belongs to the shared pool, not to embedding.** `writer_unary_stick_layout_interleaved_start_id.cpp` reads CTA slot 0 and then hardcodes `TensorAccessorArgs<2>()` (`:17-18`), so slot 1 is never read. All four co-borrowers dutifully fill it with a page size — `embeddings_rm_program_factory.cpp:231`, `embeddings_tilized_indices_program_factory.cpp:169`, `concat_program_factory.cpp:214`, `copy_same_memory_config_program_factory.cpp:135` — and each also passes the same value as a runtime arg, which is where the kernel actually gets `stick_size` (`:13`). The slot is a required placeholder given the hardcoded `<2>`, so it cannot be dropped by one caller; it routes to whoever owns `ttnn/cpp/ttnn/kernel/`.

- **Unused include.** `embedding_ind_tilized.cpp:11` includes `api/debug/dprint.h`; the file contains no `DPRINT` use.

- **A `GENERIC`-config compile-time arg with no consumer.** All three readers read the `c_2` CB index into `cb_id_in2` unconditionally (e.g. `embeddings.cpp:25`) and pass it to `prepare_local_cache`, whose body is empty unless `PADDED` or `BINARY` is defined (`embeddings_common.hpp:36-53`). Under `GENERIC` the host allocates no `c_2` CB, so the arg carries an index to a buffer that does not exist. Harmless today — nothing dereferences it — but it is the kind of arg that stops being harmless once a DFB binding is attached to it, so the port should keep the CTA conditional alongside the DFB spec.

- **A stale comment.** `embeddings_rm_program_factory.cpp:35` and `embeddings_fused_program_factory.cpp:36` both carry a `// Grayskull Device Setup` banner over code that is not Grayskull-specific and on an architecture that is no longer a target.

## Questions for the user

1. **The tilized-indices pad-token slot mismatch:** should this be routed to the ops team as a bug fix before the subset port begins, or should the porter be told explicitly to preserve the current (wrong) slot mapping and leave the fix to a follow-up? The brief currently says preserve-and-flag, on the reasoning that a port must not change behavior — but the result is that a named-arg port deliberately reproduces a defect, which is worth your call. Context: `embedding_ind_tilized.cpp:42` versus `embeddings_tilized_indices_program_factory.cpp:215-217`.

## Recipe notes

1. **The offset-base recognition model assumes a host-side fold; this op folds in the kernel, and the two are not equivalent across types.** [Offset base pointers](#offset-base-pointers) frames every recognition signal around `buffer()->address() + <offset>` appearing in the factory, and the *"No fold, op in the tables"* outcome describes the cleared shape as *"a bare `->address()` base + a separate scalar offset arg, added in the kernel"* → **GREEN**. `EmbeddingsFusedProgramFactory` has precisely that shape — clean `Buffer*`, separate scalar, addition in the kernel — and yet it is a genuine wall, because the sum is the accessor's base rather than a raw NoC address. Read literally, the GREEN bullet clears it; read structurally, Type 2 catches it. I went with Type 2.

   Suggestion: make the Type-2 recognition independent of *where* the addition happens — something like *"the accessor's base is not the tensor's base, whether the offset was added on the host or in the kernel"* — and qualify the GREEN bullet as applying to a **raw-consumption (Type 1)** site only. As written, an auditor who leads with the reconcile-against-the-doc table can land on GREEN for a site the taxonomy blocks.

2. **The brief-emission rule and the config-scoped exception are stated in different places, and the stronger-sounding one is not the operative one.** [Output: the two documents](#output-the-two-documents) says the brief is *"emitted only on a fully GREEN audit"* and *"On any RED there is no brief — there is no port yet."* The exception for a config-scoped gate lives earlier, in a parenthetical inside the **GATE** role bullet, and is referenced again from the Red scoping rule. Since the Output section is where an auditor goes when it is time to write files, a one-clause pointer there (*"except a config-scoped gate with a surviving clean subset — see Code-path scope"*) would remove the need to reconcile the two.

3. **A small gap in the report template for a subset brief.** The `METAL2_PREPORT_AUDIT.md` skeleton has a single *Port-work summary* and *Heads-ups* pair that *"mirrors the brief"*, but on a subset port those sections describe only part of the op. I labelled them *scoped to the clean subset* and kept the blocked factory's facts out of them (its Device 2.0 and Appendix A results are in *Gate detail*, where the gate-bearing subjects put them, and its donor coupling is in *Team-only* marked as blocked). Flagging in case the template should say so explicitly, since the alternative reading — one flat set of port-work bullets covering all factories — would hand the porter work on a factory that has no brief.
