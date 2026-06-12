# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads`

Single `DeviceOperation` in this directory. Two program factories:

- **`NlpCreateHeadsDeviceOperation`**
  - `Interleaved` (`nlp_create_qkv_heads_program_factory.cpp` — `Interleaved::create_descriptor`)
  - `Sharded` (`nlp_create_qkv_heads_program_factory.cpp` — `Sharded::create_descriptor`)

Kernels in scope:
- Own dataflow kernels (all in `device/kernels/dataflow/`):
  - `reader_tm_tile_layout_nlp_create_qkv_heads.cpp` — used by `Interleaved`
  - `writer_tm_tile_layout_nlp_create_qkv_heads.cpp` — used by `Interleaved`
  - `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` — used by `Sharded` (instantiated as both reader and writer with different compile-time args)
- Borrowed compute kernel (from shared pool `ttnn/cpp/ttnn/kernel/compute/`):
  - `transpose_wh.cpp` — used by `Interleaved` (when `transpose_k_heads == true` only)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads` |
| **Overall** | YELLOW |
| **DOps / Factories** | `NlpCreateHeadsDeviceOperation` → `Interleaved`, `Sharded` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `Sharded` factory — Q output CB (c_16), K output CB (c_17), V output CB (c_18) used as address sources in `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`; workaround applies |

---

## Result

**YELLOW → open question on Sharded tensor-binding classification.** All gates pass. The sole open question is whether the Sharded factory's raw-address input access pattern is genuinely Case 2 (exotic NoC walk) or Case 1 (re-expressible via `TensorAccessor`). The op comment at `nlp_create_qkv_heads_program_factory.cpp:413–423` explicitly documents the reason: K and V base addresses are computed as byte-offset arithmetic into the input (or KV) buffer rather than page-index walks, making a clean `TensorAccessor` registration non-trivial today. See Questions for the user below.

The `Interleaved` factory is clear for porting (tensor bindings use the `Buffer*` framework form; all gates green).

---

## Gate detail

### ProgramDescriptor

GREEN. Both `Interleaved::create_descriptor` and `Sharded::create_descriptor` populate a `ProgramDescriptor` and use `KernelDescriptor`, `CBDescriptor`, `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`, `ComputeConfigDescriptor` — the full `ProgramDescriptor` API. No imperative `host_api.hpp` calls (`CreateProgram` / `CreateKernel` / `SetRuntimeArgs` / `CreateCircularBuffer`).

Source: `nlp_create_qkv_heads_device_operation.hpp:38–48` (factory signatures), `nlp_create_qkv_heads_program_factory.cpp` throughout.

### Device 2.0 (every kernel used)

GREEN. All dataflow kernels use the Device 2.0 headers exclusively:

- `reader_tm_tile_layout_nlp_create_qkv_heads.cpp`: includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`. Uses `Noc noc`, `CircularBuffer cb_*`, `TensorAccessor`, `TensorAccessorArgs` — Device 2.0 idioms throughout. No legacy `InterleavedAddrGen`, `noc_async_read`, or raw CB-index free functions.
- `writer_tm_tile_layout_nlp_create_qkv_heads.cpp`: same header set. `Noc`, `CircularBuffer`, `TensorAccessor` — Device 2.0 throughout.
- `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`: includes `api/dataflow/endpoints.h` additionally. Uses `Noc`, `CircularBuffer`, `UnicastEndpoint` — Device 2.0 throughout.
- `transpose_wh.cpp` (borrowed compute kernel from `ttnn/cpp/ttnn/kernel/compute/`): uses `api/compute/transpose_wh.h` and LLK compute-side CB calls (`cb_wait_front`, `cb_reserve_back`, `cb_push_back`, `cb_pop_front`). These are standard **compute** kernel LLK primitives, not the data movement CB free functions subject to Device 2.0 migration. Compliant.

`get_tile_size(cb_id)` appears in `reader_tm_tile_layout_nlp_create_qkv_heads.cpp:46–47` and `writer_tm_tile_layout_nlp_create_qkv_heads.cpp:49–50` — sanctioned Device 2.0 free function (no wrapper form exists); not a holdover.

### Feature compatibility

Per-entry scan of all Appendix A entries:

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `global_circular_buffer` field, or related signal anywhere in op or kernels |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `Sharded` factory sets `.buffer` on three `CBDescriptor`s — Q (c_16, line 316), K (c_17, line 332), V (c_18, line 348) output CBs — each backed by the corresponding output tensor buffer. Port uses `borrowed_from`. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set anywhere; `Interleaved` and `Sharded` factory `CBDescriptor` literals omit the field entirely |
| Aliased Circular Buffers | N/A | Every `CBDescriptor` has a single-element `format_descriptors` initializer. No multi-index aliased pattern. |
| GlobalSemaphore | N/A | No `GlobalSemaphore` type, `CreateGlobalSemaphore`, or `global_semaphore.hpp` anywhere |
| Non-zero semaphore initial value | N/A | No semaphores in this op at all |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` token in host code |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` anywhere |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` has fixed-count named tensors; no `std::vector<Tensor>` and no kernel loop over `get_compile_time_arg_val(i)` with variable index |

Feature gate: GREEN — no UNSUPPORTED features fire.

---

## Port-work summary *(mirrors the brief)*

### Tensor bindings

**`Interleaved` factory:**

- **`input_q` (in0_buffer)** — `Buffer*`-binding form: `reader_rt.push_back(in0_buffer)` (`nlp_create_qkv_heads_program_factory.cpp:241`). Framework auto-registers as `BufferBinding`. Kernel-side: `in0_tensor_addr = get_arg_val<uint32_t>(0)` then `TensorAccessor(in0_args, in0_tensor_addr)` — already consuming via `TensorAccessor`. **Case 1** (re-express via `TensorParameter`; `TensorAccessorArgs` plumbing and `Buffer*` RTA disappear, kernel builds `TensorAccessor(ta::in0)` directly). Not a correctness hazard.
- **`input_kv` (in1_buffer)** — `Buffer*`-binding form: `reader_rt.push_back(in1_buffer)` / `reader_rt.push_back(uint32_t{0})` (`nlp_create_qkv_heads_program_factory.cpp:242–246`). Kernel-side: `in1_tensor_addr = get_arg_val<uint32_t>(1)` then (under `READ_FROM_INPUT_TENSOR_KV`) `TensorAccessor(in1_args, in1_tensor_addr)` — also via `TensorAccessor`. **Case 1** (re-express via `TensorParameter`). Optional tensor — port must handle the `nullptr` / zero placeholder when `input_kv` is absent.
- **`q_buffer`, `k_buffer`, `v_buffer` (outputs)** — `Buffer*`-binding form in writer RTAs (`nlp_create_qkv_heads_program_factory.cpp:255–257`). Kernel-side: `q_tensor_addr = get_arg_val<uint32_t>(0)` etc., then `TensorAccessor(q_args, q_tensor_addr)`. **Case 1** (re-express via `TensorParameter`).

**`Sharded` factory:**

- **`input_q` (q_base_addr, q_start_addr) and `input_kv` (k_base_addr / k_start_addr, v_base_addr / v_start_addr)** — raw `uint32_t` addresses baked from `buffer()->address()` plus byte-offset arithmetic (`nlp_create_qkv_heads_program_factory.cpp:356–363, 436–437, 445–446`). Kernel-side (`reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`): the kernel reads these addresses via `get_arg_val<uint32_t>` and uses them as raw remote NoC addresses (`UnicastEndpoint src_ep`, `{.addr = q_src_addr}`). No `TensorAccessor` is used. The pattern is cross-core NoC reads with direct byte-level addressing into a shard (sub-page offset arithmetic). See Questions for the user — **suspected Case 2**, classification deferred to user.

### Custom hash

None — no `compute_program_hash` override found.

---

## Heads-ups *(mirrors the brief)*

- **Dynamic CircularBuffer (borrowed memory):** `Sharded` factory uses `.buffer` on Q (c_16 @ `nlp_create_qkv_heads_program_factory.cpp:316`), K (c_17 @ line 332), V (c_18 @ line 348) CBs. Port declares these as `DataflowBufferSpec` with `borrowed_from = <output_tensor_parameter_name>`.

- **Fake CBs (address-only):** In the `Sharded` factory and `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`:
  - **(c_16 / Q output CB, reader RISC):** `cb_q_out.get_write_ptr()` at `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp:41` is called without a preceding `reserve_back` and without a following `push_back`. There is no `wait_front` / `pop_front` consumer. Q output CB is used purely as a base address to DMA into — it is a **fake CB**. The port resolves this with the sanctioned fake-CB workaround (see the port recipe); does not gate.
  - **(c_17 / K output CB, reader RISC):** `cb_kv_out.reserve_back(num_kv_tiles)` and `cb_kv_out.push_back(num_kv_tiles)` at `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp:87, 114` — producer side present. However, no `wait_front` / `pop_front` consumer for this CB exists in either kernel instance. The writer RISC also calls `reserve_back/push_back` on what resolves to c_18 (V output CB) under writer CTAs. **Treat as fake CB** — the reserve/push are present but no kernel consumes via `wait_front`; the borrowed-memory L1 region is the final output, not a producer→consumer FIFO. Port resolves with fake-CB workaround.
  - **(c_18 / V output CB, writer RISC):** same pattern as K — `reserve_back/push_back` without a consumer. Fake CB.
  
  Note: the `Interleaved` factory uses plain (non-borrowed) CBs c_0, c_1, c_16 — these are real DFBs with producers (reader) and consumers (writer / compute). No fake CB in the `Interleaved` path.

- **RTA varargs:** `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp:29–30` — `get_arg_addr(19)` and `get_arg_addr(19 + num_x)` access per-core arrays of NOC x- and y-coordinates whose length (`num_x`, `num_y`) is itself a runtime arg (arg index 18). This is a vararg RTA pattern: the host appends `noc_x_coords` (length `num_cores_x`) and `noc_y_coords` (length `num_cores_y`) to the RT arg vector (`nlp_create_qkv_heads_program_factory.cpp:450–451`). Metal 2.0 supports RTA varargs — this does not gate — but the porter will need to choose between named RTAs per coordinate (recommended, if the count is bounded) and Metal 2.0's RTA vararg mechanism (if the count is genuinely runtime-varying). Here `num_cores_x` and `num_cores_y` are determined from the shard spec grid, so they're fixed at factory construction time for a given shape — named RTAs are the preferred endpoint.

- **Cross-op / shared kernel:** `transpose_wh.cpp` from `ttnn/cpp/ttnn/kernel/compute/` (shared pool) — used by `Interleaved` when `transpose_k_heads == true`. Metal 2.0 rewrite of this compute kernel is a **port-together** event: `nlp_create_qkv_heads_boltz`, `split_query_key_value_and_split_heads`, and `nlp_create_qkv_heads_vit` all instantiate the same file. Any rewrite must land in all co-users simultaneously or the non-ported users break.

---

## Team-only

### TensorAccessor convertibility

**`Interleaved` factory bindings (`in0_buffer`, `in1_buffer`, `q_buffer`, `k_buffer`, `v_buffer`):** All are straightforwardly Case 1 — the `TensorAccessorArgs(buffer).append_to(compile_time_args)` plumbing is already in place and the kernels already build `TensorAccessor` from those args. Conversion to named `TensorParameter` is mechanical.

**`Sharded` factory input bindings (`q_base_addr` / `q_start_addr`, `k_base_addr` / `k_start_addr`, `v_base_addr` / `v_start_addr`):** Suspected Case 2, but deferring to user per recipe's self-classification rule. The access pattern: each core DMA-reads a per-core slice of a multi-head input (or KV) shard using raw `{.noc_x, .noc_y, .addr}` addressing. The base address (`q_base_addr`) is `buffer()->address()`; the per-core start address (`q_start_addr`) is `q_base_addr + remote_q_head_start_idx * head_size` (computed at factory construction time). The host comment at `nlp_create_qkv_heads_program_factory.cpp:413–423` explicitly notes the difficulty: K and V addresses are themselves offsets into the input buffer (not standalone buffers), which makes clean `BufferBinding` registration non-trivial. The kernel uses `UnicastEndpoint` with raw `{.noc_x, .noc_y, .addr}` arguments — this is cross-core NoC access to specific addresses within a sharded buffer, not tile-indexed iteration. If this is confirmed Case 2, the bridge is: declare input tensors as `TensorParameter`, pull base address via `TensorAccessor::get_bank_base_address`, and leave the existing offset arithmetic in place. The op's comment is a strong signal toward Case 2, but the user should confirm.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** `✓ clean` for function-call escapes — all `#includes` in the dataflow kernels resolve to framework `api/` headers (`tt_metal/*` class, not cross-op). No cross-family donor functions.

**Summary table:**

| Op kernel | Include / borrowed file | Class | Shape |
|---|---|---|---|
| `reader_tm_tile_layout_nlp_create_qkv_heads.cpp` | `api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | `tt_metal/*` (LLK/HAL) | ✓ no concern |
| `writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | same as above | `tt_metal/*` (LLK/HAL) | ✓ no concern |
| `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` | `api/dataflow/*`, `api/core_local_mem.h`, `api/dataflow/endpoints.h` | `tt_metal/*` (LLK/HAL) | ✓ no concern |
| `transpose_wh.cpp` | `api/compute/transpose_wh.h` | `ttnn/cpp/ttnn/kernel/compute/` (shared pool) | ✓ no Device-2.0 concern; compute LLK primitives |

No per-call detail needed — all donors are `tt_metal/*` or shared-pool LLK headers with no function-call escapes requiring shape classification.

**Borrowed kernel files (file-path instantiation):**

- `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` — owned by shared pool `ttnn/cpp/ttnn/kernel/compute/`, instantiated by `Interleaved` factory (`nlp_create_qkv_heads_program_factory.cpp:126, 136`). Other ops instantiating the same file (from grep): `nlp_create_qkv_heads_boltz/device/nlp_create_qkv_heads_boltz_program_factory.cpp`, `split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.cpp`, `nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_program_factory.cpp`. **Port-together set** for this kernel: any Metal 2.0 rewrite of `transpose_wh.cpp` must be adopted by all four ops in a single coordinated change.

### Relaxation candidates

No custom hash to mine. Default strict.

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` calls `create_device_tensor` three times (`nlp_create_qkv_heads_device_operation.cpp:235–237`) for Q, K, V — but these are the declared output tensors (`tensor_return_value_t`), not hidden intermediates. No scratch or intermediate tensors beyond the op's declared outputs.

2. **MeshWorkload needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` anywhere in the op. No op-owned tensors (Q1 = No), so the false-positive MeshWorkload-artifact path does not apply either.

3. **Pybind `create_descriptor`?** No. `nlp_create_qkv_heads_nanobind.cpp` uses `ttnn::bind_function<"nlp_create_qkv_heads", "ttnn.experimental.">` binding the op function itself — normal op-function binding. No `nb::class_<…ProgramFactory>(...).def_static("create_descriptor", …)` or similar.

4. **Other migration-risky pybind?** None. The nanobind file exposes no `DeviceOperation`, factory, or param classes — only the top-level op function with standard argument bindings.

5. **Custom hash?** No `compute_program_hash` override found in either `nlp_create_qkv_heads_device_operation.hpp` or `nlp_create_qkv_heads_device_operation.cpp`.

6. **Custom override-runtime-args?** No `override_runtime_arguments` found in either factory.

---

## Misc anomalies *(team-only, non-gating)*

- `nlp_create_qkv_heads_program_factory.cpp:380–384`: The `Sharded` factory declares a `writer_desc` with `kernel_source` pointing to `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` (same file as `reader_desc`) and `config = WriterConfigDescriptor{}`. Both reader and writer use the identical kernel file, differentiated only by compile-time args (reader uses `{q_output_cb_index, k_output_cb_index}`, writer uses `{q_output_cb_index, v_output_cb_index}`). This is intentional — one RISC handles Q+K, the other handles Q+V — but is visually surprising (a writer descriptor pointing to a file named `reader_…`). Worth noting for the porter.

- `nlp_create_qkv_heads_program_factory.cpp:335–337`: `v_shard_spec` and `v_cores` are initialized from `std::get<0>(output).shard_spec().value()` and `q_shard_spec.grid` respectively (the **Q** shard spec, not V). This appears intentional (Q and V share the same shard grid), but the variable name `v_shard_spec` bound to the Q shard spec value may confuse readers. Not a bug — validated by `TT_FATAL` checks — but could be made clearer.

- `nlp_create_qkv_heads_program_factory.cpp:413–423`: The existing comment explains the correctness-vs-performance tradeoff for the Sharded factory's raw address RTAs (no `BufferBinding` registered; Contract-1 adapter falls back to slow-path rebuild on cache hit). This is useful context for the port.

---

## Questions for the user

1. **Sharded factory — input binding Case 1 vs. Case 2:** The `Sharded` factory reads input Q, K, and V data via raw `uint32_t` addresses derived from `buffer()->address()` plus per-core byte-offset arithmetic (`q_start_addr = q_base_addr + remote_q_head_start_idx * head_size`, similarly for K/V). The kernel uses `UnicastEndpoint` with `{.noc_x, .noc_y, .addr = q_src_addr}` for cross-core NoC reads at specific byte offsets within input shards. No `TensorAccessor` is used.

   Per the audit recipe: "The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support."

   The op's own comment (`nlp_create_qkv_heads_program_factory.cpp:413–423`) documents the issue: K and V addresses are byte-offset slices into the input (or KV) buffer, not standalone page-indexed buffers, which makes clean `TensorParameter` registration non-trivial with the current kernel ABI. **Is this a genuine Case 2 (exotic NoC walk that cannot be expressed via `TensorAccessor`)? Or is it a Case 1 pattern that should be re-expressed?** The determination affects port planning for the `Sharded` factory only — the `Interleaved` factory is unambiguously Case 1.

---

## Recipe notes

- The recipe's fake-CB litmus ("does the CB have a producer *and* a consumer?") required some interpretation for the `Sharded` factory: the K and V output CBs have `reserve_back/push_back` calls (producer side) but no `wait_front/pop_front` consumer in any kernel. The borrowed-memory DFB is the final output destination, and the "consumer" is the downstream user of the tensor — not a kernel CB consumer. The recipe's litmus as written focuses on in-kernel FIFO semantics; the absence of any `wait_front` makes these clearly fake. No ambiguity in the end, but the case of "has a push_back but no pop_front" isn't explicitly covered by the recipe's litmus example (which focuses on "no producer" as the canonical fake-CB case).
