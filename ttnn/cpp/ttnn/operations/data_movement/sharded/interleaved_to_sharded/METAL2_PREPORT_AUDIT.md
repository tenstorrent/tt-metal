# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded`

- **`InterleavedToShardedDeviceOperation`**
  - `InterleavedToShardedProgramFactory` (`device/interleaved_to_sharded_program_factory.cpp`)

The factory is a single `create_descriptor` method; there is no variant selection — the same factory handles all layouts (TILE / ROW_MAJOR) and all output-buffer types (L1 sharded / DRAM) via conditional branches.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded` |
| **Overall** | RED |
| **DOps / Factories** | `InterleavedToShardedDeviceOperation` → `InterleavedToShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No |
| *Prereqs* — Cross-op escapes | issue (see Gate detail + Out-of-directory coupling) |
| *Feature Support* — overall | GREEN (no UNSUPPORTED feature in use) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

## Result

**RED → blocked on Device 2.0 gate.** Three of the five dataflow kernels used by this op broadly use Device 1.0 NoC idioms (free-function `noc_async_read` / `noc_async_write` / `noc_async_read_barrier` / `noc_async_write_barrier`, manual transaction-ID management). These kernels are shared across many ops in the `data_movement/sharded/` family. The port is blocked until the Device 2.0 migration lands for those kernels; route to the Device 2.0 team.

Scoped subset: the TILE-layout, L1-output path (tile reader + `writer_unary_sharded.cpp`) involves only Device-2.0-compliant kernels and would be portworthy in isolation, but the shared kernel files mean the family must migrate together anyway.

## Gate detail

### ProgramDescriptor

GREEN — the factory (`device/interleaved_to_sharded_program_factory.cpp`) uses `ProgramDescriptor`, `CBDescriptor`, `KernelDescriptor`, and the `KernelDescriptor::RTArgList` / `emplace_runtime_args` path throughout. No imperative `host_api.hpp` builder calls.

### Device 2.0 (every kernel used)

**RED — three kernels broadly on Device 1.0.**

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 37 | `noc_async_read(src_noc_addr, dest_write_addr, block_width_bytes)` | `Noc noc` — absent |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 37,75,76,94 | `noc_async_read(...)` (multiple sites in aligned + unaligned paths) | none |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 73 | `noc_async_read_set_trid(active_trid)` | none |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 91 | `noc_async_read_set_trid(active_trid)` | none |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 85 | `ncrisc_noc_read_with_transaction_id_flushed(noc_index, active_trid)` | none |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 100 | `ncrisc_noc_read_with_transaction_id_flushed(noc_index, active_trid)` | none |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 93 | `get_noc_addr(scratch_write_addrs[slot] + aligned_offset)` — local-L1 address to NOC addr | none |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 109 | `noc_async_read_set_trid(0)` (reset) | none |
| `writer_unary_sharded_blocks_start_id.cpp` | 49 | `noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes)` | `CircularBuffer cb_out` in scope; no `Noc noc` |
| `writer_unary_sharded_blocks_start_id.cpp` | 56 | `noc_async_write_barrier()` | none |
| `writer_unary_sharded_stick_layout_start_id.cpp` | 42 | `noc_async_write(l1_read_addr, dst_noc_addr, block_width_bytes)` | `CircularBuffer cb_out` in scope; no `Noc noc` |
| `writer_unary_sharded_stick_layout_start_id.cpp` | 47 | `noc_async_write_barrier()` | none |

All three paths above are the kernels' **primary NoC data movement** — the Device 1.0 calls are not isolated holdovers but the core of what the kernel does. This is a RED gate (not YELLOW).

**Compliant kernels (no action needed on the Device 2.0 track for these):**

- `reader_unary_sharded_blocks_interleaved_start_id.cpp` — fully Device 2.0 compliant (`Noc noc`, `CircularBuffer cb_in`, `TensorAccessor`, `noc.async_read`, `noc.async_read_barrier`).
- `writer_unary_sharded.cpp` — Device 2.0 compliant (only CB operations: `CircularBuffer::wait_front`; no NoC).
- `eltwise_copy.cpp` — compute kernel (LLK APIs for tile ops; Device 2.0 gate applies to dataflow APIs, not compute LLKs).

**Kernel ownership note.** All five dataflow kernels live in `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/`, one level above the `interleaved_to_sharded` subdirectory. These are shared with multiple other ops; the Device 2.0 migration must be coordinated across the family (see Out-of-directory coupling).

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` reference anywhere in the op or its kernels |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `push_i2s_cb_pair(..., dst_buffer)` at factory:166–173; `.buffer = bound_buffer` at factory:46; non-DRAM L1-output path. Port uses `DataflowBufferSpec::borrowed_from`. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set anywhere |
| Aliased Circular Buffers | N/A | Every `CBDescriptor` has a single `CBFormatDescriptor` (one `push_back` per CB) |
| GlobalSemaphore | N/A | No semaphores created or used |
| Non-zero semaphore initial value | N/A | No semaphores |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | `TensorAccessorArgs(*src_buffer)` with no `ArgConfig::Runtime*` second argument |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` calls |
| Variable-count compile-time arguments (CTA varargs) | N/A | Fixed input tensor count (`InterleavedToShardedInputs` has one input tensor); no variable-count CTA loop |

Feature support overall: **GREEN** — no UNSUPPORTED feature is in use; one LANDED feature (Dynamic CB) is present and clean.

## Port-work summary *(mirrors the brief — no brief issued on RED)*

- **Tensor bindings:**
  - `input_tensor` / `src_buffer` — **Case 1** (re-express). Host: `src_buffer` pushed as `Buffer*` into `KernelDescriptor::RTArgList` (arg 0 of reader). Compile-time args include `TensorAccessorArgs(*src_buffer)`. Kernel: `src_addr = get_arg_val<uint32_t>(0)` fed to `TensorAccessor(src_args, src_addr)`. The `Buffer*` form is the framework's interim binding; re-express via `TensorParameter` / `TensorBinding`. Applies to both reader kernels.
  - `output_tensor` / `dst_buffer` (DRAM-output path) — **Case 1** (re-express). Host: `dst_buffer` pushed as `Buffer*` into `KernelDescriptor::RTArgList` (arg 0 of DRAM writers). Kernel: `dst_addr = get_arg_val<uint32_t>(0)` fed to `experimental::ShardedAddrGen::bank_base_address`. Re-express via `TensorParameter` / `TensorBinding`. Applies to `writer_unary_sharded_blocks_start_id.cpp` and `writer_unary_sharded_stick_layout_start_id.cpp`.
  - `output_tensor` / `dst_buffer` (L1-sharded-output path) — **clean** (borrowed-memory DFB). The factory sets `CBDescriptor::buffer = dst_buffer`; `writer_unary_sharded.cpp` reads via `CircularBuffer::wait_front`. Port uses `DataflowBufferSpec::borrowed_from`.
- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). See `device/interleaved_to_sharded_op.cpp:141`.

## Heads-ups *(mirrors the brief — no brief issued on RED)*

- **Notable constructs:** Dynamic CB (borrowed-memory DFB) used for L1 sharded output at `factory:166–173`. Port uses `DataflowBufferSpec::borrowed_from = <output TensorParameter name>`.
- **Cross-op / shared kernels:** All five dataflow kernels live in the parent `sharded/device/kernels/` directory (not in `interleaved_to_sharded/`). They are instantiated by at least 10 other ops (see Out-of-directory coupling → Borrowed kernel files). A Metal 2.0 rewrite is a single shared rewrite — all co-borrowing ops must adopt it together.
- **RTA varargs:** None.
- **TTNN factory analysis (porter-relevant):** No `create_descriptor` pybind; no other risky pybind; no `override_runtime_arguments`.

## Team-only

### TensorAccessor convertibility

- `input_tensor` (Case 1): standard interleaved page-by-page read via `TensorAccessor::get_noc_addr(page_id)` — already uses `TensorAccessor` end-to-end in the tile-layout reader. The RM reader also uses `TensorAccessor::get_noc_addr(stick_id)` (line 35, 74). Clean `TensorAccessor` patterns; the Case 1 classification is because the host side uses the `Buffer*` interim binding rather than the full `TensorParameter` / `TensorBinding` chain.
- `output_tensor` DRAM path (Case 1): kernel uses `experimental::ShardedAddrGen::get_noc_addr(id)` rather than `TensorAccessor`. This is a mapping-table-based sharded accessor, not a standard interleaved `TensorAccessor`. Whether the standard `TensorAccessor` can address a DRAM-sharded output directly is worth confirming — this may be convertible-but-requires-verification. Flagged as Case 1 pending user confirmation; if the `TensorAccessor` iteration model cannot express the sharded DRAM access pattern, this would escalate to Case 2.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ⭐ blocked — the Device 2.0 gate already captured these. The out-of-directory coupling is the primary scheduling dependency.

**Summary table** (per-kernel, function-call escapes):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_unary_sharded_blocks_interleaved_start_id.cpp` | `api/dataflow/dataflow_api.h`, `circular_buffer.h`, `noc.h`, `noc_traits.h` | LLK/HAL | ✓ no concern |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `api/dataflow/dataflow_api.h`, `circular_buffer.h` | LLK/HAL | ✓ no concern (these are the legacy free-function paths it calls — no cross-family headers) |
| `writer_unary_sharded.cpp` | `api/dataflow/dataflow_api.h`, `circular_buffer.h` | LLK/HAL | ✓ no concern |
| `writer_unary_sharded_blocks_start_id.cpp` | `ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` | Cross-family (CCL) | ⚠ see below |
| `writer_unary_sharded_stick_layout_start_id.cpp` | `ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` | Cross-family (CCL) | ⚠ see below |
| `eltwise_copy.cpp` | `api/compute/common.h`, `api/compute/tile_move_copy.h`, `api/compute/eltwise_unary/eltwise_unary.h` | LLK/HAL | ✓ no concern |

**Per-call detail — `sharding_addrgen.hpp` consumers:**

The DRAM-output writer kernels include `ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` (CCL, cross-family) and use:
- `experimental::ShardedAddrGen<tensor_shard_info>` instantiation with `.bank_base_address = dst_addr`
- `experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(8))` (or `get_arg_addr(6)` in the RM writer)
- `s.get_noc_addr(tile_id)` / `s.get_noc_addr(stick_id)` — returns `uint64_t`

The return type of `get_noc_addr` is `uint64_t` (a pre-composed NOC address). In the Device 2.0 / Metal 2.0 world, this raw `uint64_t` does not attach to a typed endpoint. After Device 2.0 migration of the writer kernels, the `get_noc_addr` output will need to be consumed via a `UnicastEndpoint`/`Noc::async_write` path. The `sharding_addrgen.hpp` header itself carries a comment at line 267 noting this: "Consumers migrating to Device 2.0 should use `get_noc_addr()` / its components together with `Noc::async_read` directly." The CCL team owns `sharding_addrgen.hpp`; coordination is needed for whether the donor exposes a Device-2.0-compatible form or whether the consumers adapt the returned address.

Shape classification: the function returns a `uint64_t` NOC-encoded address — "sem addr" column equivalent for the tensor-accessor domain — mapping to `✗ not OK` or `⚠ workable` depending on whether a `get_sharded_addr`-equivalent that returns an endpoint object is added to the CCL donor. Route to the CCL + Device 2.0 teams to determine the forward path.

Note: `experimental::ShardedAddrGen` here is the CCL kernel-library sharding helper (`ttnn/operations/ccl/kernel_common/`), **not** the pre-Device-2.0 `tt_metal` address generator. It is a distinct type and does not trigger the Shape 4 / Device 2.0 gate. The Device 2.0 gate fires for the DRAM writer kernels on the `noc_async_write` / `noc_async_write_barrier` calls, not on the addr-gen type.

**Borrowed kernel files (file-path kernel instantiation):**

All five dataflow kernels referenced by the factory are owned by the shared `data_movement/sharded/device/kernels/dataflow/` pool, not by `interleaved_to_sharded` itself.

| Kernel file | Owning pool | Other known consumers |
|---|---|---|
| `reader_unary_sharded_blocks_interleaved_start_id.cpp` | `sharded/device/kernels/dataflow/` | `interleaved_to_sharded_partial`, `tilize`, `tilize_with_val_padding`, `transpose_wh_sharded`, `untilize`, `reduction/generic` |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `sharded/device/kernels/dataflow/` | `interleaved_to_sharded_partial`, `tilize`, `untilize` |
| `writer_unary_sharded_blocks_start_id.cpp` | `sharded/device/kernels/dataflow/` | `interleaved_to_sharded_partial`, `experimental/padded_slice`, `experimental/transformer/nlp_kv_cache_load_slice`, `reduction/generic` |
| `writer_unary_sharded_stick_layout_start_id.cpp` | `sharded/device/kernels/dataflow/` | `interleaved_to_sharded_partial`, `experimental/padded_slice` |
| `writer_unary_sharded.cpp` | `sharded/device/kernels/dataflow/` | `tilize`, `tilize_with_val_padding`, `untilize` (multiple factories) |

**Port-together coupling:** any Metal 2.0 rewrite of these kernels is a single shared rewrite. The full port-together set (ops sharing these kernel files) must migrate in one coordinated change. This affects the scheduling and scope of the eventual port for `interleaved_to_sharded`.

### Relaxation candidates (mined from custom hash before deletion)

The custom hash at `device/interleaved_to_sharded_op.cpp:141` keys on:
- `output_mem_config`, `output_dtype`, `keep_l1_aligned`
- `input.dtype()`, `input.memory_config()`, `input.layout()`, `input.padded_shape()`

The hash does **not** include `input.tensor_spec()` (which captures tile shape, page config). Since the op validates that tiles are standard 32×32, a cache miss on non-standard tiles is handled by validate. However, `TensorSpec` also encodes page config and buffer layout details beyond what `padded_shape` captures — the custom hash may be subtly incomplete. **FALLIBLE — verify before drawing relaxation conclusions from this hash.** The default hash after deletion is correct-by-construction.

Potential relaxation candidate: `match_padded_shape_only` might be safe for the input tensor if only `padded_shape` and `dtype` govern the program, but the custom hash's omission of `TensorSpec` makes this uncertain. Verify against the factory's actual branch conditions before applying any relaxation.

### TTNN factory analysis

1. **Op-owned tensors:** No. The `create_output_tensors` method at `device/interleaved_to_sharded_op.cpp:130` creates the op's declared output tensor via `create_device_tensor`. No intermediate or scratch tensors are allocated in the factory.

2. **MeshWorkload needed:** No. The op has no op-owned tensors and does not define `create_mesh_workload` / `create_workload_descriptor` / `cached_mesh_workload_t`. No genuine cross-device or multi-program need.

3. **Pybind `create_descriptor`:** No. `interleaved_to_sharded_nanobind.cpp` binds the op via `ttnn::bind_function<"interleaved_to_sharded">` — the normal user-facing function surface. No `nb::class_<...ProgramFactory>` binding.

4. **Other migration-risky pybind:** None. The nanobind file only binds the two public API overloads. No DeviceOperation or factory internals are exposed.

5. **Custom hash:** Yes, at `device/interleaved_to_sharded_op.cpp:141`. See Custom program hash subject. Port deletes this.

6. **Custom `override_runtime_arguments`:** No. No `override_runtime_arguments` definition anywhere in the op.

## Misc anomalies *(team-only, non-gating)*

- **Dead RTA in RM reader:** `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:12` declares `block_height = get_arg_val<uint32_t>(2)` (index 2), skipping index 1. The host (`factory.cpp:391`) pushes `num_units_per_row` at index 1, but the kernel never reads it. This `num_units_per_row` RTA is dead on the kernel side. Route to the op owner; the port does not act on it.

- **`keep_l1_aligned` forced true:** `factory.cpp:65`: `bool keep_l1_aligned = true;  // operation_attributes.keep_l1_aligned;` — the operation attribute is declared, passed through the hash, but never actually read (always forced to `true`). The commented-out line indicates this was intentional for backward compatibility (issue #32752), but the attribute continues to flow through `compute_program_hash` even though it has no effect on the program. This is a dead-but-hashed attribute pattern. Route to the op owner.

## Questions for the user

1. **DRAM-output TensorAccessor / ShardedAddrGen (Case 1 vs Case 2):** The DRAM-output writer kernels use `experimental::ShardedAddrGen` (CCL kernel library, `sharding_addrgen.hpp`) to compute NOC addresses for writing tiles/sticks to a DRAM sharded output. The standard `TensorAccessor` supports interleaved DRAM access; it is unclear whether it can address a DRAM-sharded destination (i.e., whether `TensorAccessor` can express the same shard-mapping logic). Is the DRAM-sharded write pattern genuinely exotic (→ Case 2: bind as TensorParameter, extract base via `get_bank_base_address`, leave ShardedAddrGen in place), or can it be re-expressed via `TensorAccessor` (→ Case 1)? `writer_unary_sharded_blocks_start_id.cpp:27-38` and `writer_unary_sharded_stick_layout_start_id.cpp:22-33`.

## Recipe notes

- The recipe's Device 2.0 compliance check (Check 2) says to confirm whether `get_local_cb_interface(cb_id)` is "sanctioned" — the recipe explicitly lists it as sanctioned alongside `get_tile_size(cb_id)`. The stick-layout reader uses `get_local_cb_interface(cb_id_in1).fifo_page_size` at line 53. Per recipe guidance this is sanctioned (not a holdover), and I recorded it as such.
- The `experimental::ShardedAddrGen` in `ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` shares a name with the legacy `ShardedAddrGen` in the Device 2.0 forbidden list but is a completely different type (different namespace, different header, different design). The recipe's Shape 4 check calls out `ShardedAddrGen` generically — an auditor pattern-matching on name alone could flag this incorrectly. Suggest the recipe add a note distinguishing `tt_metal::ShardedAddrGen` / the legacy addr-gens (forbidden) from `experimental::ShardedAddrGen` in the CCL kernel library (separate, newer construct). I confirmed the distinction by reading the header and did not flag it as Shape 4.
