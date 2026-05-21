# Pre-port audit: `ttnn/cpp/ttnn/operations/data_movement/clone/`

Op-local kernels only — no cross-op kernels.

- **`CloneOperation`**
  - `ProgramFactory` (`device/clone_program_factory.cpp`) — single factory, branches internally on `tilized` × `is_sharded` to select kernel files.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**YELLOW** — primary blocker resolves with user override on Check 1 framing; one substantive YELLOW under Check 3 affects the sharded code paths. Three questions for the user (see [Questions for the user](#questions-for-the-user)).

### Clean subset (interleaved paths)

The two interleaved kernel pairs (`read_kernel.cpp`/`write_kernel.cpp` for tilized; `read_kernel_rm.cpp`/`write_kernel_rm.cpp` for row-major) use `TensorAccessor` cleanly — Check 3 GREEN for these. If the user opts for a scoped-subset port, this is the clean path.

### Sharded subset (yellow side-issue)

The four sharded kernels (`read_kernel_sharded.cpp` / `write_kernel_sharded.cpp` / `read_kernel_rm_sharded.cpp` / `write_kernel_rm_sharded.cpp`) take a `buffer.address()` RTA and call `get_noc_addr(input_buffer_address)` directly, then loop with `local_l1_read_addr += stick_size`. No `TensorAccessor`. The access is inherently iterable (it's a contiguous stride), so it does not fit the "exotic NoC walks" case the audit's Check 3 YELLOW guidance describes as `TensorAccessor`-incompatible — but the kernel as written doesn't use `TensorAccessor`. See [Questions for the user](#questions-for-the-user) Q2.

## Porting prerequisites

### ProgramDescriptor API: **RED (per the audit doc) — but see Q1 below**

Per the audit doc's Check 1 strict reading, this op uses the imperative `host_api.hpp` builder style — not `ProgramDescriptor` — so Check 1 is RED. Definitive signals:

- `clone_program_factory.cpp:23`: `Program program;` (default-construction)
- `clone_program_factory.cpp:104`: `CreateCircularBuffer(program, all_cores, src_cb_config);`
- `clone_program_factory.cpp:148–152`: `CreateKernel(program, read_kernel_path, all_cores, ReaderDataMovementConfig(...))` and same for writer
- `clone_program_factory.cpp:164`: `CreateKernel(program, ..., ComputeConfig{...})`
- `clone_program_factory.cpp:191`, `199`, `208`, etc.: `SetRuntimeArgs(program, read_kernel_id, core, {...});`

No `ProgramDescriptor`, no `KernelDescriptor`, no `CBDescriptor`, no `desc.cbs.push_back(...)`. The header `clone_device_operation.hpp:28–47` declares the oldest `ProgramFactoryConcept` shape (`cached_program_t create(...)` + `override_runtime_arguments(...)`).

The audit doc's prescribed action for Check 1 RED is: "Report to the user that `ProgramDescriptor` migration is a prerequisite — substantial, standalone, separate PR. Do not attempt it as part of this audit. Do not bundle it with anything. Do not propose a partial conversion."

**This audit's user has explicitly instructed a direct port to `ProgramSpecFactoryConcept`** — skipping the `ProgramDescriptor` intermediate stop entirely. See Q1.

### Device 2.0 DM: **YELLOW — isolated holdovers across all kernels**

All nine clone kernels include `api/dataflow/circular_buffer.h` and use the `CircularBuffer` wrapper (the lowercase form, not the `experimental::CircularBuffer` form — but it's the same Device 2.0 wrapper class). Most calls use the wrapper instance methods (`src_cb.reserve_back(1)`, `src_cb.get_write_ptr()`, etc.), which is Device 2.0 compliant.

The legacy CB-index-keyed free-function holdover is:

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `device/kernels/read_kernel.cpp` | 21 | `get_tile_size(src_cb_id)` | `src_cb` (line 18) |
| `device/kernels/write_kernel.cpp` | 21 | `get_tile_size(dst_cb_id)` | `dst_cb` (line 18) |
| `device/kernels/read_kernel_sharded.cpp` | 15 | `get_tile_size(src_cb_id)` | `src_cb` (line 13) |
| `device/kernels/write_kernel_sharded.cpp` | 15 | `get_tile_size(dst_cb_id)` | `dst_cb` (line 13) |
| `device/kernels/compute_kernel.cpp` | 15 | `get_tile_size(src_cb_id)` | `src_cb` (line 14) |

Each is a 1-line mechanical fix (`get_tile_size(src_cb_id)` → `src_cb.get_tile_size()`). Default recommendation per the audit doc: fold into the Metal 2.0 port as port-time cleanup.

### TensorAccessor usage: **YELLOW — clean for interleaved kernels; sharded kernels read tensor memory without `TensorAccessor`**

The four interleaved kernels use `TensorAccessor` cleanly:
- `device/kernels/read_kernel.cpp:16`: `constexpr auto src_args = TensorAccessorArgs<1>();` + `TensorAccessor(src_args, input_buffer_address)` (line 20)
- `device/kernels/read_kernel_rm.cpp:17`: `constexpr auto input_args = TensorAccessorArgs<2>();` + `TensorAccessor(input_args, input_buffer_address)` (line 21)
- `device/kernels/write_kernel.cpp:16,20`: matching writer
- `device/kernels/write_kernel_rm.cpp:17,21`: matching writer

These are Check 3 GREEN.

The four sharded kernels read tensor memory **without** `TensorAccessor`:
- `device/kernels/read_kernel_sharded.cpp:9,15`: `input_buffer_address = get_arg_val<uint32_t>(0); local_l1_read_addr = get_noc_addr(input_buffer_address);`
- `device/kernels/read_kernel_rm_sharded.cpp:9,15`: same pattern
- `device/kernels/write_kernel_sharded.cpp:9,16`: same pattern (output side)
- `device/kernels/write_kernel_rm_sharded.cpp:9,16`: same pattern

These kernels take the buffer's local L1 base address as an RTA and self-reference via NoC. They iterate by simple offset advance (`local_l1_read_addr += tile_size;` or `+= stick_size;`). The access is fundamentally a contiguous stride through the local shard's L1 — NOT one of the "exotic NoC walks; sub-page access; address arithmetic the iterators don't support" cases the audit's Check 3 YELLOW guidance describes.

Causal-link gate (per the audit doc): these kernels do NOT read tensor data through `cb_*.wait_front` / `cb_*.get_read_ptr` from a borrowed-memory CB. The factory does not construct any borrowed-memory CBs (`CBDescriptor::buffer` set, or `set_globally_allocated_address`, etc.) — confirmed below. So the borrowed-memory causal-link exemption does not apply.

The compute kernel (`device/kernels/compute_kernel.cpp`) reads only from CBs and not from tensor memory. Out of scope for Check 3 per the audit doc.

The right answer for the sharded kernels is one of:
- **(a) Convert to `TensorAccessor`** in a prior, separate Device 2.0 / TensorAccessor PR. The access pattern is iterable; this is what the audit doc's Check 3 RED case prescribes.
- **(b) Use borrowed-memory DFB** (LANDED in Metal 2.0). Declare the input and output as borrowed-memory DFBs backed by their respective `TensorParameter`s. Kernel-side: read/write through `dfb::*` handles rather than a raw L1 address. **This requires kernel-side restructure beyond the recipe's "sanctioned kernel changes" set** (introducing new DFB wrapper access where there was none).
- **(c) Keep the buffer-address RTA** as an escape hatch (the audit doc's documented Check 3 YELLOW override path). The kernel keeps the `get_arg_val<uint32_t>(0)` line; the host declares a buffer-address RTA. This works but defeats Principle 2 (named bindings) — it's the documented escape hatch, not the right shape long-term.

See Q2 below.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer` references in the op |
| Dynamic CircularBuffer (CB on borrowed memory) | N/A | No `CBDescriptor::buffer` set; no `set_globally_allocated_address`. (Imperative API in use here, but CB construction is just `CircularBufferConfig` + `set_page_size` — no borrowed memory.) |
| CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` literal; imperative `CircularBufferConfig` chain has no `set_address_offset` call |
| Aliased Circular Buffers | N/A | Each `CircularBufferConfig` has a single buffer index (`src_cb_id` or `dst_cb_id`) |
| GlobalSemaphore | N/A | No semaphores at all |
| Non-zero semaphore initial value | N/A | No semaphores at all |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | grep for `ArgConfig::Runtime` in the op directory finds nothing |
| `UpdateCircularBuffer*` | GREEN | grep finds nothing |

No detail sections required — all rows are GREEN or N/A.

## Path forward

The substantive choices are around the Check 1 framing question (Q1) and the sharded-kernels question (Q2). On the user's explicit override of Q1 plus a choice on Q2:

- **If the user picks (a) for Q2** (convert sharded kernels to `TensorAccessor` first, in a separate prior PR), this audit ends YELLOW with a documented prerequisite. Re-audit after that lands.
- **If the user picks (b) for Q2** (borrowed-memory DFBs for sharded paths, with kernel restructure), the port is feasible but the kernel changes are larger than the recipe's "sanctioned kernel changes" list contemplates. This is itself a finding worth surfacing back to the porting docs.
- **If the user picks (c) for Q2** (keep buffer-address RTA as escape hatch on the sharded paths), the port can proceed mechanically, accepting the buffer-address RTA as documented escape. The interleaved paths get clean `TensorBinding`/`TensorAccessor(ta::name)`; the sharded paths still surface a buffer-address slot. This is the lowest-friction option and aligns with the "escape hatch" wording in the audit doc.
- **If the user opts for a scoped-subset port** (interleaved-only): port the four interleaved kernels and the factory's interleaved branches; leave the sharded branches on the legacy imperative API in a hybrid factory shape. Reported as YELLOW (scoped-subset) per the audit doc.

## Questions for the user

1. **Check 1 framing — direct port from `host_api.hpp` to `ProgramSpecFactoryConcept`, skipping `ProgramDescriptor`?**

   The audit doc's Check 1 categorically reds-out ops still on imperative `host_api.hpp` builder style with the rationale that `ProgramDescriptor` migration is a substantial, standalone prerequisite PR. The clone op is on that older style (signals listed under [ProgramDescriptor API](#programdescriptor-api-red-per-the-audit-doc--but-see-q1-below)). Your task description explicitly directs a direct jump to `ProgramSpecFactoryConcept`, bypassing the intermediate stop. Confirm — and if confirmed, this is a doc-evolution data point: the audit doc's Check 1 doesn't currently account for cases where a direct jump is sanctioned. (Suggested doc resolution: either soften Check 1's prescribed action to "stop and ask" rather than "refuse," or add a new tier for the bypass case.)

2. **Sharded kernels — which Check 3 resolution?**

   The four sharded kernels (`read_kernel_sharded.cpp`, `read_kernel_rm_sharded.cpp`, `write_kernel_sharded.cpp`, `write_kernel_rm_sharded.cpp`) take a buffer-address RTA and self-reference via `get_noc_addr(buffer_address)`. They do not use `TensorAccessor`. Per the audit doc's three options for Check 3 YELLOW resolution: (a) convert to `TensorAccessor` in a prior PR; (b) use borrowed-memory DFB with kernel restructure; (c) keep buffer-address RTA as escape hatch.

   The default reading of the audit doc says (a) is correct ("the previous case — kernel laziness or pre-`TensorAccessor` cruft — assume the previous case until the user confirms otherwise"). My read is the same: the access is iterable; `TensorAccessor` should fit. But (c) is the lowest-friction escape if the goal is to land the port quickly.

3. **Device 2.0 DM holdovers — port-time cleanup or separate PR?**

   The five `get_tile_size(cb_id)` holdovers listed under [Device 2.0 DM](#device-20-dm-yellow--isolated-holdovers-across-all-kernels) are trivially mechanical (`get_tile_size(cb_id)` → `cb_obj.get_tile_size()`). Default per audit doc is port-time cleanup. Confirming.
