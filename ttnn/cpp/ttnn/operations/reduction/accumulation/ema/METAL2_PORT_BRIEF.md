# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/accumulation/ema`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports` *(carry this line into the port report's Provenance section)*

## What you are porting

One device operation, one factory, three kernels — all owned by this op:

- **`EmaDeviceOperation`** → **`EmaProgramFactory`** (`device/ema_program_factory.cpp:21-196`)
- `kernels/dataflow/ema_reader.cpp` · `kernels/dataflow/ema_writer.cpp` · `kernels/compute/ema_compute.cpp`

Two things to know before you open the files:

- **The kernels live at `ema/kernels/`, not `ema/device/kernels/`** — unlike the sibling `accumulation` op. The factory's `kernel_source` strings (`ema_program_factory.cpp:143`, `:153`, `:166`) carry the real paths.
- **The factory has no configuration branch.** One core-range set, one kernel triple, three CBs, no sharded/interleaved fork, no split reader, no multicast. Everything below is therefore a single unconditional decision, not a per-config table. Input shape and requested grid change CTA/RTA *values* only (`ema_program_factory.cpp:30-64`), never the program's structure.
- **The op's kernels are already on `DataflowBuffer`** (not the older kernel-side `CircularBuffer` wrapper) — see `ema_reader.cpp:37`, `ema_writer.cpp:37`, `ema_compute.cpp:82-84`. That is the more-migrated state, not something to undo.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returns a plain `tt::tt_metal::ProgramDescriptor` (`device/ema_device_operation.hpp:24-27`).
- **Op-owned tensors:** none. The output is allocated through the ordinary TTNN path (`create_output_tensors` → `create_device_tensor`, `device/ema_device_operation.cpp:91-97`).
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus **other migration-risky pybind**, which surfaces as a `safe` warning that also fails the gate. All `no` on this op. The nanobind file exposes only the user-facing `ttnn::ema` function (`ema_nanobind.cpp:70-80`).

## Construct — to do

**Tensor bindings** (per binding):

- **`input`** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::…)`.
  - Today: the factory pushes the `MeshTensor` itself as arg 0 of the reader's per-core RTAs — `reader_desc.emplace_runtime_args(core, {input, src_start_tile})` (`ema_program_factory.cpp:184`). The kernel reads it as `src_base_addr` (`ema_reader.cpp:21`) and feeds it to `TensorAccessor(src_args, src_base_addr)` (`:34`); all access goes through that accessor (`:43`).
  - After the port: the address arg **and** the accessor-args CTA plumbing both disappear — `TensorAccessorArgs(input).append_to(reader_compile_args)` (`ema_program_factory.cpp:125`) on the host, `constexpr auto src_args = TensorAccessorArgs<1>()` (`ema_reader.cpp:17`) in the kernel. Note the CTA index shift this causes: `total_tiles_per_core` currently occupies CTA 0 with the accessor args starting at 1 (`ema_program_factory.cpp:124`, `ema_reader.cpp:16-17`).
- **`output`** — **Case 1** (via `TensorAccessor`) → same shape on the writer side: `ema_program_factory.cpp:185` → `ema_writer.cpp:21` → `:34` → `:43`, with the CTA plumbing at `ema_program_factory.cpp:128` and `ema_writer.cpp:17`.

Neither binding is Case 2 — no kernel does raw address arithmetic on a base pointer, so **no `get_bank_base_address` bridge is needed anywhere in this op**. Neither is a correctness fix either: the `MeshTensor` overload of `emplace_runtime_args` already auto-registers a binding the framework patches on cache hits (`tt_metal/api/tt-metalium/program_descriptors.hpp:161-163`, `:192-194`), so this is routine translation, not a stale-pointer repair.

**Remaining runtime args after the bindings land** — all nameable, no varargs:

| Kernel | Arg | Legacy site | Note |
|---|---|---|---|
| `ema_reader.cpp` | `src_start_tile` | `get_arg_val<uint32_t>(1)` (`:22`), set at `ema_program_factory.cpp:184`/`:186` | per-core page index into the accessor; keep as a named RTA |
| `ema_writer.cpp` | `dst_start_tile` | `get_arg_val<uint32_t>(1)` (`:22`), set at `ema_program_factory.cpp:185`/`:187` | same |
| `ema_compute.cpp` | — | — | reads no runtime args at all |

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessor constructions are two-argument (`ema_reader.cpp:34`, `ema_writer.cpp:34`). Nothing to drop.

**CB endpoints:** three CBs; `c_0` and `c_1` are already legal 1:1, `c_2` needs the self-loop.

| CB | Factory name | Kernel name | Binding to declare |
|---|---|---|---|
| `c_0` | `src_cb_index` (`ema_program_factory.cpp:78`, CB at `:92-100`) | `src_cb_idx` | reader **PRODUCER** (`ema_reader.cpp:42`, `:45`) + compute **CONSUMER** (`ema_compute.cpp:102`, `:107`). Roles already fixed by the FIFO ops — nothing to decide. |
| `c_1` | `dst_cb_index` (`:79`, CB at `:102-110`) | `dst_cb_idx` | compute **PRODUCER** (`ema_compute.cpp:122`, `:126`) + writer **CONSUMER** (`ema_writer.cpp:42`, `:45`). Same. |
| `c_2` | `prev_cb_index` (`:80`, CB at `:112-120`) | `trp_cb_idx` | **self-loop** — bind the compute kernel **both PRODUCER and CONSUMER**. |

Why `c_2` is a self-loop and not a two-toucher: the compute kernel is its **only** toucher, and it drives both FIFO ends itself — it packs the EMA result into `c_2` (`ema_compute.cpp:109`, `:111`, `:113`), then re-unpacks it from `c_2` to transpose back before packing into `c_1` (`:116`, `:118`, `:120`). A tile round-trip through SRAM to get a second transpose. There is no second kernel to assign a role to, so 1P+1C does not apply; the self-loop is legal on Gen1 for compute kernels, leaves the kernel code untouched, and runtime behavior is identical. *(A DM self-loop would be rejected only at the later Quasar-uplift stage; this is a compute kernel, and either way that is not a Gen1 concern.)*

Also worth knowing while you are in that file: **`c_2`'s host-side name is misleading.** The factory calls it `prev_cb_index` / `prev_cb_size`, but it does **not** hold the previous EMA output — that lives in an SFPU register cleared by `ema_clear_previous_output()` (`ema_compute.cpp:99`). The kernel's own name for it, `trp_cb_idx` (`:80`), is the accurate one. Don't take the host name as evidence about the buffer's role; and don't rename it either — the naming mismatch is recorded as an anomaly for the ops team, not port work.

**Tile-size lookups move onto the object.** `get_tile_size(src_cb_idx)` (`ema_reader.cpp:30`) and `get_tile_size(dst_cb_idx)` (`ema_writer.cpp:30`) have a `DataflowBuffer` method equivalent (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167`), so the port moves them onto the bound DFB handle. The free-function form is *sanctioned* at the Device 2.0 stage — it is not a gate you are working around, just the whitelisted Metal 2.0 tidy-up.

**Leave the compute-side CB-index calls alone.** `compute_kernel_hw_startup(src_cb_idx, dst_cb_idx)` (`ema_compute.cpp:93`), `transpose_init(src_cb_idx)` (`:95`), `transpose_tile(cb, …)` (`:104`, `:118`), and `pack_tile(dst, cb)` (`:111`, `:124`) are compute LLK APIs with **no** `DataflowBuffer` method equivalent. They take the CB index; that is their current API surface. They are not Device 2.0 holdovers and not yours to change — pass the bound handle where its implicit `uint32_t` conversion applies, and otherwise leave the call shape as it is.

## Watch for

- **CB endpoints (multi-binding):** none. All three multi-toucher faces were hunted and came back negative: no kernel in this op calls `get_write_ptr()`, `get_read_ptr()`, or `get_local_cb_interface(...).fifo_*_ptr` on any CB (so no hidden raw co-fill and no raw second reader), the op declares **no semaphores at all** (so the coordination mechanism a hidden co-fill needs does not exist), and each of the three `KernelDescriptor`s has a **distinct** `kernel_source` (`ema_program_factory.cpp:143`, `:153`, `:166`), so there is no dual-instance work-split. You do not need to re-run this hunt.
- **Cross-op / shared kernels:** none — **no fork needed, and no fork convention to apply.** This op owns all three kernel files and is their only binder: a filename census over `ttnn/cpp/ttnn/operations/` returns exactly one hit each (`ema_program_factory.cpp`). No `_metal2` sibling exists beside any of them, and none exists anywhere under `ttnn/cpp/ttnn/operations/reduction/`. Convert the three kernels **in place**; there is no sunset list, no pointer comment to leave, and no other consumer to coordinate with.
  - The one out-of-directory kernel include is `../../../device/kernels/accumulation_common.hpp` (`ema_reader.cpp:11`, `ema_writer.cpp:11`, `ema_compute.cpp:9`) — the in-family constants header shared with `cumsum`/`cumprod`. The EMA kernels call **no function** from it; they use only `ONE_TILE`. It declares no CB id, no semaphore, no accessor, and no `CircularBuffer&` in any signature, so no named Metal 2.0 handle ever has to bridge into it. **It is not yours to modify**, and porting these kernels does not require touching it. (It also defines `CB_IN`/`CB_OUT`/`CB_ACC` aliases for the same three buffers the EMA kernels declare locally — ignore them; the local declarations are what the code uses.)
  - **Do not look for a precedent under `ttnn/cpp/ttnn/operations/experimental/quasar/`.** Nothing there applies to this op, and its `_metal2` files are pre-port copies that carry idioms the current whitelist forbids.
- **RTA varargs:** none — prefer named RTAs throughout. Each dataflow kernel reads exactly two args at **literal** indices `(0)` and `(1)` (`ema_reader.cpp:21-22`, `ema_writer.cpp:21-22`); the compute kernel reads none. No counted loop over args, no data-selected index. After the tensor bindings land, one named RTA per dataflow kernel remains (the `*_start_tile` page index above).
