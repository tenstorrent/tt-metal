# Metal 2.0 Audit Findings — `moreh/moreh_mean_backward`

Single device-operation, single program factory:

- **`MorehMeanBackwardOperation`**
  - `MorehMeanBackwardOperation (single-descriptor)` (`device/moreh_mean_backward_program_factory.cpp`)

Kernels referenced by the factory (all op-owned, under `device/kernels/`):

- `reader_moreh_mean_backward.cpp` (`ReaderConfigDescriptor`)
- `writer_moreh_mean_backward.cpp` (`WriterConfigDescriptor`)
- `moreh_mean_backward.cpp` (compute; instantiated twice over disjoint core groups `core_group_1` / `core_group_2`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehMeanBackwardOperation` → single-descriptor factory |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — own kernels + shared `moreh_common.hpp` helpers all on `DataflowBuffer`/`Noc`/`TensorAccessor` |
| *Prereqs* — Cross-op escapes | Ok — only shared-pool (`ttnn/cpp/ttnn/kernel/`) + LLK/HAL includes |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (CTAs read at constexpr offsets only) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases) |
| *Port work* — Tensor bindings (per binding) | `output_grad` Case 1 · `input_grad`/output Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd arg at any site) |
| *Port work* — CB endpoints | 4 legal 1:1 · `c_24` self-loop |

## Tensorless-dispatch check (framework block?)

**Not tensorless — GREEN.** `tensor_args_t` is:

```cpp
struct tensor_args_t {
    const Tensor& output_grad;              // required, non-optional
    const std::optional<Tensor>& input_grad;
};
```

`output_grad` is a mandatory non-optional `Tensor&` that is always present at dispatch (it is the first positional argument of `ttnn::prim::moreh_mean_backward` and of the pybind), so the MetalV2 factory adapter can always source the `MeshDevice` from it. Only `input_grad` is optional. There is **no** optional-only / empty-`tensor_args` dispatch path, so the tensorless-dispatch framework BLOCK does **not** apply.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept (`Is able to port? = yes`) ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. No portable-subset scoping needed (single factory). Port work is two Case-1 tensor bindings, one self-loop CB, and an RTA-varargs heads-up on the reader.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (`moreh/moreh_mean_backward`, `MorehMeanBackwardOperation (single-descriptor)`): `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, **`Is able to port? = yes`**. Cross-check against code, all consistent:
  - `Concept = descriptor` ✓ — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/moreh_mean_backward_program_factory.cpp:58`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override in the device op (`device/moreh_mean_backward_device_operation.hpp/.cpp`).
  - `Runtime-args update = no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments` anywhere in the op.
  - `Pybind descriptor = no` ✓ — `moreh_mean_backward_nanobind.cpp` binds the host free function `ttnn::moreh_mean_backward` via `ttnn::bind_function`, not `create_descriptor` / a device-op `nb::class_`.
- **Device 2.0 (every kernel used):** **GREEN.** All three op-owned kernels are structurally Device 2.0:
  - `DataflowBuffer` objects for every CB (`reader:85/89/95`, `writer:24`, `compute:20-28`), `Noc noc;` + `noc.async_read`/`noc.async_write` (`reader:94,103`, `writer:23,31`), `TensorAccessor(args, addr)` (`reader:92`, `writer:21`). No legacy addr-gen (`InterleavedAddrGen*`/`ShardedAddrGen`), no raw `noc_async_read`, no manual CB-index FIFO management.
  - The only CB-index free functions are `get_tile_size(cb_id)` (`reader:96`, `writer:25`) — **sanctioned** (Device 2.0 keeps it) — and `get_compile_time_arg_val` / `get_arg_val` (arg fetch, not a CB holdover). No `get_read_ptr(cb_id)` / `get_write_ptr(cb_id)` free-function holdovers.
  - Shared donor `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (`fill_cb_with_value`, `ArgFetcher::get_next_arg_val`) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (`*_init_short_with_dt`, `pack_tile_with_dt`, `copy_tile_init_with_dt`) — the functions the op calls all take `DataflowBuffer` objects and use wrapper methods (`cb.reserve_back`/`push_back`/`get_write_ptr()`/`get_id()`). Device 2.0 native.
- **Feature compatibility:** all four Appendix A entries **N/A** (no `GREEN` row exists — every entry is UNSUPPORTED).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | plain `CBDescriptor` literals only; no `.global_circular_buffer`, no remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set on any `CBDescriptor` (all default 0) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | CTAs read only at constexpr offsets — `get_compile_time_arg_val(0/1/2)`, `TensorAccessorArgs<N>()`; `tensor_args_t` is a fixed 2-tensor tuple (no `std::vector<Tensor>`); no runtime-varying CTA index |

- **CB endpoints (GATE-free):** per-node census (see Port-work summary). `c_0`/`c_1`/`c_2`/`c_16` are plain 1P+1C (legal); `c_24` (intermed) is single-toucher → self-loop. The two compute `KernelDescriptor`s (`compute_desc_1`/`compute_desc_2`) share one source but cover **disjoint** core groups, so each node runs one compute instance — this is the disjoint-node-set split, not the dual-instance work-split; no extra touchers per node.
- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base. The reader pushes `output_grad.buffer()` (the `Buffer*` overload, `program_factory.cpp:251`) and the writer pushes `input_grad.buffer()` (`:267`) as clean bases; the per-core tile offset is passed as a **separate scalar** (`tile_offset`/`start_id`, `:253` and `:267`) and applied on-device as a page index, never folded into the address. No `->address() + <offset>` expression anywhere. No `ttnn::narrow`.
- **TensorAccessor 3rd argument:** **GREEN / N/A.** Both accessor sites are 2-arg: `TensorAccessor(output_grad_args, output_grad_addr)` (`reader:92`) and `TensorAccessor(input_grad_args, input_grad_addr)` (`writer:21`). No explicit page-size third argument at any site.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, both **Case 1 — via `TensorAccessor`**):
  - `output_grad` — delivered as `output_grad.buffer()` (`Buffer*` overload → framework `BufferBinding`, re-patched on cache hits; the factory comment at `program_factory.cpp:245-249` documents this is deliberate for AdamW-style loops that pass a fresh `output_grad` each step). Kernel feeds the base into `TensorAccessor(output_grad_args, output_grad_addr)` (`reader:92`) and reads by `page_id`. Port: express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA + `TensorAccessorArgs<1>()` CTA plumbing disappear. This is the correct-on-cache-hit interim `Buffer*` form, **not** the silent-wrong `->address()` hazard.
  - `input_grad` / output — same `Buffer*` shape (`program_factory.cpp:267`), fed into `TensorAccessor(input_grad_args, input_grad_addr)` (`writer:21`). Case 1, identical treatment.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, config)` — this op has a single config):
  - `c_0` (input, 2 tiles): reader `reserve_back`/`push_back` (producer) + compute `wait_front`/`pop_front` (consumer) → **1P+1C, legal**.
  - `c_1` (zero, 1 tile): reader `fill_cb_with_value` → `reserve_back`/`push_back` (producer) + compute `wait_front`/`pop_front` (consumer) → **1P+1C, legal**.
  - `c_2` (scalar, 1 tile): reader `fill_cb_with_value` (producer) + compute reads it as the bcast-scalar operand (`compute:64-65`, role-free reader) → **1P+1C, legal**.
  - `c_24` (intermed, 1 tile): compute only — `reserve_back`/`push_back` **and** `wait_front`/`pop_front` in the same kernel (`compute:52,58,63,75`). Single toucher → **self-loop** (bind the compute kernel PRODUCER and CONSUMER).
  - `c_16` (output, 2 tiles): compute `reserve_back`/`push_back` (producer) + writer `wait_front`/`pop_front` (consumer) → **1P+1C, legal**.

## Heads-ups  *(mirrors the brief)*

- **RTA varargs (reader):** `reader_moreh_mean_backward.cpp:48-60` reads three per-dimension blocks — `output_grad_dim[i]`, `input_grad_dim[i]`, `need_bcast_dim[i]` — each in a loop bounded by `input_grad_rank` (CTA arg 0). Per RTA-varargs rule (a), a CTA-bounded count still varies across instantiations, so these three blocks are **genuine RTA varargs** — port them via the kernel-side vararg mechanism, not by naming each element. The four leading scalars (`output_grad_addr`, `num_output_tiles`, `start_id`, `num_dim`, `:42-45`) are fixed named fields. The writer's three args (`:14-16`) are all fixed named fields (non-signal).
- **Cross-op / shared kernels:** the three kernels `#include` `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` and `.../compute/moreh_common.hpp` (shared moreh kernel pool) plus `api/...` LLK/HAL headers. No cross-family donor, no file-path instantiation of another op's kernel. The shared `moreh_common.hpp` is broadly used across the moreh family — a Metal 2.0 syntax rewrite of those helpers is a family-wide port-together unit (though they are already `DataflowBuffer`-typed, so the coupling is light). See Team-only.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**.

  | Op kernel | Donor file | Class | Shape / status |
  |---|---|---|---|
  | reader / writer / compute | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared pool (`ttnn/cpp/ttnn/kernel/`) | funcs take `DataflowBuffer` — Device 2.0 native, ✓ |
  | compute | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared pool | funcs take `DataflowBuffer` — Device 2.0 native, ✓ |
  | all | `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`, `api/compute/*.h` | `tt_metal/*` LLK/HAL | no concern |

  No borrowed/file-path kernel instantiation: all three kernel `.cpp` files are op-owned under `device/kernels/`. The shared `moreh_common.hpp` headers form a moreh-family port-together set for the eventual Metal 2.0 syntax rewrite (they are consumed by many moreh ops).
- **TTNN factory analysis:** op-owned tensors — none (sheet blank; `descriptor` concept cannot carry them). No custom hash, no custom `override_runtime_arguments`, no pybind `create_descriptor`, no smuggled pointer, `Is safe to port? = yes`. Target concept `MetalV2FactoryConcept`.

## Misc anomalies

- None observed. (All RTAs the reader unpacks are consumed; `num_dim` feeds the `1/num_dim` scalar fill at `reader:88`. The `Buffer*`-overload RTA choice for `output_grad`/`input_grad` is intentional and documented in-code, not an anomaly.)

## Recipe notes

- None. The recipe's checks applied cleanly to this single-factory `descriptor` op.
