# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/topk`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**What you are porting:** one device operation, `TopKDeviceOperation`, with two program factories, both
`descriptor` concept:

- `TopKSingleCoreProgramFactory` (`device/topk_single_core_program_factory.cpp`) — 3 kernels, 8 CBs,
  no semaphores. Despite the name it spreads `Ht` rows across several cores via `split_work_to_cores`.
- `TopKMultiCoreProgramFactory` (`device/topk_multi_core_program_factory.cpp`) — 6 kernels, 10 CB
  descriptors over two overlapping core ranges, 2 semaphores, plus a cross-core NoC transfer between
  local cores and one final aggregation core.

All nine kernel sources are owned by this op. Nothing is borrowed; no `_metal2` fork exists yet
anywhere in this op's kernel directories; there is no `experimental/quasar` copy of this op.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both factories define
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  (`device/topk_device_operation.hpp:25-36`).
- **Op-owned tensors:** none. The op allocates its two outputs through `create_output_tensors`
  (`device/topk_device_operation.cpp:293-307`), which is ordinary output allocation.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash ·
  `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported
  replacement) · pybind `create_descriptor` · other migration-risky pybind. All confirmed both on the
  readiness sheet and by grep of the op directory.

## Construct — to do

**Tensor bindings** — eight bindings, **every one Case 1**. In each case the base address is fed to a
`TensorAccessor` and all memory access goes through the accessor, so each becomes a `TensorParameter` /
`TensorBinding`, the kernel builds `TensorAccessor(tensor::name)`, and both the address argument and
its `TensorAccessorArgs` plumbing disappear.

| Factory | Binding | Delivered at | Consumed at |
|---|---|---|---|
| single-core | `input` | reader RTA 0 (`topk_single_core_program_factory.cpp:263`) | `reader_create_index_tensor.cpp:40` |
| single-core | `indices` (optional) | reader RTA 3 (`topk_single_core_program_factory.cpp:266-270`) | `reader_create_index_tensor.cpp:33` |
| single-core | `values` output | writer RTA 0 (`topk_single_core_program_factory.cpp:275`) | `writer_binary_interleaved.cpp:30` |
| single-core | `indices` output | writer RTA 1 (`topk_single_core_program_factory.cpp:276`) | `writer_binary_interleaved.cpp:31` |
| multi-core | `input` | reader_local RTA 0 (`topk_multi_core_program_factory.cpp:503`) | `reader_create_index_local_topk.cpp:33` |
| multi-core | `indices` (optional) | reader_local RTA 4 (`topk_multi_core_program_factory.cpp:507-511`) | `reader_create_index_local_topk.cpp:44` |
| multi-core | `values` output | writer_final RTA 0 (`topk_multi_core_program_factory.cpp:536`) | `writer_final_topk.cpp:30` |
| multi-core | `indices` output | writer_final RTA 1 (`topk_multi_core_program_factory.cpp:537`) | `writer_final_topk.cpp:31` |

No Case 2 binding exists, so you will not need the `get_bank_base_address` bridge anywhere. No
borrowed-memory CB exists either, so no `DataflowBufferSpec::borrowed_from`. Note the current delivery
mechanism is already `KernelDescriptor::emplace_runtime_args(core, {mesh_tensor, …})` rather than a raw
`->address()` RTA, so you are replacing a framework-patched binding with the typed one.

**One quirk on the single-core `indices` binding:** the factory hardcodes `GENERATE_INDICES` to `"1"`
(`topk_single_core_program_factory.cpp:198-200`, tracked as GH issue 36329), so the kernel's indices
accessor is compiled out and that RTA plus the appended indices `TensorAccessorArgs` are dead today.
Port the binding as written; do not "fix" the define, and do not delete the binding on the grounds
that it is currently unreachable. The multi-core factory does honour the flag
(`topk_multi_core_program_factory.cpp:348`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` in this op passes exactly two arguments, so
there is nothing to drop.

**CB endpoints.** No multi-binding advanced option is needed anywhere in this op, and there is no dead
CB to drop. Actions, per factory and per node:

*Single-core factory* (one config; all eight CBs over the same core range; every core runs reader,
writer, and compute):

- **self-loop** (compute-only scratch, produced and consumed by the same kernel): `c_2`
  (`transposed_val`), `c_3` (`transposed_ind`), `c_4` (`result_prep_val`), `c_5` (`result_prep_ind`).
- **legal 1P+1C, no action**: `c_0` and `c_1` (reader produces, compute consumes), `c_6` and `c_7`
  (compute produces, writer consumes).

*Multi-core factory, local-core nodes* (reader_local, writer_local, compute_local resident):

- **self-loop**: `c_2` and `c_3` (compute-only scratch); `c_4` and `c_5`, which writer_local touches
  **only** as a raw pointer peek (`final_values_dfb.get_write_ptr()`,
  `writer_local_topk.cpp:45-46`) with no FIFO operation, so it is role-free and one toucher.
- **legal 1P+1C, no action**: `c_0` and `c_1` (reader_local produces, compute_local consumes), `c_8`
  and `c_9` (compute_local produces, writer_local consumes).

*Multi-core factory, final-core node* (reader_final, writer_final, compute_final resident):

- **self-loop**: `c_6` (`final_values`) and `c_7` (`final_indices`), both compute-only scratch.
- **legal 1P+1C, no action**: `c_4` and `c_5` (reader_final produces, compute_final consumes), `c_8`
  and `c_9` (compute_final produces, writer_final consumes).
- **`c_0` and `c_1` have zero endpoints on this node** and are live on every local core. This is **not**
  a dead CB and must **not** be dropped. See the first Watch-for item.

**Two descriptors share one CB index.** `values_cb_index` (`c_8`) is declared twice in the multi-core
factory over disjoint core ranges with **different data formats**: local cores use
`compute_cb_data_format` (`topk_multi_core_program_factory.cpp:273-281`), the final core uses
`value_cb_data_format` (`:283-291`). The comment at `:261-271` explains why the formats differ (bf16
on the local side survives the transposed-layout NoC transfer; the output dtype on the final side
matches the DRAM write). Keep them as two separate specs with their distinct formats; collapsing them
into one changes numerics for bfp8/bfp4 inputs.

## Watch for

- **`c_0` / `c_1` on the multi-core final core — decide the DFB's declared range before you write the
  specs.** Both are declared over `all_cores_range_set`
  (`topk_multi_core_program_factory.cpp:171-191`) but no final-core kernel references either index
  (reader_final and compute_final both use `c_4` / `c_5`; writer_final uses `c_8` / `c_9`). Two
  readings are open: keep the range and accept a DFB spanning nodes where nothing binds it, or narrow
  the two specs to `local_cores_range_set`. Narrowing changes the final core's SRAM layout, which the
  factory deliberately arranged (`:158-168`), so it is not a free simplification. This is Question 1
  in `METAL2_PREPORT_AUDIT.md`; get it answered rather than picking a reading. Whatever you do, do not
  drop these CBs — they are fully live on the local cores.
- **Cross-core SRAM address assumption — verify this first, because its failure is silent.** The
  multi-core transfer works by having a local core read the write pointer of its **own** `c_4` / `c_5`
  instance and use that value as the destination address on the final core
  (`writer_local_topk.cpp:45-50`, used at `:69` and `:89`). That is valid only while a buffer declared
  over a core range set is placed at one common address on every core in that range; the factory
  documents the legacy allocator behaviour it depends on and orders its allocations for it
  (`topk_multi_core_program_factory.cpp:158-168`). Confirm the Metal 2.0 DFB allocator gives the same
  guarantee (Question 2 in the audit). If it does not, the multi-core factory mis-addresses without
  any error.
- **CB naming is actively misleading in the multi-core kernels. Take every DFB binding name from the
  factory, never from the kernel-side variable it lands in.** Three places where the kernel's own name
  contradicts the factory:
  - `reader_final_topk.cpp:22-23` and `writer_local_topk.cpp:25-26` name CTAs 9 and 10
    `final_values_dfb_index` / `final_indices_dfb_index`, but the factory passes
    `gathered_values_cb_index` (`c_4`) and `gathered_indices_cb_index` (`c_5`)
    (`topk_multi_core_program_factory.cpp:377-378`, `:403-404`). The factory's real `final_*` CBs are
    `c_6` / `c_7`, which neither kernel touches.
  - `topk_final.cpp:47-50` names CTAs 0 and 1 `input_dfb_index` / `index_dfb_index` — those are `c_4` /
    `c_5`, the gathered buffers — and CTAs 2 and 3 `input_transposed_dfb_index` /
    `index_transposed_dfb_index`, which are `c_6` / `c_7`. Neither `c_0` nor `c_1` reaches this kernel.
  - `topk_local.cpp` and `topk_final.cpp` share `topk_common_funcs.hpp`, whose parameters carry the
    same names while bound to different CBs in each caller.
- **A raw co-fill exists, but across cores, and it does not need the multi-binding flag.**
  `writer_local_topk.cpp:64-92` writes tile bytes straight into `c_4` / `c_5` under `sender_sem` /
  `receiver_sem` control, while `reader_final_topk.cpp:34-57` does the FIFO bookkeeping
  (`reserve_back` / `push_back`) for those same buffers without writing the data itself. The audit
  already worked this through: the raw writer sits on a different node than the instance it fills, so
  the final core stays at one producer plus one consumer. Do not re-derive it into a multi-binding.
  Both semaphores are plain `SemaphoreDescriptor`s (`topk_multi_core_program_factory.cpp:322-333`)
  that port as ordinary `SemaphoreSpec`s.
- **Cross-op / shared kernels:** `device/kernels/compute/topk_common_funcs.hpp` is **lent** — this op
  owns it and two other ops include it:
  `experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp:13`
  and
  `experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp:24`.
  That list is a **sunset and coordination list, not authorization to convert the header in place.**
  No fork is needed and none should be created: every function in the header takes its CB index as a
  `uint32_t` parameter, and a `dfb::name` token carries a constexpr cast to `uint32_t`, so you can
  pass your named tokens into the existing signatures and leave the header untouched. Both consumers
  are gated on the readiness sheet today, so a bundled conversion is not available anyway.
  `device/kernels/dataflow/topk_dataflow_common.hpp` is private to this op (both of its includers are
  topk readers), and its `generate_index_tile(const uint32_t dfb_id, …)` takes a CB index the same way.
- **RTA varargs:** none. Every kernel reads a fixed set of arguments at constant indices, so all
  runtime args become named args. Argument counts to expect: single-core reader 3 named plus 1 that is
  dead under the forced define, single-core writer 4, single-core compute 1; multi-core reader_local 4
  plus 1 optional, writer_local 1, compute_local 1, writer_final 2, reader_final none.
- **Dead compile-time args you will see while inventorying** (recorded for the ops team in the audit's
  Misc anomalies, **not** yours to remove): `Ht` is unused in `reader_create_index_tensor.cpp:23` and
  `writer_binary_interleaved.cpp:20`, and `sorted` is unused in `topk_local.cpp:108` and
  `topk_final.cpp:60`. Port them across as they stand.
