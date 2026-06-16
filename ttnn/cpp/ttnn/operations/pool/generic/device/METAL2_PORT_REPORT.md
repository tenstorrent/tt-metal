# Metal 2.0 Port Report — Pool2D (`pool/generic`)

Factory: `Pool2D::MultiCore` — ported `create_workload_descriptor` (WorkloadDescriptorFactoryConcept)
→ `create_program_spec` (MetalV2FactoryConcept / `ttnn::device_operation::ProgramArtifacts`).

Scope as scoped by the invoker: the **L1 config path** (`!config_tensor_in_dram`), non-`return_indices`
path (`reader_pool_2d.cpp` + `compute_pool_2d.cpp`). The DRAM-config path and the MPWI / return_indices
path are gated out of the spec build and remain BLOCKED (see below).

## TTNN ProgramFactory

- Concept realized: `MetalV2FactoryConcept` (single-program, structurally identical per mesh coord —
  confirmed by the legacy comment at the old per-coord copy loop).
- Custom `compute_program_hash`: **KEPT** (per invoker instruction — it is shared and load-bearing for
  the pool family). This is a deliberate deviation from the recipe's default "delete the custom hash"
  rule; recorded here as instructed.
- Pybind entry points removed: none (the op's nanobind does not reference the factory method directly).

## Op-owned tensors (NEW capability exercised)

Two host-populated, sharded device tensors are pushed into `ProgramArtifacts::op_owned_tensors`:

1. **reader_indices** — the per-core sliding-window halo lookup table built on host
   (`sliding_window::move_config_tensor_to_device`). Sharded L1 placement. Backs the
   `reader_indices` DFB via `borrowed_from`.
2. **scalar_config** (avg-pool, `!one_scalar_per_core` only) — the per-output-stick scalar config
   table (`create_scalar_config_tensor`), uploaded to L1_SMALL sharded. Backs the `config` DFB via
   `borrowed_from`. Omitted entirely when `one_scalar_per_core`.

Both are populated ttnn::Tensors with sharded placement — pushed straight in, NOT re-allocated empty
(per the branch's op-owned capability). TensorArguments referencing them are built against the
vector's elements after population (identity footgun honored).

## Dropped / changed plumbing

- **Live-allocator introspection DROPPED** (legacy `pool_multi_core_program_factory.cpp:996-1012`):
  `input.device()->allocator()->get_statistics(L1).total_allocated_bytes` and the
  `actual_global_cb_size` / `is_graph_capture_no_dispatch_mode` TT_FATAL guards. This is a host-side
  sanity guard with no functional effect on the program, and the framework gap list names
  live-allocator introspection as unsupported. The local-CB-size TT_FATAL that did not need the
  allocator is also dropped because it was paired with the global one (both were debugging guards on
  the legacy CB accounting, which no longer exists once CBs become DFBs).
- **`buffer->address()` through CTAs GATED OUT** (legacy `:791-794`): `config_buffer->address()`,
  `config_buffer->page_size()`, `reader_indices_buffer->address()`, `reader_indices_buffer->page_size()`
  fed CTA slots 33/34/35/36 and were consumed ONLY under `if constexpr (config_in_dram)` in the kernel
  (the DRAM path). In the L1 path they are dead. The ported spec hard-sets `config_in_dram = 0` and
  passes these four CTAs as named `0` constants, so no raw address crosses the CTA channel. The DRAM
  path STAYS BLOCKED.
- **`TensorAccessorArgs(...).append_to(reader0_ct_args)` DROPPED** (legacy `:816-819`): only the DRAM
  path's `load_config_tensor_if_in_dram` consumed these (`TensorAccessorArgs<reader_tensor_args_index>`).
  Gated out with the DRAM path.
- Magic CB indices in CTAs → `DFBBinding` (every `*_cb_id` CTA slot).
- Positional CTAs → named CTAs.

## Blockers (precise)

- **DRAM-config path** (`config_tensor_in_dram == true`): threads `reader_indices_buffer->address()`
  and `scalar_config_buffer->address()` through CTAs into the kernel's
  `load_config_tensor_if_in_dram<dram_addr, page_size, tensor_args_index, cb_id>()` +
  `TensorAccessorArgs<N>()` plumbing. This is raw-address-through-CTA into a data-movement kernel — a
  hard stop signal. The spec build `TT_FATAL`s if `config_tensor_in_dram` is requested.
- **return_indices / MPWI path** (`reader_mpwi.cpp` + `compute_mpwi.cpp`): out of scope for this port;
  the spec build `TT_FATAL`s if `return_indices` is requested. (Not a framework blocker — just not done.)

## Applied patterns

- Borrowed-memory DFBs (`borrowed_from`) for input (`raw_in`), output (`out`), reader_indices, config.
- Aliased DFBs (`advanced_options.alias_with`) for the pre_tilize / fast_tilize pair (legacy aliased CB).
- Conditional / optional DFB bindings: `in_scalar_1` (split-reader second scalar), `in_1`
  (split-reader second input), `pre_tilize`/`fast_tilize` (tiled output), `config` (avg-pool).
- Preserved multiplicity: reader0 / reader1 are two KernelSpecs of the same source when split_reader.

## Open items for downstream

- `raw_in` DFB is a borrowed-memory **tensor-local-view** fake-CB: the reader reads it only as an
  address source (`in_shard_cb.get_read_ptr()`), never as a FIFO. Bound as a borrowed-from DFB with a
  CONSUMER endpoint on the reader. (No producer kernel writes it — it is the input shard.) Flag for the
  eventual local-`TensorAccessor` migration.
- The DRAM-config path and the return_indices/MPWI path remain on the legacy WorkloadDescriptor concept
  and need a follow-up port (the latter needs the same treatment for reader_mpwi/compute_mpwi).
