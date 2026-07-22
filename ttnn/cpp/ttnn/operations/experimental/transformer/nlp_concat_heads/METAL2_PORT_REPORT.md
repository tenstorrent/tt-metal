# Metal 2.0 Port Report — nlp_concat_heads

## Outcome

**PORTED** — the single `NLPConcatHeadsProgramFactory` (both the interleaved and sharded paths of its
`if (in_sharded)` branch) converted to `MetalV2FactoryConcept::create_program_artifacts`. Build green
(`./build_metal.sh --build-tests`); the confirmed no-regression baseline passes **219 passed, 2 skipped,
0 failed** (`test_nlp_concat_heads.py` — interleaved dtype/DRAM-L1 sweep + program-cache test — and
`test_sharded.py::test_sharded_concat_heads` — sharded). The 2 skips are pre-existing config skips,
not port-induced.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — as the audit specified. The device operation keeps its single-type
`program_factory_t = std::variant<NLPConcatHeadsProgramFactory>` and no `select_program_factory`
(single-variant auto-selects). `NLPConcatHeadsProgramFactory::create_descriptor` (returning
`ProgramDescriptor`) was replaced by `create_program_artifacts(const NlpConcatHeadsParams&,
const Tensor&, Tensor&)` returning `ProgramArtifacts`. No device-operation-class edits were forced.

### Device-op-class edits
- Custom compute_program_hash deleted: none (op had no custom hash).
- Pybind entry points removed: none (pybind binds a plain function, not `create_descriptor`).

### Open items
- **Stale factory comments preserved (not fixed — out of scope).** `nlp_concat_heads_program_factory.cpp`
  still carries the legacy `// 142` (a Falcon-specific example on the `per_tensor_tiles` line) and the
  `// Grayskull Device Setup` banner (Grayskull is deprecated / not a Gen1 target). Preserved verbatim
  per scope discipline; flagged here for the owner (matches the audit's "Misc anomalies").
- **RTA→CRTA candidates (not converted — separate cleanup).** In the sharded path every core gets the
  *same* per-instance RTAs (`nheads`, `start_read_offset_bytes`, `start_write_offset_bytes` are constant
  across `all_cores`), and in the interleaved path the writer's args vary per node but the reader/writer
  schemas are per-node by construction. The sharded RTAs are really CRTAs (`common_runtime_arg_values`);
  converting would improve dispatch efficiency but changes dispatch semantics, so left as node-first RTAs
  per the recipe. Candidate for a later name-first / RTA→CRTA pass.
- **Relaxation candidates:** none observed (strict tensor-arg matching is correct here).

## Handoff points
- **Sharded `OUT0` conditional-alloc vs unconditional-bind (ops team).** The sharded kernel references
  `cb_out0` → `OUT0` (`borrowed_from = OUTPUT`) unconditionally, while the legacy factory allocated CB 16
  only under `if (out_sharded)`. `validate_on_program_cache_miss` permits sharded-in with a
  non-HEIGHT_SHARDED (possibly interleaved) output, which would leave `OUT0`'s borrowed backing without
  L1-sharded storage. The audit confirmed sharded-in ⇒ sharded-out holds in practice (a borrowed output
  CB requires L1-sharded output), and the sharded test only exercises sharded output — so the port
  reproduces the legacy behavior faithfully by allocating `OUT0` unconditionally in the sharded branch.
  This is a **pre-existing latent coupling**, not introduced by the port; an explicit assertion in the
  device op (`validate_on_program_cache_miss`) would make it safe, but that is out of the porter's scope
  (device-op-class edit). Reference: audit "Questions for the user" #1 and "Misc anomalies".

## Successes
- **Borrowed-memory DFB needs no `TensorBinding` (migration guide "Borrowed-memory DFBs" +
  `DataflowBufferSpec::borrowed_from`).** The sharded path binds neither input nor output via a
  `TensorAccessor`; both are reached through borrowed CBs. The docs say `borrowed_from` names a
  `TensorParameter` and the address resolves from `tensor_args` — but it wasn't explicit whether the
  borrowed-only `TensorParameter` (no kernel `TensorBinding`) satisfies the "every TensorParameter needs
  ≥1 binding" validator rule. The recipe steered right: declare the `TensorParameter` + `TensorArgument`,
  set `borrowed_from`, and add **no** `TensorBinding`. (Cross-checked against the shipped binary_ng
  Metal 2.0 factory, which does exactly this for its sharded/borrowed operands.) This is what let the
  sharded path stay a clean borrowed-DFB port with zero tensor accessors.
- **Two-toucher → 1P+1C, not the multi-binding flag ([Two-toucher DFB → assign 1P+1C]).** The sharded
  dual-instance work-split (reader-config + writer-config over one grid) looked at a glance like it
  might need `allow_instance_multi_binding`; the recipe's census rule (zero real `push_back`/`pop_front`
  on either CB → role-free → cosmetic 1P+1C) was exactly right and kept the port off the Gen2-forbidden
  flag.

## Friction

### Gaps
- **DFB metadata member getter vs DM kernels (whitelist §A / recipe rule 7).** Rule 7 says prefer the
  `dfb.get_tile_size()` member getter over the free function `get_tile_size(cb_id)`. The whitelist §A
  qualifies that the member getter is only available `#ifdef DFB_DESCRIPTORS_DEFINED` (chlkc_descriptors.h
  present). For a **data-movement** kernel it's non-obvious whether that macro is set — and the one
  already-ported reference (quasar matmul `reader_bmm...`, a DM kernel) used the free-function shim
  `get_tile_size(cb_id)` with `dfb::in0`, not the member getter. Resolved by tracing the legacy free
  function `get_tile_size(operand)` (`dataflow_api.h:280`): it reads `unpack_tile_size[]` under
  `#ifdef DATA_FORMATS_DEFINED`, so the JIT descriptor arrays are already present for this DM kernel —
  which also triggers `DFB_DESCRIPTORS_DEFINED` (same chlkc_descriptors.h). A one-line note in the
  whitelist ("§A getters are available on a DM kernel when it already reads `get_tile_size(cb)` /
  `DATA_FORMATS_DEFINED`") would remove the verification detour.

### Confusion
- **`dfb::`-namespace vs a variable named `dfb`.** The donor writer's `DataflowBuffer` variable was
  literally named `dfb`; writing `DataflowBuffer dfb(dfb::out)` fails to parse (the just-declared
  variable `dfb` shadows the `dfb::` namespace in its own initializer). Renamed to `dfb_out`. The
  kernel-side whitelist rule 1's `cb_* → dfb_*` rename guidance doesn't call out that a bare `dfb`
  identifier is now reserved-ish; worth a sentence.
- **`-Werror,-Wunused-variable` on legacy locals whose only consumers were dropped plumbing.**
  `in0_buffer` (fed the input address RTA + borrowed-CB `.buffer`) and `out_sharded` (gated the
  `if (out_sharded)` CB-16 alloc) became unused once their plumbing was replaced by bindings, and the
  Release build's `-Werror` failed on them. These removals are legitimately part of "Dropped Plumbing"
  but the recipe's "don't clean up variables beyond the API rename" guidance made me hesitate; a note
  that *dead locals left behind by dropped plumbing must be removed* (they aren't a "cleanup", they're
  the tail of the plumbing drop) would clarify.

## Open items for downstream

### Cross-op kernel touches
- **`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`**
  — broadly-shared donor (46 op `.cpp` files reference it: typecast, bcast, concat, copy, permute,
  reshape_on_device, slice, tilize*, transpose*, unary_backward, embedding, examples, …).
  - **Path taken: fork.** New file
    `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
    alongside the legacy copy, carrying the Metal 2.0 rewrite (CB→DFB via `dfb::out`; `dst_addr` RTA +
    `TensorAccessorArgs<1>` CTA → `TensorBinding(OUTPUT)` / `TensorAccessor(tensor::output)`;
    `get_local_cb_interface(cb).fifo_page_size` → `dfb_out.get_entry_size()`; named `num_pages` /
    `start_id` RTAs). The `#ifdef OUT_SHARDED` / `#ifdef BACKWARDS` structure is preserved so future
    co-migrators that set those defines can adopt the fork. `nlp_concat_heads` sets neither.
  - **Remaining unmigrated consumers:** the other ~45 op directories keep the legacy copy.
  - **Sunset:** delete the legacy copy when the last co-borrower ports; until then, bug fixes to the
    legacy copy should be mirrored to the `_metal2` fork (drift discipline).
- Both in-op kernels (`reader_tm_tile_layout_nlp_concat_heads.cpp`,
  `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`) were converted **in place** — they are not shared
  outside this op.

### Other
- **hw_config:** all four kernel specs use the arch-agnostic default helpers
  (`ttnn::create_reader_datamovement_config` / `create_writer_datamovement_config`), matching the legacy
  `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` resolved triples exactly. No custom DM config,
  no compute kernel (so no compute-config / `unpack_modes` concerns).
