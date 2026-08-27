# Metal 2.0 Port Report — tilize

## Outcome
**PORTED** — 5 factories {Default, SingleCore, Sharded, ShardedRetile, Retile} converted to
`CustomProgramSpecFactoryConcept`. `TilizeMultiCoreBlockProgramFactory` left on the legacy descriptor
concept (blocked: "Per-node CB size"). `test_tilize.py`: 424 passed / 76 skipped (pre-existing skips) / 0
failed, with the Metal 2.0 legality checks forced on (`METAL2_CHECKS_FORCED` present in the test log). All
five ported factories exercised (Sharded/ShardedRetile via `test_tilize_retile` shard_layout params).

## Provenance
- **Recipe docs (this port):** `7d5ddd43e0e 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `7d5ddd43e0e 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory
### Concept realized
CustomProgramSpecFactoryConcept for {Default, SingleCore, Sharded, ShardedRetile, Retile}. Block factory + dead compute/tilize.cpp stay on the legacy descriptor concept. Override returns `tensor_args` for both io TensorParameters (INPUT, OUTPUT) on every factory — mirrors legacy `patch_tilize_kernel_slot0` + `cb_addr_only` refresh (addresses only; all shape-derived args baked).

### Device-op-class edits
- Pybind entry points removed: none (no pybound `create_descriptor`).
- Custom compute_program_hash: none.
- Op has `program_factory_t` already → no direct-descriptor (exception 3) restructure.

## Handoff points
- **ShardedRetile under-supplies a compute RTA (latent legacy bug; Metal 2.0 forces it visible).**
  `retile.cpp` reads 3 runtime args (`num_input_blocks`, `num_real_input_rows`, `num_real_output_rows`);
  the 3rd was added by #52180 (shrink-case output clamp), which updated only the interleaved
  `TilizeMultiCoreRetileProgramFactory` and left `TilizeMultiCoreShardedRetileProgramFactory` emitting
  only 2 args (`tilize_multi_core_sharded_retile_program_factory.cpp:253` legacy:
  `emplace_runtime_args(core, {num_input_tile_rows, num_input_tile_rows})`). At legacy runtime slot 2 is
  read unset (undefined). The Metal 2.0 named-arg validator requires every declared arg to be set, so the
  port MUST supply `num_real_output_rows`. `num_real_output_rows` is a non-limiting cap in the sharded
  case (no grow-case height padding within a shard — the factory's own comment states "all rows are
  real"), so the port supplies `num_output_tile_rows` (the documented intent). This is a required arg the
  legacy under-specified, not a discretionary fix; the value cannot change numerics versus a
  correctly-working legacy run (the cap never fires). **Ops team: propagate the #52180 3rd-RTA fix to the
  legacy ShardedRetile factory.**

## Successes
- **Aliased DFBs (`advanced_options.alias_with`)** — the audit's flagged highest-uncertainty construct
  (retile `mid`/`mid_view`, two views over one L1 region with different tile geometry) mapped cleanly to
  two `DataflowBufferSpec`s with mutual `alias_with`. `advanced_options.hpp`'s "same total size" rule was
  exactly what the legacy single-`CBDescriptor`-two-format-descriptors expressed; going to the header
  (recipe's "headers first") settled it definitively. `retile.cpp:38`, factories' `mid_dfb`/`mid_view_dfb`.
- **Cursor-surgery whitelist (§D)** — `get_local_cb_interface(mid_cb).fifo_rd_ptr` read → `mid.get_read_ptr()`,
  `get_local_cb_interface(mid_view_cb).fifo_rd_ptr = …` → `mid_view.evil_set_read_ptr(…)`. Units matched
  (16-byte words), so the kernel's own arithmetic was untouched. `retile.cpp:108,119`.
- **CustomProgramSpecFactoryConcept override collapse** — legacy `patch_tilize_kernel_slot0` +
  `cb_addr_only`/`apply_descriptor_runtime_args` refresh reduced to a two-entry `tensor_args` on every
  factory. `ttnn_factory.md`'s "the override owns the tensor bindings too" warning fired correctly — the
  temptation to omit `tensor_args` (assuming the framework patches) is exactly the silent cache-hit bug it
  describes.

## Friction
- **Gap — unpack_modes rejection is gated on `enable_32_bit_dest`, not width alone.** The recipe
  (`Compiler`/`Hardware configuration`) paraphrases the Gen1 rule as "a ≤16-bit format with UnpackToDest —
  rejected on Gen1 as a pure perf loss," reading as absolute. tilize sets `UnpackToDestFp32` on narrow
  (bf16/fp8/uint8) input CBs in several cases, which looked like a blocker. The actual validator
  (`program_spec.cpp:1065`) permits UnpackToDest on ANY format when `enable_32_bit_dest` is true — and here
  the same `fp32_llk_acc` flag drives both, so the mechanical `UnpackToDestFp32 → UnpackToDest` mapping is
  always safe. A one-line note that the ≤16-bit rejection is `enable_32_bit_dest=false`-only would save the
  detour into the validator source.
- **Gap — aliased DFBs with *different* `entry_size`.** The retile alias pair has different entry sizes
  (input-tile vs output-tile geometry) but equal total size. `num_entries` for the second view had to be
  derived as `total_size / entry_size` by hand. The patterns-catalog alias examples (kv_cache, layernorm)
  alias *equal*-entry_size DFBs, so there was no worked example of the unequal-entry_size case the audit
  flagged as highest-uncertainty. A catalog example would help the next porter of a retile-shaped op.

## Open items for downstream
- **Shared-kernel reuse (5 forks, all pre-existing — this op joins each fork's #52228 sunset list, does not authorize in-place conversion):**
  - ttnn/cpp/ttnn/kernel/compute/tilize.cpp → tilize_metal2.cpp (reused)
  - eltwise/unary/.../writer_unary_interleaved_start_id.cpp → _metal2 (reused)
  - untilize/.../reader_unary_start_id.cpp → _metal2 (reused)
  - eltwise/unary/.../reader_unary_sharded.cpp → _metal2 (reused)
  - data_movement/sharded/.../writer_unary_sharded.cpp → _metal2 (reused, #52228)
- **Dead file** `device/kernels/compute/tilize.cpp` unreferenced (all compute paths use the shared-pool `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`) — candidate for deletion, route to ops team; port does not touch it.
- **Block factory** stays legacy pending "Per-node CB size" framework work.
