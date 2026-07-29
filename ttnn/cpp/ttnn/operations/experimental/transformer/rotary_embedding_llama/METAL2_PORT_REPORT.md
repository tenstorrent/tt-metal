# Metal 2.0 Port Report — rotary_embedding_llama

## Outcome

**PORTED** — all three factories (`RotaryEmbeddingLlamaMultiCore` interleaved-prefill,
`RotaryEmbeddingLlamaMultiCorePrefillSharded` sharded-prefill, `RotaryEmbeddingLlamaMultiCoreSharded`
decode) converted to `MetalV2FactoryConcept` and verified. Build clean (`./build_metal.sh --build-tests`,
wormhole_b0). Tests: pre-port baseline 13/13 passed; post-port sweep **410 passed, 5 skipped, 0 failed**
(`test_rotary_embedding_llama.py`, all factories + every factory-2 config path: borrowed fast-path,
reload, interleaved, per-head, cos-padding tail). A follow-up run of prefill 8k/16k added 13 more passes
(423 total, 0 failed); only prefill 128k was left unrun (very slow, single head config, same factory-1
reload path as the covered sizes).

## Provenance

- **Recipe docs (this port):** `156b384a2cf 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`
- **Audit docs (inherited):** `156b384a2cf 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

Verification arch: **wormhole_b0** (the only device on the host is a WH n150; the op's tests are all
`@skip_for_blackhole`, so BH would false-green). Confirmed with invoker.

## TTNN ProgramFactory

### Concept realized
MetalV2FactoryConcept — three sibling factories (`RotaryEmbeddingLlamaMultiCore`,
`RotaryEmbeddingLlamaMultiCorePrefillSharded`, `RotaryEmbeddingLlamaMultiCoreSharded`), each
`create_program_artifacts` → `ProgramArtifacts`. All three ported in this change (they share kernels).

### Device-op-class edits
- Custom compute_program_hash deleted: none (op had none).
- Pybind entry points removed: none (`create_descriptor` was not pybound; only the plain op function is).

### Open items
- **TensorParameter relaxation candidates:** none applied (kept strict). No `ArgConfig::Runtime*` in any
  kernel; the ops are shape-specialized via CTAs, so strict `TensorSpec` matching is correct.
- **Factory-2 placement narrowing (see Friction):** worth a doc note; it is the one non-pure-syntax
  decision in this port.

## Handoff points

None. No capitulation; no shared-kernel `_metal2` fork (the two intra-op shared kernels — writer and
`compute/rotary_embedding_llama.cpp` — convert in place because factories 1 & 2 both port in this
change); no pybind entry point removed (`create_descriptor` was never pybound); no custom-hash deletion;
no out-of-directory kernel edits; no `sem::`/`tensor::` boundary violations (the op has no semaphores and
compute kernels bind no tensors).

## Successes

- **hw_config Style-B catch ([Hardware configuration — compute kernels](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)).**
  The recipe's warning that the two compute-config styles default opposite ways fired correctly. The op
  resolves a TTNN `ComputeKernelConfig` (Style A) but the legacy `ComputeConfigDescriptor` used only two
  of its fields (`math_fidelity`, `fp32_dest_acc_en`) and left the rest at descriptor defaults — crucially
  `math_approx_mode=false` (Precise) even though `init_device_compute_kernel_config(..., /*default*/ true, ...)`
  resolves it to `true`. Routing through `to_compute_hardware_config` would have silently flipped
  `sfpu_precision_mode` to Approximate. Building `ComputeGen1Config` directly (Style B) with only the two
  fields reproduces the legacy compute config exactly. `program_factory.cpp` (all three factories, `hw_config`).
- **Borrowed-DFB validator behavior ([program_spec.cpp:533-552](../../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp)).**
  Confirmed the spec validator registers a `TensorParameter` as "used" via a DFB `borrowed_from` link — so
  the decode factory's five borrowed io tensors need no kernel `TensorBinding` (compute kernels cannot bind
  a `TensorAccessor` anyway). Made the decode port a clean all-borrowed + all-self-loop shape.
- **`get_entry_size()` on DM kernels.** The legacy free helper `get_tile_size(cb_id)` (CB→DFB whitelist §A
  `get_tile_size()` is descriptor-gated / compute-oriented) maps cleanly to the DM-safe FIFO getter
  `dfb.get_entry_size()` (§B), which equals the entry size I set == legacy page size. Used in both readers
  and the writer.

## Friction

### Gaps
- **Merged / partial-shard CBs have no 1:1 Metal 2.0 shape.** Factory 2's legacy fast path emits *two*
  `CBDescriptor`s sharing one `buffer_index` over disjoint core ranges (a borrowed CB on the shard grid + a
  plain placeholder on the remaining cores) because the kernels are placed on **all** cores and a shard may
  not cover them all. Metal 2.0 has no per-node borrowed/plain split for a single DFB (placement is derived,
  one `borrowed_from` per spec). The port resolves this by **narrowing the work unit to the active cores**
  (the cores assigned real work) — a subset of the shard grid whenever the legacy borrow-eligibility
  conditions hold, so every placed node has its own shard to borrow and the placeholder split disappears.
  This is the one place the port is not a pure syntax swap; it is output-identical (legacy idle cores
  produced nothing) but a reviewer/doc-maintainer should know the recipe currently has no worked pattern for
  "legacy all-cores-with-idle-noops + partial-shard borrowed CB." Suggest a patterns-catalog entry.

### Confusion
- **Borrowed DFB over a *subset* of the tensor's shard grid** (factory 2 fast path, partial shard, e.g. the
  `cs_sharded_32` sweep row when it lands on the fast path). This is the first port to borrow where the DFB
  placement (active cores) is a strict subset of the backing tensor's shard grid, rather than exactly equal
  (as in decode). Expected to resolve per-node; verified at test time. If a future framework change requires
  placement == shard grid for a borrowed DFB, this factory's fast path would need revisiting.
- **DFB and tensor accessors sharing a name.** The input DFB accessor and the input `TensorParameter`
  accessor are both named `"input"` (→ `dfb::input` and `tensor::input`). Relied on the `dfb::` / `tensor::`
  namespaces being independent; confirmed OK by build. Worth an explicit note in the migration guide that
  same-named accessors across resource namespaces are fine.

## Open items for downstream

- **Shared kernel touches**: `writer_rotary_embedding_llama_interleaved_start_id.cpp` and
  `compute/rotary_embedding_llama.cpp` are bound by factories 1 & 2. Both factories converted in this
  same change (rung 3, invoker-assigned whole-unit port) — **no `_metal2` fork created**, converted
  in place. No remaining unmigrated consumers.
- Stale `CoreArgs` comment (`multi_core_program_factory.cpp:295`) and redundant pre-loop `matmul_init`
  (`compute/rotary_embedding_llama_sharded.cpp:47`) — routed to ops team, not port work.
