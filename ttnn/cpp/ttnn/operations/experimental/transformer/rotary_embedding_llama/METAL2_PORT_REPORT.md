# Metal 2.0 Port Report — rotary_embedding_llama

## Outcome

**PORTED** (with one documented framework-gap deviation — see [Handoff points](#handoff-points)) — all
three factories (`RotaryEmbeddingLlamaMultiCore` interleaved-prefill,
`RotaryEmbeddingLlamaMultiCorePrefillSharded` sharded-prefill, `RotaryEmbeddingLlamaMultiCoreSharded`
decode) converted to `MetalV2FactoryConcept` and verified. The port is a faithful syntax swap everywhere
**except** Factory 2's partial-shard fast path, which Metal 2.0 cannot express (single `borrowed_from`,
derived placement — no per-node borrowed/plain split) and which is therefore routed to the pre-existing
reload path: output-identical, a performance-only deviation, recorded as a framework-gap handoff. Build
clean (`./build_metal.sh --build-tests`, wormhole_b0). Tests: post-fix full sweep **423 passed, 5 skipped,
0 failed** (`test_rotary_embedding_llama.py`, all factories + every factory-2 config path: full-shard
borrowed fast-path, partial-shard reload, interleaved, per-head, cos-padding tail); only prefill 128k left
unrun (very slow, single-head, same factory-1 reload path as the covered sizes).

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
- **Factory-2 borrow restricted to full-shard (see Friction):** the port borrows cos/sin/trans_mat only
  when the shard grid covers all cores (faithful all_cores placement) and routes partial shards to the
  reload path. Worth a patterns-catalog note; it is the one place the legacy op has no 1:1 Metal 2.0 shape.

## Handoff points

- **Framework gap — no per-node borrowed/plain DFB split (partial-shard fast path).** Tagged
  "Framework: missing per-node borrowed-memory split / GlobalDataflowBuffer." Owner: Metal 2.0 host-API team.

  *What legacy does:* Factory 2's fast path (`RotaryEmbeddingLlamaMultiCorePrefillSharded`) emits, for a
  **partially**-sharded cos/sin (or trans_mat), a single CB `buffer_index` that is **borrowed** on the shard-grid
  cores and a **plain placeholder** on the remaining cores, with the reader/writer/compute kernels placed on
  `all_cores`. It selects this fast (resident-L1-view) path whenever the shard covers the active cores
  (`num_active <= shard_cores`), even when the shard grid is smaller than the device grid.

  *Why it can't be ported faithfully:* a Metal 2.0 `DataflowBufferSpec` has a single `borrowed_from`, and its
  placement is **derived** from the union of its bound kernels' work-unit nodes — there is no per-node
  borrowed/plain split. So the legacy merged CB is inexpressible for a partial shard: you cannot borrow on some
  placed nodes and be plain on others, and you cannot place the borrowed DFB on `all_cores` because the backing
  shard does not exist on the non-shard nodes (`AttachBorrowedDFBBuffers` would fail the per-bank check —
  `program_run_args.cpp`; the spec-time validator only checks existence/L1/size, `program_spec.cpp:1506-1552`).

  *What the port does instead (deviation on record):* borrow **only when the shard grid covers all cores**
  (`shard_spec()->grid.num_cores() == num_cores`), where borrowed-over-`all_cores` is an exact syntax swap;
  route a **partial** shard to the pre-existing reload / `TensorAccessor` path (layout-agnostic, `all_cores`
  placement, output-identical). This is **output-identical but a performance deviation**: a partial-shard config
  that legacy served from the fast L1 view now takes the slower reload path. It is *not* a pure syntax swap for
  that config class; a strict-recipe port would capitulate on Factory 2's partial-shard fast path. We chose the
  reload fallback (over capitulation) because it introduces no new construct, keeps the op fully ported, keeps
  faithful `all_cores` placement, and matches the op's own test contract (docstring: `N`-core shard `-> reload
  path`; `-1` all-cores `-> fast globally-allocated CB path`). Revisit if Metal 2.0 gains a per-node
  borrowed-memory split or a `GlobalDataflowBuffer` that can back a DFB on a subset of its placed nodes.

No other handoffs: no shared-kernel `_metal2` fork (the two intra-op shared kernels — writer and
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
  one `borrowed_from` per spec), so the legacy merged CB is genuinely inexpressible for a *partial* shard.
  **Resolution (recipe-faithful):** borrow cos/sin/trans_mat **only when the shard grid covers all cores**
  (`shard_spec()->grid.num_cores() == num_cores`), where borrowed-over-all_cores is an exact syntax swap of
  the legacy borrowed CB; route a **partial** shard to the existing reload / `TensorAccessor` path, which is
  layout-agnostic and runs on all_cores with no placement change. Placement therefore stays **all_cores**
  (faithful to legacy: idle cores kept, RTAs zero-filled). The only observable deviation from legacy is that
  a partial-shard config legacy would have served from the fast L1 view now takes the (output-identical,
  slower) reload path — which matches the op's own test contract (`N`-core shard `-> reload path`, `-1`
  all-cores `-> fast globally-allocated CB path`). *(An earlier revision instead narrowed the work unit to
  the active cores; that was a behavior change — a non-syntax-swap improvisation the recipe forbids, plus a
  latent runtime-hang risk since the spec validator does not check DFB node-coverage vs shard grid, only
  existence/L1/size, with the per-bank check deferred to attach time — and has been reverted.)* Suggest a
  patterns-catalog entry for "legacy all-cores-with-idle-noops + partial-shard borrowed CB → borrow only on
  full shard, else reload."

### Confusion
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
