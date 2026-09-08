# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/pool/upsample` · `UpsampleBilinearProgramFactory`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
> Scope: **`UpsampleBilinearProgramFactory` only**. The three sibling factories (`UpsampleMultiCoreInterleaved`, `UpsampleMultiCoreSharded`, `UpsampleNearestFloat`) are already MetalV2 — leave them alone.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (N/A)

**Recipe docs:** `23861284522 2026-09-08 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`.

- **Current concept:** `descriptor` — `UpsampleBilinearProgramFactory::create_descriptor(...)` returns a `ProgramDescriptor` (`upsample_bilinear_program_factory_multicore.cpp:30`).
- **Op-owned tensors:** none (this factory declares none; op-owned tensors exist only in the sibling sharded factory, which you are not porting).
- **Target concept:** `ProgramSpecFactoryConcept` (base — `Override runtime args method? = no`, so the framework refreshes bindings on cache hit and you write one method).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. No custom hash, no `override_runtime_arguments`, no pybound `create_descriptor` are present either (none of those gate; here they're simply absent, so nothing to preserve or delete).

## Construct — to do

**Tensor bindings** (per binding) — both **clean** (borrowed-memory DFB), no `TensorAccessor`, no address RTA:

- `input` (halo) — **clean / borrowed-memory DFB.** CB `c_0` currently sets `.buffer = halo_in.buffer()` (`upsample_bilinear_program_factory_multicore.cpp:113`); the reader/writer read it raw via `halo_dfb.get_read_ptr()`. Port: declare a `TensorParameter` for the input and make the DFB `borrowed_from` it. Kernel side is unchanged (already `DataflowBuffer halo_dfb(...)` + raw offset arithmetic).
- `output` — **clean / borrowed-memory DFB.** CB `c_5` sets `.buffer = output.buffer()` (`upsample_bilinear_program_factory_multicore.cpp:184`); compute writes it. Port: `TensorParameter` + DFB `borrowed_from`.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none — no accessor in scope.

**CB endpoints** (single config: `HEIGHT_SHARDED` bf16, integer scale — the factory's only supported path):

- **self-loop** `out_cb` (`c_5`): one toucher (compute produces via `pack_untilize_dest` + a raw `fifo_wr_ptr` advance in `llk_push_pages_bilinear`, nothing consumes). Bind compute PRODUCER **and** CONSUMER.
- **assign 1P+1C** `halo_cb` (`c_0`): two role-free touchers — the reader-config and writer-config instances of the same kernel source both raw-read it. Bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1).
- **legal 1:1** (no action): `tilize_reduce_cb_0` (`c_1`, reader→compute), `tilize_reduce_cb_1` (`c_2`, writer→compute), `in_scalar_cb_id1` (`c_3`, reader→compute), `in_scalar_cb_id2` (`c_4`, writer→compute).

No multi-binding flag, no dead-CB drop, no conditional DFB.

**CB-ID CTAs → tokens.** The factory passes CB indices as compile-time args (reader CTAs 6–8, compute CTAs 0–4). Under the port these become `dfb::name` tokens; the kernels already hold them as `DataflowBuffer` locals, so the change is at the binding/CTA layer, not the kernel body.

**Runtime args.** Per-core loop (`upsample_bilinear_program_factory_multicore.cpp:316–350`) sets `{start_output_idx, min_input_offset, out_sticks_this_core}` on reader+writer and `{out_sticks_this_core}` on compute (with a `{…,0,0}` / `{0}` no-work path for empty cores). All are fixed named scalars → name each in the `runtime_arg_schema`; translate the loop into the `ProgramRunArgs` schedule.

## Watch for

- **Kernel already modernized:** this is a **binding-layer port, not an idiom rewrite.** The kernels are already on `DataflowBuffer` / `Noc` / `UnicastEndpoint` / `experimental::local_addr`. Do not rewrite the raw L1 addressing in the reader — it is the intended borrowed-memory access pattern.
- **`get_local_cb_interface` in `bilinear.cpp:18`** (inside `llk_push_pages_bilinear`) is a **sanctioned** Device-2.0 free function — leave it. Per the kernel-side whitelist, a Metal 2.0 port *may* move that lookup onto the DFB object (`out_dfb`), since `DataflowBuffer` exposes the tile/format metadata; confirm the equivalent rather than swapping blind, and treat it as optional cleanup, not required.
- **CB endpoints (multi-binding):** none — no hidden second writer (the factory declares no semaphores; the `in_scalar` / `tilize_reduce` CBs are cleanly 1-producer/1-consumer, split between the reader-instance and writer-instance CBs). Do not reach for the multi-binding flag on `halo_cb`; two role-free touchers is 1P+1C.
- **Cross-op / shared kernels:** none — both bilinear kernels are op-owned and used by no other op; no `_metal2` fork exists or is needed, no sunset list. Header helpers (`experimental_device_api.hpp`, `fixed_point_arithmetic.hpp`) are in-family, Device 2.0 native — no donor-side work.
- **RTA varargs:** none — all args are named-able.
- **Dead compute CTAs (optional):** `bilinear.cpp` CTA[6] `in_ntiles_hwc` and CTA[7] `window_size_hw` are unused; you may carry them across unchanged. Pruning them is an ops-team cleanup, not port work — do not fold a functional change into the port diff.
