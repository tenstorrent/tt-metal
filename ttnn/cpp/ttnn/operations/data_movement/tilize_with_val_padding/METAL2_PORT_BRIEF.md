# Metal 2.0 Port Brief — `data_movement/tilize_with_val_padding` · `TilizeWithValPaddingMultiCoreBlockInterleavedFactory`

> Audit cleared all gates for this factory. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **Scope:** this brief covers **only** the `TilizeWithValPaddingMultiCoreBlockInterleavedFactory`. The op's three sibling factories (SingleCore / MultiCoreDefault / MultiCoreSharded) are out of scope here (each carries `Is able to port? = yes` on the sheet but was not audited in depth).

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Why this factory is now portable:** it was the one blocked by the per-region DFB-sizing wall of issue [#51305](https://github.com/tenstorrent/tt-metal/issues/51305) (closed 2026-09-04). The family-wide refactor (issue's Option 1) split the work into two independently-sized `BlockBufferSet`s with **distinct** CB indices — `full` = c_0 / c_1 / c_16, `cliffrow` = c_2 / c_3 / c_17 — so every named CB now has **one uniform size across all its nodes** (`push_buffer_set`, `common.cpp:808-863`). That is what makes it expressible as `DataflowBufferSpec` (one `entry_size` + `num_entries` per DFB).

**Recipe docs:** `d51708326b5 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the factory ports to `ProgramSpecFactoryConcept`.

- **Current concept:** `descriptor` — `create_descriptor` returns a `ProgramDescriptor` (`tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp:14-17`).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (the base concept — the factory declares **no** `override_runtime_arguments`, so the framework refreshes the tensor bindings on a cache hit and the factory writes one `create_program_spec` method).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (cell is `none`) · `get_dynamic_runtime_args` (absent). No custom hash, no `override_runtime_arguments`, no pybound `create_descriptor` are present either (none of those gate, but noting they are all absent means the wiring is the simplest form).

## Construct — to do

**Tensor bindings** (per binding) — both are the `Buffer*`-binding form today (the factory pushes the `Buffer*` object as RTA slot 0 and relies on framework `BufferBinding` auto-registration); replace each with a typed `TensorParameter`:

- **input** (`src0_buffer`) — **Case 1** (via `TensorAccessor`). Bind as `TensorParameter` (e.g. `tensor::src`); the reader builds `TensorAccessor(tensor::src)` instead of `TensorAccessor(src_args, src_addr)`. Remove: reader RTA slot 0 (`.cpp:294`), the `src_addr = get_arg_val<uint32_t>(0)` read (reader `:34`), and the `TensorAccessorArgs(*src0_buffer).append_to(...)` CTA (`.cpp:145`) + the `TensorAccessorArgs<8>()` CTA read (reader `:32`).
- **output** (`dst_buffer`) — **Case 1** (via `TensorAccessor`). Bind as `TensorParameter` (e.g. `tensor::dst`); the writer builds `TensorAccessor(tensor::dst)` instead of `TensorAccessor(dst_args, dst_addr)`. Remove: writer RTA slot 0 (`.cpp:306`), `dst_addr = get_arg_val<uint32_t>(0)` (writer `:11`), and `TensorAccessorArgs(*dst_buffer).append_to(...)` (`.cpp:161`) + `TensorAccessorArgs<4>()` (writer `:20`).

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none — both accessors are 2-arg; nothing to drop.

**CB endpoints** (per set; identical shape on `full` and `cliffrow`):

- **staging** `c_1` (full) / `c_3` (cliffrow) — **self-loop**: one toucher (the reader `reserve_back(1)`/`get_write_ptr`/`push_back(1)` then uses `temp_addr` as scratch, reader `:42-45`). Bind the reader as **both** PRODUCER and CONSUMER. This is a **DM self-loop** — legal on Gen1; it becomes a Quasar-uplift item, not a Gen1 blocker.
- **input** `c_0` (full) / `c_2` (cliffrow) — **1P+1C**: reader PRODUCER, compute CONSUMER.
- **output** `c_16` (full) / `c_17` (cliffrow) — **1P+1C**: compute PRODUCER, writer CONSUMER.

**CB → DFB index mapping** (from `make_block_plan`, `common.cpp:919-936`) — keep the two sets on **distinct** DFB names; do **not** collapse them (that would reintroduce the #51305 corruption):

| Set | staging | input | output | size (`block_tiles`) |
|---|---|---|---|---|
| `full` | c_1 | c_0 | c_16 | `single_sub_block_size` |
| `cliffrow` | c_3 | c_2 | c_17 | `single_block_size_cliff_row` |

**Kernel binding tokens** — all three kernels currently take the CB index as a numeric CTA and build the DFB from it; swap each numeric CTA for the `dfb::` token:

- reader `dfb_id_in0` / `dfb_id_in1` (`get_compile_time_arg_val(6/7)`, reader `:28-29`) → `dfb::in` / `dfb::stage` tokens (per set).
- writer `cb_id_out` (`get_compile_time_arg_val(0)`, writer `:16`) → `dfb::out`.
- compute `dfb_id_in` / `dfb_id_out` (`get_compile_time_arg_val(3/4)`, compute `:19-20`) → `dfb::in` / `dfb::out`.
- The remaining CTAs on each kernel are scalars (counts / dims) — carry them as named CTAs.

**Preserved-multiplicity wiring** (not multi-binding): the factory emits 4 compute KernelSpecs (`.cpp:217-232`) and a reader+writer per set (`.cpp:173-176, 323-330`) over **disjoint** core ranges. For each shared DFB, list it per group with disjoint `target_nodes` (see `metal2_port_gotchas`), and let the framework derive placement. Per node there is exactly 1 reader + 1 compute + 1 writer on each I/O CB.

**RTAs → named args:** all RTAs are nameable (no varargs). Reader slots 0-8, writer slots 0-3. Note the reader re-reads slots 3-8 each `third_dim` iteration at constant indices — these are the same named args, read in a loop; name them once.

## Watch for

- **Shared kernels — create `_metal2` forks, do NOT convert in place** (each is co-borrowed by **tilize**'s still-legacy block factory; converting the shared file in place would break it):
  - reader `device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` — owned by this op; **no fork yet** — this port creates it beside the original.
  - writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` — **no `_metal2` fork of the `_wh` variant** (the adjacent `writer_unary_interleaved_start_id_metal2.cpp` is a fork of the *non-`_wh`* kernel — do **not** reuse it). This port creates the first `_wh` fork. **Preserve the `BACKWARDS` `#ifdef`** (writer `:31-41`) — the untilize direction relies on it.
  - compute `data_movement/tilize/device/kernels/compute/tilize_wh.cpp` — **no `_metal2` fork** (`ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp` is a fork of a *different* tilize compute kernel). This port creates the first fork.
  - **Sunset list** (bind the same fork, delete legacy when the last consumer migrates): **{tilize_with_val_padding block factory (this one), tilize block factory}**. This list is *not* an authorization to port tilize's block factory in the same diff — it is the coordination/sunset set.
- **Per-region sizing is a correctness property here** (#51305): keep `full` and `cliffrow` on separate DFB names sized from their own `block_tiles`. The factory's own assertions (`.cpp:197-202, 281-287`) enforce that a set's buffers match the block width its cores are fed; carry that invariant through — don't unify the two sets or size a DFB to a max/LCM.
- **Cache-miss-only split:** `make_block_plan` is not reproducible on a cache hit (`common.hpp:113-117` — the block-size limit folds in live L1 occupancy). Anything the cache-hit path needs must be recorded at miss time. For the base `ProgramSpecFactoryConcept` this is only the tensor bindings (framework-refreshed), so no custom cache-hit logic is required — but do not add a cache-hit recompute of the split.
- **Reader uses `PrecomposedUnicastEndpoint` / `CoreLocalMem` / `tt_memmove`** for the DRAM-misalignment staging path (reader `:82-108`). These are Device-2.0 idioms already; the port is a binding-layer change (CB index CTA → `dfb::` token, address RTA → `tensor::` binding), **not** an idiom rewrite. Leave the staging algorithm untouched.
