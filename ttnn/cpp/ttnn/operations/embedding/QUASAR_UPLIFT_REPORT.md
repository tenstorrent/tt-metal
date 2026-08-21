# Quasar Uplift Report — `ttnn/cpp/ttnn/operations/embedding`

Recipe followed: `docs/source/ttnn/ttnn/ai/quasar_porting.md`, plus the canonical passes it
extends on branch `akertesz/op-porting-recipe`
(`ai/audit/quasar_audit.md`, `ai/post_port/style/sync_free_dfbs.md`,
`ai/post_port/semantic/dm_self_loop_dfbs.md`).

**Leave this file uncommitted; delete it before merge.**

---

## Status: **AMBER** — two of three program factories uplifted; two configurations are RED and stopped

The op dispatches three program factories, and they are in three different states. Nothing about
the op's directory or namespace changed; nothing was copied into `experimental/quasar/`; no
`::qsr` namespace was invented.

| Factory | Reached when | Verdict |
|---|---|---|
| `EmbeddingsTilizedIndicesProgramFactory` | index tensor is `TILE_LAYOUT` | **Uplift done; blocked on a defect outside the op.** Gen1 green. Quasar output is wrong, isolated to a Gen2 runtime defect reproduced without any embedding code — see [RED 3](#red-3) |
| `EmbeddingsRMProgramFactory`, interleaved output | index tensor `ROW_MAJOR`, `tilized=false`, interleaved output | **Uplift done; same downstream block** — same shared writer and DFB geometry, see [RED 3](#red-3) |
| `EmbeddingsRMProgramFactory`, **sharded** output | same, but height-sharded output | **RED — stopped.** Borrowed-memory DM self-loop; see [RED 1](#red-1) |
| `EmbeddingsFusedProgramFactory` | index tensor `ROW_MAJOR`, `tilized=true` | **RED — stopped.** Still on the legacy `ProgramDescriptor` API; see [RED 2](#red-2) |

The uplift was not a no-op. The audit found a construct that Quasar rejects **at program creation,
in every configuration of both Metal 2.0 factories**, so those paths could not have run on Quasar at
all before this change. Details in [What the audit found](#what-the-audit-found).

Both uplifted factories now get past program creation on Quasar and execute, which they could not do
before. What they hit next is a Gen2 runtime defect this uplift did not cause: on Quasar, a
`DataflowBuffer` breaks when a producer issues **more NoC transactions than the slots it announces**
with `push_back` **and** the consumer is a **data-movement kernel**. Both conditions are required. The
consumer is then released before the producer has filled the entry. That is [RED 3](#red-3), reproduced
in a loopback with no embedding code, no scratchpad and no ttnn, passing on Wormhole hardware and
failing on Quasar. The same defect was found independently while porting `indexed_fill`, and that
investigation's repro measures the credit surplus directly; RED 3 cites it.

Read that section before forming a theory. Three earlier theories in this report were wrong and are
recorded there as ruled out, one of which was the natural reading of this op's symptom: that the
scratchpad was the trigger. It is not. The scratchpad read is simply one NoC transaction more than the
reader announces, and an unrelated destination does exactly the same damage.

**Gen1 parity is confirmed by test, not merely argued:** `test_embedding_tiled_input` passes on Gen1
with these changes. See [Parity](#parity-with-wormhole--blackhole).

---

## What the audit found

### The blocking construct: data-movement self-loop DataflowBuffers

Both Metal 2.0 factories bound two DataflowBuffers with the **reader kernel as both `PRODUCER` and
`CONSUMER`**:

- `index_scratch` — the reader stages a block/page of token indices in it and decodes tokens
  straight back out.
- `weight_cache` — under `PADDED`/`BINARY`, the reader caches one or two weight rows in it and
  serves matching tokens from the cache instead of refetching.

A data-movement kernel holding both endpoints of one buffer is legal on Gen1, where a DFB lowers to
a plain circular buffer that one RISC can fill and drain. On Gen2 the credit machinery needs the two
endpoints on **different** RISCs, so the shape cannot be lowered. It is rejected outright:

`tt_metal/impl/metal2_host_api/program_spec.cpp:1433-1439`

> `DataflowBuffer '{}' is self-looped by data-movement kernel '{}' (bound as both PRODUCER and
> CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures.
> Consider using a scratchpad or LocalTensorAccessor instead.`

This is a hard `TT_FATAL` in the spec legalizer, not a runtime symptom, so it is not one of the
recipe's reactive §7–§8 fixes — it fires deterministically the moment the program is built on
Quasar. Both buffers are also unreachable as legal DFBs by any other route: the legalizer requires
exactly one producer **and** one consumer instance per node
(`program_spec.cpp:1368-1400`), so dropping the unused endpoint is not an option either.

Neither buffer was doing any real work as a FIFO. On both, every FIFO call was a no-op given the
buffer's own configuration:

- `index_scratch` was declared with `num_entries = 1`. The reader's `reserve_back(1)` on an empty
  one-entry buffer returns immediately; its single `push_back(1)`, placed at the very end of the
  kernel purely to leave the buffer balanced, posts a credit nobody reads before the kernel exits.
- `weight_cache` was `reserve_back(1)` or `reserve_back(2)` on a buffer with exactly that many
  entries, and was **never** pushed at all. The write pointer never moved.

So on both buffers the write pointer sat at the allocation's base for the kernel's whole lifetime,
and the two "endpoints" never exchanged a credit that anything observed. They are scratchpads by
nature, which is what they have been converted to — the fix the legalizer's own message names, and
the one `dm_self_loop_dfbs.md` prescribes.

### Checks that came back clean

| Check | Result |
|---|---|
| `quasar_audit.md` check 2 — non-zero-init semaphores | **Clean.** The op creates no semaphores at all. |
| `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for` (§7) | **Clean, and left that way.** Never set. `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)` already return `DataMovementGen2Config` on Quasar with implicit sync on, so the `gen2_hardware_configs.md` pass was already done for these kernels. Note that implicit sync being on puts the op in the configuration issue #50328 describes; setting the opt-out was tried as a diagnostic and did not change the symptom, so it is not the cause here. See [RED 3](#red-3). |
| `fifo_page_size` / `get_local_cb_interface` (§5, §8.3) | **Clean.** No occurrence anywhere in the op. |
| `evil_set_read_ptr` / `evil_set_write_ptr` (§7, §8.3) | **Clean.** Not used, so the missing-Gen2-rewind-API blocker does not apply. |
| `dfb_run_overrides` | **Clean.** Neither factory overrides a DFB size at runtime. |
| `get_entry_size()` on a converted buffer | **Clean.** Not called on either converted buffer, which removes `dm_self_loop_dfbs.md`'s most common stop condition. |
| `async_write_zeros` / DFB as multicast destination | **Clean.** Neither appears; both would have rejected a `Scratchpad`. |
| Quasar has Int32, no uint16/uint32 device format (§7) | **Nothing to guard.** The `BFP16`/`UINT32` defines select a C++ integer width for RISC loads of the index page (`input_token_t`), not an LLK tile format. The op forwards the index tensor's `DataType` without a format branch of its own, which §7 says is the format/LLK layer's concern, not the op's. Converting `index_scratch` to a scratchpad also drops the only place the op declared a `UInt32` `data_format_metadata`. |
| `compute_kernel_hw_startup` exactly once (§7) | **N/A to the uplifted paths.** The op's only compute kernel (`kernels/compute/tilize_chunked.cpp`) belongs to the RED fused factory. It does call it once at the top of `main()`. |
| `reserve_back`→`push_back` / `wait_front`→`pop_front` with no intervening TDMA op (§8.5) | **Clean.** Every such pair in both readers and both writers has a NoC transfer between the two calls. |
| `data_format_metadata` consulted by a converted buffer's kernel | **Clean, checked rather than assumed.** Every use of both handles is a raw base-address grab or a NoC operand; nothing reads the declared format. Safe to drop, as `ScratchpadSpec` has no counterpart field. |

---

## Files changed

All changes are in place, in the op's existing directory and namespace.

| File | Change |
|---|---|
| `device/embeddings_rm_program_factory.cpp` | `index_scratch` and `weight_cache` converted from `DataflowBufferSpec` + self-loop `DFBBinding` pairs to `ScratchpadSpec` + `ScratchpadBinding`. Dropped the now-unused `input_data_format`. Noted on the sharded-output `OUTPUT` binding that a DM kernel holding both endpoints is Gen1-only. |
| `device/embeddings_tilized_indices_program_factory.cpp` | Same conversion. Dropped the now-unused `input_data_format` and `input_element_size_bytes`. **`index_scratch`'s size changed** — see [the one parity exception](#the-one-parity-exception). |
| `device/kernels/dataflow/embeddings.cpp` | RM reader: `DataflowBuffer dfb_in1` + raw `volatile` cast replaced by `Scratchpad<volatile input_token_t> indices`, passed directly as the NoC read destination and subscripted for token reads. Dropped the trailing balancing `push_back`. |
| `device/kernels/dataflow/embedding_ind_tilized.cpp` | Same conversion. Also removed a stray `api/debug/dprint.h` include with no `DPRINT` call behind it (§9 "strip DIAG"; on the emulator a spare debug include can push a kernel over the size limit). |
| `device/kernels/dataflow/embeddings_common_metal2.hpp` | `prepare_local_cache` now takes a `ScratchpadBindingToken` and builds a `Scratchpad<uint8_t>`; the cached rows are written through the scratchpad as a NoC destination with an explicit byte offset instead of a rebuilt `CoreLocalMem`. `read_token_async` is unchanged. |

Nothing outside the op directory was touched. The shared Metal 2.0 writer
`ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp` was
audited and needs no change: it is a genuine cross-kernel FIFO consumer, takes its write size from
`stick_size` rather than the buffer's entry size, and has a NoC write between its `wait_front` and
`pop_front`.

### A documented deviation from `dm_self_loop_dfbs.md`

That pass lists "a FIFO call made through anything other than a `DataflowBuffer` declared in this
kernel — a helper taking the DFB or its id" as a **stop**, and `weight_cache`'s calls are made inside
`prepare_local_cache` in `embeddings_common_metal2.hpp`. It was converted anyway. The reasoning:

- The stop exists because the pass's translation rewrites FIFO calls sitting beside the handle and
  "has nothing to say about a call made inside a callee." Here the callee is a twenty-line function
  in the op's own directory, included only by the op's two Metal 2.0 readers, so the whole state
  machine is visible and the translation was applied inside it.
- There is no state to replay. The helper reserves once and never pushes, so both indices stay at
  zero; the translation is `reserve_back` deleted and `get_write_ptr()` → `get_base_address()`, with
  no stride, no wrap, and no index variables.
- Unlike the Gen1 style/semantic passes, whose worst case is a missed cleanup, leaving this buffer
  a DFB keeps the op un-runnable on Quasar in every `PADDED`/`BINARY` configuration.

Worth a reviewer's attention, since it is a deliberate departure rather than an oversight.

---

## Parity with Wormhole / Blackhole

**Confirmed by test.** `test_embedding_tiled_input` passes on Gen1 with these changes. That is the
case that exercises both converted scratchpads *and* the one non-neutral change, the index-buffer
resize, so it is the run that mattered most. The structural argument below is retained because it is
per-buffer and covers configurations the single test does not reach; the remaining commands are in
[How to verify](#how-to-verify).

The changes are **not** `ARCH_QUASAR`-guarded, and deliberately so: a guard would leave the Gen1
path on a construct that is a misdescription on Gen1 too, and the conversion is behaviour-preserving
on Gen1 rather than Gen2-specific. The argument that WH/BH behaviour is unchanged is per-buffer:

- **`weight_cache`** — byte-for-byte equivalent. Its `reserve_back` was a no-op on a buffer with
  exactly that many free entries, it was never pushed, and `get_write_ptr()` on a never-advanced
  pointer returns the allocation base, which is what `get_base_address()` returns. The NoC reads
  target the same offsets from that base as before, in the same order, with the same sizes. Region
  size unchanged (`entry_size × num_entries` carried across as the product).
- **`index_scratch` in the RM factory** — same argument, and the region size is carried across
  unchanged (`block_height * index_page_size`). The reads and subscripts are identical.
- **`index_scratch` in the tilized-indices factory** — same argument for the accesses, but the
  declared size changed. This is the one exception, below.
- **Allocation order** — scratchpads are allocated from the same L1 region as DFBs, so a converted
  buffer's own address may land elsewhere. `sync_free_dfbs.md` and `dm_self_loop_dfbs.md` both state
  that nothing functional depends on that ordering.
- **Removed `push_back`s** — both were the last statement before the kernel returned, posting a
  credit no consumer reads. Nothing observes them.
- **Removed `dprint.h` include** — no `DPRINT` call existed behind it.

### The one parity exception

`EmbeddingsTilizedIndicesProgramFactory`'s index buffer was declared as
`FACE_HEIGHT * round_up_to_mul32(input_element_size_bytes)` = **512 bytes**, unchanged from the
legacy CB (`git show 09d9312c08b^` confirms the pre-Metal-2.0 CB carried the same expression). But
the reader NoC-reads a **whole index page** into it — `input.get_aligned_page_size()`, which for a
`TILE_LAYOUT` index tensor is one tile: 4096 B at `uint32`, 2048 B at `bfloat16` — and then
subscripts it across that page's face-swizzled token layout.

The reachable subscript range is exact. The kernel indexes `[token_idx + offset]` where `offset` is
always of the form `f*256 + r*16` with `f ≤ 3` and `r ≤ 15` (the host seeds it that way and every
increment in the kernel is ±16 or ±256), so `offset ≤ 1008`, and `token_idx < 16` gives a maximum
index of **1023** — exactly one tile's 1024 elements, in both dtypes. So the buffer was
**under-declared by 8× (uint32) or 4× (bfloat16)**, and the correct size is one input page.

This is a pre-existing defect on `main`, not something the port or this uplift introduced. It has
been benign so far because a CB is an unchecked L1 region and the overflow evidently lands on
allocation slack or on a neighbour whose contents are rewritten anyway — `test_embedding_tiled_input`
and `test_tiled` pass on `main` with an exact `assert_equal`.

It could not be carried across as-is. A `Scratchpad`'s size is its contract: `operator[]` is
bounds-checked against it, so keeping 512 B would have declared a region the kernel provably
overruns and turned a silent overflow into an assertion failure in watcher / lightweight-assert
builds — on WH/BH as well as Quasar. The scratchpad is therefore sized
`a.buffer()->aligned_page_size()`, which is the same quantity the kernel reads and the exact bound
its subscripts need. With that size the bounds check is satisfiable at every reachable index.

**What a reviewer should decide:** whether this belongs in this PR or a separate one. It raises L1
use on the affected path by ~3.5 KB per core (uint32; ~1.5 KB at bfloat16), which is small against
the ~1 MB budget and against the same factory's own `2 × rounded_weight_page_size` output buffer,
but it is a real change on Gen1 and the only thing here that is not behaviour-neutral. Worth its
own issue either way — the under-declaration is on `main` today.

---

## RED — stopped, not forced through

### RED 1

**`EmbeddingsRMProgramFactory` with a sharded output: borrowed-memory data-movement self-loop.**

When the output is height-sharded, the factory creates no writer kernel: the `OUTPUT` DFB is
declared `borrowed_from = OUTPUT_PARAM`, so it *is* the output shard, and the reader stages its rows
directly into it. That leaves the reader as the buffer's only endpoint, holding both roles — the
same DM self-loop Gen2 rejects.

Unlike the two converted buffers, this one is a real fake-FIFO: the reader calls
`reserve_back(1)` / `get_write_ptr()` / `push_back(1)` once per output row, and the write pointer
genuinely advances so that row *i* lands at shard offset *i × entry_size*. So:

- It is not a site for the sync-free pass (it calls the FIFO machinery).
- It is not a site for the DM self-loop pass either: that pass's survey step 5 stops when
  `borrowed_from` is set, because a `ScratchpadSpec` is a fresh private allocation while this is a
  window onto a tensor the op owns — swapping one for the other silently points the kernel at
  different memory. The pass states plainly that "fake-FIFO bookkeeping over borrowed memory is a
  combination nothing in this suite has examined."
- Removing the unused endpoint is illegal: the legalizer requires one producer and one consumer per
  node.

The plausible end state is a `LocalTensorAccessor` over the output tensor with the row stride written
out explicitly, and the reader already has that stride as its `chunk_size` compile-time argument
(when sharded, `use_chunked` is false, so `chunk_size == rounded_weight_page_size`, which is the
DFB's `entry_size`). What makes it an owner decision rather than a mechanical edit is where that
stride comes from: `rounded_weight_page_size` is `tt::align(weight_page_size, alignment)` where
`alignment` is the **index** tensor's buffer alignment, not the output's. The factory's own
`TT_FATAL` and comment at
[embeddings_rm_program_factory.cpp:144-153](ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp#L144-L153)
already flag that these two alignments can disagree. Re-deriving that stride in explicit kernel
arithmetic risks a silently different value writing to the wrong offsets in the output tensor, with
nothing to catch it.

**Left exactly as it is. Nothing was worked around** — in particular the unused endpoint was not
bound to an unrelated kernel, which `dm_self_loop_dfbs.md` calls out as the specific workaround it
exists to prevent. This configuration will `TT_FATAL` at program creation on Quasar.

Also worth flagging for whoever picks this up: §7's rule that a Quasar row-major shard width must be
16-byte aligned (bfloat16 ⇒ a multiple of 8 elements) applies to this path and is not validated
anywhere today. It is moot while the path is blocked.

### RED 2

**`EmbeddingsFusedProgramFactory` is not Metal 2.0 yet.**

This is the recipe's first RED-stop condition verbatim. The factory is still on
`create_descriptor` / `ProgramDescriptor`, with `CBDescriptor`, `tt::CBIndex::c_0`, positional
`get_arg_val<uint32_t>(i)` and address-RTA `TensorAccessorArgs<10>()` in its kernels
(`kernels/dataflow/embeddings_tilize.cpp`, `kernels/compute/tilize_chunked.cpp`, and the legacy
`kernels/dataflow/embeddings_common.hpp` fork they share).

This is known and intentional: per the Metal 2.0 port commit (`09d9312c08b`, PR #53425), the
pre-port audit blocked this factory on an offset-base-pointer prerequisite for the ops team, so
`EmbeddingsDeviceOperation` deliberately carries a mixed set of factory concepts. The Quasar uplift
starts from an already-Metal-2.0 factory, so **the base Metal 2.0 port has to land here first**
(`ai/port/metal2_port.md`); there is nothing for this pass to do. No files under the fused path were
touched.

Note this is the path the BERT-large demo uses, and therefore the only path in the repo that
exercises `EmbeddingsType::PADDED` and `BINARY` at all — see the coverage gap below.

Observed on craq-sim, as expected:

```
TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:418:
  DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.
  --- tt::tt_metal::CreateDataMovementKernel(...)
  --- tt::tt_metal::Program::Program(tt::tt_metal::ProgramDescriptor const&)
```

`Program::Program(ProgramDescriptor const&)` in the backtrace is the legacy descriptor path, which
is the signature of this RED rather than of anything in the uplift.

### RED 3

**Both uplifted paths run on Quasar but produce wrong output. Isolated to a minimal repro that
contains no embedding code, no scratchpad, and no ttnn, and that passes on Gen1. Not a WH/BH
regression, and not caused by this uplift.**

Gen1 passes the same test with these changes, so the op's logic and the scratchpad conversion are
sound on Gen1.

**Symptom in the op.** Correct data, wrong output pages: every row is shifted later by exactly
`num_entries` positions within each core's range, with the vacated leading pages zero. On
`test_embedding_tiled_input` at batch 1 / sentence 32 / hidden 768, split over 2 cores:

```
zero rows: tensor([ 0,  1, 16, 17])                  # num_entries = 2
zero rows: tensor([ 0,  1,  2,  3, 16, 17, 18, 19])  # num_entries = 4
```

**It is timing-sensitive, not a fixed offset.** Adding DPRINT to the reader and writer makes 4 of the
5 first cases pass. Per §9 that is a signal, not a fix, and it rules out a deterministic
initialisation error.

#### The condition, stated

A DataflowBuffer on Quasar misbehaves when **two** things hold at the same time:

1. the producer issues **more NoC transactions than the slots it announces** with `push_back`, and
2. the **consumer is a data-movement kernel**.

Remove either one and the buffer behaves correctly. Fewer transactions than announced slots is
harmless. A compute kernel in the consumer position is unaffected by the identical producer, even at a
surplus large enough to deadlock the data-movement case.

The second condition explains the scope. `wait_front` has two unrelated implementations: a compute
consumer waits through the unpacker, and a data-movement consumer spins on the buffer's occupancy from
a RISC. Only the second is affected.

#### The minimal repro

`tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp` (untracked; keep or drop
with this report) is a producer / DataflowBuffer / consumer DRAM loopback using the explicit credit
APIs. It runs on both generations, and every case below passes on Wormhole hardware, so the contract,
not the test, is what differs between the two.

Two independent checks per case, because the weaker one under-reports. Per grant, the consumer samples
the granted slot's first word the instant `wait_front` returns; slots carry distinct values, so grant
*k* must see entry *k*'s word. End to end, the delivered DRAM output must equal the input. A slot
released early only corrupts the output if the consumer also loses the race against the arriving data,
so the two checks together separate a real early release from a probe artifact.

**The transaction-to-announcement ratio is the trigger.** Nothing in these three cases involves a
scratchpad, a second destination, or anything else unusual. The producer is one kernel varying only
how it splits its reads
([dfb_ratio_probe_producer.cpp:48-87](tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_ratio_probe_producer.cpp#L48-L87)).

| Producer, per announced slot | Wormhole (hardware) | Quasar (craq-sim) |
|---|---|---|
| one full-entry read, then announce one slot | pass | pass |
| **two half-entry reads, then announce one slot** | pass | **fail**, 3 of 4 grants early, output wrong |
| one double-entry read, then announce two slots | pass | pass |

A surplus breaks it. A matching count and a deficit both behave. The failing case is
[test_dfb_gen2_credits_hw.cpp:790-814](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L790-L814),
and it is the smallest repro in the file.

**The consumer's hardware decides whether the surplus matters.** Same producer, same surplus, three
kernels instead of two so a compute kernel can hold the consumer end: producer, then a compute tile
copy, then a data-movement writer that drains to DRAM
([test_dfb_gen2_credits_hw.cpp:904-1018](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L904-L1018),
compute kernel at
[dfb_tile_copy_compute.cpp:28-52](tests/tt_metal/tt_metal/test_kernels/compute/dfb_tile_copy_compute.cpp#L28-L52)).

| Producer, per announced slot | Consumer | Wormhole | Quasar |
|---|---|---|---|
| one read | data movement | pass | pass |
| two reads | data movement | pass | **fail** |
| one read | compute | pass | pass |
| two reads | compute | pass | pass |
| eight reads | compute | pass | pass |
| eight reads | data movement | pass | **hangs** |

The last two rows are the decisive pair. Eight reads per announced slot is a surplus of seven against
a two-slot ring, which deadlocks the data-movement consumer and leaves the compute consumer's output
bit-exact. That rules out the compute kernel merely winning a race it could have lost, which a 2:1
surplus alone would not: the writer only inspects a tile after the copy, so a small surplus could have
been masked by the copy's own latency, and a surplus this large cannot be.

**The destination is irrelevant.** An earlier reading of this defect blamed the scratchpad, because the
op's reader NoC-reads an index page into one. The sweep at
[test_dfb_gen2_credits_hw.cpp:1091-1267](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1091-L1267)
varied only where an extra read lands, holding the DataflowBuffer half of the producer identical: into
a scratchpad through its binding, at the scratchpad's own address through a plain pointer, at a
scratchpad's far end, at an unrelated address with a scratchpad bound, and at an unrelated address with
no scratchpad anywhere in the program. All fail identically on Quasar and all pass on Wormhole. The one
row that passes on Quasar is the row that touches the scratchpad with ordinary load and store
instructions, because that is not a NoC transaction.
[test_dfb_gen2_credits_hw.cpp:578-654](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L578-L654)
is the same result with no scratchpad declared at all.

So the scratchpad in the op is incidental. What matters is that the index read is one more NoC
transaction than the reader announces, and that the writer draining the `output` buffer is a
data-movement kernel.

#### Why this op is unlucky and the rest of the Quasar library is not

An audit of all 27 ops under (28 directories, one of which is a shared helper) `ttnn/cpp/ttnn/operations/experimental/quasar/`, every factory and every
buffer, found that the ops carrying a real surplus almost all avoid the second condition:

- **The heavy producers feed compute.** conv2d's activation reader, halo, tilize (32 reads per
  announced slot), untilize, matmul, pool, reduction and binary_ng all place a compute kernel between
  the reader and the writer, so their surplus lands on the path that is not affected. In matmul the
  split is absolute: every buffer with a data-movement producer has a compute consumer, and every
  buffer with a data-movement consumer has a compute producer.
- **Most data-movement-to-data-movement buffers carry no credits at all.** reshard, fold's sharded
  factory, transpose's sharded path and halo's gather scratch declare a producer and a consumer only to
  satisfy the one-producer-one-consumer rule, then use the buffer purely as an address source and never
  call `push_back` or `wait_front`. Others are strictly one transaction per slot, or announce more slots
  than they fill, which is the harmless direction.
- **About ten buffers do match both conditions**, and are latent rather than safe: the four row-major
  `pad` factories, `padded_slice` on its non-aligned path, `fold`'s row-major interleaved factory at
  four reads per slot for a 2x2 fold, `interleaved_to_sharded`'s unaligned stick branch, and two `slice`
  and two `transpose` factories where the surplus appears only for width- or block-sharded row-major
  input.

Of those, only `interleaved_to_sharded` sits on the ResNet path, and its match requires the unaligned
stick branch, which a tile-layout or an aligned shard never takes. That is why a working model on
Quasar never hit this, and it is not evidence that the library as a whole is clear.

The embedding op is at the intersection of both conditions: a reader feeding a writer with no compute
kernel anywhere, a buffer with real waits, and an index read that makes transactions outnumber
announcements. The op is written to the documented contract, which is why this is reported rather than
worked around.

#### The mechanism, as far as it is pinned down

One surplus credit per surplus transaction accounts for the pattern. With one extra credit, consumer
read *k* proceeds once the producer has pushed *k* real entries, so it reads slot *k mod N*, whose last
writer was entry *k - N*. The stream comes back shifted by the ring depth with *N* pristine leading
entries, which is what the op shows at *N* = 2 and *N* = 4.

The shift cannot pin the count on its own: a shift of *N* is what any surplus from 1 to *N* produces.
The parallel `indexed_fill` investigation measured the count directly, with the producer doing its
extra transfer and posting nothing at all: a consumer waiting for one entry completes and a consumer
waiting for two blocks, so one surplus transaction is worth exactly one credit the producer never
posted. Its repro is `tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_dm_handshake_repro.cpp` in the
`indexed_fill` workspace, and it is the better instrument for the count.

Magnitude changes the symptom rather than only its severity. A surplus of one corrupts; a surplus of
seven against a two-slot ring deadlocks. Where the boundary sits is not established, and an earlier
claim in this report that it is governed by the surplus exceeding the ring depth was wrong by these
same numbers.

#### Ruled out, each by measurement

| Candidate | Ruled out by |
|---|---|
| Op-side index or page arithmetic | Gen1 passes. Checked by hand too: core 0 reads index elements 0-15, core 1 reads 256-271, and the writer's `stick_size` equals the buffer's `entry_size` (both 1536, confirmed from the write and read pointers being 1536 apart). |
| Implicit-sync double count (§8.2, issue #50328) | `disable_dfb_implicit_sync_for_all = true` on both kernels leaves the symptom bit-identical, both in the op and against the failing ratio in the harness ([test_dfb_gen2_credits_hw.cpp:816-836](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L816-L836), same per-grant samples to the bit). |
| A scratchpad, or a second destination, being required | Eight destination variants fail identically, including one with no scratchpad in the program, and the three ratio cases involve no scratchpad at all. |
| `capacity` clobbered to 0 for consumers (`dataflow_buffer.cpp:620`) | `wait_front_impl` asserts `capacity >= num_entries`; `TT_METAL_LLK_ASSERTS=1` produced no assert. |
| Stale cached read of a reused slot (§8.3, §12) | Adding `invalidate_l1_cache()` before the writer's `async_write` changes nothing. |
| The raw-write-pointer fill idiom | Changing the op's reader to pass the `DataflowBuffer` to the NoC instead changes nothing, and the harness covers both forms. |
| An unconditional Gen2 data-movement credit defect | The baseline loopback passes on Quasar across buffer depths, entry sizes, core counts, addressing styles and with implicit sync switched off. It takes a surplus of transactions to break it. |
| An address collision between the buffer and the scratchpad | Measured on a failing run: ring at `0x4C000`-`0x4C800` (via the `+4 MB` uncached alias the write pointer reports), scratchpad at `0x4CF00`-`0x4DF00`. Disjoint. In the op: scratchpad at `0x4D340` + 4096 B, ring at `0x4C700` + 3072 B. Also disjoint. |

**Superseded theories, recorded so they are not rebuilt.** That the implicit-sync interrupt handler was
responsible, falsified by the opt-out probe above. That a Gen2 data-movement consumer starts with
`capacity` credits, falsified by the baseline passing and by the payload sampling showing that slots
genuinely are filled at grant time in the baseline. That the scratchpad binding was the trigger,
falsified by the destination sweep. Cross-RISC DPRINT ordering was also misleading and should not be
treated as a happens-before.

**A disagreement with the sibling repro, resolved.** That repro's control for "extra read, no scratchpad
bound" passed, which contradicted this one. The cause was in its own harness: the `USE_SCRATCHPAD`
define is added only for modes 1, 2 and 4, so its mode 3 compiled without the scratchpad path and
tested nothing. Supplying the define makes mode 3 fail, in agreement. Its check is an expected-value
comparison and was never the problem, contrary to an earlier note in this report that guessed at check
sensitivity. The `indexed_fill` author should be told, since the vacuous control is still in their file.

#### What remains open

- The exact credit accounting, beyond "one per surplus transaction" inferred from two experiments.
- Where the surplus stops corrupting and starts deadlocking.
- The direction of the surplus. The failing pattern is a rotation by the ring depth, which reads as the
  producer running ahead and overwriting slots rather than the consumer being released early: at depth 2
  over four entries, grant 0 sees entry 2's data. Occupancy and free space are two reads of one counter
  pair, so a single accounting error can present as either side gaining credit. This is not reconciled
  with the direct measurement of a consumer-side surplus, and it survives a start delay that should
  guarantee the consumer arrives first.
- Quasar hardware. Not run.

**Substrate, stated carefully.** Reproduced on **craq-sim**, via
`TT_METAL_SIMULATOR=<qsr-sim>/libttsim.so`. craq-sim ships *as* a `libttsim.so`, so that file name does
not identify the variant; the build directory does. Both this investigation and the `indexed_fill` one
initially mislabelled their runs as the plain functional simulator on the strength of that name, and
both then listed craq-sim as untried. Neither is right: both ran craq-sim. Still untried are the plain
functional simulator and Quasar hardware. Every Wormhole control is real hardware.

A gap in the simulator's model rather than in the code it simulates therefore remains possible. It
matters less than it looks: what is measured is a violation of the buffer's documented contract, so it
is a defect on whichever layer owns it, and not in the operation.

**Where to file it.** The repro is op-independent and passes on Gen1, so it belongs with the Gen2
runtime and LLK owners rather than with this op. It is **not** issue #50328: that issue's fix is
`disable_dfb_implicit_sync_for_all = true`, and that flag leaves this symptom bit-identical. The two are
adjacent and easy to confuse, so say so explicitly when filing. It is the same defect as the
`indexed_fill` finding, so file the two together with both repros attached.

#### Repro commands

Standalone harness on Quasar, everything except the two deliberate hangs. Expect 10 passes and 4
failures (`MinimalExtraReadNoScratchpad`, `RatioTwoReadsPerAnnouncedSlot`,
`RatioTwoReadsPerAnnouncedSlotNoImplicitSync`, `ScratchpadUsePatternsThatDisturbTheBuffer`):

```bash
ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_SIMULATOR=<qsr-sim>/libttsim.so TT_METAL_SIMULATOR_HOME=<qsr-sim> \
  ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.*:-Gen2DFBCreditsTest.ScratchpadReadEveryEntry*:Gen2DFBCreditsTest.DmConsumerManyReads*"
```

Note the **single** `-`. gtest reads everything after the first `-` as the exclusion list, colon
separated, so writing `-A*:-B*` excludes only `A*` and lets `B*` run, which hangs. Verify any filter
with `--gtest_list_tests` before trusting it: the correct form above selects 14 tests, the two-dash
form selects 15.

The smallest single failing case, for a bug report:

```bash
ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_SIMULATOR=<qsr-sim>/libttsim.so TT_METAL_SIMULATOR_HOME=<qsr-sim> \
  ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.RatioTwoReadsPerAnnouncedSlot"
```

The two hanging cases, each on its own under an external timeout. gtest has no per-test timeout, which
is why they are excluded above:

```bash
timeout 120 ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.ScratchpadReadEveryEntry*"
timeout 120 ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.DmConsumerManyReads*"
```

Gen1 control, expect all to pass, including both cases that hang on Quasar:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Release/test/tt_metal/unit_tests_api --gtest_filter="Gen2DFBCreditsTest.*"
```

The op-level failure, for reference. Minimal case: batch 1, sentence 32, hidden 768, vocabulary 512.
The failure is a wrong-output `assert_equal`, not a hang. The virtualenv activation is required, not
optional:

```bash
source python_env/bin/activate && \
ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
TT_METAL_SIMULATOR=<qsr-sim>/libttsim.so TT_METAL_SIMULATOR_HOME=<qsr-sim> \
  pytest -x \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding_tiled_input"
```

**Report, do not work around.** §8.2 and §12 are explicit that a credit anomaly is to be reported. The
tempting op-level workarounds, draining `num_entries` dummy entries in the writer or forcing
`num_entries = 1`, would each encode a defect nobody has located into the op, break it again when the
real cause is fixed, and change WH/BH behaviour without a guard. Not applied. Note also that
`disable_dfb_implicit_sync_for_all`, which §7 of the porting recipe forbids, is not a candidate here in
any case: it was measured and it does not help.

---

## Deferred / follow-up items

0. **File RED 3 with the Gen2 runtime / LLK owners.** It is ready: a minimal op-independent repro
   that passes on Wormhole hardware and fails on Quasar, the two conditions the failure requires
   (a producer issuing more NoC transactions than it announces, and a data-movement consumer),
   neighbouring configurations on both sides of each condition to bound it, and eight ruled-out
   candidates each backed by a measurement. Everything is in [RED 3](#red-3). The op needs no change
   when this is fixed. Two things to get right in the report: do not file it as "Gen2 DFB credits are
   broken", which the baseline contradicts, and do not let it be closed as a duplicate of issue #50328,
   whose fix was measured here and does not help.

1. **Under-declared index buffer in `EmbeddingsTilizedIndicesProgramFactory`** (see
   [the one parity exception](#the-one-parity-exception)). Pre-existing on `main`; corrected here
   because the `Scratchpad` contract required it. Deserves its own issue, and a reviewer's call on
   whether to split it out of this PR.

2. **No test coverage for `PADDED` / `BINARY` on either Metal 2.0 path.** The `weight_cache`
   conversion — including the `prepare_local_cache` signature change — is exercised by **nothing** in
   the repo. Every `PADDED`/`BINARY` call site (`models/demos/metal_BERT_large_11/tt/embeddings.py`)
   passes `layout=ttnn.TILE_LAYOUT` with a row-major index tensor, which dispatches the RED fused
   factory. `tests/ttnn/unit_tests/operations/data_movement/test_embedding.py` only ever passes
   `EmbeddingsType.GENERIC`.

   Tests were **not** added here, deliberately: the tilized-indices `PADDED` path hands
   `starting_index` — this core's starting column within the face row — to `prepare_local_cache` as
   the pad-token value
   ([embedding_ind_tilized.cpp:33-41](ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp#L33-L41),
   whose comment hedges on exactly this), which looks wrong independently of anything in this
   change. A new test would most likely fail for that pre-existing reason and confuse the parity
   claim for this PR. Recommended follow-up: parametrize `test_embedding` (row-major output) and
   `test_embedding_tiled_input` over `embeddings_type`, and triage the `starting_index` pad-token
   question separately.

3. **Reactive §7–§8 items considered but not applied.** No device run happened here, so per §2
   these were left alone rather than pre-empted. Two are worth watching for on the first Quasar run:

   - **§6, local self-read / self-copy.** Under `PADDED`/`BINARY`, `read_token_async` serves a
     cached row by NoC-reading from the core's **own** L1 to its own L1 (`UnicastEndpoint` at
     `my_x[noc_id]` / `my_y[noc_id]`). §6 says such a loopback "can spin on `can_post` or silently
     drop the read" on the emulator, and that the fix is a direct L1→L1 RISC copy. Unchanged here:
     it is the same loopback as before, the symptom has not fired, and this path has no test
     coverage (item 2) to fire it. Expect it if `PADDED`/`BINARY` hangs or returns wrong pad rows on
     Quasar. Note that the obvious helper,
     `tt::data_movement::common::tt_memmove`, still uses a NoC loopback for non-overlapping copies;
     only its `copy_via_memmove` fallback is a true CPU copy, and that one carries an open Quasar
     cache-coherency caveat (issue #51763).
   - **§11, degenerate emulator grids.** Both uplifted factories size their work split from
     `device->compute_with_storage_grid_size()` and use no multicast, so there is no `+1` mcast
     corner to clamp. `split_work_to_cores_aligned` should degrade cleanly on a 1×3 emulator grid,
     but that is untested.

4. **`offset == tile_height * tile_height` in `embedding_ind_tilized.cpp` is unreachable.** Falls out
   of the same bound analysis as item 1: `offset` maxes at 1008. Pre-existing dead code, not touched.

---

## How to verify

Per §9, order is **BH → WH → Quasar**, and each build/run is the human's to launch. The commands:

### Build

```bash
# from the repo root
./build_metal.sh -e --enable-fake-kernels-target
```

Kernels changed, so force JIT on every run below:

```bash
export TT_METAL_FORCE_JIT_COMPILE=1
```

### Gen1 parity (Blackhole, then Wormhole)

The two uplifted factories. These must pass unchanged — a failure means behaviour changed, not that
Gen2 was enabled.

```bash
source python_env/bin/activate && pytest \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding_tiled_input" \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_tiled" \
  -v
```

```bash
# EmbeddingsRMProgramFactory: the row-major-output half of test_embedding, plus the
# row-major-specific cases
source python_env/bin/activate && pytest \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding" \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding_unaligned_RM_pages" \
  -k "ROW_MAJOR" -v
```

```bash
source python_env/bin/activate && pytest \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_moe_embedding" \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_tg_llama_sharded_rm_embedding" \
  -v
```

Then the whole file, to confirm the untouched RED fused path is unaffected:

```bash
source python_env/bin/activate && pytest \
  tests/ttnn/unit_tests/operations/data_movement/test_embedding.py -v
```

Run each of the above **twice**, once with `TT_METAL_LLK_ASSERTS` set and once unset (§9: a pass with
asserts off is not proof, and several hangs only assert with them on). The assert-on run is the one
that exercises the `Scratchpad` bounds checks, which is the point of the sizing correction above.

### Quasar

Every Quasar result in this report came from **craq-sim**, with these variables. This is the recipe to
reproduce what is written here:

```bash
export QSR_SIM=<path to the qsr-sim build>
export ARCH_NAME=quasar
export CHIP_ARCH=quasar
export TT_METAL_SIMULATOR="$QSR_SIM/libttsim.so"            # craq-sim ships as a libttsim.so
export TT_METAL_SIMULATOR_HOME="$QSR_SIM"
export TT_METAL_SLOW_DISPATCH_MODE=1                        # the small grids have no fast-dispatch cores
export TT_METAL_FORCE_JIT_COMPILE=1
```

The emulator was **not** run. If someone does run it, the variables differ only in the simulator path,
which is an `emu-quasar-*` build rather than a `libttsim.so`, and the expected outcomes below should
still hold:

```bash
export ARCH_NAME=quasar
export TT_METAL_SIMULATOR=<path to an emu-quasar-* build>   # a path, not "1"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_FORCE_JIT_COMPILE=1

source python_env/bin/activate && pytest \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding_tiled_input" \
  "tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding" \
  -k "ROW_MAJOR or tiled" -v
```

Expected outcomes on Quasar, all diagnosed rather than open:

- The sharded row-major case (`test_tg_llama_sharded_rm_embedding`) fails with
  `Self-loop DFBs are not supported for data-movement kernels on Gen2` — RED 1, not uplifted.
- Every `layout=TILE_LAYOUT` case with a row-major index tensor fails with
  `DataMovementKernel is not supported on Quasar` — RED 2, not uplifted.
- The two uplifted paths run but produce output shifted by `num_entries` pages per core — RED 3, a
  runtime/LLK bug outside the op. Reproduced on craq-sim; the emulator is untried, so treat this last
  one as expected rather than observed there.

Note that `-k` rejects `=`, so a filter written from the parametrize IDs (`-k "batch_size=1"`) is a
hard error rather than a silent empty selection. Bare substrings work: `-k "ROW_MAJOR"` selects 180
of the 265 cases, being the row-major-output half across `test_embedding`,
`test_embedding_unaligned_RM_pages`, `test_embedding_tiled_input` and `test_tiled`. It misses
`test_moe_embedding`, which is not parametrized on layout, so name that one explicitly.

Also per §9: `tt-triage`, `tt-exalens` and device-side gdb are unavailable on the emulator — debug
with DPRINT (needs `TT_METAL_LLK_ASSERTS` unset and the `DPRINT("fmt {}", args)` form), `log_debug()`,
WATCHER, LLK asserts and host-side gdb. Run with debug env both on and off; on the emulator the
debug tooling itself can push a kernel over the size limit or trip the MOP timeout.

---

## Definition-of-done checklist (§10)

- [x] Uplifted in place, existing directory and namespace. Nothing in `experimental/quasar/`, no `::qsr`.
- [~] Factories are `create_program_artifacts`/`ProgramArtifacts` and kernels use `dfb::`/`args::`/`tensor::`/`scratch::` — true for the two uplifted factories; the fused factory is RED 2.
- [x] `opt_level` untouched (an uplift must not change it; none was set before or after).
- [x] Every remaining DFB has a valid `data_format_metadata`; no kernel reads sizes via `fifo_page_size`.
- [~] DM self-loop DFBs converted to `Scratchpad` — done for `index_scratch` and `weight_cache` in both Metal 2.0 factories; the borrowed sharded-output self-loop is RED 1.
- [x] No `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for`; implicit sync left on.
- [x] No non-zero-init semaphore dependency (no semaphores at all).
- [x] **BH and WH pass unchanged** — `test_embedding_tiled_input` confirmed green on Gen1, which covers the one deliberate Gen1-visible change (the index-buffer sizing correction). Remaining commands above for the full sweep.
- [~] **Quasar builds and runs** — both uplifted factories now reach program creation and execute on craq-sim, which they could not before. Output is wrong there because of RED 3, a Gen2 DFB credit-init bug outside the op.
- [x] No DIAG/debug leftovers (a stray unused `dprint.h` include was removed).
- [x] Missing core dependencies flagged rather than bundled: RED 1 and RED 2, plus the follow-ups above.
- [x] This report written, with per-configuration status, every changed file, and the parity argument.

### Feeding back into `quasar_audit.md`

Two things this uplift hit that the audit scaffold does not yet name, worth adding:

1. **The DM self-loop check needs a `borrowed_from` fork.** The audit delegates self-loops to
   `cb_dfb_quasar_audit_helper.md`, and `dm_self_loop_dfbs.md` correctly stops when `borrowed_from`
   is set — but on Gen2 that combination is not a deferrable cleanup, it is a hard program-creation
   failure with no sanctioned conversion. It deserves to be called out as a distinct Quasar-uplift
   blocker class ("fake-FIFO over borrowed memory"), not just a stop inside a Gen1 style pass.
2. **A DM-producer to DM-consumer DFB with no compute stage is a distinct Gen2 risk class.** RED 3
   only shows up on that shape: the consumer-side wait takes the `!COMPILE_FOR_TRISC` spin on
   `llk_intf_get_occupancy`, whereas a compute consumer takes `llk_wait_tiles` and is unaffected. The
   audit should ask, for every DFB, whether *both* endpoints are data movement — the answer predicts
   whether the op is walking on ground other Quasar ops have covered. Related and worth asking in the
   same breath: is the op's shared kernel bound by anything else that has already run on Gen2? Here
   the writer was bound by no other op, so nothing had proven it.

3. **A converted buffer's declared size becomes load-bearing.** `Scratchpad::operator[]` is
   bounds-checked where a CB/DFB address was not, so a DFB whose declared size the kernel overruns
   turns a silent overflow into an assertion the moment it is converted. Checking the reachable
   index range against the declared size belongs in the survey, before the conversion — otherwise it
   surfaces as an assert in a watcher build after the diff is written.
