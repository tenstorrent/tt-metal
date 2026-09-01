# `tt/unified` -- a one-page overview

A unified kernel is **one source describing a whole Tensix pipeline**, compiled once per
baby RISC-V thread. Each statement lowers to that thread's half of the dataflow-buffer
protocol, and the halves that do not belong to a given thread compile away to nothing.

```
       INPUT           |     OUTPUT             |     INTERMED
-----------------------|------------------------|--------------------
       DM    Compute   |     DM    Compute      |     DM    Compute
-----------------------|------------------------|--------------------
  reserve <- *         |     * -> reserve       |           reserve
    write              |            write       |             write
     push ->    wait   |  wait <-    push       |              push
                read   |  read                  |              wait
        * <-     pop   |   pop -> *             |              read
                       |                        |               pop
```

The model abstracts two things that a Metalium kernel normally spells by hand: the split
across threads and dataflow buffers, and the tile loop over the DST register file. It adds
escape hatches so that neither abstraction is a wall.

Layering is `core` -> `adaptor.hpp` (binds intrinsics to metal) -> `api.h` (contracts) ->
`math.hpp` (fusion kinds, strategies) -> `expr.hpp` (op-agnostic tree) -> `impl.hpp`.

---

## 1. Abstracting multiple threads and DFBs

### 1.1 Lineage

Heavily inspired by Stas's single-threaded proposal: one program text for the whole core,
rather than a reader, a writer and a compute kernel maintained in lockstep. The departure
is that the thread assignment stays **visible and deliberate** rather than inferred --
see below.

### 1.2 Data movement carries an explicit thread parameter

Every data-movement call names its thread as a template argument, and it is up to the
programmer to assign them:

```cpp
u::ComputeBlock a = u::noc_load<0>(a_storage, a_acc, block).wait();   // read on thread 0
u::noc_store<1>(out_storage.store(a.exp()), out, block);              // write on thread 1
```

The call compiles away entirely on every other thread, which is what lets the three
projections share a source.

Note the `.wait()`, `noc_load` actually returns a `NocAsyncReadTx` object which
gives the programmer fine grain control for manually synchronizing the noc.

### 1.3 `Block` as the central DFB abstraction

`Storage<S>` is a dataflow buffer: an id plus the `Shape` of one block, checked at
construction against the depth the host configured. It is agnostic to how many
back buffers it has, and since it is shared across threads, double-buffering is
still the right rule of thumb so that DM threads can start fetching the next
`Block` which compute is using the current.

`Block<S>` is **move-only evidence that a Storage was produced into** -- it comes back from
anything that has already pushed. Move-only so it reaches exactly one consumer; consumers
take it by value. A `Block` must be moved on to either

- a **DM thread**, via `noc_store<thread>(block, ...)`, which waits, writes and pops; or
- the **compute thread**, via `ComputeBlock<S>`, which waits in its constructor and pops in
  its destructor.

That destructor is the whole protocol in one place: the pop happens at end of scope, so the
kernel never writes `cb_pop_front` and never forgets it. `ComputeBlock` is also the
expression leaf, so the value flows straight into the compute layer.

In an assert build a `Block` that is destroyed without reaching a consumer aborts -- a
dropped output block is otherwise a silent hang. `RetainedBlock<S>` is the escape for state
carried across a loop (the running max, sum and output of an online softmax): it *moves* the
obligation into a slot that outlives the iteration instead of discharging it, so the
diagnostic is relocated rather than switched off. It costs nothing in a release build.

`Accumulator<S>` holds the state a multi-block matmul's k-loop would otherwise carry, so the
kernel keeps the loop and the operands stay streamable.

---

## 2. Abstracting compute and the DST register file

### 2.1 Expression-tree metaprogramming

A compute expression is a tree encoded in its own type, so there is no tree to *build* --
only to walk:

```cpp
u::ComputeBlock out = out_storage.store(x * cos + rot * sin);
//                                      ^ four leaves, one pass
```

`expr.hpp` is deliberately op-agnostic: it knows nothing about dataflow buffers, the NOC or
Tensix. It supplies the tree shapes, a compile-time register allocator, the emission walk
and the method spelling of the ops; `math.hpp` supplies the policies -- which ops have an
FPU form, what a reduction's output shape is, how a broadcast is checked.

The shape of the result is derived from the expression and checked against the destination
`Storage`, which is the check that replaces every hand-derived page count.

### 2.2 It loop-tiles the data over DST automatically

DST holds 8 tiles (4 under a 32-bit Dest), and the library owns the loop that fits the work
into it. Register allocation is Sethi-Ullman numbering:

```
need(leaf)   = 1
need(unary)  = need(child)
need(binary) = max(need(L), 1 + need(R))
```

so a left-associated chain of any length costs two slots. Every slot number is a template
parameter, so the emitted code contains only compile-time constants -- no base-offset
arithmetic at run time.

A `Strategy` per fusion kind then picks the loop:

- **SFPU tree** -- every operand needs its own slot, so the driver either walks tile by tile
  or, when the tree is narrow enough that a whole group of tiles fits, hoists the leaves out
  and amortises the reconfigures.
- **FPU elementwise** -- operands stay in L1 and the FPU reads them itself, so DST holds only
  results and the whole group fits in one acquire.
- **FPU (matmul)** -- a single acquire when the output subblock fits the budget, and a banded
  walk when it does not.
- **Broadcast** and **reduce** have their own.

The kernel author writes none of this, and does not write `tile_regs_acquire` /
`commit` / `wait` / `release` either -- the strategies bracket every pass.

### 2.3 The same lever could tile over Quasar's Tensix engines

Because the loop is *generated* rather than written, its extent is the library's to choose.
Quasar has four independent compute kernels per node; distributing the same tile loop across
them is a change inside `Strategy`, not in any kernel. This is a consequence of the design,
not something the current implementation does -- `api.h` still assumes one compute
projection today (see `unified_metal2_spec.md` 5.4).

---

## 3. Escape hatches

The model is a layer over the base Metalium API, not a replacement for it, and every hatch
is written in terms the raw API already uses. The built-in overloads are implemented through
the same hatches, so these paths carry the same weight rather than being a side door.

### 3.1 Custom data movement

`noc_load` / `noc_store` take a routine instead of an accessor. The harness keeps the
protocol -- reserve, the write pointer, the read barrier, push -- and the routine owns the
traffic. It receives `L1Entries`: this core's page base, the page size, and the count the
handle will push.

```cpp
u::ComputeBlock a = u::noc_load<0>(a_storage, [&](u::L1Entries pages) {
    for (uint32_t p = 0; p < pages.count; ++p) {
        const uint32_t row = i * mt + p / kt;      // strided gather the built-in
        const uint32_t col = b * kt + p % kt;      // overload cannot express
        noc_async_read(a_acc.get_noc_addr(row * ktot + col), pages.addr(p), pages.entry_bytes);
    }
}).wait();
```

Rules: issue only reads (or only writes, for `noc_store`), only on this thread's NOC, and
loop on `pages.count` -- that is what gets pushed, whatever the routine actually wrote.

### 3.2 Custom compute

`custom_compute` hands a routine the raw dataflow-buffer ids of blocks the model still owns.
Everything inside is the base compute API:

```cpp
u::custom_compute(a, b, [&](uint32_t a_dfb, uint32_t b_dfb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    cb_reserve_back(kDfbOut, tiles);
    ckernel::sub_init(a_dfb, b_dfb);
    for (uint32_t t = 0; t < tiles; ++t) {
        ckernel::tile_regs_acquire();
        ckernel::sub_tiles(a_dfb, b_dfb, t, t, 0);
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ckernel::pack_tile(0, kDfbOut);
        ckernel::tile_regs_release();
    }
    cb_push_back(kDfbOut, tiles);
#endif
});

// The routine pushed the pages, so this only names them; a DM thread drains it
// exactly as it would a Block from Storage::store.
u::noc_store<1>(u::Block<Blk>{out_storage}, out, 0);
```

The harness waits the input blocks and pops them at end of scope. That is all. DST
bracketing, the output reserve/pack/push, and putting the unpacker, math and packer
configuration back the way it was found are the routine's own -- the last one bites, because
whatever the routine leaves set, the next unified op inherits.

`#if IS_COMPUTE_THREAD` inside the lambda is required, not optional: the body is *compiled*
on all five projections even though it is only *called* on one, so every name it mentions
has to resolve everywhere. `adaptor.hpp` stands in for the intrinsics a given projection
does not have.

### 3.3 Compatibility with the base API

Nothing above is exclusive. Buffer slots arrive as ordinary named compile-time values
(`get_named_compile_time_arg_val("dfb_<name>")`), `Storage` takes a raw id, and the library
names no host-side token of its own -- so a unified body can be composed into a kernel
beside untouched hand-written ops, on buffers the host allocated normally. That property is
what makes incremental adoption possible; see `unified_blaze_integration_spec.md` 1.

---

## Where to look next

| | |
|---|---|
| `tt/unified/api.h` | the surface: declarations, contracts, and the hazard each one exists for |
| `tt/unified/expr.hpp` | the tree, the allocator, the emission walk |
| `tt/unified/math.hpp` | fusion kinds and the per-kind DST strategies |
| `unified_kernels/` | worked kernels -- `eltwise_add_exp` to start, then `attention`, `matmul_blocked`, `flash_attention` |
| `unified_api_hazards.md` | the failure modes the design is shaped by |
| `unified_metal2_spec.md` | the Metal 2.0 port, and what Quasar would still need |
