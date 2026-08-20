<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 10 proposal: flash chunking / online softmax

The first phase since the shape refactor that needs new library surface. Written up before
implementing because the draft turned up three gaps and one bug in itself.

## 1. What the algorithm needs

K and V stream in chunks so the score block never exists in full. Per query row, three
running values are carried across the chunk loop:

    m   the running maximum score
    l   the running sum of exp(score - m)
    o   the running UNNORMALISED output

When a chunk raises the maximum, everything already accumulated under the old one has to be
corrected before the new chunk folds in.

`Accumulator` cannot express this. It carries a running TOTAL and adds each matmul's product
to it; nothing in it can RESCALE the total between steps. That is the new idiom, and it is
the only thing on the llama-prefill path that was never a missing op.

## 2. The formulation to use, and why not the textbook one

The obvious writing computes `p = exp(s - m_new)`, which needs `m_new` in the same iteration
that produces it. That does not survive contact with circular buffers: reading a buffer pops
it, so a value both used now and kept for later would be consumed before the next iteration
sees it. **My first draft had exactly that bug** -- it stored `m` and then built a
`ComputeBlock` over it, whose destructor popped the state at the end of every iteration.
Caught by review, not by a run.

The equivalent formulation below normalises each chunk by its OWN row max and folds the
difference into the corrections, so `m_new` is written as state and never read in its own
iteration:

    rm      = rowmax(s)                  this chunk's max
    p       = exp(s - rm)                bounded by 1 by construction
    m'      = max(m, rm)                 the new state -- written, not read
    c_old   = exp(m  - m')               rescales everything accumulated so far
    c_new   = exp(rm - m')               rescales THIS chunk's contribution
    l'      = l * c_old + rowsum(p) * c_new
    o'      = o * c_old + (p @ V) * c_new

Every exponent is non-positive, so nothing can overflow -- which matters here, because phase
7 established that this SFPU's `exp` has a finite input domain and returns garbage outside it.

## 3. Library additions -- SETTLED

Two, not three, and the second one replaced both halves of what I first proposed.

### 3a. `max_(a, b)` -- landed

`MaxOp` over metal's `binary_max_tile`, a clone of `AddOp`. A named function rather than an
operator, since `max` has no punctuation, with the same `is_operand` SFINAE and the same
FPU-fusion rejection as `operator+`. Covered in `test_unified_binary.py`, where it measures
PCC 1.000000 and relative error 0.00000 -- exact, as a max of two bfloat16 values must be.

### 3b. `RetainedBlock<S>` -- landed, and it subsumes the rest

My proposal was two additions: `Storage::store_retained()` returning void, and an explicit
`ComputeBlock(const Storage&)`. `RetainedBlock` replaces both and is a better idea, because
of one distinction I had wrong.

**The obligation should TRANSFER, not be discharged.** `store_retained()` would have pushed
the pages and simply stopped tracking them -- switching off the check that catches a dropped
output. `RetainedBlock` moves the `Block` into a slot that carries the obligation itself, so
`~RetainedBlock` asserting it is empty says "you pushed pages and nobody ever waited on
them". Same diagnostic, relocated rather than thrown away.

```cpp
RetainedBlock<Vec> m;                            // OUTSIDE the loop: that is the lifetime
for (uint32_t j = 0; j < chunks; ++j) {
    if (j == 0) {
        m = m_storage.store(first(...));         // moves the obligation into the slot
    } else {
        ComputeBlock<Vec> prev = m.release();    // the consumer: waits, then pops
        m = m_storage.store(update(prev, ...));
    }
}
```

Neither of the original additions is needed: `m = storage.store(node)` is the write side and
`m.release()` hands a `Block` to the existing `ComputeBlock` constructor.

**It costs nothing in a release build.** Every member is assertion-only, so
`sizeof(RetainedBlock) == sizeof(Block) == 4`; with assertions armed it is 12 against 8. The
reasoning is that `Block`'s destructor is itself assertion-only, so in a release build there
is nothing to run and therefore nothing to track -- and that premise is pinned by a
`static_assert` on `is_trivially_destructible<Block>` rather than left to a comment, so a
future `Block` with a real destructor breaks the build instead of leaking.

**Three assertions, one invariant.** `held` says whether the slot is occupied, and each
operation states its precondition: `emplace` asserts empty, `release` asserts occupied, the
destructor asserts empty. Overwriting without releasing is a hard error rather than something
to recover from, because it cannot un-push the old pages -- the buffer would hold two and the
next reader would get the stale one, so it is always a protocol bug.

Verified by three misuse probes in `tmp.cpp`, each aborting under `-DASSERT_ENABLED=1`:
dropping a held block, assigning twice without releasing, and releasing an empty slot.

### 3c. `copy(block)` -- still open, still needed

Chunk 0 seeds the state with `m = rm`, which is a copy from one buffer to another, and the
model cannot spell one. `store(as_node(b))` works today but `as_node` is an internal hook.
Three lines would make it honest; the alternatives are `max_(rm, rm)` or
`rm * bcast<Axis::Both>(one)`, which both work and both read as tricks.

Not blocking: the kernel can use `max_(rm, rm)` and this can be revisited.

### What the demonstration turned up on the way

`tmp.cpp` exists to show the hazard, and building it found that the selftest's
circular-buffer balance check **had never matched a single line** -- it searched for the tag
`"cbN)"` while every trace line writes `"cbN,2"`. Every "protocol balanced" this model has
ever printed was vacuous. Fixed, extended to all 32 buffers, and strengthened with the rule
that catches this class of bug: for a buffer one thread both produces and consumes, every
pushed block must be waited on exactly once. `reserve == push` together with `wait == pop`
says nothing about whether the two sides agree.

## 4. Kernel shape

`unified_kernels/flash_attention.cpp`, drafted but NOT working yet. Around 200 lines and 23
circular buffers -- Q resident, then per chunk K, V and mask, six scratch buffers, and two
each for m, l and o alternating by parity.

Two buffers per state variable rather than one: with the state written and the old value not
yet popped, a single buffer holds both and `wait_front` returns the older.

Bounded by the DST budget, as attention is: `Sq*Sk <= 8` and `Sq*D <= 8`, where Sk is now the
CHUNK width rather than the whole sequence -- which is the point. A longer sequence costs more
chunks, not a bigger block.

## 5. The test that matters

**Chunk invariance.** The same Q, K, V and mask, run at 1, 2 and 4 chunks, must give the same
answer -- and all three must match torch. That is the whole claim of the online algorithm: the
result does not depend on how the sequence was carved up. A correction that is wrong, applied
in the wrong order, or skipped shows up as a difference between chunk counts even when each
individual run looks plausible.

Plus what the non-flash test already gates on: max absolute error against torch, and the
causal identity `out[0] == V[0]`.

## 6. Still open

1. **`copy(block)`** for seeding chunk 0's state, or `max_(rm, rm)` and add nothing.

2. **Whether `RetainedBlock` should also own the double buffering.** It holds one value; the
   kernel still picks which of two circular buffers each iteration writes, by parity. A
   `State<S>` over a pair of buffers would absorb that and put "read the old, write the new,
   retire the old in that order" in one place. Worth deciding once there is a second caller --
   designing it against one is how the first proposal went wrong.
