<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 6 outline: `BcastFusion`

Planned changes for phase 6 of [unified_llama_prefill.md](unified_llama_prefill.md).
Nothing implemented yet.

## 1. Why the expression tree cannot do this

Worth stating precisely, because it looks at first as though it could. An SFPU tree
leaf already receives the tile index, so a "broadcast leaf" could map block tile `t` to
vector tile `t % Wt` and let `SubOp` do the rest. That handles the INTER-tile mapping and
misses the whole point: a reduce result is `Shape<Ht,1>` whose data sits in column 0 of
each tile, and subtracting it from a full block needs that column replicated across all
32 columns WITHIN the tile. `copy_tile` + an SFPU binary cannot do that; the unpacker's
broadcast mode can, and only the FPU bcast ops reach it.

So it is a third kind, as planned -- and its precedent already exists in the tree:
`Strategy<FPUFusion>::bias_finish` is a tile loop of `add_tiles_bcast_rows`.

## 2. The dimension is DECLARED and the shape is CHECKED

An earlier draft of this outline inferred the broadcast dimension from the vector's
shape. **That was unsound and this section replaces it.** The flaw: a `Shape` is measured
in TILES, and the broadcast dimension is a property of the ELEMENT layout inside a tile.
A one-tile vector holding a row (32 values in row 0), a column (32 values in column 0),
or a single scalar at [0,0] is `Shape<1,1>` in all three cases. The shape simply does not
carry the distinction.

Inference appeared to work only because the three vector shapes happen to differ when the
block's tile extents differ. They collide as soon as either extent is 1:

| block (tiles) | rows vec | cols vec | scalar | distinct? |
|---|---|---|---|---|
| 4x6 | 1x6 | 4x1 | 1x1 | yes |
| 1x6 | 1x6 | 1x1 | 1x1 | **no** |
| 4x1 | 1x1 | 4x1 | 1x1 | **no** |
| 1x1 | 1x1 | 1x1 | 1x1 | **no -- all three** |

`Shape<1,N>` blocks are not exotic: that is exactly what a reduction produces, so
broadcasting against one is an ordinary thing to want.

### The fix: an explicit marker on the operand being broadcast

```cpp
x - u::bcast<u::Axis::Cols>(m)
```

The axis is stated because it is information the shape lacks; the shape is then verified
against it, which is information the axis lacks. Two different jobs, neither guessed:

| spelling | requires the vector to be | metal op | vector tile for block tile `t` |
|---|---|---|---|
| `bcast<Axis::Rows>(v)` | `Shape<1, block.cols>` | `_bcast_rows` | `t % cols` |
| `bcast<Axis::Cols>(v)` | `Shape<block.rows, 1>` | `_bcast_cols` | `t / cols` |
| `bcast<Axis::Both>(v)` | `Shape<1, 1>` | `_bcast_scalar` | `0` |

### One axis vocabulary, shared with reduce

The axis names what is EXPANDED, which makes it the same word `reduce` uses for what it
COLLAPSES -- so a reduce and the broadcast that undoes it read alike and cannot drift:

```cpp
u::ComputeBlock m = m_storage.store(u::reduce_max<u::Axis::Cols>(x, one));   // collapses Cols
u::ComputeBlock e = e_storage.store((x - u::bcast<u::Axis::Cols>(m)).exp()); // expands Cols
```

They also match metal's own suffixes: `Axis::Cols` -> `sub_tiles_bcast_cols`.

Planned: `enum class Axis { Rows, Cols, Both }` with `using ReduceAxis = Axis;` kept so no
existing code changes. Writing `ReduceAxis::Cols` inside a broadcast would read badly,
and one vocabulary for both halves is worth the alias.

### The phase-5 pairing benefit survives

The reduce -> bcast check is still automatic, just enforced from the other side: a
`reduce<Cols>` result is `Shape<Ht,1>`, and `bcast<Cols>` demands exactly that. Write
`bcast<Rows>` over it and the shape requirement fails. What changed is that the axis is no
longer *derived* from a shape it cannot be derived from.

### Why a wrapper type rather than `bcast_sub<Axis>(x, v)`

The marker sits on the operand whose layout it describes, which is where the property
actually lives -- and it lets the existing operators carry it, so no `bcast_add` /
`bcast_sub` / `bcast_mul` names are needed at all. Dispatch is on the wrapper TYPE, not on
a shape mismatch, so it does not weaken the strict-equality rule phase 5 chose for
`operator-` between two ordinary blocks.

Constraint to enforce: metal's bcast ops require the broadcast operand to be **in1**, so
only `block <op> bcast<A>(vec)` is provided. The reverse order gets a `static_assert`
naming the reason rather than a lookup failure. The block operand must also be a
`ComputeBlock` rather than an expression, because the FPU reads both operands from
circular buffers -- an SFPU tree lives in DST and has to be stored first.

## 3. Direction: metal's own documentation contradicts itself

`add_tiles_bcast`'s doc says, for `BroadcastType::COL`, that B is "a single tile with a
filled 0-column" and then that the result is `C[h,w] = A[h,w] + B[w]`. Those disagree: a
filled column 0 means the values are indexed by `h`, not `w`. The same mismatch appears in
the `Dim::C` paragraph.

So the direction is taken from evidence, not from the doc:

- **rows** -- CONFIRMED here. `bias_finish` calls `add_tiles_bcast_rows(acc, bias, ...)`
  with a bias whose data is in row 0, and `test_unified_matmul_bias.py` passes. So
  `_bcast_rows` means "B is a row, replicated down the rows".
- **cols** -- inferred, NOT yet confirmed. ttnn's softmax uses `sub_tiles_bcast_cols` for
  `x - rowmax`, where `rowmax` comes from a `REDUCE_ROW` and is therefore a column. The
  phase-6 test must pin this down empirically rather than assume it, and the test has to
  be built so that getting the direction backwards FAILS -- which means a non-square
  block and a vector whose entries differ along its length.

## 4. Types

```cpp
enum class Axis { Rows, Cols, Both };
using ReduceAxis = Axis;                      // the existing name, kept

// The marker. Holds no data beyond the buffer -- its job is to carry the axis in the
// TYPE, which is the thing a Shape cannot express.
template <Axis A, typename S>
struct Broadcast {
    static constexpr Axis axis = A;
    using shape = S;
    uint32_t cb_id;
};

template <Axis A, typename S>
Broadcast<A, S> bcast(const ComputeBlock<S>& v);

struct BcastFusion {};

template <typename Op, Axis A, typename SB, typename SV, typename Chain>
struct BcastNode : expr::Fluent<BcastNode<Op, A, SB, SV, Chain>> {
    using fusion_kind = BcastFusion;
    static constexpr Axis axis = A;
    using block_shape = SB;
    using vec_shape = SV;
    using chain = Chain;
    using shape = SB;                         // a broadcast is shape-preserving

    // The check the axis cannot make, now that the axis is the check the shape cannot.
    static_assert(
        same_shape_v<SV, bcast_vec_shape<SB, A>>,
        "a broadcast vector has the wrong shape for the axis it declares: Axis::Rows needs "
        "Shape<1, cols>, Axis::Cols needs Shape<rows, 1>, Axis::Both needs Shape<1, 1>");

    uint32_t block_cb;
    uint32_t vec_cb;
};
```

`bcast_vec_shape<SB, A>` is the small metafunction giving the shape an axis requires --
`with_hw<SB, 1, SB::cols>`, `with_hw<SB, SB::rows, 1>`, or `Shape<1,1>`.

`shape = SB` means `Storage::store` conformance already covers the destination and
`node_shape`'s primary template needs no specialisation.

The operator overloads, one per op, taking the marker only on the right:

```cpp
template <typename SB, Axis A, typename SV>
auto operator-(const ComputeBlock<SB>& block, Broadcast<A, SV> vec);   // and + and *
```

`is_operand<Broadcast<...>>` stays FALSE so the existing SFPU `operator-` cannot swallow
it, and a `Broadcast` on the left gets a `static_assert` explaining that metal requires
the broadcast operand to be in1.

## 5. Strategy

Per TILE, not per block:

```
reconfig_data_format(block_cb, vec_cb)
<op>_bcast_<dim>_init_short(block_cb, vec_cb)
cb_reserve_back(out_cb, N)
for t in 0..N:
    acquire
    <op>_tiles_bcast_<dim>(block_cb, vec_cb, t, vec_tile(t), t)
    Chain::apply_in_place(0)
    commit; wait; pack_tile(0, out_cb); release
cb_push_back(out_cb, N)
```

Per-tile because `bias_finish` packs a whole block into DST, which caps it at 8 tiles --
and attention's score block is 16. Per-tile costs one DST slot whatever the block size.
Neither operand is popped here: both are `ComputeBlock`s whose destructors pop them, the
same contract `Strategy<SFPUFusion>` follows.

## 6. API

No new function names beyond `bcast` itself -- the existing operators carry it:

```cpp
x - u::bcast<u::Axis::Cols>(m)
(x - u::bcast<u::Axis::Cols>(m)).exp()          // exp(x - rowmax), one pass
e * u::bcast<u::Axis::Cols>(r)
scores * u::bcast<u::Axis::Both>(scale)
```

This replaces the `bcast_add` / `bcast_sub` / `bcast_mul` free functions an earlier draft
proposed. Dispatch is on the `Broadcast` wrapper TYPE, which is an explicit marker the
caller wrote, not on an accidental shape difference -- so the strict shape equality phase 5
chose for `operator-` between two ordinary blocks is untouched.

No `div`: metal has no broadcast divide, and softmax normalises with `recip` then a
broadcast multiply.

## 7. Naming trap in metal's init functions

The `init_short` names are not uniform, so a macro over `{add,sub,mul} x {rows,cols,scalar}`
breaks on two cells:

```
add_bcast_rows_init_short     sub_bcast_rows_init_short     mul_bcast_rows_init_short
add_bcast_cols_init_short     sub_bcast_cols_init_short     mul_bcast_cols_init_short
add_bcast_scalar_init_short   sub_tiles_bcast_scalar_init_short   mul_tiles_bcast_scalar_init_short
                                  ^^^^^^                              ^^^^^^
```

`add`'s scalar init omits `tiles_`; `sub`'s and `mul`'s include it. The per-op traits have
to spell all nine out rather than compose the name.

## 8. Verification plan

- [ ] Selftest: a bcast example per dimension, plus the free/method spellings of the
      chained form. Trace changes -- an intentional re-baseline.
- [ ] `test_unified_bcast.py` on device: {add, sub, mul} x {rows, cols, scalar} against
      torch, on a NON-SQUARE block with a vector whose entries vary along its length, so
      a rows/cols mix-up cannot pass.
- [ ] Prove the shape guard: a vector that is neither a row, a column, nor a scalar must
      fail to compile.
- [ ] Prove the reduce -> bcast pairing end to end: `reduce_max<Cols>` then
      `bcast_sub`, in one kernel, matching `x - x.max(dim=-1)`.
- [ ] **A bcast followed by an SFPU op in the same kernel.** The bcast init leaves the
      unpacker in a broadcast mode; if the following op does not reset it the result is
      quietly wrong. Phase 4 gave SFPU leaves their own `copy_tile_to_dst_init_short`,
      which should cover it -- this test is what makes that "should" a "does".

## 9. Open questions

1. `Shape<1,1>` block: resolve the three-way match to scalar. Agree?
2. ~~Should `bcast` also accept a `Storage` for the vector?~~ **DECIDED: `ComputeBlock`
   only.** A `ComputeBlock`'s constructor performs the `cb_wait_front` that makes reading
   the buffer legal at all; a `Storage` is a cb id and a shape with nothing having waited
   on it. Accepting one would also force `bcast` to decide whether to pop, and both
   answers are wrong -- popping breaks a resident vector that every tile re-reads, not
   popping leaks the pages. `ComputeBlock` puts the wait in the constructor, the pop in
   the destructor, and the lifetime in the variable's scope, which is why `reduce_*`
   takes one for its scaler.

   Checked while deciding: `.bias()` duck-types on `operand.get_cb_id()`, and `Storage`
   exposes `cb_id` as a public MEMBER with no such accessor -- so `.bias(some_storage)`
   already fails to compile. `get_cb_id()` is serving as a de-facto "this was waited on"
   proof, and requiring `ComputeBlock<S>` in the signature just makes that visible.

   What none of this fixes is LIFETIME: a loop-scoped `ComputeBlock` passed as a bias or
   a broadcast vector compiles and is then popped at the end of the first iteration,
   leaving the next waiting on a refill nobody issues. Only a distinct resident type
   would catch that, which was considered and rejected. Documented at every resident
   operand instead, with the selftest's circular-buffer balance check as the backstop.
3. Refactoring `bias_finish` onto `BcastFusion`: the plan proposed it with a
   byte-identical trace. That will NOT hold -- `bias_finish` packs a whole block and
   lives inside the FPU strategy's restore dance, while this packs per tile. Evaluate
   after `BcastFusion` exists; do not promise it.

## 10. What the API looks like in use

### The minimal case

```cpp
using In  = u::Shape<4, 6>;
using Row = u::Shape<1, 6>;
using Col = u::Shape<4, 1>;

u::ComputeBlock x = u::noc_load<1>(x_storage, x_acc, b).wait();
u::ComputeBlock v = u::noc_load<1>(row_storage, v_acc, 0).wait();   // v : Row

u::noc_store<0>(out_storage.store(x + u::bcast<u::Axis::Rows>(v)), out_acc, b);
```

Declaring `Axis::Cols` there is a compile error -- `Axis::Cols` requires `Shape<4,1>` and
`v` is `Shape<1,6>`. Declaring the axis is what makes the one-tile case work too:

```cpp
using One = u::Shape<1, 1>;         // x and v are both a single tile
u::ComputeBlock x = ...;            // 32x32 values
u::ComputeBlock v = ...;            // values in row 0 only

x - u::bcast<u::Axis::Rows>(v)      // unambiguous; nothing here is guessed
```

### Softmax, which is the whole point

Row-wise softmax over a `Ht x Wt` block. Every intermediate shape is derived, and the axis
is written once per direction:

```cpp
using In  = u::Shape<kHt, kWt>;
using Vec = u::reduce_shape<In, u::Axis::Cols>;      // Shape<kHt, 1>

u::Storage<In>  x_storage(kCbX),  e_storage(kCbE),  out_storage(kCbOut);
u::Storage<Vec> m_storage(kCbM),  s_storage(kCbS),  r_storage(kCbR);
u::Storage<u::Shape<1, 1>> one_storage(kCbOne);

u::ComputeBlock one = u::fill_reduce_scaler<1>(one_storage);       // kernel scope

for (uint32_t b = 0; b < num_blocks; ++b) {
    u::ComputeBlock x = u::noc_load<1>(x_storage, in, b).wait();

    u::ComputeBlock m = m_storage.store(u::reduce_max<u::Axis::Cols>(x, one));
    u::ComputeBlock e = e_storage.store((x - u::bcast<u::Axis::Cols>(m)).exp());
    u::ComputeBlock s = s_storage.store(u::reduce_sum<u::Axis::Cols>(e, one));
    u::ComputeBlock r = r_storage.store(u::recip(s));

    u::noc_store<0>(out_storage.store(e * u::bcast<u::Axis::Cols>(r)), out, b);
}
```

`Axis::Cols` appears in both the reductions and the broadcasts, and means the same thing in
both: the axis being collapsed, then re-expanded. Change it to `Rows` in one place and the
shapes stop agreeing.

Two things this shows beyond the ops. `x` is read TWICE -- by the reduction and by the
subtract -- and `e` twice likewise; both are held as `ComputeBlock`s for the iteration, so
each is waited once and popped once at scope end. That is the resident-operand idiom the
fused bias and the reduce scaler already use. And `Vec` is written once and reused for
three buffers, so changing `kHt` moves everything together.

### The scalar case: attention's 1/sqrt(d)

```cpp
u::Storage<u::Shape<1, 1>> scale_storage(kCbScale);
u::ComputeBlock scale = u::fill_scalar<1>(scale_storage, u::bf16_pair(1.0f / std::sqrt(kHeadDim)));

... scores * u::bcast<u::Axis::Both>(scale) ...
```

See section 11 on `fill_scalar` -- `fill_reduce_scaler` may fill more of the tile than
`_bcast_scalar` reads.

### For comparison, the same subtract written against metal directly

```cpp
reconfig_data_format(cb_x, cb_v);
sub_bcast_cols_init_short(cb_x, cb_v);
cb_wait_front(cb_x, Ht * Wt);
cb_wait_front(cb_v, Ht);
cb_reserve_back(cb_out, Ht * Wt);
for (uint32_t t = 0; t < Ht * Wt; ++t) {
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_x, cb_v, t, t / Wt, 0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
}
cb_push_back(cb_out, Ht * Wt);
cb_pop_front(cb_v, Ht);
cb_pop_front(cb_x, Ht * Wt);
```

versus

```cpp
e_storage.store((x - u::bcast<u::Axis::Cols>(m)).exp());
```

The `t / Wt`, the `_cols` suffix agreeing with it, and the two pop counts are four things a
caller can get wrong and the compiler cannot see. All four are derived above; the one thing
the caller still states -- the axis -- is the one thing that genuinely cannot be derived.

## 11. One more open question, from writing the examples

`fill_reduce_scaler` writes the constant into row 0 of each of the tile's four faces,
which is the pattern `reduce_tile` folds in. `bcast_scalar` is documented as wanting "a
filled single value at location [0,0], zeros elsewhere". Those differ. Either the extra
values are ignored in SCALAR broadcast mode -- likely, but an assumption -- or the scalar
case needs its own filler. Resolve before using `bcast_mul` with a `Shape<1,1>` vector:
write `fill_scalar` if the shared filler turns out to be wrong, and test the scalar
dimension against torch either way.
