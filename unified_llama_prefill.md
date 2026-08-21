<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Llama prefill on unified kernels

Ordered least to most code impact. Each phase is independently testable, and the
first five are library work that phases 6+ only consume.

**Target for the first milestone** (phases 1-7): single-head scaled dot-product
attention, one core, non-flash, post-RoPE inputs, host-supplied additive causal
mask.

```
out = softmax(Q@Kt * scale + mask) @ V      Q,K,V : S x D   (S=128, D=128 -> 4x4 tiles)
```

At S=128 the score block is 16 tiles (32KB), so it fits L1 and needs no flash
chunking. Checkable against `torch.nn.functional.scaled_dot_product_attention`.

## Already works -- verified, no action needed

- `matmul<Geom>` + `Accumulator<Dst|L1>` + `.bias()` -- QKV and output projections.
- `reduce_max` / `reduce_sum` over `ReduceAxis::Cols` -> `REDUCE_ROW` -> `Ht x 1`.
  That is exactly the row-statistic shape softmax wants, and the packer edge masks
  zero everything outside column 0.
- `exp_` over a block.
- Elementwise add of two blocks from **different** CBs. `TileSource::emit` is
  `copy_tile(cb_id, tile, dst)` per leaf (`tt/unified/math.hpp:79`), so `a + b`
  already spans two CBs -- an additive causal mask needs nothing new.
- Holding a `ComputeBlock` at kernel scope while it is read twice (the resident
  operand idiom from the fused bias). Softmax needs `x` for both the row max and
  the subtract that follows.

---

## Phase 1 -- Unary SFPU ops: `recip`, `rsqrt`, `sqrt`

Each is a clone of `ReluOp` (`tt/unified/math.hpp:125`). `recip` is what softmax
normalizes with; `rsqrt` is what RMSNorm needs.

- [x] `tt/unified/adaptor_v1.hpp` -- include `eltwise_unary/recip.h`, `sqrt.h`, `rsqrt.h`
- [x] `tt/unified/math.hpp` -- three op structs + free-function templates over expr nodes
- [x] `tt/unified/expr.hpp` -- `fluent_recip` / `fluent_rsqrt` / `fluent_sqrt` hooks
- [x] `tt/unified/api.h` -- `ComputeBlock` overload declarations
- [x] `tt/unified/impl_v1.hpp` -- definitions
- [x] `unified_selftest.cpp` -- ckernel stubs + extend the permanent syntax probe
- [x] `unified_kernels/unary.cpp` + `test_unified_unary.py`

**DONE.** Selftest passes `-Werror` on all three projections; the probe grew to 121
COMPUTE instructions and the free/method spellings agree. On device: recip, sqrt,
rsqrt and a `recip(sqrt(x))` chain all match torch, max relative error 0.0039-0.0051
(bf16 is 2^-8 = 0.0039, so this is at the format's floor). All six pre-existing
unified tests still pass.

Facts worth keeping:

- **Naming.** A trailing underscore only where the name shadows `<cmath>`: `exp_`
  and `sqrt_` have one, `relu`, `recip` and `rsqrt` do not. The method spelling is
  never shadowed, so it stays bare -- `x.exp()`, `x.sqrt()`.
- **PCC would have passed the sabotage.** Deleting the `sqrt` from the chain leaves
  PCC at 0.9958, above the 0.99 threshold; max relative error goes to 0.414. The
  gate is relative error, and the test says so.
- **The JIT cache does not mask header edits.** A sabotaged `RecipOp` in `math.hpp`
  failed identically with the cache warm and with it deleted, so device sabotage
  runs do not need a cache wipe.
- **Five sites per unary op** -- `ReduceNode`, generic tree, `MatmulNode`, the
  fluent hook, and the `ComputeBlock` decl/def pair. Written out longhand here
  because that is what the phase called for, and because no function-like macros
  exist anywhere in `tt/unified`. At five ops the duplication is ~20 lines each and
  still readable; if phase 2's binaries and a later `silu`/`gelu`/`tanh` push it to
  ten, an X-macro over the overload set is the obvious fix. **Revisit at phase 2,
  not before.**

## Phase 2 -- Binary SFPU ops: `mul`, `sub` (and maybe `div`)

Clones of `AddOp` (`tt/unified/math.hpp:87`) over `mul_binary_tile` /
`sub_binary_tile` / `div_binary_tile`, plus `operator*` and `operator-` alongside
the existing `operator+`.

- [x] `tt/unified/math.hpp` -- op structs + `operator-` / `operator*` / `operator/`
- [x] `tt/unified/math.hpp` -- one FPU-fusion guard per operator, message factored
      into `reject_fpu_operand<A>()` so the four cannot drift
- [x] `unified_selftest.cpp` -- stubs + probe
- [x] `unified_kernels/binary.cpp` + `test_unified_binary.py`
- [x] **`div` decided: kept.** It is ~12 lines, it completes `+ - * /`, and for a
      genuine elementwise divide it is ONE SFPU pass where `a * recip(b)` is two.
      Softmax still normalises with `recip` + a broadcast multiply, which is a
      different shape entirely (phase 6) -- `div` does not serve that.

**DONE.** Single ops land at 0.0038-0.0058 max relative error across four seeds,
which is bfloat16's own 2^-8 floor; the mixed chain at 0.025-0.028.

Facts worth keeping:

- **Binaries were far cheaper than the unaries.** No `expr.hpp`, `api.h` or
  `impl_v1.hpp` work at all: `operator+` is generic over `as_node`, and a binary is
  not a method, so it never touches the `Fluent` mixin. Three sites each, all in
  `math.hpp`. **This settles the macro question deferred from phase 1: no.** The
  duplication did not grow the way the unaries' did.
- **`tt/unified/impl_v1.hpp` and `api.h` were untouched by this phase**, which the
  plan had expected to need mirroring. They did not.
- **Operand order is gated explicitly.** For `sub` and `div` the SWAPPED reference
  must fail. Swapping the operands inside `operator-` reports
  `swapped rel err = 0.004 (MATCHES)` and fails -- a test comparing against one
  reference could not tell that apart from correct.
- **The chain's tolerance is 0.05 for a reason, not for slack.** `(a + b) - a`
  cancels, amplifying relative error by `|a + b| / |b|`, which reaches 5 over the
  input range: `5 * 2^-8 = 0.020` before the mul and div round again. Measured
  0.025-0.028. Documented in the test so nobody tightens it and gets a flake.
- **All four FPU-fusion guards verified by compiling a violation** -- `matmul(a,b)`
  as an operand of each of `+ - * /` fails with the shared message.

## Phase 3 -- Matmul transpose flag (Q@Kt)

**DONE.** Landed after the shape refactor, which is what gave it a home: the transpose
is the one matmul property shapes cannot derive, so it is the only thing left to state.

- [x] `enum class TransposeB { No, Yes }` -- not a bool, so call sites read
      `matmul<u::TransposeB::Yes>(a, b)` rather than `matmul<true>(a, b)`
- [x] `MatmulGeometry<SA, SB, Tr>` carries it; `MatmulNode<SA, SB, Tr, Chain>` threads it
- [x] `matmul_init<SA, SB, Tr>` -- the transpose became compile-time here too
- [x] Both `kTranspose` constants now read `G::transpose`, covering all four downstream
      sites in one place
- [x] `unified_kernels/matmul.cpp` names the flag ONCE as `kTransposeB` and passes it to
      both `matmul_init` and every `matmul()`
- [x] `test_unified_matmul_transpose.py` -- a real A@B.T over 11 configurations

### What the flag actually is

A PER-TILE transpose of B, and nothing more. The tile grid is untouched. So the flag
alone gives neither A@B nor A@B.T once B is wider than one tile; a true transpose needs
the reader to place page (r, c) at slot (c, r) as well. The test supplies that half from
the host and matches torch's `a @ b.T` to 0.005-0.009 across all 11 rows.

### Coverage boundary, stated because it is not obvious

Forcing each site to `transpose=0` shows what each is worth:

| site | what fails |
|---|---|
| `matmul_block` | every transposed row |
| Dst reload restore | **only** `k_blocks>=2` in Dst mode -- all five single-block rows pass |
| `bias_finish` restore | nothing |
| L1 biased-finish restore | nothing |

The last two are unreachable rather than untested: both restore the FPU "for the next
output block", and no kernel here emits more than one output block per launch. Their
transpose argument is correct by construction and unverified by execution. The first
kernel that loops output blocks -- attention, over Q chunks -- makes them live and they
must be re-verified then.

The middle row is the one that justifies the multi-block cases: a single-block suite
passes a missing transpose in the Dst reload path.

### Deferred

`matmul_init`'s transpose must match every `matmul()`'s, and nothing can check that
across two calls. Resolved the way ttnn resolves it: one named constant in the kernel,
used twice. A `matmul_nt(a, b)` paired with a grid-transposing loader -- so the two
halves cannot be separated -- is the sugar worth revisiting once the attention kernel
has written the pairing once by hand.

## Phase 4 -- Data-format reconfig between SFPU tree leaves

**DONE.** It was a real, silent bug, not a latent one: an SFPU tree whose leaves live in
circular buffers of different data formats returned garbage, with no hang and no assert.

- [x] Decided: reconfig per leaf, not a static assert. A static check needs the format in
      the type system, which the model does not carry -- and reconfiguring makes mixed
      formats WORK rather than merely refusing them.
- [x] `TileSource::emit` takes a `reconfigure` flag; when set it calls the ONE-argument
      `reconfig_data_format_srca(cb_id)` plus `copy_tile_to_dst_init_short(cb_id)`
- [x] `expr.hpp` gained `leaf_count_v<Node>` -- structural, beside `Need` -- and threads
      `reconfigure` down the walk without knowing what it means
- [x] `test_unified_mixed_format.py`: in0 bfloat16, in1 float32, one expression over both

### Why the one-argument form

`copy_tile` carries no format, and `copy_tile_to_dst_init_short` explicitly "does not
reconfigure the unpacker data types" -- so ttnn's `where` kernel, which uses it across
three buffers, is relying on the same single-format constraint this model had. The
conditional `_with_dt(old, new)` form is cheaper but needs the PREVIOUS operand, which
would mean tracking hardware state across the tree walk. ttnn's binary_ng does thread it,
which it can afford because it batches all of one operand's tiles together; this per-tile
loop cannot. The one-argument form is unconditional and needs no history, so a leaf stays
self-sufficient.

### What it costs, and what it does not

Gated on leaf count, so the common cases pay nothing:

| tree | reconfigs |
|---|---|
| one leaf (`exp_(a)`, `relu(a)`) | one, before the first tile |
| two or more leaves | one per leaf per tile |

Verified in the trace: `exp(in)` over two tiles reconfigures on tile 0 only.

### The sabotage that gives the test meaning

Forcing `reconfigure` to false -- the pre-phase-4 behaviour -- gives:

    mixed-format (in1 float32)    max rel err = inf     FAIL
    binary, add_exp, unary        all bfloat16          PASS

So the bug was invisible to every test in the suite, which is why a dedicated
mixed-format test had to exist rather than a note in a comment.

### Still assumed

Uniform TILE GEOMETRY -- one 32x32 tile per circular-buffer page. The model already
assumes this everywhere (see `Storage::store`), and a buffer with different face geometry
would need `is_tile_dim_reconfig_en` as well.

## Phase 5 -- Static block shapes (makes both geometries moot)

`Storage`, `Block` and `ComputeBlock` carry a dynamic 1-D `num_pages`. Every kernel
then hand-derives it from a geometry it also states separately, and nothing ties the
two together. `reduction_tree.cpp` is the worst case:

```cpp
using PerCore = u::ReduceGeometry<in_ht, in_wt>;
constexpr uint32_t reduced_tiles_per_block = PerCore::out_tiles(kAxis);
u::Storage tmp1_storage(kCbTmp1, reduced_tiles_per_block * num_cores_y);  // hand arithmetic
u::Block per_core_sum = tmp0_storage.store(RT_REDUCE(PerCore, a));        // geometry restated
```

That `* num_cores_y` is where the out-block-index collision lived.

### Why static, and why rank 2

An audit of all 1294 kernel sources under `ttnn/cpp/ttnn/operations/` settles both
questions empirically:

| | compile-time | runtime | % CT |
|---|---|---|---|
| Block shape (`in0_block_w`, `out_subblock_h/w`, `Sq_chunk_t`, `DHt`, `num_tiles_per_cycle`, `per_core_M/N`) | **311** | 38 | **89%** |
| Iteration count (`num_tiles`, `num_rows`, `tile_start/freq`, `num_tiles_per_core`) | 80 | **335** | 19% |
| `Ht`/`Wt` (serve both roles) | 129 | 156 | 45% |

**Shape is static, count is dynamic** -- a deliberate convention, stated outright in
`ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`:

```cpp
template <uint32_t block_width_tiles, uint32_t input_dfb, uint32_t output_dfb, ...>
ALWI void tilize(uint32_t num_blocks, ...);   // shape in the template, count in the argument
```

The same split holds even in eltwise, the most runtime-dominated family: `num_tiles`
is `get_arg_val`, `num_tiles_per_cycle` is `constexpr get_compile_time_arg_val`.

**Rank 2 is enough.** Of 1294 kernels, 916 carry no shape quantity, 172 carry one, 142
carry two, and only 64 carry three or more -- all matmul-like, where the operands' own
shapes supply them. Nothing in the op set motivates general rank.

**What carrying it buys**, stated by ttnn's own most-optimised kernel.
`bmm_large_block_zm_fused_bias_activation.cpp` takes 19 compile-time args, five of
which are products it could derive, each with the formula in a comment beside it:

```cpp
constexpr uint32_t in0_block_num_tiles    = get_compile_time_arg_val(2);   // out_subblock_h*in0_block_w*in0_num_subblocks;
constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
```

There is not one `static_assert` in that file: five unchecked host-maintained
invariants with their derivations written down in prose.

### The type

`Shape` is taken by the multicast rectangle, so:

```cpp
template <uint32_t H, uint32_t W>
struct Tiles {
    static constexpr uint32_t h = H;
    static constexpr uint32_t w = W;
    static constexpr uint32_t num_pages = H * W;  // one tile per page in v1
};
```

`num_pages` stays the derived name, so the CB-facing vocabulary does not change.

### Both geometries become moot

`MatmulGeometry<RtDim, CtDim, KtDim, NumBlocks, In1RowStride, Transpose>`:

| parameter | fate |
|---|---|
| `RtDim`, `CtDim`, `KtDim` | inferred from operand shapes; a `Kt` disagreement becomes a compile error, which **nothing checks today** |
| `NumBlocks` | **deleted.** Used nowhere in the library -- only as a kernel loop bound, and the audit puts counts at 81% runtime |
| `In1RowStride` | **deleted.** Never passed non-default, and subsumed: the stride is `b`'s shape width |
| `Transpose` | the only residue |

`ReduceGeometry<Ht, Wt>` disappears outright: `out_tiles`, `elements`, `group` and
`contributor` are all pure functions of `(Ht, Wt, axis)`, so the operand's shape *is*
the geometry. `reduce_sum<RG, Axis>(a, sc)` becomes `reduce_sum<Axis>(a, sc)`.

### Shape of an expression

`tt/unified/expr.hpp` stays shape-agnostic -- it does not name ops and must not name
shapes either. Instead a metafunction in `math.hpp`, where both operand types are
already in hand:

```
node_shape<TileSource>      -> its own
node_shape<Un<Op, C>>       -> child's
node_shape<Bin<Op, L, R>>   -> lhs, with a static_assert that lhs == rhs
node_shape<MatmulNode>      -> Tiles<lhs.h, rhs.w>, static_assert lhs.w == rhs.h
node_shape<ReduceNode>      -> collapsed per axis
```

Today `Tiles<2,2> + Tiles<1,4>` is silently fine, because only the page count is
compared.

### Staging

The lever: **no stage changes a single emitted instruction**, so the selftest trace
must stay byte-identical throughout. That is a stronger net than any refactor here has
had, including bias -> bcast.

- [ ] **Stage 1 -- mechanical.** Add `Tiles<H,W>`; template `Storage<S>`, `Block<S>`,
      `ComputeBlock<S>`, `Accumulator<Mode,S>`, the four `NocAsync*Tx<thread,S>` types.
      Keep both geometries and every existing template argument. Add no checks. ~35
      signatures across `api.h` and `impl_v1.hpp`. Trace byte-identical.
- [x] **Stage 2 -- the checks.** `node_shape<>`; `store` conformance; strict eltwise
      shape equality; matmul operands against the geometry; `noc_core` fit-and-tile;
      scaler as `Shape<1,1>`; `.bias()` compile-time. Each proven by a violation that
      must fail to compile, plus a control for the legal gather. See *What Stage 2
      found* in unified_shape_outline.md -- the core-copy invariant turned out not to
      be equality.
- [x] **Stage 3 -- infer matmul.** Done. `MatmulGeometry` went from five `uint32_t`
      parameters to two SHAPES and is now derived, not declared -- no kernel writes it.
      `matmul<Geom>(a, b)` is `matmul(a, b)`; `matmul_init<Geom>` is
      `matmul_init<In0, In1>`; `NumBlocks` and `In1RowStride` are gone. Found a real
      hole doing it: `Accumulator` bypasses `store`, so its shape agreement was
      unchecked -- a `Shape<2,1>` accumulator on a `Shape<1,2>` matmul ran *correctly on
      device* because the page counts matched. Now a `static_assert`.
- [x] **Stage 4 -- infer reduce.** Done. `ReduceGeometry<Ht, Wt>` is now
      `ReduceGeometry<S>`, derived from the operand's shape -- every member was already
      a pure function of `(rows, cols, axis)`. `reduce_sum<RG, Axis>(a, sc)` is
      `reduce_sum<Axis>(a, sc)`, and `reduction_tree`'s `RT_REDUCE(Geom, x)` macro loses
      its geometry argument along with both `using PerCore/PerColumn` lines and the
      hand-derived `reduced_tiles_per_block`. A batched reduce is now an explicit
      `static_assert` rather than silently wrong: `reduce_shape` handles a leading
      extent correctly but `Strategy<ReduceFusion>` walks one 2-D grid.

### Risks

- `is_operand<ComputeBlock>` becomes a partial specialisation and `as_node` a template.
  These are the ADL hooks that produced the `Fluent` ordering trap once already, so
  expect that class of error.
- `Block`'s moved-from poison sets both `cb_id` and `num_pages` to `kMovedFrom`. Only
  `cb_id` can be poisoned now. The contract survives; one line goes.
- Error quality is the deliverable, not a side effect. Every mismatch must be a
  `static_assert` with a written message, not an overload-resolution failure. The
  existing DST-budget assert is the model to match.
- A dynamic extent is a foreseeable need -- 45% of `Ht`/`Wt` in ttnn are runtime. Not
  building it, but leave room for a `Dynamic` sentinel rather than precluding one.

### Open question

Spelling for the transpose residue once geometry is gone: `matmul<TransposeB>(a, b)`,
or a named `matmul_nt(a, b)` on the BLAS convention. Note an in0 transpose is NOT
symmetric with it -- ttnn does that with a separate materialised pass into an extra CB
(`transpose_init` into `cb_in0_transposed`), so B-transpose is a free flag while
A-transpose costs an op plus a buffer.

## Phase 6 -- `BcastFusion` (the one structural gap)

**DONE.** Spelled with the ordinary operators and an explicit marker on the operand being
broadcast:

```cpp
u::ComputeBlock m = m_storage.store(u::reduce_max<u::Axis::Cols>(x, one));
e_storage.store((x - u::bcast<u::Axis::Cols>(m)).exp());     // exp(x - rowmax), one pass
out_storage.store(e * u::bcast<u::Axis::Cols>(r));
```

- [x] `enum class Axis { Rows, Cols, Both }`, with `ReduceAxis` kept as an alias. One
      vocabulary: the axis a broadcast expands is the axis a reduction collapses.
- [x] `Broadcast<A, S>` marker + `bcast<A>(ComputeBlock)`; `is_operand` stays false for it
      so the SFPU operators cannot swallow one
- [x] `BcastNode<Op, A, SB, SV, Chain>` with `shape = SB`, so `store` conformance and
      `node_shape` need nothing new
- [x] `Strategy<BcastFusion>`: hoisted `reconfig_data_format` + `init_short`, then one DST
      slot per tile -- unlike `bias_finish`, which packs a whole block and is therefore
      capped at 8 tiles
- [x] Nine `(op, axis)` traits spelled out, because metal's init names are not uniform
- [x] `unified_kernels/bcast.cpp` + `test_unified_bcast.py`, 12 cases

### The axis is DECLARED, and that was the design correction

The first draft inferred the axis from the vector's shape. That is unsound: a `Shape`
counts TILES and the distinction lives inside one -- a single tile holding a row, a column,
or a lone value at `[0,0]` is `Shape<1,1>` in all three cases. Inference only appeared to
work because the three vector shapes differ when the block's tile extents differ; they
collide for `1x6`, `4x1` and `1x1`, and `Shape<1,N>` is exactly what a reduction produces.

So the axis is stated because it is information the shape lacks, and the shape is then
checked against it because that is information the axis lacks. Four guards, each proven by
a violation that must fail to compile: wrong vector shape for the axis, marker on the left,
an expression as the block operand, and a leading batch extent.

### The direction was measured, not read

Metal's `add_tiles_bcast` doc asserts both that B has "a filled 0-column" and that the
result is `C[h,w] = A[h,w] + B[w]` -- opposite claims. `test_unified_bcast.py` settles it:
`_bcast_cols` reads B as a COLUMN, so metal's "0-column" wording was right and its `B[w]`
was the typo. Swapping the rows/cols traits fails all six directional cases with errors of
0.59-1.53 while leaving `Both` untouched, which is what makes the test a measurement.

Two properties make it unable to pass wrongly: a non-square 2x3 block, so the two vector
shapes are different types; and vector entries that VARY along their length, so a wrong
direction cannot coincidentally agree.

### A broadcast followed by an SFPU op

The outline flagged this as the test that turns phase 4's "should" into "does", and it
earned its place. Dropping just `copy_tile_to_dst_init_short` from the SFPU leaf -- keeping
the format reconfig -- leaves all NINE plain broadcast cases passing and breaks only the
follow-on ones: `Rows` returns `err = 1.0` and `Cols` hangs the device outright. Nothing
else in the suite notices.

## Phase 7 -- Attention kernel + test

**DONE.** `unified_kernels/attention.cpp` computes one head of

    out = softmax(Q @ Kt / sqrt(d) + mask) @ V

on one core, non-flash, with a host-supplied additive causal mask. It composes the whole
set built in phases 1-6: a transposed matmul, a scalar broadcast, a two-buffer elementwise
add, a row max and row sum, a broadcast subtract with a fused exp, a reciprocal riding a
reduction's epilogue, a broadcast multiply, and a second matmul.

`test_unified_attention.py` passes 10 configurations against torch, max absolute error
0.0009-0.0040, and the causal identity holds EXACTLY.

### Two bugs, both real, both found by the test

**The second matmul had no block-dimension init.** `Strategy<FPUFusion>` emitted
`matmul_block_init` only from the restore paths; the normal path relied entirely on the
one-time `matmul_init` at kernel entry. That is fine for a kernel whose only compute is
matmul, and wrong the moment a broadcast, a reduction or an SFPU pass runs in between --
each reconfigures the unpack and math units for itself, so attention's second matmul ran
against another op's state. **Matmul was not composable with the other fusion kinds**, and
nothing before phase 7 had ever put them in one kernel.

Fixed in the strategy rather than the kernel: `run()` now programs its own block dimensions.
The first attempt was a kernel-level `matmul_init` before the second matmul, which HUNG the
device -- `compute_kernel_hw_startup` is MMIO plus a pack-sync init and must run exactly
once, as this model's own api.h comment says. Only the `matmul_block_init` half may repeat.

**The mask value was outside exp's domain.** `-1e4` -- the obvious choice, and what a
reference implementation would use with `-inf` -- gave a max error of 616. The SFPU's `exp`
has a finite input range and `exp(-1e4 - rowmax)` leaves it rather than underflowing to
zero. `-30` is ample: `exp(-30)` is 1e-13 against a row sum of at least 1.

### What the test gates on, and why not PCC

Softmax rows sum to 1 and every probability is O(1/S), so a global scale error or a per-row
offset correlates almost perfectly -- the blind spot that let a bias offset and a mean scale
factor through at 0.9999 earlier in this model. The checks that carry information:

  max absolute error vs torch    catches magnitude errors PCC cannot see
  out[0] == V[0] under a causal mask    an IDENTITY, not a tolerance: position 0 attends
                                       only to itself, so the first output row is V's first
                                       row exactly. It holds to 0.0000, and it depends on
                                       the mask, the softmax and both matmuls all being
                                       right at once.

### Bounded by the DST budget, not by L1

Both matmul output blocks must fit 8 tiles, so `Sq*Sk <= 8` and `Sq*D <= 8`. That caps a
single-shot head at 64x64 with a 2x2 tile score block. Larger S needs the output subblocked
-- which is also what would make phase 3's two unreachable restore sites live.

## Phase 8 -- RMSNorm

**DONE, with zero library changes** -- which was the prediction, and it held on the first
run. `unified_kernels/rmsnorm.cpp` is the phases 1-6 ops rearranged:

```cpp
u::ComputeBlock sq      = sq_storage.store(x * x);
u::ComputeBlock mean    = mean_storage.store(u::reduce_mean<kAxis>(sq, inv_n));
u::ComputeBlock inv_rms = rsqrt_storage.store((mean + u::bcast<u::Axis::Both>(eps)).rsqrt());
u::ComputeBlock normed  = normed_storage.store(x * u::bcast<u::Axis::Cols>(inv_rms));
out_storage.store(normed * u::bcast<u::Axis::Rows>(w));
```

Ten configurations against torch, max absolute error 0.023-0.033.

### What it adds over attention

**Both broadcast axes in one kernel.** Cols for the per-row reciprocal RMS, Rows for the
per-feature weight. Attention only ever broadcast along Cols.

**The case that needs the axis declared.** The epsilon is a scalar broadcast onto the
`Shape<Ht,1>` mean vector -- and against an `Ht x 1` block, both `Axis::Both` and
`Axis::Rows` require a `Shape<1,1>` vector, so no shape could tell them apart. This is the
collision from the design discussion, appearing in the first real kernel that needed it.

**`x * x` as an ordinary tree** whose two leaves are the same buffer.

### The check that carries the information

With `weight == 1`, every output row must have RMS 1. That is the op's definition rather
than a tolerance, and it is precisely what a scale error survives. Feeding `reduce_mean` a
scaler of 1 -- the classic mistake, turning a mean into a sum -- makes the row RMS collapse
to exactly `1/sqrt(N)`: measured 0.1765 at W=32 (1/sqrt(32) = 0.1768), 0.1245 at W=64,
0.0880 at W=128. The absolute-error check also fails, but only the RMS check names the bug.

### Settled along the way

`fill_reduce_scaler` IS a valid filler for a `bcast_scalar` vector. It writes the constant
into row 0 of all four faces where `_bcast_scalar` is documented to read only `[0,0]`, and
the outline flagged the difference as a blocking unknown. Two kernels now depend on it --
attention's `1/sqrt(d)` and this kernel's epsilon -- and both match torch, so no separate
`fill_scalar` is needed.

## Phase 9 -- RoPE

**DONE, with zero library changes** -- the second phase in a row where the prediction held.

    out = x * cos + (x @ M) * sin

M is one 32x32 tile with `M[2i][2i+1] = +1` and `M[2i+1][2i] = -1`, so `x @ M` sends each
adjacent pair `(x[2i], x[2i+1])` to `(-x[2i+1], x[2i])` -- the rotate-half as a matmul, which
is how ttnn does it. Six configurations, max absolute error 0.004-0.005.

### Why it fits the model unchanged

**The rotation is PER TILE**, because the pairing never crosses a 32-element boundary. A
block matmul expresses that exactly when `kt_dim == 1`: `out(rt x 1) = A(rt x 1) @ B(1 x 1)`
has no sum over k, so each output tile is one input tile times the single M tile. ttnn spells
the same thing as a `matmul_tiles` loop; here it is one `matmul(x, m)`.

**And because the op is per-tile, the block's 2-D shape is irrelevant.** A chunk of N tiles
is declared `Shape<N, 1>` whatever the sequence and head dimensions are. N is capped at 8 by
the matmul's DST budget, so the kernel walks the tensor in chunks -- and the chunk size being
a free parameter is what the test sweeps.

**The rest is one SFPU pass.** `x * cos + rot * sin` is a four-leaf tree needing three DST
slots -- the deepest expression this model has run, and the first to exercise phase 4's
per-leaf format reconfig across four distinct buffers.

It also alternates matmul and SFPU work every iteration, which is precisely what phase 7's
composability fix made legal.

### Two gates, and neither is sufficient alone

That is the finding worth keeping, established by sabotage rather than asserted:

| sabotage | max abs error | pair-norm deviation |
|---|---|---|
| sign-flipped M (device only) | **0.98 FAIL** | 0.004 pass |
| `cos` used where `sin` belongs | 0.70 FAIL | **0.61 FAIL** |

A rotation preserves the length of every `(x[2i], x[2i+1])` pair, so `pair_norm(out)` must
equal `pair_norm(x)`. That is a property of the op rather than of any reference -- but a
SIGN error is still a rotation, so it sails through. Conversely the error-versus-reference
check cannot catch a misunderstanding shared by the kernel and the reference, which is why
the reference applies the permutation directly and never multiplies by M. The two together
cover both; either alone has a blind spot.

## Phase 10 -- Flash chunking / online softmax

**DONE.** `unified_kernels/flash_attention.cpp` streams K and V in chunks so the score block
never exists in full. Ten checks pass, max absolute error 0.0023-0.0029 against torch, and
the answer is the same at 1, 2 and 4 chunks.

### The new idiom, and the one library type it needed

`Accumulator` carries a running TOTAL and nothing in it can rescale that total between steps,
which is exactly what an online softmax does when a chunk raises the maximum. So the state
lives in circular buffers across the loop, held by `RetainedBlock<S>` -- the obligation moves
into a slot rather than being discharged, so `~RetainedBlock` asserting empty still catches a
state that was pushed and never waited on. See unified_flash_outline.md.

One buffer per state variable, sized 2x the block: `release()` waits on the live value,
`store()` reserves the free half, and the pop lands at the end of the iteration, so the next
one finds the new value at the front. No parity bookkeeping.

`copy(block)` earned its place twice over -- seeding chunk 0's maximum, and copying the new
maximum from its scratch buffer into the state slot.

### The formulation, and a correction to the outline

Each chunk is normalised by its OWN row max and the difference folded into two corrections:

    rm = rowmax(s); p = exp(s - rm)
    m' = max(m, rm);  c_old = exp(m - m');  c_new = exp(rm - m')
    l' = l * c_old + rowsum(p * c_new)
    o' = o * c_old + (p * c_new) @ V

The outline claimed this means `m'` is "written as state and never read here". **That was
wrong** -- both corrections read it. Caught by review before the first run: reading it back
out of the state buffer would have taken the FRONT, which is the OLD value, and popped it. So
`m'` goes to a scratch buffer, both corrections read that, and `copy()` moves it into the
slot. What the formulation does buy is that the Sq x Sk exponential never needs it.

### The gate is chunk invariance, and the first version of it was vacuous

Matching torch is not enough: a single-chunk run never rescales anything, so it passes with
the correction machinery entirely broken. Every sabotage below leaves the 1-chunk row at
0.0024 and is caught only by the comparison BETWEEN chunk counts.

But invariance alone was not enough either, because the test data made the corrections
irrelevant. With uniform random Q and K every chunk's row maximum is nearly identical, so
`exp(m_old - m_new)` is already 1 -- and forcing it to 1 changed the answer by 0.005 against a
0.02 tolerance. **It passed.** The keys are now RAMPED along the sequence so later ones score
far higher and the maxima genuinely jump between chunks. The ramp is a function of position in
the full sequence, not of the chunk, so every chunk count still sees the same problem.

| sabotage | 1 chunk | invariance 1 vs 4 |
|---|---|---|
| `c_old` forced to 1 | 0.0024 ok | **0.201 FAIL** |
| `c_new` forced to 1 | 0.0024 ok | **0.095 FAIL** |
| corrections read `rm` instead of `m'` | 0.0024 ok | **0.040 FAIL** |

At the original 5x ramp the first of those measured 0.025 and only just tripped a 0.02
threshold; at 21x it is a 10x margin.

## Benchmarking

`unified_bench.py` measures DEVICE time using metal's real-time profiler, which streams a
`ProgramRealtimeRecord` per completed program over the existing dispatch path: start and end
timestamps plus the device frequency. Records carry `kernel_sources`, which is how a
measurement is attributed to the op that produced it.

That indirection is necessary rather than fancy. These kernels run in tens of microseconds and
host dispatch is tens of microseconds, so wall-clock timing measures the dispatcher. It also
confirms the model's structure from the outside: a unified kernel's record lists the same
source three times, once per descriptor.

`bench_attention.py` puts our kernels next to ttnn's SDPA at one shape. S=128, D=64, causal,
one head, ONE CORE each -- ttnn's `SDPAProgramConfig` takes the grid size, so pinning it to
`CoreCoord(1, 1)` is what makes the comparison mean anything. ttnn is also pinned to
`MathFidelity::HiFi2` with `math_approx_mode=True`, so the rows below compare at equal settings
rather than comparing our defaults against its cheaper ones:

| | device time |
|---|---|
| ours: flash, 2 chunks, HiFi4 exact (metal defaults) | 50.9 us |
| ours: flash, 2 chunks, HiFi2 approx | 36.5 us |
| ours: flash, 4 chunks, HiFi4 exact | 72.0 us |
| ours: flash, 4 chunks, HiFi2 approx | 53.2 us |
| ttnn SDPA, q32/k128, HiFi2 approx | 21.0 us |
| ttnn SDPA, q128/k128, HiFi2 approx | 19.9 us |

**ttnn is about 1.8x faster on one core at the same shape and the same fidelity.** That is down
from 3.9x, and the way it came down is worth more than the number.

### Where the time actually goes

The first diagnosis recorded here was that per-pass circular-buffer overhead dominates, and it
was wrong. `unified_kernels/passcost.cpp` exists to settle it: `out = in` through N identity
passes, each `copy` into its own scratch CB, so with the math pinned at zero the slope in N is
the cost of one L1 round trip plus its CB handshake. It is dead linear at **0.79 us per pass over 8 tiles**.

Quoting that as 0.099 us per tile, as an earlier revision of this file did, hides the
part that matters. Sweeping the WIDTH at a fixed pass count separates the two terms:

| block width | one pass | per tile |
|---|---|---|
| 1 tile | 0.289 us | 0.289 |
| 2 tiles | 0.367 us | 0.183 |
| 4 tiles | 0.510 us | 0.128 |
| 8 tiles | 0.793 us | 0.099 |

A pass is **~0.217us fixed plus ~0.072us per tile**, and the fixed half is charged
whatever the block. It is 27% of an 8-tile pass but 43% of the 4-tile vector passes that
the online-softmax state update is made of, so per-tile averages understate what a
narrow pass costs. Fusing narrow passes is worth more than fusing wide ones.

Against that baseline, running the same one-load-one-pass-one-store structure with real math
prices each SFPU op directly (8 tiles, one core, metal's default exact mode):

| op | total | math above the copy baseline |
|---|---|---|
| copy (zero math) | 3.03 us | -- |
| exp | 8.38 us | 0.665 us/tile |
| sqrt | 10.26 us | 0.904 us/tile |
| rsqrt | 11.29 us | 1.033 us/tile |
| recip | 12.56 us | 1.192 us/tile |

**The math outweighs the plumbing that carries it by roughly ten to one.** Fusing passes was
never going to close a 3x gap: every pass removed from the flash kernel is worth 0.8 us, and
there are not enough passes in it to matter. Two hypotheses died on this measurement:

- **Pass fusion** did work, and its effect was small and in the predicted direction. Folding the
  Q scale onto the host and normalising `p` straight to the new maximum took the chunk from 15
  passes to 12, for 81.4 -> 69.1 us.
- **DST batching was a regression and was reverted.** Each tile paid its own
  `tile_regs_acquire/commit/wait/release`, a cross-thread sync, where DST holds 8 tiles; batching
  the whole group into one acquire should have been free money. It made binary-chain 8-tile work
  27.9 -> 31.4 us and 4-chunk flash 112.6 -> 118.3 us. The handshake is not what costs, and the
  unrolled group code schedules worse than the simple loop.

### The one that mattered: math_approx_mode was silently dead

`ComputeConfigDescriptor` defaults to `MathFidelity::HiFi4` with `math_approx_mode = false` --
the most accurate and slowest settings metal offers -- and neither was reachable from our
harness. Threading them through was worth 3-5% for the fidelity half and almost nothing for
approx, which is the tell: metal takes `approx` as an explicit **template parameter** defaulting
to `false`, so `ckernel::exp_tile_init()` hardcoded exact mode and the config flag never reached
the SFPU at all. Passing metal's generated `APPROX` constant (see `ExpOp` in `math.hpp`) makes
approx exp **5.8x cheaper: 0.665 -> 0.115 us/tile**, which is what took flash to 54.3 us.

`sqrt_tile_init` already reads `APPROX` internally, and `recip`/`rsqrt` expose no such knob, so
exp is the only op with a flag to thread. It is also the one flash spends two passes per chunk on.

The accuracy cost is real and is the reason this is a knob and not a new default. Normalised max
absolute error on flash goes from 0.008 (HiFi4 exact) to 0.031 (HiFi2 approx), and approx exp
alone takes a single exp from 0.6% to 3.3% max relative error -- which is why
`test_unified_unary.py`, whose gate is 2% relative error, would fail under it. The default stays
exact; ttnn's SDPA runs at the cheap settings, so that is what the comparison uses.

### Every op is cheap, and the kernel is not

`passcost.cpp` prices the rest of the op set the same way, by baseline subtraction
against the zero-math copy control. `bcast` and `matmul` are shape-preserving, so they
chain and the slope method applies directly; `matmul` uses the IDENTITY as its second
operand, which makes the chain exact -- every product but one is a zero -- while the
FPU still does the full inner product, because the hardware does not shortcut a zero.
A reduction collapses and cannot be chained, so it is swept by shape instead: widening
`cols` at fixed `rows` prices one more input tile, and raising `rows` prices input and
output together, so the difference is the per-output-tile part alone.

| | measured | against a copy of the same tiles |
|---|---|---|
| copy (the control) | 0.097 us/tile/pass | -- |
| broadcast, as `bcast - copy` | 0.012 us/tile | +12% |
| matmul | 0.133 us/tile-MAC (0.267 us/output tile) | 2.7x |
| reduce, per input tile | 0.153 us | 1.6x |
| reduce, per output tile alone | 0.082 us | the acquire and pack |

Two things fall out. The per-output-tile cost of a reduction is 0.082us, so **batching
`Strategy<ReduceFusion>`'s one-acquire-per-output-tile could not win more than that**,
and the SFPU batching that already failed says even that is optimistic -- the open
question in the notes is answered, and the answer is not to bother. And a broadcast is
very nearly free, which retires the idea that the bcast-heavy softmax tail is where
flash spends itself.

**The sum of the parts does not come to the whole, and that is now the finding.** For
flash's 4-chunk shape the measured per-op costs add to about 9.4us per chunk, against a
measured marginal 20.4us -- the kernel costs 2.2x what everything in it costs. Four
candidates are now dead: SFPU math (measured, and the reason approx exp mattered), CB
plumbing (0.097us/tile/pass), per-tile acquire batching (tried, a regression), and
reconfiguration between passes. That last one is worth stating because it was the
obvious suspect: a homogeneous chain is the best case for the hardware, so PC_ALT
alternates broadcast and copy to make every pass change kind, and a broadcast leaves
the unpacker in a mode the next SFPU op has to restore. It costs **+0.005us per pass**.
Switching kinds is free.

### Double buffering works, and was missing where it mattered

Sweeping every CB between one block of pages and two is the sanity check for that, and
it turned up a live bug rather than confirming a working system: **flash shipped with
its streaming CBs single-buffered.** Only m, l and o carried 2x, and that is an
aliasing requirement -- `store()` reserves while the old value is still resident, so
halving THOSE deadlocks -- not a pipelining choice. Giving K, V and mask a second block
is worth a measured **5-6%** (54.3 -> 51.6us at 2 chunks, 95.0 -> 89.7us at 4) for L1
pages and nothing else, with the error unchanged, so it is now the default.

The same sweep shows no difference at all in unary or passcost, and that is the
expected answer rather than a broken knob: 8 blocks of 1 tile through an exact recip
comes to 11.73us either way, because the recip is ~9.5us of it and the DRAM reads are
small enough to hide behind that whatever the reader is allowed to do. A second block
only pays when there is latency to overlap.

### Still unaccounted for

Even with the fixed per-pass term above and the streaming fix, the parts do not sum to
the whole: flash's 2-chunk shape models at ~28us against a measured 51.6. Five
candidates are now dead with numbers attached -- SFPU math, CB plumbing, per-tile
acquire batching, reconfiguration between passes, and DRAM buffering. The honest state
is that roughly half of flash's time is not yet attributed to anything measured.

The one switch NOT yet priced is the one flash actually makes most: PC_ALT alternates
broadcast and copy, both SFPU-side, and found it free. It never tests matmul against
SFPU, which is a change of compute unit and carries a `matmul_block_init`. That is the
next measurement, and on current evidence it is the last cheap explanation left.

### Does the wrapper generate worse code?

Static analysis of the built ELFs, ours against ttnn's SDPA, both on the math thread
(trisc1) at matched HiFi2 + approx -- the config is recorded in each variant's generated
`chlkc_descriptors.h`, so the pair is matched rather than assumed. The TRISC firmware
contributes almost nothing to `.text` (a trivial passcost kernel is 812 bytes), so
these numbers are essentially kernel code.

| | ours (flash) | ttnn (sdpa) |
|---|---|---|
| `.text`, math thread | 13560 B | 5460 B |
| largest function | `kernel_main` 8688 B | `normalize_row_streaming` 2140 B |
| instructions (static) | 3390 | 1365 |
| Tensix + SFPU ops | 1510 | 655 |
| scalar | 1756 | 670 |
| **scalar per work op** | **1.16** | **1.02** |
| stack spills | 96 | 48 |
| non-inlined calls | 31 | 3 |

**More code, but not materially worse code.** 2.4x the text, and the answer to why is
structural rather than a codegen failure: our `kernel_main` is ONE 8688-byte inlined
function because every pass is written out and every tile loop unrolls -- the trip count
is a compile-time `Shape` constant -- while ttnn outlines its work into
`normalize_row_streaming` and two clones of `blocked_matmul_and_pack` and loops over
them. The quality number is the scalar-instructions-per-Tensix-op ratio, and at 1.16
against 1.02 the wrapper costs about 14% more scalar work per unit of real work. Spills
scale with size rather than outpacing it (96 against 48, for 2.3x the code).

**Code size is not the missing time.** passcost spans 812 to 5568 bytes of `.text` across
its variants, a 6.9x range, and its pass sweep is dead linear over exactly that range --
if instruction fetch cost anything material at these sizes, the 8-pass point would bend
upward. It does not, and the kernel size limit is 1432KB, so 8.7KB is not near any wall.

Two real findings, both small:

- `TileSource<Shape<4,1>>::emit` is a 640-byte function that did NOT inline, and is
  called 6 times. It is the leaf of the expression machinery, so it sits in the hot
  path; worth an `always_inline` and a re-measure.
- `Strategy<FPUFusion>::bias_finish` is emitted at 900 bytes with **zero call sites** in
  a kernel that uses no bias. Dead text rather than dead time, but it should not be
  there.

The caveat that keeps this from being a verdict on dynamic cost: static counts weight
every instruction once. Our straight-line body makes static ~ dynamic per chunk, while
ttnn's loops mean its static count understates what it actually issues. So the 2.4x size
gap emphatically does NOT mean we execute 2.4x the instructions -- if anything the
comparison flatters ttnn's dynamic count. **The wrapper is not where the missing 2x is.**

### always_inline on TileSource::emit: tried, reverted

The static analysis above suggested forcing the expression-tree leaf inline, since a
640-byte function reached from 6 call sites in the hot path looks like an obvious win.
It is not. Reproducibly, across a stash-and-repeat A/B:

| | without | with |
|---|---|---|
| flash, 2 chunks | 51.61 us | 52.82 us |
| flash, 4 chunks | 89.68 us | 89.50 us |

2.3% slower where it moved at all. The mechanism is visible in the ELF and is the
opposite of the intent: the 6 calls to `emit` do go away, but the two reconfig calls
that live inside its branch get replicated at every inlined site, so **total calls go
31 -> 43 and reconfig calls 12 -> 19**. `kernel_main` even shrinks slightly (8688 ->
8220 bytes), so this is not a size effect. GCC's estimate was right and the attribute is
reverted.

The useful part is what that proves: **the `reconfigure` branch does not fold.** The
hope was that inlining would let the caller's compile-time `Shape` bound turn
`kEveryTile || i == 0` into a constant per unrolled iteration and delete the reconfig
entirely. Instead the count went up, which means the branch survives as runtime work at
every leaf of every tile of every pass. That makes reconfiguration a target in its own
right rather than a side effect to be optimised away by inlining.

### Which thread is the bottleneck: the math TRISC

Answered directly rather than by inference. The build already has `ENABLE_TRACY=ON`, so
metal's device profiler gives per-RISC kernel spans, and `TT_METAL_PROFILER_SUM=1` adds
its own accumulating stall zones -- `CB-COMPUTE-WAIT-FRONT` on unpack and
`CB-COMPUTE-RESERVE-BACK` on pack. Profiling costs about 1% here (a 50.98us span against
51.63us unprofiled), so the numbers can be read at face value. flash, 2 chunks, HiFi2 +
approx, one core:

| thread | span | measured stall | slack |
|---|---|---|---|
| NCRISC (reader) | 9.02 us | -- | **idle for 42 us** |
| TRISC_0 (unpack) | 49.48 us | 21.17 us on `cb_wait_front` | 21 us |
| TRISC_1 (math) | **50.04 us** | **none measurable** | **none** |
| TRISC_2 (pack) | 49.74 us | 0.59 us on `cb_reserve_back` | 0.6 us |
| BRISC (writer) | 50.55 us | -- | waits at the end |

**The math thread is the constraint**, at 50.04us of a 50.98us kernel with no measurable
stall. The reasoning that separates it from pack, which also looks ~96% occupied, is a
zone placed on `tile_regs_acquire` -- where math blocks if pack has not released DST.
That came to **0.47us**. Math is not waiting for pack, so pack is not the constraint;
pack's apparent busyness is mostly `tile_regs_wait`, which is not a CB zone and so is
counted as busy. Unpack has 21us of slack and the reader has 42us.

Two hypotheses die here. **DRAM is not the bottleneck** -- the reader finishes its loads
in the first 9us and idles for the remaining 82% of the kernel, which also explains why
double buffering the streaming CBs was worth only 5%. And the DRAM prefetch idea recorded
earlier as "the leading candidate" is now dead; it was wrong, like the three before it.

**And the math thread is not busy with arithmetic.** The per-op costs model ~28us of real
work for this shape against ~50us of occupancy. Putting a sum zone on the reconfiguration
branch inside `TileSource::emit` prices it at **9.46us, stable to 0.01us across three
runs**, against **1.78us** for the `copy_tile` it guards -- reconfiguration costs over
five times the work it protects. Some of that is the zone's own overhead, which the
CPTILE number bounds at under 1.8us, so call it 8us net: **roughly 16% of the kernel
spent re-pointing the unpacker.** That is the same reconfiguration the always_inline
experiment proved cannot be folded away, and it is now measured rather than suspected.

Repeating this: `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1`, with
`DeviceZoneScopedSumN1/N2` around the region of interest and the define injected by
rebinding `unified_program` in the test module, so no test or library source is touched
for a commit. One trap worth knowing -- there are only **two sum slots per RISC**, and
metal's built-in CB zones occupy one on unpack and one on pack, so a custom zone can
silently land on an unexpected thread. Slot contention is why the first attempt attributed
a pack-side zone to unpack.

### Splitting the reconfiguration cost, and what it does to the plan

The leaf does two unconditional things, and they are not equally expensive. Zones on each,
with a calibration run where the same zones wrap NOTHING at the same invocation count, so
the subtraction removes the instrument rather than trusting it:

| on the math TRISC | measured | calibration | **net** |
|---|---|---|---|
| (a) `reconfig_data_format_srca` | 3.632 us | 0.375 us | **3.26 us** |
| (b) `copy_tile_to_dst_init_short` | 6.130 us | 0.224 us | **5.91 us** |
| total | | | **9.17 us** |

Cross-checks: the total agrees with the 9.46us measured earlier as one combined zone, and
the calibration confirms the zones cost 0.2-0.4us, so these are real. The same split on
the unpack thread puts (b) at 11.01us, but unpack has ~22us of slack so it does not gate.

**This inverts the recommendation.** (b) is 1.8x (a), and (b) -- reprogramming the
unpacker MOP and the math datacopy -- is the one metal offers **no conditional variant**
for. So:

- **Option 1 alone is now a minor win.** Metal's two-argument conditional reconfig
  addresses (a) only: 3.26 of 9.17us, about 36% of the reconfiguration cost and ~6% of
  the kernel. Worth having, not worth leading with. Note that
  `copy_tile_to_dst_init_short_with_dt` looks like the fix and is not -- it makes (a)
  conditional and still runs (b) every time.
- **Option 3 becomes the recommendation.** Leaf-outer batching -- reconfigure once per
  leaf, `copy_tile` all G tiles of that leaf, then apply the op -- cuts (a) AND (b)
  together by the group factor, with no format tracking and so no silently-wrong failure
  mode. For a 2-leaf 8-tile pass at G=4 that is 16 pairs down to 4, taking ~9.2us to
  ~2.3us. Bounded by `G * leaf_count <= kMaxDstTiles`.
- **Option 2 is only for what survives.** Skipping (b) on unchanged formats is the sole
  route to zero, and it still rests on the unproven claim that same format plus uniform
  tile geometry means an identical MOP. Reach for it after 3, if the remainder justifies
  the risk.

Option 3 sits next to the DST batching that already regressed, and the difference is worth
stating: that attempt kept whole-tree-per-tile order, so it never reduced reconfiguration
at all -- it only moved the acquire. Leaf-outer changes what is actually expensive here.

### Leaf-outer emission: reconfiguration once per leaf per group

Implemented, and it is the first structural win in this file. The interleaved walk in
`expr.hpp` re-points the unpacker once per leaf per TILE, because a Bin loads its left
leaf, applies, then loads its right. Turning the loops inside out -- load every tile of
leaf 0, then every tile of leaf 1, and only then apply the ops -- pays that once per leaf
per GROUP.

| | before | after |
|---|---|---|
| flash, 2 chunks, HiFi2 approx | 51.60 us | **46.56 us** (-9.8%) |
| flash, 4 chunks, HiFi2 approx | 89.72 us | **80.20 us** (-10.6%) |
| flash, 2 chunks, HiFi4 exact | 66.55 us | 60.41 us |
| `reconfig_data_format_srca`, math thread | 3.632 us | **1.573 us** |
| `copy_tile_to_dst_init_short`, math thread | 6.130 us | **4.330 us** |
| `copy_tile_to_dst_init_short`, unpack thread | 11.21 us | **1.68 us** |

Error is unchanged to four decimals at both chunk counts, and the profiler confirms the
mechanism rather than just the wall clock: reconfiguration on the bottleneck thread nearly
halved, 9.76us to 5.90us. Worth noting the two halves did not fall proportionally even
though they fire together -- the calls that survive are the genuine format changes at each
leaf's first tile, while the ones removed were the cheap repeats, so the average survivor
costs more.

The cost is DST slots. `Emit` REUSES them, folding a Bin into its left operand, so a
leaf's slot is overwritten before a later leaf is read -- which is exactly why leaf-outer
cannot share that allocation and needs one slot per leaf. The layout that works is leaf
`j` of tile `k` at `k*leaf_count + j`, with a subtree folding into its LEFTMOST leaf's
slot; subtrees own disjoint leaf ranges, so folding left can never land on a slot another
subtree still needs.

**When it is NOT taken, and why that is not tuning.** A group of one reconfigures exactly
as often as the interleaved walk, so a tree wide enough to leave no room for a second tile
gains nothing and pays for the longer path. Measured: the five-leaf binary chain
(`((a+b)-a)*b/a`, which gives `G = 8/5 = 1`) came out **5% slower**, 27.84 -> 29.32us.
Gating on `leaf_count * 2 <= kMaxDstTiles` put it back at 27.85us exactly. Single-leaf
trees keep the interleaved path too -- they already reconfigure once per pass, so batching
them could only re-introduce the acquire penalty that the earlier DST experiment measured.
Wide trees that will not fit at all also fall back, so no expression that compiled before
stops compiling.

This is the same territory as the DST batching that regressed, and the difference is now
demonstrated rather than argued: that attempt kept whole-tree-per-tile order and never
reduced reconfiguration, only moving the acquire. Reordering the loops is what mattered.

The gates are not vacuous, which took a sabotage to establish. Dropping the reconfigure
entirely fails `test_unified_mixed_format.py` and correctly PASSES
`test_unified_binary.py` -- uniform formats genuinely do not need it, so only the
mixed-format test can see this class of bug. A wrong stride, where different tiles' leaves
collide, fails both binary and flash.

Remaining on this thread: ~5.9us of reconfiguration still on the math TRISC, of which the
MOP init is the larger part and has no conditional form. Metal's two-argument conditional
reconfig (option 1 in the earlier proposal) is still available for what is left of the
cheaper half.

### Conditional reconfiguration (option 1): tried, does not pay, reverted

The other half of the plan was metal's comparing form, `reconfig_data_format_srca(old,
new)`, which re-points srcA only if the two formats actually differ. That needs the leaf
to know what the hardware is currently pointed at, so `bool reconfigure` became a
threaded `uint32_t& prev_cb` carrying three cases: unknown (re-point outright), our own
buffer (do nothing at all), or another buffer (compare and skip if they match). `prev_cb`
resets once per PASS, never per group, since srcA survives a tile_regs cycle and only
another strategy having run makes the state opaque.

It works and it is slower where it matters:

| | leaf-outer only | + conditional | + conditional, decision hoisted |
|---|---|---|---|
| flash, 2 chunks | **46.57 us** | 48.79 us | 49.75 us |
| flash, 4 chunks | **80.16 us** | 79.52 us | 85.24 us |
| binary chain, 5 leaves | 27.85 us | 27.20 us | **26.93 us** |

The split is consistent and explains itself. Leaf-outer has already cut the
reconfiguration count to once per leaf per group, so there are only a handful of calls
left to make conditional -- and the comparing form is not free: it reads four entries out
of the unpack format tables before deciding. Paying that to skip a few re-points is a net
loss. The interleaved path still reconfigures once per leaf per TILE, forty times in a
five-leaf eight-tile pass, and there the same trade wins 2-3%.

Hoisting the decision out of the tile loop -- splitting the leaf into `prepare` and
`load` so the check runs once per leaf rather than once per leaf-tile -- was the obvious
repair and made flash worse still, 49.75us. That result is not explained, and rather than
narrate a theory the honest summary is: the mechanism was measured three ways, two of
them regress the target kernel, and it is reverted.

What would make it worth revisiting is the shape it helps: many leaves over many tiles,
where the interleaved fallback runs. No kernel in llama prefill has that shape, so
carrying two leaf protocols to win 3% on an expression we do not ship is not a trade
worth making today.

The remaining ~5.9us of reconfiguration on the math TRISC is now mostly the MOP init,
which has no comparing form at all. That leaves option 2 -- skipping the init when the
format is unchanged -- as the only route further, and it still rests on an unproven claim
about MOP equivalence. It should be attacked, if at all, with the mixed-format sabotage
already shown to catch exactly this class of bug.

### Where the bottleneck thread's time goes, by strategy

With the math TRISC established as the constraint, the useful question is which strategy
owns its time. A sum zone on each `Strategy<...>::run`, two per run because there are only
two slots per RISC. flash, 2 chunks, HiFi2 + approx, math thread span 45.85us:

| strategy | math thread | share |
|---|---|---|
| `SFPUFusion` | **25.79 us** | **56%** |
| `FPUFusion` (matmul) | 9.53 us | 21% |
| `BcastFusion` | 5.90 us | 13% |
| `ReduceFusion` | 3.46 us | 8% |
| accounted | 44.68 us | **97%** |

97% of the bottleneck thread is accounted for, which is the first time anything in this
file has closed. The FPU/SFPU pair was measured twice in separate runs and agreed to
0.03us. Unpack and pack now show ~23us of stall each -- leaf-outer moved the balance, and
they have slack to spare.

**SFPU is the heavy hitter, and it is not the arithmetic.** Of its 25.79us,
reconfiguration is 5.9us and the per-pass fixed cost about 2.6us; the rest is leaf loads
and packing. flash runs six SFPU passes per chunk and, counting tiles times leaves, about
64 `copy_tile`s per chunk -- every one of them a tile fetched out of L1 because the
previous pass put it there.

That reframes the remaining list. `matmul_init` was a suspect worth a measurement, but the
whole FPU strategy is 9.53us, so even eliminating its setup entirely cannot be the 2x.
Reduce and bcast are 9.4us combined and were already measured cheap per op. The lever is
the SFPU pass count and the leaf count within each pass:

- `m_state = copy(m_now)` is a pure copy pass that exists only because the running maximum
  needs a scratch buffer -- 4 tiles of pure overhead per chunk. Alternating two CBs for the
  state would remove the pass outright.
- `l = l_prev * c_old + rs` is three leaves over 4 tiles, twelve `copy_tile`s for four
  output tiles.
- `sm = s + mask` is the largest single SFPU pass: 8 tiles, 2 leaves, 16 `copy_tile`s. It
  exists only to add the mask to the matmul's result, and the matmul already has an
  epilogue mechanism -- `Strategy<FPUFusion>::bias_finish` -- that applies a resident
  operand to DST before packing. A full-block add is not the same shape as a broadcast
  bias, but the place to put it exists. Folding it in removes a whole pass AND the L1 round
  trip of `s`.

That last one is the heaviest single item on the list and is the same fusion ttnn appears
to do -- its `normalize_row_streaming` name suggests the softmax normalisation happens in
one streaming pass over DST rather than the six this kernel spends.

### Why #2 and #1 both need a decision first

**#2, removing `m_state = copy(m_now)`, needs a new ownership state.** `ComputeBlock` is
constructed from a `Block`, has every copy and move deleted, and pops in its destructor, so
a block is either read this iteration or retained for the next -- never both. `m_now` needs
both: it is read twice (by `c_old` and by `p`) and must survive as next chunk's `m_prev`.
Alternating two state CBs does not help, because the pop is what loses it, not the aliasing.
Restructuring the algebra to avoid reading `m_now` is worse, not better: `c_old` can be had
from `m_prev` and `rm` alone as `exp(-relu(rm - m_prev))`, but `p` then needs
`exp(sm - bcast(m_prev)) * bcast(c_old)`, which is two broadcast passes where there was one.
So #2 costs a "read but not consumed" operation in the core API -- the same class of
decision `RetainedBlock` was -- to save one 4-tile single-leaf pass per chunk, about 1.0us
or 2%.

**#1, folding `sm = s + mask` into the matmul, does not work the way the list assumed.**
`bias_finish` is not a DST epilogue. It packs the matmul total to `acc_cb`, then takes a
fresh `tile_regs_acquire` and reads it back out of L1 to add the bias. Routing the mask
through it would buy exactly nothing -- it is the same L1 round trip the separate SFPU pass
already pays.

A true DST epilogue, adding the mask while the product is still in DST, is blocked by
capacity: flash's scores block is 4x2 = 8 tiles, which is the entire half-sync DST budget,
so there is no slot left to unpack a mask tile into. Subblocking the matmul to 4 tiles would
free room at the cost of doubling matmul invocations and their `matmul_block_init`s.

The route that does work is the **packer's L1 accumulate**, which is already live and tested
(`MM_ACC_L1` in `test_unified_matmul.py`, used by `matmul.cpp` and `matmul_mcast.cpp` for
partials). `pack_reconfig_l1_acc(1)` makes `pack_block` ADD into the destination rather than
overwrite it. So if the reader seeds the scores CB with the mask instead of a CB of its own,
the matmul's own pack lands on top of it and `s + mask` costs nothing at all: no extra pass,
no extra DST pressure, and one fewer L1 round trip for `s`.

Measured payoff: `sm = s + mask` is 16 of the ~64 `copy_tile`s per chunk, so roughly 5us of
the 25.79us SFPU budget -- about 11% of the kernel, the largest single item left.

Two things to settle before building it. The push/pop dance the L1 path already documents
is load-bearing -- `pack_block` advances `fifo_wr_tile_ptr` and only `cb_push_back` resets
it -- so seeding means the reader pushes the mask and the compute side pops it, without
reading, purely to wind the pointers back. And the numerics move: today `s + mask` is summed
in DST and packed once, where L1 accumulate packs `s` to bfloat16 first and adds it to a
bfloat16 mask, so the error gate has to be re-checked rather than assumed.

### FPU vs SFPU for add, sub and mul: the FPU wins everywhere

add, sub and mul exist on both units, and the two forms differ in a way that matters
given where the time actually goes. The SFPU form takes two DST slots, so every operand
needs a `copy_tile` to get there; the FPU form reads both operands straight out of
circular buffers and needs no copy at all. Since the SFPU budget is dominated by leaf
copies rather than arithmetic, that is the right thing to attack.

Priced through the real code path -- `Strategy<FpuEltwiseFusion>` behind an explicit
`u::fpu_add/sub/mul`, not a hand-rolled approximation -- as the slope in tiles at one
pass, so the program floor cancels:

| op | SFPU us/tile | FPU us/tile | ratio |
|---|---|---|---|
| add | 0.532 | 0.311 | **0.58x** |
| sub | 0.524 | 0.314 | 0.60x |
| **mul** | **1.123** | **0.332** | **0.30x** |

Identical at HiFi2 + approx to within noise, and every check gates on exact equality, so
the two units agree bit for bit on inputs where nothing rounds.

**The caution recorded before this measurement was wrong in an instructive way.** The
worry was that FPU eltwise runs at MATH_FIDELITY, so `mul_tiles` might take four passes
at our HiFi4 default and lose to the SFPU. The opposite holds: the SFPU multiply is the
most expensive op in the set at 1.123us/tile -- more than twice the SFPU add -- and the
FPU does it in 0.332us at either fidelity. So mul is the BIGGEST winner rather than the
exception, no fidelity-dependent predicate is needed, and the "measure before building"
step earned its keep for the third time today.

What this implies for flash. Of six SFPU passes per chunk, four are FPU-fusable: `s +
mask` (8 tiles), `os + pv` (8), `l_prev*c_old + rs` (4, mul then add), and
`(m_prev - m_now).exp()` (4, sub with an SFPU epilogue). At 0.21us/tile saved for add and
sub and 0.79us for mul, that is roughly 4-5us per chunk, so **9-10us of a 46.6us kernel,
about 20%** -- the largest single item identified anywhere in this file.

Also worth recording: `eltwise_binary.h` was simply never included by
`adaptor_v1.hpp`, which had only the `_sfpu` variant. The FPU forms of the three ops the
model already supports were not reachable at all.

### Trees choose their own unit

Implemented. `expr::kind_of` inspects an expression tree at store() and sends it to the
FPU when it can, the SFPU otherwise, so a kernel writing `a + b` gets whichever unit can
do it and nothing in any kernel changed.

| config | 2 chunks | 4 chunks | flash error |
|---|---|---|---|
| SFPU only (`TT_UNIFIED_NO_FPU_ELTWISE`) | 46.55 us | 80.16 us | 0.0312 / 0.0271 |
| FPU add/sub only | 41.81 (-10%) | 64.75 (-19%) | 0.0312 / 0.0271 |
| **FPU add/sub/mul (default)** | **36.47** (-22%) | **53.21** (-34%) | 0.0312 / 0.0271 |

The predicate: every binary op must have an FPU form, AND every binary node must have at
least one LEAF child. Two non-leaf children would put two operands in DST and no
instruction takes that, so left-deep chains qualify and `(a+b)-(c+a)` does not. A unary
must wrap a non-leaf -- on a bare leaf it would need a copy_tile, the thing being
avoided. Emission is OP-OUTER over the group, one init per op rather than per tile, the
same lesson leaf-outer already taught on the SFPU side. DST holds one slot per output
tile whatever the tree's size, since operands never occupy it.

**Two things this turned up that measurement alone would not have.**

A short init is not enough. `add_tiles_init` programs the math unit and the unpackers for
an operand pair but NOT the hardware data formats, which came from `compute_init` at
kernel entry for one specific pair. A mixed-format pair therefore read garbage --
`test_unified_mixed_format` went to **inf** error, not to a slightly worse number. The
fix is `reconfig_data_format(cb0, cb1)` before the seed and the single-sided form before
a chain link, which is what `bias_finish` already does for the same reason. The full init
that would have covered it, `binary_op_init_common`, carries hw_configure and
pack_sync_init and must not run twice -- the same split matmul needs.

And the FPU is not uniformly better. Per-op max relative error, measured:

| op | FPU | SFPU | |
|---|---|---|---|
| add | 0.00389 | 0.00583 | FPU more accurate AND faster |
| sub | 0.00389 | 0.00389 | equal, and faster |
| **mul** | **0.01023** | **0.00380** | **2.7x worse**, 3.4x faster |

add and sub are free wins and need no knob. mul is a real accuracy-for-speed trade, and
the FPU is now the **default** for it too: the op-level cost does not propagate, since
flash's error is unchanged to four decimals either way -- that kernel's error comes from
approx exp and the bfloat16 chain, not from one multiply -- while the time is worth
12-16% of the kernel. `-DTT_UNIFIED_SFPU_MUL` takes the accurate form back.

Loosening `test_unified_binary`'s mul limit from 0.01 to 0.015 to accommodate that would
have been the weak move on its own, so the test grew instead. It now re-runs add, sub and
mul with the FPU **disabled** and holds that path to the original 0.01, so a regression in
the accurate implementation cannot hide behind the FPU's allowance. And it asserts the
ORDER -- that the FPU multiply is the less accurate of the two -- which is what notices
dispatch silently changing rather than just drifting. Forcing mul back to the SFPU makes
that check fail, which is how it was confirmed to bite.

Coverage came with it. The harness had only two-leaf adds and unaries on bare leaves, so
the predicate and the chaining would have shipped untested; `example_fpu_eltwise` now
walks one shape per rule -- chain with DST left, chain with DST right (the direction that
decides whether a subtraction comes out backwards), FPU seed with an SFPU epilogue, the
op rule falling back on `max_`, and the SHAPE rule falling back on `(a+b)-(c+a)` where
every op does have an FPU form.

### Real model shapes, with the PV matmul banded

`bench_models.py` sweeps the smallest published config of each family. Per HEAD on ONE
core, causal prefill measured AS triangular: the kernel's mask handles S_q < S_k, so one
launch is one q-chunk of a prefill and a head is the SUM over q-chunks, chunk i attending
to (i+1)*S_q keys. Both sides at HiFi2 + approx, which the script enforces.

| model | d | S | sq | ours/head | ttnn/head | ratio |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 64 | 512 | 8 | 189.9 us | 122.6 us | 1.55x |
| Qwen2.5 0.5B | 64 | 512 | 8 | 189.8 us | 122.6 us | 1.55x |
| TinyLlama 1.1B | 64 | 512 | 8 | 189.9 us | 122.6 us | 1.55x |
| Phi-3 mini 3.8B | 96 | 512 | 8 | 213.0 us | 132.1 us | 1.61x |
| Qwen3 0.6B | 128 | 512 | 4 | 235.4 us | 142.8 us | 1.65x |
| Llama 3.2 3B | 128 | 512 | 4 | 235.5 us | 142.8 us | 1.65x |
| Gemma 3 1B | 256 | 512 | 4 | 322.7 us | 200.1 us | 1.61x |

**Banding the single-shot matmul is what moved these**, from 2.5-3.1x to 1.5-1.65x. An
output block wider than the 8-tile DST budget is now walked in row bands, which lifts TWO
limits at once, and the second one turned out to matter more than the first:

- `sq * dt <= 8`, the PV matmul's block, which pinned a 256-wide head to sq=1: 32 query
  rows per launch, 32 launches for a 1024-long prefill.
- `sq * sk <= 8`, the scores matmul's block, which capped the KEY chunk. This is the one
  that dominated. At sq=8 the old rule allowed sk=1, so the kernel ran one-tile key chunks
  and paid a pass per tile.

Measured on Llama 3.2 1B at S=512, holding everything else equal:

| | total |
|---|---|
| sq=8, sk<=8 -- both caps lifted | **189.8 us** |
| sq=4, sk<=2 -- what the old rule allowed | 314.1 us |
| sq=8, sk<=1 -- one cap lifted, the other left in place | 469.8 us |

That middle-to-bottom row is worth keeping: raising sq while leaving the sk cap alone is
WORSE than not raising it. A first pass at this sweep did exactly that and produced
"bigger sq hurts, because causal work grows with the q-chunk". The causal-work argument is
true -- a prefill in chunks of Q computes S*(S+Q)/2 tiles of scores -- but it was not what
the numbers were showing. They were showing a cap I had left in the benchmark.

sq is now bounded by L1 rather than DST, since q, o, pv and scores all scale with it, so
the sweep probes the ceiling instead of deriving it: 8 for 64- and 96-wide heads, 6 for
128, 4 for 256. It then SEARCHES sq within that, because the causal-work effect is real
even if it was not the story here -- for 128-wide heads sq=4 beats the sq=6 ceiling.

**The gap no longer widens with sequence length**, which the pre-banding numbers did
(2.30x to 3.58x on Gemma from S=128 to S=1024):

| | S=256 | S=512 | S=1024 |
|---|---|---|---|
| Llama 3.2 1B (d=64) | 1.52x | 1.55x | 1.44x |
| Gemma 3 1B (d=256) | 1.63x | 1.61x | 1.53x |

That growth was the sk cap biting harder as sequences got longer, not a structural
disadvantage. It is flat now, and slightly improving.

Heads are independent, so the script reports a layer as grid ROUNDS: at 32 heads on 64
cores that is one round, making the per-head number the per-layer number -- 190us for
Llama 3.2 1B at S=512, 235us for 3B.

Still missing for a real layer: GQA head mapping, multi-core partitioning, and a q-loop,
so a prefill is N launches where ttnn's is one. All phase 11.

### Is the multi-launch structure making the comparison unfair?

Fair question, since a causal head is N launches for us and one program for ttnn. Measured
rather than assumed: hold the key-chunk size at sk=8 and vary how many chunks one launch
does, and the slope is the marginal cost of a chunk while the intercept is what a launch
pays regardless.

| | marginal per k-chunk | per-launch fixed |
|---|---|---|
| d=64, sq=8 | 51.5 us | 17.4 us |
| d=256, sq=4 | 48.3 us | 18.8 us |

17-19us per launch looks like it would matter -- 2 launches for Llama at S=512, 4 for
Gemma. But almost none of it is amortizable, because of WHAT it is:

- kernel startup, bounded by the whole-program floor of ~1.6us that passcost measures;
- the Q load, which is per q-chunk by definition -- each chunk has its own queries;
- the softmax tail, likewise: `recip(l)` on sq tiles then `o * bcast(recip)` on sq*dt,
  and each q-chunk has its own output.

Only the startup goes away in a fused q-loop, so the honest figure is about 2us times
(N-1): **1% for Llama at S=512, 2% for Gemma.** The 1.5x is a real compute gap, not a
dispatch artifact.

One thing that measurement DID surface: the tail's `recip` is **9.53us at sq=8**, which is
a directly measured number (8 tiles of recip against the copy baseline) and 5% of Llama's
190us. recip is SFPU-only at 1.19us/tile and, unlike exp, exposes no approximation
parameter, so there is no cheap version of it to switch to.

What this does NOT establish: the real dispatch gap between launches. Timing the launches
together gives a span 3.0ms wide for two launches, but that is the test harness rebuilding
tensors and the program descriptor on every call, not the pipeline. Isolating it would need
the programs built once and enqueued back to back, which is phase 11's shape anyway. The
sum-of-device-times used throughout is the right basis for comparing compute, and it is
what both sides are measured with.

### Are the two flash attentions functionally equivalent? No.

The core algorithm is the same, and that is the part the ratio compares: Q@K^T, mask,
online softmax with a running max and sum and the rescaling that goes with them, P@V, a
final normalise. Same structure, same numerics.

Everything around it differs. From `sdpa.cpp`'s compile-time arguments and
`compute_common.hpp`, ttnn's op carries:

| | ttnn SDPA | ours |
|---|---|---|
| GQA (`NQH` vs `NKH`) | yes | one head, no mapping |
| batch and head loops | in the kernel | none |
| q-chunk and k-chunk loops | both in one program | k-chunk only; q-chunk is a launch |
| causal | internal, with `causal_k_limit` skipping fully-masked k-chunks | host picks each q-chunk's key extent |
| the mask | GENERATED on device: `apply_causal_mask_lightweight` stamps a neginf tile and a diagonal tile through L1 accumulate, and rows below the diagonal get nothing | a full S_q x S_k tensor materialised in DRAM and loaded per chunk |
| user-provided mask, padded mask | both | mask is the only input |
| chunked prefill / KV-cache continuation | `is_chunked`, `chunk_start_idx` | no |
| sliding window | `sliding_window_size` | no |
| attention sinks | `use_attention_sink` | no |
| multi-core | `num_cores`, `core_id`, zigzag load balancing | single core |
| subblocking | explicit qk and out subblock params | derived from the shape |
| the scale | folded into exp via `exp_tile_init<true, scale_fp32, ...>` | pre-applied to Q on the host |
| V head dim != K head dim | `vDHt` separate from `DHt` | assumed equal |
| second implementation | `use_streaming_compute` path | one |

So ours is the core of a flash attention and ttnn's is a production attention op. The 1.5x
is measured on the intersection, which is the fair thing to compare, but "1.5x off ttnn"
should not be read as "1.5x off a like-for-like kernel".

**Does the difference flatter us?** The one place the work genuinely differs is the mask,
and it runs against us: ttnn stamps a band and skips below-diagonal rows entirely, where we
load a scores-sized tensor and add it over the whole block. Measured, though, that costs
only about **1.65us per k-chunk** -- roughly 3% -- because on the thread that binds, the
math TRISC, an FPU add is one instruction per tile. The reader has slack too (20.8us busy
of a 120.7us span), so the mask's DRAM traffic is free as well. The functional gap is real;
it is not what the performance gap is made of.

Worth recording alongside it: the per-op us/tile rates measured with passcost are
WHOLE-PIPELINE throughput, taken from kernels where the unpacker was often the limit. They
are the right numbers for comparing ops against each other, and the wrong ones for
attributing time on a critical path that a different thread owns -- which is why removing
the mask add saved 1.65us where those rates predicted 15.

### What to test next

Ordered by expected value, with what each result would actually mean. Six candidates are
already dead (SFPU math, CB plumbing, acquire batching, SFPU-side kind switching, DRAM
buffering, wrapper codegen), and roughly half of flash's time is still unattributed.

1. ~~**Which thread is on the critical path.**~~ **Answered above: the math TRISC**, at
   98% occupancy with no measurable stall, while unpack has 21us of slack and the reader
   idles for 42us. This re-ranks everything below it: work that shortens the math
   thread's critical path counts, and work that helps unpack, pack or the reader does
   not. It also promotes item 3 to the top, since reconfiguration is now measured at
   ~8us of math-thread time.

2. **matmul against SFPU, the switch flash actually makes.** PC_ALT alternates broadcast
   and copy, both SFPU-side, and found switching free -- that result does NOT cover a
   change of compute unit. Add a PC_ALT variant alternating matmul and copy. Related and
   possibly the same finding: `llk_math_matmul_init` is a **1372-byte outlined function
   with 4 call sites** in a kernel with 2 matmuls per chunk, so the question is whether
   it is being re-run per matmul rather than hoisted. If it is, that is a large fixed
   cost on every FPU pass and the fix is in `Strategy<FPUFusion>`.

3. **Reconfiguration per leaf.** `kEveryTile` is `leaf_count > 1`, so a 2-leaf expression
   reconfigures on EVERY tile while a 1-leaf reconfigures once. That is a controlled
   experiment already latent in the design: compare a 1-leaf copy pass against a 2-leaf
   add of two blocks at equal tile counts, and the difference is the per-tile reconfig
   cost. Most of flash's passes are multi-leaf, so if this is expensive it is expensive
   twelve times per chunk. The always_inline result above says the branch is real work.

4. **Push granularity.** `store()` does one `cb_reserve_back(num_tiles)` and one
   `cb_push_back(num_tiles)` around the whole block, so a downstream consumer cannot
   start until the entire block is packed. Pushing per tile would let the next stage's
   unpack overlap the current stage's pack -- these are different threads, so there is
   real concurrency to win. Test by pushing per tile in one strategy and measuring.

5. **Are we doing more Tensix work than ttnn at all?** The gap may be algorithmic rather
   than overhead, and nothing here has checked. ttnn's `normalize_row_streaming` name
   suggests it fuses the whole softmax normalisation into one streaming pass over DST,
   where we spend 12 separate L1 round trips. Counting dynamic Tensix ops for the same
   shape on both sides answers whether we are slower at the same work or doing more of
   it -- and those call for completely different fixes.

6. **CB count and L1 pressure.** flash declares 19 CBs. Whether the count itself costs
   anything (config space, reserve/push bookkeeping) is untested and cheap to check with
   a kernel that does fixed work through varying numbers of buffers. Low expected value,
   listed because it is nearly free to rule out.

7. **A hand-written metal kernel for the same math.** The definitive test of whether the
   model's structure -- not its codegen, which the static analysis clears -- is what
   costs. Expensive to write, and worth doing only if 1 through 5 leave the gap
   unexplained, at which point it stops being optional.

### The full grid was never the interesting number

ttnn on 64 cores is 13.8 us against 21 us on one, because at S=128 with a single head there are
only a handful of q-chunks of work and most cores idle. Any per-core figure derived from the full
grid is fiction, which is why the table pins both sides to one core.

Not measured, and worth stating: ttnn is solving the general problem -- batches, heads, GQA,
arbitrary sequence lengths -- while these kernels do one head at one shape, and the shapes our
DST budget allows are far below where ttnn is designed to operate.

## Phase 11 -- Full block orchestration

Host-side and kernel-loop work, not model gaps.

- [ ] GQA head mapping (n_heads != n_kv_heads)
- [ ] Multi-core work partitioning across heads
- [ ] Head concat + output projection wired to the attention core
- [ ] End-to-end single-layer prefill against a reference

---

## Open questions to settle before phase 6

- ~~Does `BcastGeometry` reuse `ReduceGeometry`, or does the pairing deserve its own
  type so a mismatched reduce/bcast pair fails to compile?~~ **Answered by phase 5.**
  `ReduceGeometry` is deleted; `reduce<Cols>` on `Tiles<Ht,Wt>` yields `Tiles<Ht,1>`
  and `bcast_cols` demands exactly that, so the pairing checks itself. This is the
  reason phase 5 comes before phase 6 rather than after.
- Should the scale fold into the softmax scaler the way ttnn's SDPA does
  (`compute_common.hpp:1713` notes it gets scaling "for free" inside `exp`), rather
  than costing a separate bcast-scalar pass?
