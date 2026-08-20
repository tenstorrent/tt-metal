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
| ours: flash, 2 chunks, HiFi4 exact (metal defaults) | 69.0 us |
| ours: flash, 2 chunks, HiFi2 approx | 54.3 us |
| ours: flash, 4 chunks, HiFi4 exact | 112.7 us |
| ours: flash, 4 chunks, HiFi2 approx | 95.0 us |
| ttnn SDPA, q32/k128, HiFi2 approx | 21.0 us |
| ttnn SDPA, q128/k128, HiFi2 approx | 19.9 us |

**ttnn is about 2.6x faster on one core at the same shape and the same fidelity.** That is down
from 3.9x, and the way it came down is worth more than the number.

### Where the time actually goes

The first diagnosis recorded here was that per-pass circular-buffer overhead dominates, and it
was wrong. `unified_kernels/passcost.cpp` exists to settle it: `out = in` through N identity
passes, each `copy` into its own scratch CB, so with the math pinned at zero the slope in N is
the cost of one L1 round trip plus its CB handshake. It is dead linear at **0.79 us per pass over
8 tiles, or 0.099 us per tile per pass**.

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

What passcost structurally does NOT have is DRAM traffic: it loads one block and
everything after that is L1-resident, so its slope can never include a load. Flash
reads K, V and mask per chunk and calls `.wait()` on each immediately. At 4 chunks that
is 8 tiles of DRAM per chunk against an 11us shortfall, or ~1.4us per tile, which is
the right order for an unhidden DRAM round trip. That makes prefetch the next thing to
try -- the CBs are already double-buffered, but issuing chunk j+1's loads before
computing chunk j is what would let the latency overlap. **Not yet measured, and stated
as the leading candidate rather than a conclusion** -- the last three confident
diagnoses in this file were all wrong, which is why each now comes with a number.

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
