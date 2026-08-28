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

- [x] `tt/unified/adaptor.hpp` -- include `eltwise_unary/recip.h`, `sqrt.h`, `rsqrt.h`
- [x] `tt/unified/math.hpp` -- three op structs + free-function templates over expr nodes
- [x] `tt/unified/expr.hpp` -- `fluent_recip` / `fluent_rsqrt` / `fluent_sqrt` hooks
- [x] `tt/unified/api.h` -- `ComputeBlock` overload declarations
- [x] `tt/unified/impl.hpp` -- definitions
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
  `impl.hpp` work at all: `operator+` is generic over `as_node`, and a binary is
  not a method, so it never touches the `Fluent` mixin. Three sites each, all in
  `math.hpp`. **This settles the macro question deferred from phase 1: no.** The
  duplication did not grow the way the unaries' did.
- **`tt/unified/impl.hpp` and `api.h` were untouched by this phase**, which the
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
      signatures across `api.h` and `impl.hpp`. Trace byte-identical.
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
`adaptor.hpp`, which had only the `_sfpu` variant. The FPU forms of the three ops the
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
| Llama 3.2 1B | 64 | 512 | 8 | 178.5 us | 122.5 us | **1.46x** |
| Qwen2.5 0.5B | 64 | 512 | 8 | 178.5 us | 122.5 us | 1.46x |
| TinyLlama 1.1B | 64 | 512 | 8 | 178.5 us | 122.5 us | 1.46x |
| Phi-3 mini 3.8B | 96 | 512 | 8 | 201.5 us | 132.2 us | 1.52x |
| Qwen3 0.6B | 128 | 512 | 8 | 223.2 us | 142.8 us | 1.56x |
| Llama 3.2 3B | 128 | 512 | 8 | 223.2 us | 142.8 us | 1.56x |
| Gemma 3 1B | 256 | 512 | 4 | 313.2 us | 200.1 us | 1.57x |

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

### Where the remaining 1.5x is, measured at the shape the sweep uses

The old budget (SFPU 56%) is stale twice over: it predates the FPU eltwise dispatch and it
was taken at sq=4/sk=2. Re-measured with the current library at sq=8/sk=8, 2 chunks, on the
math TRISC, which is still the constraint (117 of 119us, no stalls; unpack has 29us of
slack, the reader 100us):

| | math thread | share |
|---|---|---|
| matmul (banded) | 33.8 us | ~26% |
| **reduce** | **33.8 us** | **~26%** |
| broadcast | 27.9 us | ~21% |
| FPU eltwise | 16.8 us | ~13% |
| SFPU tree | 16.6 us | ~13% |

Shares are approximate, and the reason is worse than instrument overhead. A sum zone around
a strategy measures how long that REGION lasts on the thread it is read from, not how much
work that thread did in it -- and the regions OVERLAP, because the threads are pipelined:
while math is inside strategy N+1, pack is still finishing N. So the columns sum past the
span (here ~110%, and a later re-measure reached 127us against a 115us span), and a zone's
absolute value is not comparable across runs whose pipeline balance differs. The same zone
reading ~51us on BOTH math and pack is the giveaway. Use these to rank, never to subtract.
For a number to trust, ablate: change one thing and measure the whole kernel, which is how
the mask fold got its 1.65us.

**There is no dominant item left**, which is what 1.5x looks like from the inside. The old
56% concentration is gone -- the FPU dispatch moved eltwise work off the SFPU, and the
larger score block moved weight onto reduce and broadcast. Reduce went 8% -> 26%, because
its cost scales with the INPUT block: 64 score tiles fold to 8, so 64 `reduce_tile` calls,
twice per chunk (max and sum).

Ranked by measured size and by how much is known rather than hoped:

1. **Row-sum as a matmul, not a reduce (~13% available).** ttnn does this and the evidence
   is in its kernel: `matmul_reduce<Sq_chunk_t>(cb_col_identity, ...)`, a matmul against a
   column of ones, with `N = 1  // Result of reduce is 1 column`. A row sum IS a matvec, so
   `reduce_sum<Cols>` can be `matmul(p, ones)` -- 8 MACs per output tile against 64
   `reduce_tile` calls. It only works for the SUM; `reduce_max` has no matmul form and stays.
   That halves the reduce line.
2. **Fold the mask into the matmul (a whole 64-tile pass).** The packer's L1 accumulate
   makes `pack_block` add into the destination, so seeding the scores buffer with the mask
   has the matmul land on top of it and `s + mask` disappears -- no pass, no DST pressure,
   one fewer L1 round trip for `s`. Note the earlier 1.65us measurement does NOT bound this:
   that replaced the add with a copy, which is still a pass. This removes the pass.
3. **Defer `o / l` into the output projection (~8%).** The tail's `recip` is 9.53us at
   sq=8, SFPU-only with no approximation parameter, so it cannot be made cheaper in place.
   But `diag(1/l) @ (o @ W_o) == (o/l) @ W_o`, so the normalisation commutes with the
   output projection and can move there. That takes it out of attention entirely rather than
   optimising it. Layer-level, so it belongs with phase 11.
4. **A binary chain on broadcast.** `os = o_prev * bcast(c_old)` then `o = os + pv` is two
   passes over the output block; as one expression it is one. Same shape of gap as the FPU
   eltwise chaining, in `BcastFusion`.
5. **The state copy**, worth ~2%, still blocked on the read-but-not-consumed ownership
   question.

And the structural one behind all of it: a pass per `store()` means every intermediate is
packed to L1 and unpacked back. Items 2 and 4 are individual instances; the general fix is
a way to produce more than one result from one acquire, which is a real API question rather
than an optimisation.

Not worth revisiting, all measured: eltwise math (done, -22%/-34%), reconfiguration (halved
by leaf-outer, ~5% left), acquire batching (a regression, twice), dispatch overhead (~1-2%),
and the mask's DRAM traffic (the reader has 100us of slack).

### Row sum as a matmul, and the mask folded into the product

Both from the list, both implemented, and only one of them was worth anything -- which is
the useful part.

| | d=64 sq=8 | d=256 sq=4 |
|---|---|---|
| before | 189.8 us | 322.7 us |
| row sum as a matmul | 179.2 (**-5.6%**) | 315.2 (-2.3%) |
| mask folded in as well | 178.6 (-0.3%) | 313.3 (-0.6%) |

**Row sum as a matmul needed no library change at all.** `matmul(p, col_ones)` with
`col_ones : Shape<sk, 1>` already produces `Shape<sq, 1>`, exactly what
`reduce_sum<Cols>` produced, so it is a kernel edit. The ones sit in COLUMN 0 only, which
reproduces the reduction's contract -- sum in column 0, zeros elsewhere -- where all-ones
would have put the sum in every column. Nothing downstream reads those columns, but
`bcast<Cols>` taking column 0 is a contract not worth quietly changing. The operand is
built on the host rather than by a filler: a column of ones has to land in column 0 of each
of a tile's four 16x16 faces, and getting that packing wrong by one half-word is a silently
wrong sum. `reduce_max` stays a reduction; a maximum has no matmul form.

**The mask fold is a wash on time, and the reason corrects an earlier claim in this file.**
The list said folding it "removes a whole 64-tile pass", which is true, and predicted a win,
which was wrong. `add` replaces the add pass's FPU add with a dest-reuse add: the MATH
thread issues one instruction per output tile either way. What actually disappears is 64
packs and 128 unpacks -- and pack and unpack have slack, while math is the constraint. So
**removing L1 round trips does not help while math is the bottleneck**, and the general
"fewer passes" framing recorded earlier is wrong for this machine. What matters is
math-thread work, which is why the row sum (fewer tile-ops) paid and the mask fold did not.

It is kept anyway, for L1 rather than time: the scores buffer is gone, 64 pages at sq=8/sk=8,
and sq is L1-bound. That lifted 128-wide heads from sq=6 to **sq=8** and is most of why
Qwen3 0.6B and Llama 3.2 3B moved 235.4 -> 223.2us.

`add` is a genuine addition to the matmul node, distinct from `bias` -- a whole block of the
output's shape rather than one row broadcast down it -- and applies in both the plain and
banded paths, before the epilogue chain so `matmul(a,b).plus(m).relu()` is `relu(A@B + m)`.
Two things it has to do that the trace pins: put `matmul_block_init` back afterwards, since
the reuse op reprograms the math unit and a later band would otherwise run a matmul against
eltwise state; and carry `addend_cb` through `bias()`, or `.add(m).bias(v)` would drop the
addend and look like a plain biased matmul rather than an error.

### Deferred: merging bias into add

`plus` is now `add`. Merging `bias` into it -- writing a bias as
`matmul(a, b).add(bcast<Axis::Rows>(bias))` -- is specced but not done. The spelling works:
`reduce_shape<out_shape, Axis::Rows>` IS `Shape<1, ct_dim>`, which is exactly the shape
`bias()` demands, so the type system already agrees that a Rows-broadcast of that shape
belongs to this output block.

What stops it being cosmetic is that the two forms cannot cost the same. Metal's
`add_tiles_bcast_rows` reads BOTH operands from circular buffers and writes DST; there is
no DST-operand form. So a block addend is in place and free (one FPU op per tile), while a
broadcast addend has to stay `bias_finish`: pack the product out, re-acquire, read it back,
broadcast-add, pack again. One API, two costs about 2x apart. The LLK does template
`src_b_bcast_type` and `binary_reuse_dest` independently, so an in-place broadcast add is
plausible -- but metal exposes no wrapper for the combination and this library has only ever
called public `ckernel::` entry points, so that is a separate spike.

The real argument for the merge is that `bias_cb` and `addend_cb` are runtime fields every
node-construction site must remember to copy, and sites keep forgetting:

- the five unary chain builders dropped `addend_cb`, so `matmul(q, k).add(mask).relu()`
  silently produced `relu(A@B)`. **Fixed here**, and `example_matmul_add` now covers the
  chained form: reverting one builder takes the trace from 10 add_reuse lines to 5.
- `bias()` dropped `addend_cb` -- fixed when `add` landed.
- `run_banded` ignores `bias_cb` entirely, so a single-shot `store(matmul(a, b).bias(v))`
  with a >8-tile output silently dropped the bias. **Fixed**, and without touching
  `MatmulNode`'s signature: `bias()` knows `geometry::out_subblock_num_tiles` at compile
  time, so it static_asserts there. The error lands at the call site instead of the shape
  being mishandled downstream, and nothing is lost -- the accumulating path already refused
  the same shape, so this only extends the refusal to the single-shot case. A compile-only
  probe confirms both halves: the 4x4 bias fails with the intended message, and the same
  16-tile output WITHOUT a bias still compiles and bands.

Three instances of one bug class, all silent, all now closed. Two needed the field threaded
through by hand; the third turned out to be expressible as a static_assert because the
geometry is already in the type. The remaining argument for making the addend part of
`MatmulNode`'s TYPE is that it would have prevented the first two rather than requiring them
to be found.

### Where the perf exploration stands

Selftest traces are current -- byte-identical across all three projections. The per-strategy
BUDGET is not, and re-measuring it produced a number that says more about the method than
about the kernel: `reduce` came out at 51us having had HALF its work removed (the sums are
matmuls now), and the same zone read 52us on the pack thread. Strategy regions overlap
across a pipelined set of threads, so the zones cannot be summed or compared across runs.
They rank; they do not attribute. Ablation is the method that gives a number.

Two lessons now constrain what is worth trying, both learned the expensive way:

- **Only math-thread work counts.** Math is the constraint; unpack and pack have slack. So
  removing L1 round trips does nothing, which is why the mask fold was a wash.
- **Swapping one math op for another is therefore neutral.** `add` replaced an FPU add with
  a dest-reuse add: same instruction per tile. This retires the broadcast-chain item
  (`o_prev * bcast(c) + pv` as one pass) before it is written -- 16 bcast-muls plus 16 adds
  either way.

What is left, ranked by how much math work it actually removes:

1. ~~**The wasted half of the diagonal chunk.**~~ **Measured, and it is gated behind the
   q-loop -- see below.**
2. **Defer `o / l` into the output projection (~8%).** `recip` is 9.53us at sq=8, SFPU-only
   with no approximation parameter. `diag(1/l) @ (o @ W_o) == (o/l) @ W_o`, so it can leave
   attention entirely rather than be optimised. Layer-level, so it lands with phase 11.
3. **`reduce_max`** is the last large reduction, 128 `reduce_tile` calls for a 2-chunk pair,
   and has no matmul form. Dropping the running maximum altogether is what would remove it,
   and that is a numerics change flash exists to avoid -- ttnn tracks it too.
4. **The state copy**, ~2%, still waiting on the read-but-not-consumed ownership question.
5. **GQA and multi-core** -- throughput rather than per-core time, and the only items that
   make a real model runnable. Phase 11.

### The diagonal waste is real, and cannot be taken without the q-loop

Priced first, which is what made the answer clear. Widening the key extent at fixed sq adds
sq score tiles per step, so the slope is what a score tile costs with everything that
touches it -- matmul, fused mask, exp, reduce_max, the row-sum matmul:

**0.685 us per score tile.** At S=512 with sq=8 the kernel computes 192 score tiles where
the causal triangle needs 136, so **56 wasted tiles = 38.3 us of a 178 us head, 22%**. The
largest single item left, confirmed by measurement rather than by counting tiles.

Then the surprise. Finer K chunks cannot help -- the waste is masked ROWS inside a chunk,
not whole chunks, so any rectangular tiling of the same key range computes the same 192.
Finer Q chunks can, and do, and it makes things worse:

| | q-chunks | k-chunks | score tiles (136 needed) | total |
|---|---|---|---|---|
| sq=8 | 2 | 3 | 192 | **178.5 us** |
| sq=4 | 4 | 6 | 160 | 182.8 us |
| sq=2 | 8 | 12 | 144 | 222.4 us |

sq=2 computes almost exactly the triangle and is 25% SLOWER. Subtracting the score work at
the measured price leaves the non-score cost, and it fits `13.1us fixed per q-chunk +
1.30us per sq tile` (predicts 15.7us at sq=2 against 15.5 measured). The fixed 13.1us is
per-LAUNCH: kernel startup, program setup, the fixed part of the tail. Eight q-chunks pay
it eight times, which swamps the 38us of waste they avoid.

**So the diagonal waste is not a separate item -- it is the q-loop's payoff.** With the
per-q-chunk fixed cost down to kernel startup alone (~2us), the same three plans model as:

| | with a q-loop |
|---|---|
| sq=2 | ~135 us |
| sq=4 | ~138 us |
| sq=8 | ~156 us |

against 178.5us today: **roughly 24%**, and it inverts which sq wins. That is the whole
diagonal waste plus the launch overhead, from one change that phase 11 needs anyway and
that makes the dispatch structure match ttnn's.

The alternative -- ragged computation inside a rectangular block -- was considered and is
worth less. The banded matmul already walks one output row at a time and `ct_dim` is a
RUNTIME argument to `matmul_block`, so band r could compute only its needed columns; DST
would be seeded with the mask row first so skipped tiles carry -inf and `exp` sends them to
zero, and the full sk tiles still get packed so the block stays rectangular. But it saves
only the MATMUL share of a score tile -- about 0.27us of the 0.685 -- because exp, reduce
and the row-sum matmul still run on all sk tiles. Call it 9% of the kernel for a change to
the banded path, against 24% for the q-loop.

### The q-loop: built, correct, and the prediction was wrong

One launch is now one head -- a query-chunk loop around the key-chunk loop, with the
causal walk in the kernel so query chunk i visits only `k_offset + (i+1)*sq` key tiles and
never touches a chunk wholly above the diagonal. Verified against the FULL S x S reference
rather than one query chunk, which is a stronger check than the old shape allowed, and
`num_q=1` reproduces the old behaviour exactly.

The predicted ~24% did not appear:

| | per-q-chunk launches | q-loop |
|---|---|---|
| sq=8 | 178.5 us | **172.7 us** (-3.3%) |
| sq=4 | 182.8 us | 197.4 us (WORSE) |
| sq=2 | 222.4 us | 338.2 us (much worse) |

The prediction also said the optimum would invert to sq=2. It did the opposite, and the
reason is that the fixed cost was attributed to the wrong thing. Holding sq and the key
range constant -- so the score-tile count is identical at 192 -- and varying only the chunk
width:

| | k-chunks | score tiles | total |
|---|---|---|---|
| sk=8 | 3 | 192 | 172.7 us |
| sk=4 | 6 | 192 | 204.4 us |
| sk=2 | 12 | 192 | 281.0 us |

**12.0 us of fixed cost per K-CHUNK**, with score work held exactly constant. The earlier
fit of "13.1us per q-chunk" matched its three points just as well, because in those plans
the two counts moved together; the q-loop is what separated them, and it separated them
the wrong way for the hypothesis. Causally the k-chunk count grows as n(n+1)/2 while
q-chunks grow as n, so finer query chunks multiply the dominant cost quadratically.

**So the diagonal waste is not reachable by chunking in either dimension.** That is now
measured twice, and the second time with the launch overhead removed, which was the only
remaining excuse. The waste is 22% and sk is already at its ceiling (the banded matmul
needs one row band to fit DST, so sk <= 8).

What the q-loop is still worth keeping for: 3.3% at the best shape, and it removes the
"N launches versus one program" asymmetry that has qualified every ttnn comparison here.
GQA and multi-core both need it.

**The new target is that 12.0 us.** At sq=8/sk=8 with 3 k-chunks it is ~36us of 172.7 --
21% -- and it is fixed per chunk regardless of block size, so it is setup, not arithmetic:
per-pass fixed costs (~0.217us x ~10 passes) plus the strategy inits, of which the biggest
suspects are the four matmul_block_inits a chunk now runs and the per-band init the fused
mask forces after every addend.

### Matmul swept against ttnn: two holes, and neither is arithmetic

`bench_matmul.py` sweeps output rows and columns in tiles, the inner dimension, the number
of k-blocks accumulated over, and how the running total is carried -- the axes of ttnn's
own single-core matmul microbenchmark (`test_moreh_microbenchmark.py` -> `test_compute_mm`).
Both sides pinned to one core and HiFi2. 160 configs.

It asks the library whether a shape is expressible by TRYING it, and classifies the failure
from the library's own assert text, rather than re-deriving the rule in Python where it
would go stale and then lie in the direction that matters.

**Where we are fine.** Every shape that fits takes 1.02x to 1.28x, and the single-shot
banded path is FASTER than ttnn at several wide ones -- 8x8 at kt=2 is 0.90x, 4x8 is 0.92x,
2x8 is 0.96x. There is no shape-dependent cliff.

**Hole 1, functional: a large output block cannot be ACCUMULATED.** FIXED -- see
"Subblocking the output" below, which also lifted ct > 8. Kept here for the shape of the
finding. `rt*ct > 8` is refused
on both accumulating modes, which is 48 of the 160 cells. Single-shot covers all sixteen
rt x ct combinations via row banding, so the gap is specific: big output block AND several
k-blocks together. A real matmul with a long K and a wide output has to split the output
today.

**Hole 2, rate: each k-block costs about 1.3us, and ttnn's k-steps cost nearly nothing.**
RETRACTED -- this was self-inflicted, and the sweep's own axes are what hid it. The sweep
varied k_blocks and only ever at kt in {2, 8}, so it never asked what ONE call with a large
kt costs. It costs nothing extra: kt is not a DST dimension. Nothing here is wrong about
what was measured, but "ttnn's k-steps cost nearly nothing" and ours cost 1.3us is a
comparison between ttnn doing the arithmetic in one call and us choosing to split it. The
sweep now sweeps kt instead, and the retraction is worked through under "The k-loop was the
mistake" below. Left in place because the reasoning that followed from it is instructive. K = 8 tiles as ONE block against the same K as FOUR blocks, at 1x1 output:

| | ours | ttnn | ratio |
|---|---|---|---|
| kb=1, kt=8 | 3.36 us | 2.96 us | 1.13x |
| kb=4, kt=2 | 7.32 us | 2.96 us | **2.47x** |

Identical work. ttnn is identical too -- 2.96us either way, as it should be, since the
blocking is our construct and not its. We pay 3.96us for three extra k-blocks, so ~1.3us
each, and every rate hole the sweep flagged is a kb=4 kt=2 cell for exactly this reason.

That is the same shape of cost the flash work kept hitting: a fixed price per chunk that is
setup rather than arithmetic. Here it is isolated to `Accumulator::accumulate`, whose
per-call work is a `matmul_block_init`, the reload or the L1-accumulate push/pop dance, and
the reconfigurations around them. Isolating it in a matmul with no softmax around it makes
it much easier to attack than it was in flash.

Two caveats on the harness. The per-MAC filter flags small shapes at 15x the best per-MAC
rate, and that is not a hole -- a 1x1 kt=2 matmul is two tile-multiplies against a fixed
program cost, and no implementation makes that efficient. And a few ttnn references are
missing (the `-` cells): its one-core matmul declines some shapes, which is recorded rather
than worked around.

### Diagnosing the 1.3us k-block, and what fixing it would take

Measured first, and it killed the obvious hypothesis. With total K held at 8 tiles and only
the blocking changed, the cost is dead linear in the number of `accumulate` calls:

| | calls | ours |
|---|---|---|
| kb=1, kt=8 | 1 | 3.35 us |
| kb=2, kt=4 | 2 | 4.89 us |
| kb=4, kt=2 | 4 | 7.33 us |
| kb=8, kt=1 | 8 | 11.80 us |

**1.21 us per call.** Three things then locate it:

- **It is not `matmul_block_init`.** Ablating the per-call init entirely -- the 1372-byte
  function the static analysis flagged, which looked like the answer -- moved kb=8 from
  11.88 to 11.58us. That is 0.04us per call, 3% of it. The suspicion was wrong.
- **It is not the data.** Per-call cost against output block size is 1.32us at one tile,
  1.42 at four, 1.50 at eight -- essentially flat with a ~0.025us/tile slope. A cost that
  does not scale with the block is synchronisation, not movement.
- **It is almost exactly a whole matmul PASS.** passcost prices a standalone single-shot
  matmul pass at 1.066us. So an `accumulate` call in a k-loop costs what an independent
  matmul costs: the loop buys nothing.

That last one is the diagnosis. Every call is a self-contained pass -- `tile_regs_acquire`,
matmul, pack, `cb_push_back`, then `cb_wait_front` and `cb_pop_front` on the buffer it just
wrote, then release. The wait is the serialisation: the math thread blocks until the packer
has landed the partial before the next k-block may start, so consecutive k-blocks cannot
overlap. Both modes pay it, which is why Dst and L1 measured within 0.01x of each other --
Dst through its reload, L1 through the push/pop that rewinds the write pointer. The comment
already says that push/pop pair is load-bearing for correctness, and it is; it is also what
costs.

**The fix proposed here was to hold the partial in DST across calls, and it was the wrong
fix.** `kb=1, kt=8` is the fast case, which is the right observation; the wrong conclusion
was that a blocked K has to be made to behave like an unblocked one. It can just BE
unblocked -- `kt_dim` has no DST limit. See "The k-loop was the mistake" below, which
supersedes the rest of this section.

Expected impact, taking the per-call cost down to the matmul issue itself (~0.05us):

| | now | fixed | vs ttnn now | vs ttnn fixed |
|---|---|---|---|---|
| kb=4, 1x1 | 7.33 us | ~3.5 us | 2.47x | ~1.2x |
| kb=8, 1x1 | 11.80 us | ~3.5 us | ~4x | ~1.2x |

**That prediction was wrong. The section below has the built result and the real cause.**

**Where this matters is not attention.** Flash's matmuls are single-shot, so it is untouched
by this. What it touches is every projection and FFN matmul, where K is the hidden size --
2048 elements is 64 tiles, far past what one block holds, so those matmuls are necessarily
blocked and pay this on every step. That is most of a transformer's arithmetic.

**What it costs to do.** Today each `accumulate` call is deliberately self-contained; the
strategy's own comment explains that it reprograms the block dimensions rather than trusting
earlier state, because a broadcast or reduction between calls would otherwise leave the
units configured for something else. Holding DST across the loop gives that up: the
Accumulator would own the register file from first call to finish, and any other op
interleaved into the loop would corrupt the partial silently. So the change is not the
arithmetic, it is the ownership -- the acquire and release move into the Accumulator, and the
"nothing else may touch DST here" rule has to be expressed rather than assumed. It also does
NOT lift hole 1: the partial living in DST still needs rt*ct <= 8.

### The k-loop was the mistake, not the accumulator

I built `AccumulatorMode::DstResident` to hold the partial in DST across k-blocks, measured
14% instead of the ~4x predicted, and have reverted it. The abstraction was answering a
question that should not have been asked, and the reason is worth recording because it
invalidates the "fix" proposed in the section above.

**`kt_dim` is not a DST dimension.** DST budgets the OUTPUT block -- `rt_dim * ct_dim`, which
is what the strategy's `static_assert` checks. K never occupies a register: `matmul_block`
walks its k-loop internally and accumulates every step into the same `idst` slots. So there
is no DST reason to block K at all, and `MatmulGeometry` has never had a limit on `kt_dim`.
Measured, one accumulate call, 1x1 output:

| kt | K in elements | PCC | ours |
|---|---|---|---|
| 8 | 256 | 0.999969 | 3.40 us |
| 32 | 1024 | 0.999843 | 8.20 us |
| 64 | 2048 | 0.999575 | 14.50 us |
| 128 | 4096 | 0.998927 | 27.16 us |
| 256 | 8192 | 0.997457 | 52.49 us |

K=8192 in a single call, correct. The PCC drift is ordinary bf16 accumulation error over more
terms, not a limit being hit.

At the shape that actually matters -- K=64 tiles, which is llama's 2048 hidden, 2x2 output --
blocking is pure loss:

| | ours | PCC |
|---|---|---|
| kb=1, kt=64 | **29.99 us** | 0.999587 |
| kb=2, kt=32 | 31.31 us | 0.999623 |
| kb=4, kt=16 | 33.85 us | 0.999616 |
| kb=8, kt=8 | 39.31 us | 0.999617 |
| kb=16, kt=4 | 50.45 us | 0.999621 |

**40% for nothing, 1.36us per block.** DstResident recovered 0.33us of that per block; not
blocking recovers all of it, needs no new mode, no DST ownership rule, and no guard on every
acquire site in the library. So the k-loop is not the general case that projections and FFNs
have to live with -- it is a streaming device for when K genuinely does not fit in L1, and
the operand blocks are `rt*kt` and `kt*ct` tiles, so at 2KB a tile a 64-tile K at 2x2 is
256KB an operand. That fits. Nothing in a single-core transformer layer needs the k-loop.

What survives from the DstResident measurement, because it is about movement and not the
accumulator: with the operand loads ablated out of the k-loop, kb=8 went from 10.10 to
2.72us, so **~1.05us of each blocked step is the per-block DRAM read of the operands** --
read latency, not bandwidth, since it is the same total bytes either way. That is the real
reason fine blocking hurts, and if a K ever is too large for L1, prefetch depth (issue
several blocks' reads before barriering on the first) is the lever, not the accumulator
mode. The ablation was a six-line `#if` in the kernel that reused block 0's operands for
every step -- a deliberately wrong answer, kept only long enough to take the measurement,
and removed with the revert.

Two lessons, and the second is the one I keep relearning. First: check whether the
constraint you are designing around is real -- I never verified that a large `kt` was a
problem before building machinery to avoid needing one. Second, for the fourth time this
session: I priced the mechanism I had just been reading about without checking what else was
in the same measurement.

### Re-swept on kt: no rate holes left, and parity with ttnn at size

With kt as the axis and k_blocks pinned to 1, `bench_matmul.py` covers 240 cells across
rt, ct in {1,2,4,8}, kt in {1,2,8,32,64} and all three carry modes (I first wrote 332 here,
which was a miscount -- the grep that produced it also counted the per-MAC report's lines). The result is much
duller than the previous sweep, which is the point -- the interesting holes it used to
report were ones we were making ourselves.

**64 holes, and 60 of them are the one real limit:** `rt*ct > 8` refused on the two
accumulating modes. Single-shot covers those shapes by row banding, so the gap remains
exactly hole 1 above. Of the remaining four, three are new and correct: at kt=64 the
operands are rt*kt and kt*ct tiles, so 4x8, 8x4 and 8x8 exhaust L1 -- 512 tiles an operand
is 1MB, and there are two of them. That is the ceiling that actually bounds kt, it is a
chip limit rather than a library one, and the sweep now names it. The last is one transient
"no profiler records".

**Zero rate holes.** Nothing is above 2x; the worst cell anywhere is 1.41x. And the gap
closes as the work grows:

| kt=64 | ours | ttnn | ratio |
|---|---|---|---|
| 1x1 | 14.51 us | 10.64 us | 1.36x |
| 1x2 | 21.19 us | 17.16 us | 1.23x |
| 1x4 | 34.67 us | 30.75 us | 1.13x |
| 1x8 | 61.61 us | 61.12 us | **1.01x** |
| 8x1 | 62.98 us | 60.92 us | 1.03x |

So there is no shape-dependent cliff and no arithmetic deficit: at a real projection's worth
of work we are at parity, and the 1.2-1.4x at smaller shapes is the fixed per-pass cost the
flash work has been chasing all along, now visible without a softmax around it. A 1x1 kt=1
matmul at 2.23us against a 0.061us/MAC best rate is the same statement -- 36x the best
per-MAC cost, and no implementation makes two tile-multiplies efficient. The per-MAC filter
flagging those is a property of the filter, not a hole.

### Subblocking the output: rt*ct > 8 is gone, and so is ct > 8

Hole 1 -- a large output block could not be ACCUMULATED -- is fixed, and the fix turned out
to also lift a limit next to it that I had asserted was impossible.

**The subblock shape.** `dst_subblock(rt, ct)` is a port of tt-mlir's
`calculateOutputSubblockFactors`, and what carries over is the priority order rather than
the arithmetic: serve the INNER dimension first, so the subblock is as wide as it can be,
then spend whatever capacity is left on rows. If the inner dimension is fully consumed the
leftover is real and buys rows; if it is not, the subblock is pinned to one row. That yields
the invariant everything else leans on:

> a subblock is EITHER full-width and several rows tall, OR a single row and narrower
> than the block -- never both partial-width and multi-row.

Which matters because partial-width multi-row is the one case whose tiles are *not*
contiguous in a row-major output block. Under the invariant, walking subblocks with rows
outermost visits the output in exactly flat row-major order, so each subblock's pack lands
immediately after the previous one -- `pack_block` advances the buffer's write pointer
itself and only `cb_push_back` resets it. No addressing, no offsets, no second pass.
Checked exhaustively over every (rt, ct) pair drawn from {1..9, 11, 12, 16, 32, 64}: 196
pairs, all within the 8-tile budget, all dividing their dimension, all satisfying the
invariant.

**What had to be banded with it**, and this is the part the old assert was warning about:
the matmul itself, the reload, the pack, a fused addend, the per-step chain, the finish-only
epilogue, the L1 copy-out, and `bias_finish`. One caveat on that last one, since the tests
do not cover it: for any shape with rt*ct <= 8 `dst_subblock` returns the whole block as a
single subblock, and `bias()` still refuses anything larger, so `bias_finish`'s loop always
runs exactly once today. Its banding is therefore a no-op that no test exercises with more
than one subblock -- correct by construction and by the unchanged bias tests, but not
verified in the multi-subblock case, and it will not be until the `bias()` limit below is
lifted. The reload is the interesting one. Partials
are popped from `acc_cb` a SUBBLOCK at a time rather than a block at a time, because the
pack has to reserve pages again in the same call -- `acc_cb` holds exactly one output block,
so reads and writes chase each other around it in lockstep, and a block-sized pop would
deadlock against a subblock-sized reserve.

**`matmul_block` takes the subblock's extents but the true strides.** A's rows are `kt_dim`
apart and B's k rows are `ct_dim` apart however the output is cut; only the output extents
change. `ct_dim` doubles as B's column count and as DST's row stride, which is *why* a
partial-width subblock has to be a single row -- and with `rt_dim = 1` that stride is never
used, so the case is expressible after all.

**Which retires a claim I had put in an assert.** The old message said a column band "would
have to address partial output rows, which the packer cannot do in order". That is true of a
multi-row column band and false of a single-row one, and the algorithm only ever produces
the latter. So `ct > 8` now works, which no path allowed before:

| single-shot | PCC |
|---|---|
| 1x16, kt=2 | 0.999989 |
| 2x16, kt=2 | 0.999989 |
| 1x12, kt=2 | 0.999989 |
| 1x9, kt=1 | 0.999993 |
| 2x32, kt=1 | 0.999993 |

**Correct on the shapes that were refused**, both accumulating modes, including 64-tile
output blocks and multi-k-block runs that exercise the reload rotation -- 4x4, 8x8, 4x8,
8x4, 8x2, 2x8 all at PCC > 0.9996, with per-step, finish-only and both-at-once SFPU chains
all correct on top. All 16 unified tests still pass.

**And they are competitive**, which was not guaranteed -- a 64-tile block is eight passes:

| kt=2 | ours | ttnn | ratio |
|---|---|---|---|
| 4x8 | 8.66 us | 9.48 us | **0.91x** |
| 8x8 | 14.21 us | 15.81 us | **0.90x** |
| 8x4 | 8.55 us | 7.91 us | 1.08x |
| 8x8, kt=32 | 124.24 us | 126.10 us | 0.99x |

Faster than ttnn at several of them. Dst mode and single-shot now measure identically on
these shapes (14.21 vs 14.21 at 8x8 kt=2), which is the expected consequence: with one
k-block and an output past the budget, the accumulating path walks the same subblocks the
banded path does.

**The sweep, same 240 cells as before the change:**

| | before | after |
|---|---|---|
| holes | 64 | **9** |
| of which `rt*ct > 8` refused | 60 | 0 |
| of which L1 capacity | 3 | 9 |
| rate holes (>2x) | 0 | 0 |
| WRONG cells | 0 | 0 |

Every one of the 60 functional holes is gone, and the 9 that remain are all the L1 ceiling
at kt=64 -- 4x8, 8x4 and 8x8, now reported for all three modes rather than only single-shot,
because the other two no longer refuse those shapes before the allocator gets a chance to.
That is a chip limit: two operands of 512 tiles are 2MB against 1.5MB of L1. Across the 201
cells with a ttnn reference the ratio runs 0.90x to 1.40x with a median of 1.14x, and 20
cells are faster than ttnn.

**The bias limit is lifted too**, by folding the bias into the subblock loop with
`fpu_reuse_apply` -- and the measurement says fold in one mode and not the other.

`AddOp::fpu_reuse_apply` adds a named CB tile into a named DST slot, so tile `t` of a
subblock at (r0, c0) takes bias tile `c0 + t % cols`, the same tile re-read once per row.
The catch is that a dest-reuse add is ELEMENTWISE. The two-pass form used
`add_tiles_bcast_rows`, which broadcasts row 0 in hardware; an elementwise add reads all 32
rows, so the bias operand's row has to be replicated down each tile. That costs nothing at
runtime -- the bias is ct tiles either way, so DRAM, L1 and the NOC transfer are identical
and only rows 1..31 differ -- and it is correct for the broadcast form too, which still
finds the right value in row 0. Which is what made the two directly comparable.

**Measured, and it split by mode:**

| 2x4, kt=2 | folded | two-pass |
|---|---|---|
| Dst | **5.76 us** | 6.36 us |
| L1 | 6.59 us | **6.31 us** |

Dst wins by folding because it would otherwise pack the total to `acc_cb` purely to give the
bias pass something to read back. L1 LOSES by folding, because it has to copy the total out
of `acc_cb` regardless and the two-pass form rides along inside that copy for free -- which
is exactly what the original comment claimed ("L1 mode pays nothing for that: this replaces
the copy-out it already did"). So Dst and single-shot fold, L1 keeps the second pass. The
split held at every shape from 1x1 to 4x2 kt=8 kb=4: Dst 0.09-0.61us faster, L1 within
0.05us either way.

I also checked the condition is the mode and not the subblock count, since the fold pays per
subblock (an init and a reuse pass each) while the two-pass form pays once per block. In Dst
mode at 1, 2, 4 and 8 subblocks the fold still wins by 0.60, 0.12, 0.31 and 0.42us. It does
not decay.

Folding also gives the single-shot path a bias for the first time -- it has no accumulation
buffer and so no second pass available -- which is what finally lets `bias()` drop its
output-size assert. A bias no longer constrains the block shape on any path: 4x4, 4x8, 8x4
and 8x8 all verified with a bias, on Dst, L1 and single-shot, with a relu epilogue on top.
That also closes the gap flagged above -- `bias_finish`'s multi-subblock banding is now
exercised, by L1 at 8x8.

**Re-swept afterwards to check the shared code**, since the fold moved the chain ordering
and restructured `run_banded`: 240 cells, 9 holes, all still the L1 ceiling, ratios 0.90x to
1.43x with a 1.15x median. Compared cell by cell against the pre-bias sweep, the 231 cells
with a reference on both sides have a median delta of 0.00% and a mean absolute delta of
0.36%, the largest movement in either direction is 0.09us, and exactly one cell moves more
than 3% -- a 1x2 kt=1 *speedup* where 0.08us is 3.4%. So the bias work is neutral on
unbiased matmuls, which is what it should be: the fold sits behind a `bias_cb != kNoBias`
test that is false for every cell the sweep runs. Worth stating plainly, though -- **the
sweep does not exercise a bias at all**, so it confirms the absence of a regression and
nothing about the change itself. The bias numbers above are the evidence for that, and they
come from a separate harness.

**The cost is a real footgun.** A caller that leaves rows 1..31 zeroed gets the bias applied
to one output row in 32, and because L1 still uses the broadcast form it gets the RIGHT
answer there and the wrong one in Dst and single-shot. That is not hypothetical: the fold
broke `test_unified_matmul_transpose` immediately, at 0.37 relative error, because that file
built its bias the old way. Two call sites in my own repo and I updated one. The test caught
it, both harnesses now replicate, and `bias()` documents the requirement -- but nothing on
device can check it, so it is worth knowing about before a third caller appears.


### The sweep can measure a bias now, and the whole grid agrees with the spot check

`bench_matmul.py --bias` drives the biased harness against `ttnn.linear` instead of
`ttnn.matmul`, so both sides carry a bias. Same 240 cells, same 9 holes (the L1 ceiling),
and the fidelity pin had to be threaded into the bias harness first -- it had no way to set
it, so every biased number before this was at default fidelity and not comparable to ttnn.

**Against ttnn.linear: 0.86x to 1.41x, median 1.14x, 14 cells faster than ttnn.** That is
the same distribution as the unbiased sweep's 1.15x median, so fusing a bias costs us about
what it costs ttnn. The worst cells are 1x1 at kt=32 and kt=64 (1.37x-1.41x), which is the
usual fixed-overhead-on-a-tiny-block story and not about the bias.

**The fold holds up across the whole grid**, which is worth more than the seven shapes it
was decided on: with a bias, Dst is faster than L1 in **77 of 77** comparable cells, median
gap 0.39us. Median ratio by mode is 1.12x for Dst, 1.18x for L1, 1.12x for single-shot.

One number needs care, because read carelessly it looks like the opposite:

| median over 77 cells | unbiased | biased | cost of the bias |
|---|---|---|---|
| Dst | -- | -- | +1.18 us |
| L1 | -- | -- | +0.88 us |
| L1 minus Dst | +0.83 us | +0.39 us | |

The MARGINAL cost of a bias is lower in L1 (+0.88) than in Dst (+1.18), and adding a bias
closes the L1-to-Dst gap from 0.83us to 0.39us. That is not evidence against the fold. L1's
bias pass replaces a copy-out it was already doing in the unbiased case, so part of its
"bias cost" was already on the books; Dst has no such pass to absorb, so its bias shows up
entirely as new work. The comparison that decides the fold is fold-vs-two-pass WITHIN Dst,
which is measured above at 1, 2, 4 and 8 subblocks and favours the fold every time. And in
absolute terms -- which is what a caller actually pays -- Dst with the fold is the fastest
biased configuration in every one of the 77 cells.

**Two reporting fixes fell out of building it.** A missing ttnn reference used to print as a
bare `-` with the reason discarded; it now says why. That immediately showed
`ttnn.linear` producing zero records under the `operations/matmul` match at m=64 n=128 k=64
-- it dispatches elsewhere -- so the biased side matches on `operations` instead, which is
safe because only one op runs inside the bench call and the two agree wherever the narrow
match works. Nine shapes still have no reference, 27 cells of the 231, almost all of them
ct=1: ttnn declines to leave records for a single-tile-wide linear. Reported per cell rather
than hidden.

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

### GQA head mapping: one launch, many heads

The kernel now covers `n_heads` query heads over `n_kv_heads` key/value heads in ONE launch.
The mapping is one line -- query head `h` reads KV head `h / (n_heads / n_kv_heads)` -- and
everything else is addressing: `noc_load(storage, acc, block)` reads
`[block * num_pages, +num_pages)`, so a head dimension is just a stride in block-index
space. Q and the output stride by `num_q_chunks` per head, K and V by `k_tiles / sk` per KV
head, derived rather than passed so it cannot disagree with the causal loop bound.

MHA is `n_kv_heads == n_heads` and MQA is `n_kv_heads == 1`; both fall out of the same
expression, with a static_assert that the counts divide.

**The mask needs no head dimension.** A causal mask does not depend on the head, so the flat
counter the host lays out for one head restarts per head and the same blocks are re-read --
`n_heads` identical copies would otherwise sit in DRAM. Resetting that counter inside the
head loop rather than outside it is the one thing here that would have been a silent
cross-head corruption.

**Why one launch rather than one per head**, measured at sq=2 sk=4 dt=2 with two query
chunks:

| | fused | as separate launches | saved |
|---|---|---|---|
| 2 heads | 53.49 us | 58.49 us | 9% |
| 4 heads | 102.16 us | 116.97 us | 13% |
| 8 heads | 201.20 us | 233.94 us | 14% |

4.1us per head, and the per-head cost falls from 29.24 to 25.15us as the group grows --
`matmul_init`'s hardware startup, the reduce scaler and the column of ones are paid once for
the group. Worth noting that this is NOT the 13.1us the q-loop saved per query chunk; I
initially wrote that number into the kernel comment and extrapolated 0.4ms for llama's 32
heads, which was wrong. The head loop's saving is its own measurement, and 32 heads would
save about 130us a layer, not 400.

**Tested with per-head random data**, so reading the wrong head is a wrong answer rather than
a coincidence -- identical heads would make a broken mapping invisible. Sabotaging the
kernel's `h / kv_group` to `h % n_kv_heads` takes 4x2 and 8x2 to 0.46 error against a 0.03
tolerance, so the cases bite. The `n_kv=1` rows cannot catch it and are not expected to:
with one KV head every mapping selects it. Six (n_heads, n_kv) combinations are now in
`test_unified_flash.py`.

### Heads across cores: no communication, and the ceiling is per-core setup

Heads share nothing -- a core reads its own heads' queries and their KV heads and writes its
own output blocks -- so partitioning them needs no communication at all, only a range each.
The kernel takes `head_begin`/`head_count` as RUNTIME args and the host owns the policy
(`split_evenly` in `unified_harness.py`, spreading the remainder one unit per core so the
makespan stays at ceil rather than piling it on the last core).

Runtime args rather than a coordinate, deliberately. A range derived from
`PhysicalCoord::this_core()` would be right on the two data-movement threads and the
ORIGIN's range on compute, because `my_x`/`my_y` are never filled on a TRISC -- so the loads
and the compute would disagree about how many blocks exist and the circular buffers would
deadlock. `LogicalCoord::this_core()` is safe and would work, but the partition is a host
policy and this keeps it there.

**Scaling, 8 heads at sq=2 sk=4 dt=2 causal:**

| cores | time | speedup | efficiency |
|---|---|---|---|
| 1 | 200.54 us | 1.00x | 100% |
| 2 | 103.92 us | 1.93x | 96% |
| 4 | 57.16 us | 3.51x | 88% |
| 8 | 36.68 us | 5.47x | 68% |

The efficiency curve is fully explained by per-core fixed setup, not by contention: fitting
`T(n) = fixed + heads_per_core * per_head` gives fixed = 11.5us and per_head = 23.6us, which
predicts 200.3us at one core and 35.1us at eight against 200.54 and 36.68 measured. Every
core pays its own `matmul_init` hardware startup, reduce scaler and column of ones, and that
cost stops being amortised once a core holds only one head. 32 heads over 32 cores is 50.22us
total, 1.57us per head.

**A testing hole this turned up, and it was mine.** The output tensor was allocated
uninitialised, and since `run()` is called many times a session the allocator hands back the
same address -- so an output block that NOTHING writes holds the previous run's correct
values. Sabotaging the partition to drop one head passed cleanly at 8 heads over 8 cores and
over 4, and was only caught at 4 over 2 where the preceding run had left something else
there. The output is now pre-filled with NaN, which propagates into the error, and all three
sabotage cases fail as they should. Worth keeping in mind for every other harness here that
allocates an output without initialising it.

**What the tests can and cannot catch.** A partition that MISSES a head is caught (nothing
writes that block). A partition that OVERLAPS is not: two cores computing the same head both
write the same correct values, so the answer stays right. Over-coverage shows up only as a
missing speedup, which is why the scaling table above is evidence rather than decoration.
Partition invariance is checked exactly -- 1 core against 2, 4 and 8 agree to 0.000000, not
to a tolerance, because heads are independent and there is no reordering to excuse a
difference.

### Head concat is not an operation, and the writer does it for free

The attention kernel stores head h's query chunk as an [sq, dt] rectangle at columns
[h*dt, +dt) of one [S_q, d_model] tensor, so the heads come out already concatenated and the
output projection is an ordinary [S_q, d_model] @ Wo matmul. The concat is not a pass, not a
gather and not a k-loop -- it is where the writer aims its pages.

**It costs the attention kernel nothing**, and that is the fact the whole design rests on:
the built-in store already issues one `noc_async_write` per page, because consecutive pages
of an interleaved tensor sit on different banks. A strided store issues exactly the same
number of writes with different destination indices. Measured: 486.88us against 487.11us for
the contiguous store, which is noise.

**The route not taken, and why.** The first version left the output head-major and recovered
the concat arithmetically -- `concat(O_0..O_H) @ Wo == sum over h of O_h @ Wo_h` -- as an
accumulating matmul whose k-blocks are the heads. That is elegant and needs no strided store,
and it cost 30%:

| out 2x8 tiles, K = 8 tiles | |
|---|---|
| 4 k-blocks of 2 (the heads) | 19.67 us |
| 1 k-block of 8 (contiguous) | 13.80 us |

For a fixed query chunk the heads sit `num_q_chunks` blocks apart, so they cannot be one
operand: it was four accumulate CALLS where one would do. The k-blocking itself was not the
expense -- per-call pass overhead was, the same finding as "do not block K" from the matmul
work, reached from the other direction.

**Result at S=256, four 64-wide heads, one core:**

| | head-major + k-loop | strided store |
|---|---|---|
| attention | 487.11 us | 486.88 us |
| projection | 72.04 us | **34.36 us** |
| projection vs `ttnn.matmul` | 2.19x | **1.04x** |
| pair | 559.15 us | 521.24 us |

**I predicted ~55us and 1.67x, and it came out at 34.36 and 1.04x.** Wrong in the useful
direction, and worth understanding rather than pocketing: I had priced only the k-split, but
removing it also removed the accumulator entirely -- no partial buffer, no reload, no extra
pack -- and let Wo become resident, loaded once instead of once per (chunk, head). Three
effects, and I costed one. The same mistake in shape as the DstResident prediction, just with
the sign reversed.

**Tested with both kernels running**, the error taken from the attention REFERENCE rather
than the device's attention output so it covers both: 0.0014-0.0037 max error over four
(heads, kv) combinations including a four-chunk query loop, plus five projection shapes and
four core partitions. The flash suite's 28 checks still pass with the new layout, including
exact partition invariance -- with several cores now writing disjoint COLUMN ranges of the
same rows rather than disjoint row blocks.

**LIMIT worth naming.** Wo is one resident block of dm*dm tiles: 64 tiles at d_model 256, but
4096 tiles (8MB) at d_model 2048, far past L1. A real d_model needs the matmul k-blocked over
slices of dm, and the activation operand is then strided in exactly the way the attention
store is (row r's dm tiles are contiguous, so a k-slice of it is not), which `noc_load`'s Fn
form can express. That reintroduces the per-call cost measured above -- which is what any
implementation pays for a matmul too large to hold, ttnn included.

**Q is still head-major on the way IN.** The attention kernel reads query head h at block
h*num_q_chunks + i, which is the mirror image of the problem just fixed on the output side.
It matters once the QKV projection exists, since that will produce [S_q, d_model] and heads
will be column slices of it -- a strided LOAD, the same shape of change.

### A whole llama decoder layer, and what it took to make the test mean anything

Eleven steps, fourteen launches, every one a unified kernel: rmsnorm, three projections,
RoPE on Q and K, flash attention, the output projection, the residual, the second rmsnorm,
the gate and up projections, silu*up, the down projection and the second residual. Verified
against a torch reference built independently from the same weights. Relative L2 on the
layer output is 0.003-0.004 after eleven bf16 steps, at four (S, d_model, heads, kv)
combinations. 16.6ms wall per layer at S=64, d_model=256 -- wall clock, so host dispatch
dominates it and it is not a device number.

silu had to be added for the FFN, and metal makes that oddly awkward: there is no
`eltwise_unary/silu.h` to match the per-op headers the adaptor includes for exp, recip, relu,
rsqrt and sqrt, even though the SFPU side has its own `ckernel_sfpu_silu.h`. `silu_tile` is
declared only in the umbrella `compute_kernel_api.h`, so the adaptor now includes that.

**The test as first written was nearly worthless, and only sabotage showed it.** Checking the
layer OUTPUT and nothing else, I sabotaged three steps:

| sabotage | output pcc | caught? |
|---|---|---|
| drop the attention residual | 0.288 | yes |
| swap silu's operands, silu(u)*g | 0.999815 | **no** |
| skip RoPE on K entirely | 0.999954 | **no** |

Clean was 0.999948. Two real bugs sat inside the noise. The reason is structural: `y = h + f
@ Wd` with `h = x + ao`, and with x positive and of unit scale while the branches are scaled
smaller, an error inside a branch is diluted before it reaches y. A layer test that cannot
see a missing RoPE is not testing a layer.

So every stage is checked where it happens, not just the output -- ten checkpoints. That
caught the silu swap (0.322 against 0.016 clean). It did NOT catch the missing RoPE on K,
which needed one more step: comparing Q and K themselves. Checking RoPE through the attention
output cannot work, because with random weights the scores are near-uniform and softmax
returns roughly the mean of V whatever they are -- dropping RoPE on K moved the attention
stage from 0.018 to 0.015, in the wrong direction and by nothing. That is exactly the vacuity
the flash harness fixes with a ramp on the keys, reappearing a level up. With `rope_q` and
`rope_k` as their own stages both sabotages fail at 0.90 and 0.95 against 0.008 clean.

Clean per-stage drift, which is what sets the 0.06 tolerance: rmsnorm 0.005, rope 0.008-0.010,
v_proj 0.009-0.012, attention 0.018-0.035, out_proj 0.019-0.039, residual 0.002-0.004,
rmsnorm_ffn 0.006, silu_mul 0.015-0.021, layer 0.003-0.004. The comparisons are CUMULATIVE --
each stage is measured against a reference from exact inputs, so upstream drift is included,
which is the right thing for a composition test and is why attention sits higher than the
kernels' own tests do. Sabotage margins are 20-100x, so the tolerance is not doing the work.

**One step is not on device.** The projections produce [S, d_model], and the attention kernel
still reads Q head-major and K grid-transposed, so `to_flash_layout` rearranges on the host
between the projections and attention. It is a pure permutation with no arithmetic, so it does
not affect what the numbers prove, but the layer is not yet a pure device pipeline and should
not be described as one. The fix is the mirror of what the output side already does: Q wants a
strided load (rows [i*sq, +sq), columns [h*dt, +dt)), K wants the same with its tile grid
transposed, V the same as Q -- all three expressible with `noc_load`'s Fn form at identical
NOC cost, since a read per page is what the built-in load already issues. That would delete
the host-side `grid_transpose` entirely, which has been a wart since the first flash test.

**Still open:** the strided loads above, and fusing the attention and projection launches.

### Blocking the projection in two dimensions: d_model 2048 at 0.94x

Wo is dm*dm tiles -- 64 at d_model 256, but 4096 (8MB) at d_model 2048 -- so holding it whole
was what capped the projection. K is now split into kb blocks of kt tiles, and the two operands
split differently, which is the whole content of the change:

- **Wo block b is contiguous.** It is rows [b*kt, +kt) of a row-major [dm, dm], so an ordinary
  block load at index b.
- **The activation's k-slice is not.** It is columns [b*kt, +kt) of an [sq, dm] block, and rows
  are what is contiguous there, so it needs a custom load -- at the same page count, one read
  per page either way, exactly as with the attention store.

kt == dm gives kb == 1 and takes the single-shot path, so the d_model 256 case keeps its 1.04x
rather than paying an accumulator it does not need.

**Then the accumulator mode mattered more than the blocking.** Dst mode reloads the running
total into DST before each k-block and packs it back after, which costs O(output block) per
k-block -- and this output block is sq*dm tiles, 128 at sq=2 and d_model 2048. L1 mode lets the
packer add into the partial instead, so the total never enters DST:

| [512, 2048] @ [2048, 2048], one core, HiFi2 | Dst | L1 |
|---|---|---|
| sq=4, kt=2 (kb=32, 4 chunks) | 5764.9 us | **4527.1 us** |
| sq=2, kt=4 (kb=16, 8 chunks) | 5918.1 us | 5345.6 us |
| ttnn.matmul, same shape, one core | | 3962.7 us |

**4527us against 3962.7us is 1.14x**, which is exactly the median the matmul sweep gets --
so the projection is no worse at a real d_model than the library's matmul is anywhere else.
L1 is also the more ACCURATE mode here (PCC 0.999963 against 0.999586), because the packer
accumulates without the round trip through bf16 DST that Dst mode takes every k-block. That
inverts the bias-fold result, where Dst won: there the output block was 8 tiles and the reload
was cheap, here it is 128 and it is not. L1 is now the default and Dst is behind
`PROJ_ACC_DST`.

**Then the output columns were blocked too, and that closed it.** With `nt == dm` the whole
output width is resident, so `sq` is capped by `sq*dm` and Wo gets re-read once per query
chunk. Blocking both dimensions trades one against the other. Over `st` total row-tiles the
DRAM traffic is

    st * dm^2 * (1/sq + 1/nt)   tiles

-- every query chunk reads all of Wo, every output-column block reads all of the activation --
subject to `2*sq*nt` plus operands fitting L1. `nt == dm` makes the first term dominate.
Balancing them is the whole trick, and the model ranks the configurations before measuring:

| [512, 2048] @ [2048, 2048], one core, HiFi2 | model (tiles) | measured |
|---|---|---|
| sq=8, kt=8, nt=16 | 12288 | **3711.0 us** |
| sq=16, kt=8, nt=8 | 12288 | 3712.5 us |
| sq=8, kt=4, nt=16 | 12288 | 3938.3 us |
| **ttnn.matmul, same shape** | | **3962.9 us** |
| sq=8, kt=8, nt=8 | 16384 | 4076.9 us |
| sq=4, kt=8, nt=32 | 18432 | 4119.7 us |
| sq=4, kt=2, nt=64 (K-only) | 17408 | 4531.6 us |

**3711us against 3962.9us is 0.94x -- faster than ttnn.matmul on the same shape, on one
core.** The model orders the configurations correctly except for one inversion: 17408 tiles
measures slower than 18432, because the 17408 config has kt=2 and so kb=32 accumulate calls
where the other has kb=8. Traffic explains most of it and per-call overhead explains the rest,
which is the same pair of costs every measurement in this file keeps landing on.

Every operand is now gathered by a custom load -- the activation's k-slice, Wo's (k, n) tile,
and the output block's strided store -- because none of the three is contiguous in its backing
tensor once both dimensions are blocked. That is free: one read per page is what a contiguous
block load already issues. Gathering W also normalises its row stride to `nt`, so the matmul
geometry never has to know it came out of a wider matrix.

ttnn's 3962.9us is essentially just the arithmetic: 65536 tile-MACs at the 0.061us/tile-MAC
the sweep's best cell achieves is 3998us. Beating it slightly is not a claim about the FPU --
it is that at HiFi2 on one core this shape is bandwidth-bound, and the traffic model above is
what got the blocking right.

### The same treatment for rmsnorm and the FFN, and one kernel for all four matmuls

**The projection kernel was already a general blocked matmul, so it became one.**
`attention_proj.cpp` is now `matmul_blocked.cpp`, taking mt, ktot, ntot, kt and nt instead of
a single square d_model. That one kernel is all four of a layer's large matmuls: the output
projection at K = N = d_model, gate and up at N = ffn, and down at K = ffn. Consolidating
beat duplicating the same three gathers into a second kernel. `matmul.cpp` stays as the
single-shot path for blocks that DO fit whole, which is cheaper when it is possible.

The FFN shapes at llama-3.2-1B, S=512, one core, HiFi2 -- B is 16384 tiles (32MB) here, so
these exist only because both dimensions are blocked:

| | ours (mt=16, kt=8, nt=8) | ttnn.matmul | |
|---|---|---|---|
| gate/up, [512, 2048] @ [2048, 8192] | 14818.6 us | 21808.1 us | 0.68x |
| down, [512, 8192] @ [8192, 2048] | 14675.7 us | 15772.1 us | 0.93x |

**The 0.68x is not a speed claim.** The two shapes have identical arithmetic -- 262144
tile-MACs each -- and ours measures within 1% of itself on them (14818.6 against 14675.7),
which is the sanity check that says the implementation is symmetric. ttnn's 21808 against
15772 for the same work is its single-core program config choosing badly for the wide-N shape,
and ttnn on one core is not the configuration it is built for. What the numbers do support is
that we are at the arithmetic limit: 262144 tile-MACs in 14675.7us is 0.056us per tile-MAC,
against the 0.061 best cell in the matmul sweep.

**rmsnorm now walks rows in chunks.** Rows are normalised independently -- each one's RMS
depends on that row alone -- so a chunk height is a pure decomposition, and the test holds it
to exactly that: every chunk height agrees to the BIT, not merely closely. It could not do
S=512 by d_model 2048 before, because the kernel holds four blocks of the tensor at once (x,
its square, the normalised value and the output), so 1024 tiles resident means 8MB. At chunk=1
it is 256 tiles and it runs: 494.9us on one core, 147.5us on sixteen.

Sixteen is the ceiling, and that is worth naming: the parallel unit is the row chunk, so at
S=512 with a one-tile chunk there are only 16 of them and 48 of the 64 cores have nothing to
do. ttnn's layer breakdown puts its two layernorms at 94.3us together, and it gets there by
splitting differently. Ours would need the WIDTH split across cores as well, with a
cross-core reduction for each row's sum -- the first place in this work where a core would
have to talk to another one.

**Accuracy, since a wide row moves it:** rel-L2 goes 0.00515 at 128-wide rows to 0.00690 at
2048-wide, and max|err| 0.030 to 0.051. That is bf16 accumulation over 16x the terms, not a
chunking artifact -- chunk=1 and chunk=2 agree exactly at the wide shape. It does cross the
0.05 abs bound the narrow cases use, so the wide cases are gated on relative L2 instead; an
absolute bound was never going to scale with the row width.

**What still stops the LAYER at d_model 2048** is now only flash attention, which has not been
tried at 32 heads, and the host-side Q/K/V layout glue. Every other component scales: rmsnorm,
all four matmuls, RoPE (a flat per-tile stream), silu*up and the residuals (elementwise, any
block count). The verified layer is still d_model 256.

**Cost of these two changes, paid twice now:** changing a kernel's runtime-arg contract has no
host-side check. Adding the chunk range to rmsnorm hung the device, because
test_unified_layer.py has its own rmsnorm launcher that still passed four runtime args and the
loop bound came from whatever was in the fifth slot. The 2D blocking hung it the same way
earlier, from a compile-time arg count. Both were `tt-smi -r` and a one-line fix, but a kernel
that reads argument N while a caller passes N-1 is not a compile error on either side.

### The ttnn equivalent, and what its op breakdown says about where to spend effort

There is a direct counterpart: `models/tt_transformers/tt/decoder.py`, class `TransformerBlock`
-- one llama decoder layer -- with `tests/test_decoder_prefill.py` running it in prefill
against an HF reference. That is the same thing test_unified_layer.py does. It normally needs
a checkpoint, but `ModelArgs(..., dummy_weights=True)` reads the config from
`models/tt_transformers/model_params/<name>` and generates the weights, so it runs with no
download: `HF_MODEL=meta-llama/Llama-3.2-1B-Instruct` is enough.

Run at the real Llama-3.2-1B shape (d_model 2048, 32 heads, 8 KV heads, head_dim 64, FFN
8192), S=512, one N150, bfloat8_b weights: **1.98-2.13ms wall, 1820us of summed device time
over 22 ops.** Where that goes:

| us | share | op |
|---|---|---|
| 730.3 | 40% | `bmm_large_block_zm_fused_bias_activation` -- the big matmuls |
| 388.4 | 21% | a multicast matmul (`dm_in0_sender` / `dm_in1_sender_out`) |
| 186.4 | 10% | eltwise binary SFPU |
| 135.8 | **7.5%** | `sdpa` -- the attention itself |
| 94.3 | 5% | layernorm, twice |
| 72.5 | 4% | rotary embedding |
| 65.5 | 3.6% | `nlp_create_qkv_heads` -- the head split |
| 63.8 | 3.5% | eltwise binary |

Three things worth taking from that, and they re-rank the remaining work:

- **Matmuls are 61% of a layer and attention is 7.5%.** Our matmul sits at a 1.14x median
  against ttnn (1.04x for the output projection), and our flash attention is around 1.5x off
  SDPA. Weighted by this breakdown, the matmul gap is worth several times more than the
  attention gap -- and the attention work is where nearly all the effort has gone. A 1.5x on
  7.5% is about 4% of a layer.
- **ttnn spends 3.6% of a layer on `nlp_create_qkv_heads`**, which is exactly the head
  split/concat problem. The strided store does that for free, and the strided loads would do
  the same for Q, K and V. That is a real structural advantage of writing the whole pipeline
  as one kernel rather than a sequence of ops, and it is now quantified rather than asserted.
- **A matched-shape head-to-head is not possible yet**, so no ratio is quoted here. ttnn's
  block only runs at real model dims; ours cannot reach d_model 2048 because the projection
  holds all of Wo in L1. Ours measures 411.8us at S=64/d_model=256 and 757.1us at
  S=128/d_model=256 (15 programs, summed device time, one core, bf16). Against ttnn's number
  that is 800x less work, on 1/64 of the cores, at higher fidelity -- too many confounds to
  divide. The per-op comparisons stay the honest ones until the projection can be k-blocked.

Method note: `bench()`'s `median_us` is the median of ONE program's duration, which is what
every measurement here has wanted -- but for a 22-op layer it is not the layer's time. Summing
the rows and dividing by iterations is. Reading it the first way gave 41.3us for a layer whose
arithmetic cannot be done in under a millisecond, which is what caught it.

### Where llama prefill perf actually stands

Every component measured at the REAL llama-3.2-1B shape -- d_model 2048, 32 heads, 8 KV heads,
head_dim 64, FFN 8192, S=512 -- on ONE core at bf16/HiFi2, each in its best blocking:

| step | us | share |
|---|---|---|
| FFN gate + up, [512,2048]@[2048,8192] x2 | 29638 | 38% |
| FFN down, [512,8192]@[8192,2048] | 14676 | 19% |
| silu(gate) * up, 4096 tiles | 10972 | 14% |
| flash attention, 32 heads causal | 9387 | 12% |
| Q projection [512,2048]@[2048,2048] | 3711 | 5% |
| output projection, same shape | 3711 | 5% |
| RoPE on Q and K | 3007 | 4% |
| K and V projections [512,2048]@[2048,512] | 1872 | 2% |
| rmsnorm x2 | 990 | 1% |
| residual x2 | 374 | <1% |
| **total** | **78338** | |

**78.3ms on one core.** ttnn's TransformerBlock is 1820us of device time on the full grid at
bfloat8_b. That is 43x, and the two factors that explain it are 64 cores and a cheaper weight
format -- not per-core arithmetic, where the same components measure at or slightly better
than ttnn on one core: the output projection is 0.94x, FFN down 0.93x, and the matmuls run at
0.056us per tile-MAC against the sweep's 0.061 best.

The internal proportions also agree with ttnn's own breakdown, which is a decent check that
nothing here is structurally wrong: matmuls are 62% of our layer against ttnn's 61%, and
attention is 12% against its 7.5%.

**The blocker was multi-core, and splitting N fixed the first half of it -- see the section
below.** The original diagnosis, kept because the second half is still true:

The parallel unit was the M-block, and
the blocking that is efficient per core is the one that leaves almost nothing to parallelise.
Measured on the output projection:

| | M-blocks | cores | us |
|---|---|---|---|
| mt=8, nt=16 | 2 | 1 | 3710.7 |
| mt=8, nt=16 | 2 | 2 | 2265.9 |
| mt=1, nt=16 | 16 | 1 | 8102.7 |
| mt=1, nt=16 | 16 | 8 | 4191.8 |
| mt=1, nt=16 | 16 | 16 | 3371.7 |
| **mt=2, nt=16** | 8 | 8 | **2225.0** |

Sixteen cores at mt=1 (3371.7us) barely beats ONE core at mt=8 (3710.7us), because traffic
goes as 1/mt and shrinking mt to make blocks gives back what the extra cores win. The best
found is mt=2 on 8 cores, and that is only 1.67x over one core.

So the ceiling is not the FPU and not the blocking arithmetic -- it is that cores are handed
only M-blocks. A core should take an (m, n) TILE of the output instead: at S=512 and d_model
2048 that is 8 M-blocks x 4 N-blocks = 32 units each still holding a large mt, where today the
same shape offers either 2 fat blocks or 16 thin ones. Splitting N across cores needs no
reduction (different output columns, disjoint writes), which is why it is the next thing to do
rather than splitting K. rmsnorm has the same shape of problem from the other direction: its
row chunks cap at 16 at this shape, and going wider needs the width split with a cross-core
reduction per row.

Two smaller things the table exposes. silu*up at 10972us for 4096 tiles is 2.7us a tile for
one SFPU pass, which is worth a look -- ttnn spends 10% of a layer on its eltwise SFPU and we
spend 14%. And RoPE at 3007us is 4% here against ttnn's 4%, so it is in proportion, but 2399us
of it is Q alone at 1024 tiles.

### Splitting N across cores, and the wall behind it

The unit of work is now one OUTPUT BLOCK -- an (m, n) tile, indexed flat as m*nb + n -- rather
than an M-block. Neither dimension needs a reduction: two cores holding different m or
different n write disjoint parts of the output, and only K would need one, which is why K
stays inside a core. The host hands each core a contiguous range of that flat index.

That removes the trade that made mt fight the core count. Output projection,
[512, 2048] @ [2048, 2048], HiFi2:

| | units | cores | us |
|---|---|---|---|
| before, M-split only, best found | 8 | 8 | 2225.0 |
| mt=8 nt=16 | 8 | 1 | 3717.4 |
| mt=8 nt=16 | 8 | 8 | 932.8 |
| **mt=8 nt=8** | 16 | 16 | **814.1** |
| mt=4 nt=8 | 32 | 32 | 940.0 |
| mt=2 nt=8 | 64 | 64 | 1359.3 |

**2225 -> 814us, 2.7x**, and 4.6x over one core. But the scaling stops at 16 cores and then
REVERSES, which is the interesting part: more units means smaller mt and nt, and traffic goes
as (1/mt + 1/nt), so past 16 cores the extra traffic costs more than the extra cores earn.
The measured bandwidth tells the same story -- 26GB/s at 8 cores, 41 at 16, 43 at 32 -- so the
device is not saturated and the limit is what we ASK it to move.

**Against ttnn on the same shape, which is now the fair comparison:**

| cores | ours | ttnn.matmul |
|---|---|---|
| 1 | 3717 us | 3962 us |
| 4 | 1548 us | 1032 us |
| 16 | 814 us | 288.5 us |
| 64 | 1359 us | **118.6 us** |

We win on one core and lose by 7x on sixty-four. ttnn scales 33x where we scale 4.6x, and the
reason is structural rather than a tuning gap: **every core here fetches its own operand tiles
from DRAM, while ttnn MULTICASTS them across the grid.** Cores in a grid row share the same A
blocks and cores in a column share the same B blocks, so a weight tile is read from DRAM once
and broadcast over the NOC. At 64 cores that is roughly mtot*ktot + ktot*ntot = 5120 tiles
(10MB) against our 40960 (80MB) -- 8x the data for the same arithmetic. ttnn's 118.6us over
10MB is 84GB/s, which is consistent; our model's 80MB in that time would need 675GB/s, which
is not.

So the next thing is weight multicast, and the library already has the primitive: `noc_load`
has a `PhysicalMcast` overload with the two semaphores, and `matmul_mcast.cpp` and
`mcast_bcast.cpp` already use it. What is missing is putting it under the blocked matmul: a
core would join a row group for A and a column group for B, and the traffic term stops scaling
with the core count.

### Weight multicast: 814 -> 353us, and the ceiling is now bandwidth per read

Each operand tile is read from DRAM ONCE and broadcast to the cores that share it: A along a
grid ROW (those cores share an m-block), B down a COLUMN (they share an n-block). The traffic
term stops depending on how many cores there are, which is what the N-split could not fix.

**The library needed one addition.** Its multicast load read the operand as one contiguous
block -- fine for the block-major layout `matmul_mcast.cpp` uses, useless for the blocked
matmul, whose A is a k-slice of a row-major activation and whose B is a (k, n) tile of a wider
matrix. So the handshake moved into an Fn form where `fn` runs on the SENDER only and fills
its copy however it likes; the accessor form now delegates to it with a contiguous fill. A
gathered operand is an ordinary block once it is in L1, and the broadcast does not care how
the bytes got there -- at no extra traffic, since the built-in read issues one request per
page too.

**Measured, [512, 2048] @ [2048, 2048], HiFi2, natural layouts:**

| cores | no multicast | with multicast |
|---|---|---|
| 16 | 814.1 us | **482.1 us** |
| 32 | 940.0 us | 379.9 us |
| 64 | 1359.3 us | **353.2 us** |

It improves monotonically now, where without multicast it REVERSED past 16 cores. The whole
arc for this shape: 2225us with M-split only, 814 after splitting N, 353 after multicast --
**6.3x**, and the gap to ttnn goes from 6.9x to 3.0x.

It also beats the block-major `matmul_mcast.cpp` at the same shape (353.2 against 364.5us)
while needing no host-side rearrangement, which is the gather paying for itself.

**The price is a strict mapping**, and it is worth stating: core (r, c) owns output block
(r, c), so the grid is exactly mb x nb and each core holds one block. A multicast is
COLLECTIVE -- every core in a group must make the same calls in the same order or the
handshakes desynchronise -- which the flat unit range cannot promise, since `split_evenly`
hands different cores different counts. It also caps nb at the grid width: nt=4 gives nb=16,
wider than the 8-column device, and the program fails to build.

**It is not sharding, which was the obvious suspicion.** The ttnn baseline throughout this
file was built with a plain `ttnn.from_torch(..., device=d)`, and its memory config reads
`TensorMemoryLayout::INTERLEAVED, BufferType::DRAM` -- the same interleaved DRAM ours uses.
ttnn reaches 118.6us from the same layout we read at 353.2, so the difference is in how the
reads are issued, not where the data sits.

**And we would support sharding if we wanted it, with no kernel changes.** `TensorAccessor`
handles sharded layouts (`is_sharded`, `shard_pages_address_iterator.h`), and the harness
passes `TensorAccessorArgs(t)`, which encodes whatever layout the tensor actually has -- so
`acc.get_noc_addr(page)` resolves a shard exactly as it resolves an interleaved page.
Verified: the blocked matmul reads a HEIGHT_SHARDED L1 operand at pcc 0.999967 with nothing
in the kernel mentioning sharding. Where it would earn its keep is weights RESIDENT in L1
across the grid -- 8MB of Wo is 128KB a core on 64 cores -- which removes the DRAM read
entirely rather than making it faster. That only pays when a weight is reused across many
matmuls, which prefill does not do within one layer.

### Prefetch depth: it pays only where multicast made room for it

The reads WITHIN a block were already deep -- every page is issued before the single barrier
-- so what was missing was depth ACROSS blocks: with a one-block operand CB the data-movement
thread cannot reserve k-block b+1 until compute has popped b, so its reads and the compute
serialise. Giving the operand CBs `depth` blocks is permission to run ahead, and it needs no
kernel change at all: the kernel reserves one block at a time because `num_pages` comes from
the Shape, not from the CB. Same mechanism as the flash kernel's `stream_buffering`.

| [512,2048]@[2048,2048], multicast 8x8 | |
|---|---|
| depth 1 | 350.2 us |
| **depth 2** | **307.6 us** |
| depth 3 | 318.6 us |
| depth 4 | 309.3 us |

12%, and nothing beyond 2. **But it only helps where multicast made room for it**, which is
the part worth keeping:

| 16 cores, out-projection, NO multicast | |
|---|---|
| depth 1 | 807.8 us |
| depth 2 | 830.1 us |
| depth 3 | 842.9 us |

Steadily WORSE. Depth pays when the read path is latency-bound and costs when it is
bandwidth-bound: with multicast only 16 of 64 cores touch DRAM and there is headroom for
another read in flight; without it every core reads for itself, DRAM is already saturated,
and extra outstanding requests only add contention. So the default is 2 with multicast and 1
without, and that is measured rather than tidy. The FFN agrees: gate/up on 64 cores goes
1482.7 -> 1329.5us.

**Bigger k-blocks do not help either** -- kt=8 gives 320.1us, kt=16 352.0, kt=32 370.8 -- so
the per-block handshake is not what is left. Coarser blocks just delay the broadcast behind a
longer read and cost overlap.

### Profiling the threads: the math is not on the critical path at all

Two measurements, and the first one is a knob rather than a probe.

**Per-RISC kernel spans say nothing**, which is worth recording as a negative result. Every
one of the five threads spans essentially the whole kernel -- BRISC 284.9us median, NCRISC
266.0, the three TRISCs 272.x, against a ~310us program. Nobody is idle. But a SPAN includes
time blocked on a circular buffer, so this ranks nothing: five threads in lockstep through a
k-loop all span the loop.

**The fidelity ablation is decisive.** Math fidelity changes the MAC cost and nothing else --
LoFi is one pass per tile, HiFi2 two, HiFi4 four -- so it moves compute without touching a
byte of data movement:

| | |
|---|---|
| LoFi | 319.6 us |
| HiFi2 | 317.4 us |
| HiFi4 | 317.4 us |

**Four times the math, zero difference.** Compute is not on the critical path; it is entirely
hidden. That is a stronger statement than the bandwidth arithmetic suggested -- it had
compute at 57us of ~310, and the truth is it contributes nothing.

**So where does the movement go?** Hoisting the operands out of the k-loop -- one load and
broadcast instead of eight, the wrong answer on purpose (`MMB_ABL_HOIST`):

| multicast 8x8, mt=2 kt=8 nt=8, depth=2 | |
|---|---|
| real | 312.0 us |
| operands hoisted | **127.2 us** |

**The per-k-block operand movement is 185us of the 312, 59%**, over the seven rounds the
hoist removes -- about 26us a round for a read, a broadcast and two handshakes. And the
127.2us residual is close to ttnn's entire 118.6us, which says ttnn's per-k-block movement is
nearly free where ours is the whole cost.

**The mechanism, and it is structural rather than a tuning gap.** Per k-block the B sender
moves 64 tiles (128KB) and the A sender 16 (32KB); at NOC and DRAM rates that is maybe 12us
of traffic, and the handshakes another few. But 26us of it lands on the critical path because
`noc_load`'s multicast form does reserve, read, BARRIER, wait-for-receivers, broadcast, push
-- all inside one call. The read for k-block b+1 therefore cannot start until b's broadcast
has finished, so the DM thread runs read(b), handshake(b), read(b+1), handshake(b+1) in
series. Deeper CBs let the RESERVE happen early, which is why depth=2 bought 12%, but the
read is issued inside the call that then blocks on the handshake, so nothing overlaps the
part that matters.

I proposed splitting the multicast load into two phases -- start the read, then
finish-and-broadcast -- so a sender could have the next block's DRAM read in flight during
the current broadcast. **That diagnosis does not survive contact with ttnn's source, and the
correction is the useful part.**

### What ttnn's matmul actually does, and where the difference really is

Checked against `reader_bmm_tile_layout_in0_sender_padding.cpp` rather than inferred:

- **The loop nesting is the same.** K is the INNERMOST loop
  (`for block < num_blocks_inner_dim` inside the h- and w-block loops), not hoisted outside
  to reuse operands across an outer product. Ours is the same.
- **The per-k-step sequence is the same.** `reserve_back`, read the tiles,
  `noc.async_read_barrier()`, wait for the receivers' semaphore, `async_write_multicast`,
  multicast the flag, `push_back`. Line for line, that is our multicast load.

So ttnn serialises the read against the broadcast exactly as we do, and "split it into two
phases" cannot be what makes it 2.6x faster -- both would have the same defect.

**The difference is the flushes.** After the payload multicast ttnn does NOTHING on Wormhole:

    // Note: no need for write barrier, since these two multicasts are done on the same noc
    // id, same vc, same cmd_buf ... using NOC_CMD_STATIC_VC

Only Blackhole gets a flush there. We do TWO `noc_async_writes_flushed()` per broadcast, and
there are two broadcasts per k-block, so 32 NOC round trips per core over an eight-block
k-loop.

**They are not removable as they stand**: an ablation with both taken out DEADLOCKED the
device, which is the proof they are load-bearing rather than defensive. The reason is not
ordering -- ttnn is right that same-NOC, same-VC writes cannot reorder -- it is our
`data_sent.set(0)` immediately after. `set_mcast` sources its value from local L1, so the
reset can overwrite the flag word before the multicast has read it, and the receivers then
wait on a 1 that never arrives.

**So the obvious fix is the protocol, not the primitive** -- a flag the sender never rewrites
in the same breath, an incrementing counter the receiver compares against a block number,
needing no reset and therefore no flush to protect one.

**Built it. It bought nothing, and has been reverted.** Recorded here so nobody spends the
day on it twice.

The counter version made both semaphores running counts that nothing resets: the sender
waited for `(seq + 1) * num_dests` instead of `num_dests` then a reset, the flag became
`inc_mcast(mcast, 1)`, and the receiver did `wait_min(seq + 1)` instead of wait-1 then rearm.
It was correct everywhere and it did remove a race the old protocol only avoids by accident
-- a receiver cannot increment the ready counter for block b+1 before the sender's reset for
block b, but nothing in the protocol says so, only the surrounding order of operations does.

It measured 314.0us at 64 cores and depth 2, against 307.6, 312.0 and 315.0 taken three times
on the existing protocol. The same number. With hindsight the reason is plain: **the flush it
removes sits AFTER the flag goes out**, so its latency was already overlapped with the
receivers making progress. The flush that matters is the one between the payload and the
flag, and the counter form cannot remove it.

**Nor can any form.** ttnn skips that flush because its flag is a WRITE on the same NOC, VC
and command buffer as the payload write, so the two cannot reorder. An atomic increment does
not share the write path's ordering, so the increment form needs the flush. Going back to a
write to inherit the ordering does not work either: with a monotonic value and `wait_min`, a
flag multicast still in flight when the next block overwrites the local word delivers the
LATER value, and a receiver steps past a payload that has not arrived. **The increment is
safe without a second flush; the write is ordered without the first; neither gets both.**

Reverted because it cost an API change -- a sequence number threaded through every multicast
overload and all three call sites -- for no measurable gain. The race it fixed is real but
latent, and not worth that price on its own; if it is ever fixed it should be fixed without
the parameter.

### Three zones on the sender, and the answer

`TT_UNIFIED_MCAST_ZONES` puts a `DeviceZoneScopedN` around each of the three things a sender
does per k-block. Off by default: a zone per block per core fills the profiler buffer
quickly, and it drops records silently once it does. Note that `fn` merely ISSUES the reads,
which are asynchronous, so their cost lands in the barrier rather than in the issue.

32 cores (4x8), mt=4 kt=8 nt=8, eight k-blocks, per sender summed over its blocks:

| zone | per block | per sender | share |
|---|---|---|---|
| MCAST-READY -- waiting for receivers | 0.04 us | 0.3 us | **0.2%** |
| MCAST-DRAM -- the reads landing | 21.46 us | 127.3 us | **72%** |
| MCAST-SEND -- broadcast, flag, flushes | 7.54 us | 50.2 us | 28% |

**The handshake wait is 0.3us.** Everything spent on the counter protocol was aimed at a cost
that does not exist -- which is the strongest argument for having reverted it, and a reminder
that the flushes were never the same thing as the waiting.

**It is the DRAM read, and it is bandwidth, not per-page overhead.** The two senders disagree
too much for a fixed cost per request: the A sender moves 32 tiles at 151ns a page (13.6GB/s)
while the B sender moves 64 at 366ns (5.6GB/s). What they have in common is the moment --
twelve sender cores reading at once. Per k-block the grid demands 4x64KB of A plus 8x128KB of
B, 1280KB, and over eight blocks exactly the 10MB that A-once-plus-B-once should be. At the
B sender's 23.45us a block that is 188us of DRAM on the critical path, which is the 185us the
hoist ablation attributed to operand movement, arrived at independently.

**So: 56GB/s aggregate against ttnn's 84GB/s on the same 10MB.** Not more traffic, not more
requests, not synchronisation -- the same bytes moving more slowly. That is where the 2.6x
lives, and it is a narrower target than anything earlier in this file: the sender's read
pattern against DRAM, twelve cores at a time.

### Sharded weights: tried, 10% worse, and the cost moved rather than vanished

Reverted; the numbers are kept because a negative result that cost device time is exactly
what should not be re-run. The idea was that a sender's whole read set for a column is the
strip `[c*nt, +nt)` of B across every k-block, so WIDTH-sharding B one strip per column would
turn pages round-robining across banks into one contiguous run. The kernel needed no change
at all -- `TensorAccessorArgs(t)` carries the layout and `get_noc_addr` resolves it -- so it
was a harness-only experiment, which is itself the point worth keeping about the accessor
design: a different memory layout is not a kernel change.

| [512,2048]@[2048,2048], multicast, depth 2 | 64 cores | 32 cores |
|---|---|---|
| B interleaved | **316.7 us** | **320.3 us** |
| B DRAM width-sharded | 348.8 us | 355.7 us |
| B L1 width-sharded | out of L1 | out of L1 |

10% WORSE, at both core counts. L1 sharding does not fit: a strip is 1MB and the kernel needs
its own circular buffers on those same cores.

**And the zones say the cost moved rather than appeared.** With B sharded, the B sender's own
read actually got FASTER per page -- 278ns against 366ns, 7.4GB/s against 5.6 -- while the
program got slower. So the sharded read is not what regressed; something else absorbed more
than it saved. The likeliest explanation is bank coverage: eight strips occupy eight banks
where the interleaved layout spreads across all twelve, so peak aggregate bandwidth drops
even as one reader's slice improves. That is a guess consistent with the numbers, not a
measurement -- I did not verify which banks the shards landed on.

**What it settles is the hypothesis it was built to test.** Contiguity is not the missing
property: interleaving is what gives a single reader bank-level parallelism, and taking it
away concentrates that reader instead of helping it. ttnn's 84GB/s comes from interleaved
DRAM, and this says that is not an accident of its configuration but the better layout for
this access pattern. The 56-vs-84GB/s gap is still open, and it is not layout.

Recorded because the sequence matters: the bandwidth ratio suggested a story, the ablation
narrowed it to the per-k-block movement, and only READING THE OTHER IMPLEMENTATION showed
that the mechanism I had named was shared by both and therefore explained nothing.

**What is left is bandwidth per read, not traffic.** With multicast the traffic is A once plus
B once, 5120 tiles or 10MB, and we move it in 353.2us -- 28.4GB/s. ttnn moves the same 10MB in
118.6us, 84GB/s. So the remaining 3x is not extra data, it is that our reads are slower: the
sender issues one request per page and BARRIERS before it can broadcast, with nothing else in
flight, and the operand CBs hold exactly one block so it cannot read the next k-block while
the current one is being broadcast and consumed. That is the prefetch-depth finding from the
matmul k-block work, arrived at a third time and now the single thing standing between this
and ttnn's number.

### Digging into the sender: 95% accounted, three hypotheses dead, one lever left

**The sender's time, per sender over eight k-blocks, against a 273.9us span.** The zones
NEST -- `LOAD-ISSUE` wraps the whole multicast handshake -- so the leaf costs come from
subtracting, and they close to 95%:

| | us | share | |
|---|---|---|---|
| LOAD-RESERVE | 0.3 | 0% | backpressure; depth 2 is enough |
| issue (leaf) | 31.3 | 11% | 512 `noc_async_read` calls at ~60ns each |
| MCAST-READY | 0.3 | 0% | waiting on receivers |
| **MCAST-DRAM** | **168.1** | **61%** | the barrier: reads landing |
| MCAST-SEND | 60.6 | 22% | broadcast, flag, flush |

**Three hypotheses died cheaply, which is the point of having the zones.**

*Bank aliasing.* With `nt=8` and twelve banks, the eight senders' addresses stride by 8 and
land on only 3 distinct banks; at `nt=7` they land on 8. Predicted a large win. Measured 370
vs 359 ns/page -- **nothing**. Bank spread is not the constraint.

*Block size.* Per-page cost is flat as the block grows -- 304ns at kt=2, 312 at kt=4, 361 at
kt=8, 389 at kt=16 -- so it is a steady-state rate, not a per-block startup cost that bigger
reads would amortise.

*A per-core read ceiling.* A concurrency sweep looked like it showed one sender reading at
89.4GB/s and eight at 5.6 each. **That reading was wrong and is worth recording as a trap:**
the zone measures the BARRIER, which is read time minus whatever already overlapped. At low
concurrency the reads were hidden behind other work, not fast. The zone tells you what is
EXPOSED, not what a wire can do.

**What is left is one structural thing, and it is ours rather than a mystery.** Per block the
sender spends 22.42us at the DRAM barrier and 7.55us broadcasting, strictly in that order,
because both live inside one `noc_load` call on one thread. Nothing overlaps them: the reads
for block b+1 cannot be issued until block b's broadcast has flushed. Overlapping them would
cost max(22.42, 7.55) instead of the sum -- about 25%, or ~310us down to ~235.

That was proposed once and withdrawn on the grounds that ttnn's sender serialises identically
(verified again here in its in1 sender, which is the one that carries the weights). **That was
a reasoning error worth naming: "ttnn shares this defect" means it does not explain the gap,
not that fixing it is worthless.** The two are different claims and conflating them cost a
lever.

It needs the primitive split into begin/finish so a sender can issue block b+1's reads while
block b's flag is still going out. Before building that, one ablation bounds what it could
possibly be worth: remove the broadcast ENTIRELY -- no ready-wait, no payload multicast, no
flag, receivers skipping their wait to match (`TT_UNIFIED_MCAST_NOSEND`, wrong answer on
purpose).

| [512,2048]@[2048,2048], 8x8 mcast | |
|---|---|
| real | 312.7 us |
| reads only, broadcast deleted | **258.7 us** |

**54us, 17%, and that is the CEILING** -- perfect overlap can only hide the broadcast, never
do better than deleting it. So the begin/finish split was NOT built: it buys at most 17% and
leaves the gap at 2.2x, which does not justify a second API change of the kind that was just
reverted for buying nothing. Measuring the bound cost one ablation; building first would have
cost a day.

**And the bound says something much more useful than its own number.** With the broadcast
gone entirely, the reads alone still take 258.7us to move 10MB -- 39GB/s -- while ttnn moves
the same 10MB in 118.6us WITH its broadcasts. **Our reads by themselves are 2.2x ttnn's
entire runtime.** Every structural thing that could have explained this is now measured and
matched: same loop nesting, same per-k-step sequence, same interleaved layout, same page
size, same traffic, same number of sender cores, same NOC split across two threads. The
handshake is 0.3us, compute is free, the CBs never stall, bank spread does not matter, block
size does not matter.

What is left is the rate at which a sender's `noc_async_read` stream actually retires, and
nothing in the kernel's structure explains it. That is where this stops until someone reads
what ttnn's reader does at the NOC level -- transaction ids, command buffers, VC assignment,
`noc_async_read_one_packet_with_state` and friends -- rather than at the loop level, which is
where the last four hypotheses were formed and died.

### It was the NOC. B on NOC 0 and A on NOC 1: 308.3 -> 152.5us

Reading ttnn's reader at the NOC level -- the thing the paragraph above says to do -- turns
up nothing exotic. Same `TensorAccessor`, same `noc_async_read`, same general path (a 2KB
tile is four bursts at `NOC_MAX_BURST_SIZE` 512 for both of us), no `_with_state` variants,
no VC pinning on the reads. So instead of reading further I instrumented **ttnn's own** in1
sender with our three zones and compared like for like:

| per sender core | DRAM wait | broadcast |
|---|---|---|
| ttnn in1 sender, 32 k-blocks | **49.3 us** | 68.6 us |
| ours, 8 k-blocks | **168.1 us** | 60.6 us |

Our broadcast was already FASTER. The entire gap was one number: the DRAM wait, 3.4x, for
the same 1MB per sender. Not the loop, not the handshake, not the flushes -- all of which
had been measured and cleared -- but the reads themselves, exactly as the ablation had
narrowed it to.

Which left the one property of a read that the kernel picks and nothing above had varied:
**which NOC it goes out on.** A DM thread is bound to a NOC by its index, so choosing the
thread chooses the NOC, and we had A on thread 0 and B on thread 1 for no reason beyond
"they should not serialise". All four assignments, same shape, same everything else:

| [512,2048]@[2048,2048], 8x8 mcast, kt=8 depth=3 | |
|---|---|
| A on NOC 1, B on NOC 0 | **155.6 us** |
| A on NOC 0, B on NOC 0 | 216.2 us |
| A on NOC 0, B on NOC 1 (what we had) | 308.3 us |
| A on NOC 1, B on NOC 1 | 403.5 us |

**2.6x between the best and worst arrangement of the same reads.** The ordering separates
into two independent effects. Which NOC carries **B** is worth ~1.4x on its own -- NOC 0 is
simply better for these DRAM reads -- and B is the big operand, 8MB against A's 2MB, so it
is the one that must have it. Given that, putting A on the *other* NOC is worth another
1.4x, which is the ordinary parallelism argument and the only part we had reasoned about.
We had optimised the small effect and got the large one backwards.

The sender zones confirm the mechanism rather than merely correlating with it:

| totals over all sender cores, one launch | A:0 B:1 | A:1 B:0 |
|---|---|---|
| MCAST-DRAM (wait for the reads) | 1702.8 us | **309.6 us** |
| MCAST-SEND (the broadcast) | 602.8 us | 580.6 us |
| MCAST-READY (the handshake) | 4.9 us | 4.9 us |

**5.5x less DRAM wait; the broadcast and handshake do not move**, which is what a NOC-routing
effect should look like and what a bandwidth or contention effect should not. Per sender that
is 19.4us against ttnn's 49.3 -- **our reads are now faster than ttnn's**, and the remaining
gap is somewhere else.

Re-sweeping on top of it, since the best kt was chosen under the old assignment:

| kt | depth 2 | depth 3 | depth 4 |
|---|---|---|---|
| 1 | 188.1 | 187.9 | 188.3 |
| 2 | 165.6 | 165.1 | 165.3 |
| 4 | 155.4 | 154.8 | 155.4 |
| **8** | 153.7 | **152.5** | 152.7 |
| 16 | 158.0 | 157.7 | 158.4 |

**152.5us against ttnn's 117.5us -- 1.30x, down from 2.62x.** The whole 30-launch sweep
now sits below where the single best configuration was an hour ago, and kt=8/depth=3 is
still the peak, so nothing about the blocking choice changed.

Two things this cost, both worth naming. The hypothesis was reachable at any point --
`MMB_IN1_THREAD` already existed as a knob, and flipping it is a one-line define -- and it
was not reached because every hypothesis for a day had been about *structure*: loop nesting,
accumulator residency, block size, flushes, handshake shape, memory layout. Each was
plausible, each was measured, each was wrong, and none of them was about the wire. What
finally pointed at it was not more thinking about our kernel but instrumenting *theirs* with
the same three zones, which turned "we are 2.6x slower" into "our DRAM wait is 3.4x, our
broadcast is fine" -- a statement narrow enough that only one unexamined variable was left.
Measure the reference, not just the thing you are optimising.

### The same rule holds in every kernel, and it is not a multicast artifact

The multicast path has two senders on two threads, so its NOC assignment could plausibly
have been about the two streams interfering rather than about the NOCs themselves. It is
not. The **non**-multicast path, where every core reads its own operands and nothing is
shared, splits exactly the same way:

| [512,2048]@[2048,2048], 16 cores, NO multicast, mt=4 kt=8 nt=8 | |
|---|---|
| A on NOC 1, B on NOC 0 | **435.9 us** |
| A on NOC 0, B on NOC 0 | 449.0 us |
| A on NOC 0, B on NOC 1 | 802.0 us |
| A on NOC 1, B on NOC 1 (what we had) | 1150.2 us |

**2.6x again, same ordering.** And the old 807.8us figure recorded in the prefetch-depth
section above is the A:0/B:1 row, so that whole measurement -- and the conclusion drawn from
it, that depth hurts without multicast because DRAM is saturated -- was taken on a
misconfigured NOC. The depth conclusion may or may not survive; it has not been re-taken.

So the rule was applied to every kernel that reads DRAM, all of which had reads on thread 1
and stores on thread 0 -- the wrong way round on the reads, and by chance the wrong way round
on the writes too:

| | reads 1 / writes 0 | reads 0 / writes 1 | reads 0 / writes 0 | reads 1 / writes 1 |
|---|---|---|---|---|
| rmsnorm [512,2048], 64 cores | 148.0 us | **62.5 us** | 91.5 us | 129.0 us |
| flash attention, 8 heads, 8 cores | 355.4 us | **300.0 us** | 309.6 us | 358.6 us |

**rmsnorm 2.4x, flash 1.18x**, and reads-on-0 wins in every kernel measured. The spread
tracks how bandwidth-bound each one is: rmsnorm is almost pure streaming and moves the most,
flash is latency- and compute-bound and moves the least, and the matmul sits between. Keeping
reads and writes on *different* threads still matters on top of that -- reads 0 / writes 0
loses to reads 0 / writes 1 in both -- but it is the smaller of the two effects, and it is
the only one we had been reasoning about.

The knobs are gone again rather than left behind as tuning surface: the answer is the same
everywhere, so the kernels spell the threads literally and the rule is stated once, next to
`noc_load` in api.h, where somebody choosing a thread will actually read it. The blocked
matmul keeps `MMB_IN0_THREAD`/`MMB_IN1_THREAD` because it genuinely has two read streams of
different sizes and the assignment between them is a real choice.

### Where the layer stands after the NOC fix

Every kernel that reads DRAM was on the wrong NOC, so the whole layer moves at once. At the
real llama-3.2-1B prefill shape -- S=512, d_model 2048, ffn 8192, 32 heads over 8 KV heads --
on the full 8x8 grid, bf16, each matmul at its own best kt/depth:

| stage | us | share |
|---|---|---|
| FFN gate + up, [512,2048]@[2048,8192] x2 | 1179.8 | 47% |
| FFN down, [512,8192]@[8192,2048] | 470.1 | 19% |
| flash attention, 32 heads causal | 336.1 | 13% |
| Q projection [512,2048]@[2048,2048] | 154.7 | 6% |
| output projection, same shape | 154.7 | 6% |
| K and V projections [512,2048]@[2048,512] x2 | 104.8 | 4% |
| rmsnorm x2 | 122.0 | 5% |
| **subtotal** | **2522.2** | |

against **ttnn's whole TransformerBlock at 1820us**, which also includes RoPE, the two
residuals and silu*up -- none of which are in the subtotal above. So this is not yet a like
for like layer number, and the honest statement is that the parts measured here are 1.39x
ttnn's *entire* layer, where before this session's changes the same parts were roughly 4700us
and the comparison was not worth making. The three unmeasured stages were 18% of the old
single-core breakdown, so a full layer is plausibly around 3000us, or 1.6x -- to be measured,
not claimed.

The FFN sweeps that produced the two numbers above, both on the strict 8x8 grid:

| FFN gate/up, mt=2 nt=32 | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| kt=1 | 826.2 | 633.4 | 633.9 |
| kt=2 | 780.3 | 609.2 | 603.0 |
| **kt=4** | 761.7 | **589.9** | 627.5 |
| kt=8 | 743.2 | 671.1 | past L1 |

| FFN down, mt=2 nt=8 | depth 2 | depth 3 |
|---|---|---|
| kt=8 | 480.0 | 483.8 |
| **kt=16** | 471.5 | **470.1** |
| kt=32 | 548.8 | past L1 |

**And the bottleneck has moved.** The sender zones now put MCAST-DRAM at 309.6us against
MCAST-SEND at 580.6: the reads that were the whole problem are no longer the larger half,
and the broadcast is. That inverts the priority of the two shelved items -- the begin/finish
overlap, whose measured ceiling was 17% when reads dominated, is now aimed at the bigger
term, and it should be re-bounded before anything else is tried.

The other lever is format. Everything here is bf16 and ttnn's 1820us is bfloat8_b, so ttnn
moves half the weight bytes we do. At 60-66 GB/s effective across these shapes the matmuls
are not near DRAM peak, so halving the bytes would not simply halve the time -- but it is the
one difference against ttnn that is arithmetic rather than a matter of tuning, and it is
worth roughly the whole remaining gap if the kernels are anywhere near bandwidth-bound.

### bfloat8_b weights: the layer's matmuls go 2522.2 -> 1929.1us

ttnn's 1820us layer is bfloat8_b and everything here was bf16, so ttnn moved half the weight
bytes we did. That difference is arithmetic rather than tuning, which made it the obvious next
thing.

**It needed no kernel change at all**, which is the part worth recording. B's circular buffer
is declared bfloat8_b on the host; `matmul_block_init` reconfigures the unpacker from the CB,
and `pages.page_bytes` comes from the CB too, so the gather loop reads 1088-byte pages without
naming a format anywhere. The one host-side fix was `DTYPE_TILE_BYTES`, which had no entry for
a block format -- bfloat8_b is 1088 bytes a tile, 1024 of mantissa plus a 64-byte exponent
section, not the 1024 an element count would suggest. The mixed-format machinery that existed
for the SFPU (test_unified_mixed_format.py, where a leaf's `reconfigure` flag is what makes a
two-format tree correct) turns out to cover the FPU path as well.

Accuracy barely moves: pcc 0.999967 -> 0.999957 on the small shapes, and on the real ones
0.999929 for the projection, 0.999995 for gate/up, 0.999917 for FFN down.

| [512, ...] on the 8x8 grid, A always bf16 | ours bf16 | ours bf8 | ttnn bf8 | ours/ttnn |
|---|---|---|---|---|
| projection @[2048,2048] | 153.2 | **113.5** | 88.6 | 1.28x |
| K/V projection @[2048,512] | 52.4 | **50.1** | 42.8 | 1.17x |
| FFN gate/up @[2048,8192] | 602.5 | **422.4** | 563.3 | **0.75x** |
| FFN down @[8192,2048] | 473.4 | **299.0** | 256.2 | 1.17x |

The best kt and depth are unchanged from bf16 in every case, which is worth knowing: halving
B's bytes does not shift the blocking choice, it just moves fewer of them.

**The gate/up row is not a fluke of ttnn's heuristic, and it is not flattering to us either.**
ttnn's gate/up is identical at bf16 and bf8 weights -- 560.7 against 563.3 -- which says
weights are not what binds it. What binds it is the OUTPUT: the shape writes 8MB where FFN
down, with exactly the same 262144 tile-MACs, writes 2MB, and giving ttnn a bfloat8_b output
takes it to 439.6us. So the comparison stands three ways: against ttnn writing the same 8MB
bf16 output we do we are 1.33x faster, and against ttnn writing HALF our output bytes we are
still slightly ahead at 422.4 to 439.6. Wide-N is a shape where the gather-store -- every page
issued, then one barrier -- is doing better than what ttnn's writer does.

The whole layer's measured part:

| stage | bf16 | bf8 weights |
|---|---|---|
| FFN gate + up x2 | 1179.8 | 844.3 |
| FFN down | 470.1 | 299.9 |
| flash attention, 32 heads | 336.1 | 336.6 |
| Q + output projections | 309.4 | 227.2 |
| K and V projections x2 | 104.8 | 100.1 |
| rmsnorm x2 | 122.0 | 121.1 |
| **subtotal** | **2522.2** | **1929.1** |

Attention and rmsnorm do not move, as they should not -- neither reads a weight matrix of any
size. The three stages ttnn's 1820us includes and this does not (RoPE, two residuals, silu*up)
are still unmeasured at this shape.

**And the next lever is now visible in ttnn's own numbers**: a bfloat8_b OUTPUT was worth
123.7us to ttnn on gate/up alone. Our outputs are all bf16. That is an activation-format
change rather than a weight-format one, so it changes what the next stage consumes and cannot
be done one matmul at a time -- but the FFN's intermediate is exactly the kind of value that
does not need bf16, and it is 8MB written and 8MB read back.

### bfloat8_b activations, and a packer bug that was waiting for them

Weights were the easy half. Activations mean the OUTPUT is bfloat8_b, and that is where the
library had a real bug: **the packer's output format is programmed once**, by
`compute_kernel_hw_startup` inside `matmul_init`, for ONE circular buffer -- while a blocked
matmul packs to two, the accumulator carrying partials and the output. With both bfloat16 that
never showed. With an output in a different format the second buffer is written using the
first one's packer, and there is no assert and no hang: the bytes land, and bfloat16 read back
as bfloat8_b comes out as 1.33e36.

ttnn reconfigures at exactly these transitions -- `PACK((pack_reconfig_data_format(...)))`
around its partials and output buffers in `bmm_large_block_zm_fused_bias_activation.cpp` --
and the fix is the same call at our four: the Dst-mode pack whose destination alternates, the
L1-mode pack into the accumulator, the L1 copy-out, and `bias_finish`. The two-argument form
is guarded inside the LLK on `pack_dst_format[old] != pack_dst_format[new]`, so in the uniform
case, which is every other kernel here, it costs a table lookup and a compare.

Worth stating plainly: this was latent in the library, not in the new code. Any kernel that
packed to two buffers of different formats was silently wrong, and only a test that varied the
output format could have found it -- which is why the suite now varies it rather than only
varying the weights.

**The accumulator stays bfloat16 and must.** The packer's L1 accumulate reads back what is
already at the destination and adds to it, which a shared-exponent block format cannot do in
place.

| [512, ...] on the 8x8 grid, at each format | all bf16 | W bf8 | W+out bf8 | W+A+out bf8 |
|---|---|---|---|---|
| projection @[2048,2048] | 152.8 | 113.3 | 93.6 | **93.7** |
| FFN gate/up @[2048,8192] | 605.9 | 423.0 | 345.4 | **346.7** |
| FFN down @[8192,2048] | 474.4 | 299.2 | 280.4 | **277.6** |

1.63x to 1.75x over bf16. **Almost all of the second step is the OUTPUT, not A**: 93.6 against
93.7 on the projection, and the same everywhere. That is what multicast implies -- A is read
from DRAM once for a whole grid row, so its format hardly matters, while the output is written
in full by every core. pcc holds at 0.999863 to 1.000000 on the real shapes.

The layer's measured part, against ttnn at the SAME formats (weights and output both
bfloat8_b, A bfloat8_b):

| stage | bf16 | W bf8 | all bf8 | ttnn, same formats | ours/ttnn |
|---|---|---|---|---|---|
| FFN gate + up x2 | 1179.8 | 844.3 | **692.8** | 834.0 | **0.83x** |
| FFN down | 470.1 | 299.9 | **277.5** | 229.9 | 1.21x |
| flash attention, 32 heads | 336.1 | 336.6 | 337.5 | -- | |
| Q + output projections | 309.4 | 227.2 | **186.8** | 142.6 | 1.31x |
| K and V projections x2 | 104.8 | 100.1 | **70.6** | 56.4 | 1.25x |
| rmsnorm x2 | 122.0 | 121.1 | 121.9 | -- | |
| **subtotal** | **2522.2** | **1929.1** | **1687.0** | | |

**1.50x over bf16 for the whole measured part**, and it now sits under ttnn's 1820us whole
layer -- which still is not like for like, because that 1820 includes RoPE, two residuals and
silu*up and this does not. The per-shape column is the honest comparison: 1.21x to 1.31x on
the three ordinary shapes and 0.83x on FFN gate/up, where the wide-N output plays to the
gather-store.

Attention and rmsnorm do not move at all across three format changes, which is the correct
result and a useful check that the harness is changing what it says it is: neither reads a
weight matrix, and their activations were left bfloat16.

### The whole layer in bfloat8_b, and the packer bug turning out to be general

Running the end-to-end layer at bfloat8_b failed at the FIRST stage -- rmsnorm, inf -- which
said immediately that the packer fix committed with the matmul was too narrow. It had been
placed at the four transitions a blocked matmul makes. But **rmsnorm packs to five buffers**
(the squares, the row mean, its reciprocal square root, the normalised block and the output),
and in general every `Strategy::run` in the library packs to one buffer that nothing had
promised was the one the packer was programmed for.

So the fix moved to where it belongs: one `pack_to(cb_id)` that names the destination, called
by every pass. It keeps the currently-configured buffer in a static local -- per-RISC state,
which is exactly the scope the packer's configuration has -- and is guarded twice. Nothing
happens when the destination has not changed, and when it has, the LLK's two-argument form
does nothing further if the two buffers agree on format. A kernel with one format throughout
pays a compare per pass, and the measurements say so: the bf16 projection is 155.4us against
152.8 before, rmsnorm 61.1 against 62.5, both inside the noise of the run.

That replaced the four hand-placed calls from the previous commit with nine call sites that
cover every pack path in the library -- SFPU tree, FPU eltwise, broadcast, reduce, single-shot
matmul, both accumulator modes, the L1 copy-out and `bias_finish`.

**The general statement is worth keeping**: in this library the packer's format is per-kernel
state that a pass has to claim, exactly like the unpacker's srcA/srcB, and the library already
knew that about the unpacker (`reconfigure` on an SFPU leaf, `reconfig_data_format_srca`
around every `matmul_block_init`). The packer was simply missed, and stayed missed because
until now nothing declared two buffers of different formats in one kernel.

With that, the whole layer runs in bfloat8_b -- weights and inter-stage activations both, all
eleven stages, four shapes:

| stage | bf16 | bf8 | ratio |
|---|---|---|---|
| rmsnorm | 0.00513 | 0.01146 | 2.2x |
| rope_q | 0.01033 | 0.02607 | 2.5x |
| rope_k | 0.01011 | 0.02553 | 2.5x |
| v_proj | 0.01179 | 0.02299 | 2.0x |
| attention | 0.02604 | 0.03271 | 1.3x |
| out_proj | 0.02951 | 0.03760 | 1.3x |
| residual | 0.00313 | 0.00930 | 3.0x |
| rmsnorm_ffn | 0.00579 | 0.01349 | 2.3x |
| silu_mul | 0.01917 | 0.04913 | 2.6x |
| **layer** | **0.00365** | **0.01369** | 3.8x |

Relative L2 per stage, against a torch reference, at S=64 d_model=256. **Every stage clears
the same 0.06 threshold bfloat16 is held to**, and the finished layer is 1.4% off. Both
formats now run in the suite, on all four shapes, because eleven stages compound and a format
that is fine on one matmul is not automatically fine in series -- which is the whole reason to
run this rather than infer it from the per-op numbers.

Two honest qualifications. `silu_mul` at 0.04913, and 0.05077 on the fourth shape, is the
stage nearest the threshold -- an SFPU pass on a product of two bfloat8_b operands, which is
where the format is worked hardest. And **flash attention and the output projection are NOT
covered by this**: both cross the host on the way in (`to_flash_layout`, and the projection's
torch weight), so they build their own bfloat16 tensors regardless of what the layer asked
for. That is a gap in the coverage, not a claim about them, and it closes when the strided
loads land and the host glue goes away.

What stays bfloat16 by construction, and should: the accumulator, every row statistic, the
RoPE rotation matrix and cos/sin tables, and each kernel's scalar constants. The accumulator
is not a preference -- the packer's L1 accumulate reads its destination back and adds in
place, which a shared-exponent format cannot do.

### Where perf stands, and what the packer change cost (nothing)

**The ubench first, because the packer change touches every compute pass in the library.**
`pack_to` was added to nine call sites, and the guard that makes it free in the uniform case
is an argument, not a measurement -- so it was measured, by running the full single-core
sweep against `tt/unified/math.hpp` checked out from the commit before any packer change:

| 231 shapes, one core, HiFi2 | |
|---|---|
| total across every shape | 4763.9 -> 4765.5us, **1.0003x** |
| median per-shape ratio | 1.0000 |
| shapes slower by >3% | 2 of 231 |
| shapes faster by >3% | 0 of 231 |
| holes found | 9 both sides, identical |
| best per-MAC cost | 0.060us both sides |

**No regression.** The two shapes past 3% are the smallest in the sweep -- 2.44 -> 2.53us and
2.69 -> 2.80us -- and the reason to call that jitter rather than cost is that a real per-pass
cost would grow with the number of passes, and this does the opposite:

| shapes grouped by size | median new/old |
|---|---|
| 1-16 MACs | 1.0000 |
| 17-128 MACs | 1.0000 |
| over 128 MACs | 1.0001 |

Absolute drift is +0.008us on shapes under 5us and +0.006us on shapes over 20us -- flat, which
is what launch jitter looks like and is not what a per-pass compare would look like.

**The layer.** Re-measured after `pack_to`, everything bfloat8_b, on the 8x8 grid:

| parallelised stage | us |
|---|---|
| FFN gate + up x2 | 692.8 |
| flash attention, 32 heads | 336.3 |
| FFN down | 277.5 |
| Q + output projections | 187.5 |
| rmsnorm x2 | 121.7 |
| K and V projections x2 | 67.2 |
| **subtotal** | **1683.0** |

1687.0 before the packer change and 1683.0 after, i.e. unmoved. Against **ttnn's whole
TransformerBlock at 1820us** -- still not like for like, and here is the size of what is
missing, which has not been stated numerically before:

| stage, STILL SINGLE CORE | us on one core | /64 if it parallelised perfectly |
|---|---|---|
| silu(gate) * up, 4096 tiles | 10315.7 | 161 |
| RoPE on Q, [512,2048] | 2412.0 | 38 |
| residual x2, 1024 tiles | 353.1 | 6 |
| **total** | **13080.8** | **204** |

Neither `rope.cpp` nor `binary.cpp` has a multi-core harness, which is the whole reason these
three are not in the subtotal. So the projection for a complete layer is **1683 + at best 204,
about 1890us against ttnn's 1820** -- roughly parity, and a projection rather than a
measurement, because perfect scaling is the optimistic end and these two kernels have never
been run on more than one core.

**Outcome: 430.5us, not 204, so a whole layer is 2113.5us.** Both kernels were parallelised
in the two sections below and the optimistic end was off by 2.1x.

**silu*up is now the single largest remaining item in the layer**, at 10.3ms of one core
against the 13.1ms those three stages cost together. It is one SFPU pass over 4096 tiles, and
partitioning it is the same shape of work as the row-chunking rmsnorm already has.

One correction worth recording: the first pass at this table measured `add` where it meant
`silu_mul` and reported 674us for that row. The real figure is 10315.7us, 15x larger, and the
reason it was caught is that it disagreed with the 10972us the old single-core breakdown had
recorded for the same stage. Cross-checking a new measurement against an old one that should
agree is worth the minute it costs.

### Parallelising silu*up: 10235.9 -> 262.9us

`binary.cpp` walked all `num_blocks` blocks on one core. Blocks already share nothing -- block
b reads pages `[b*tiles_per_block, +tiles_per_block)` of each input and writes the same range
of the output -- so partitioning them needs no reduction, no ordering and no communication,
only a `block_begin`/`block_count` pair per core, which is the same shape rmsnorm and
matmul_blocked already use. `num_blocks` stays a compile-time arg because it is what the host
divides, not what a core walks.

| silu*up, 4096 tiles ([512, 8192]), 64 blocks of 64 tiles | | |
|---|---|---|
| 1 core | 10234.9 us | 1.0x |
| 2 | 5126.0 | 2.0x |
| 4 | 2572.3 | 4.0x |
| 8 | 1297.6 | 7.9x |
| 16 | 674.5 | 15.2x |
| 32 | 409.6 | 25.0x |
| **64** | **265.2** | **38.6x** |

Linear to 16 cores, then falling off. **It is not falling off because of bandwidth**, which
was the obvious guess: at 64 cores this moves 24MB in 265us, 90GB/s, while the plain `add`
below reaches 184GB/s on the same kernel and the same grid. Per tile per core it is 4.14us
against add's 2.08us, so what limits silu*up at 64 cores is the SFPU pass itself, and the next
lever there is the activation rather than the partitioning.

| residual (add), 1024 tiles | | |
|---|---|---|
| 1 core | 209.3 us | 1.0x |
| 16 | 34.0 | 6.2x |
| 64 | 33.3 | 6.3x |

The add saturates at 16 cores and 6.3x is its ceiling, not a defect: 6MB in 33.3us is 184GB/s,
which is most of what this part has ever measured, and no number of cores moves bytes faster.

Correctness is checked EXACTLY rather than to a tolerance, which partitioning permits and
should therefore be held to: the same 16 blocks over 1, 3, 8 and 16 cores are bit-identical,
max|diff| 0.000000, and 3 into 16 covers the uneven split where an off-by-one in the range
arithmetic would live. At the full 4096-tile shape every core count from 1 to 64 also matches
one core bit for bit.

**Where the layer stands after it:**

| | us |
|---|---|
| matmuls, rmsnorm, attention (parallelised, bf8) | 1683.0 |
| silu*up | 262.9 |
| residual x2 | 66.7 |
| **parallelised total** | **2012.6** |
| RoPE on Q, STILL ONE CORE | 2411.4 |
| total as it actually runs today | 4424.0 |

against ttnn's 1820us. **RoPE is now the whole problem**: at 2411.4us on one core it costs more
than everything else in the layer put together, where before this change it was second to
silu*up. It is the same change -- `rope.cpp` walks a flat per-tile stream in chunks exactly as
`binary.cpp` walked blocks -- and it is the obvious next thing.

The earlier projection is worth marking against the outcome: it put these three stages at 204us
if they parallelised perfectly, and the two that are done came in at 329.6 rather than 167.
Perfect scaling was the optimistic end and it was labelled as such; the gap is the SFPU limit
on silu and the bandwidth ceiling on add, neither of which more cores can fix.

**And it hung the device twice on the way in, the same way as the last two times.** Adding
two runtime args to a kernel is a change to a contract that lives in three places -- the
kernel and every launcher of it -- and `binary.cpp` has three launchers: its own harness,
`test_unified_mixed_format.py` and the layer. Fixing the harness leaves the other two passing
the old three arguments, and the kernel then reads `block_count` out of whatever occupies that
slot, so the loop runs for a garbage number of iterations. It does not fail to compile and it
does not assert; the device hangs and wants `tt-smi -r`.

The lesson is not "remember to update the launchers", which is what was concluded the last two
times and evidently does not take. It is that **the arg list is a contract with no compiler
behind it**, and the fix is the named-kernel-argument work that was surveyed and shelved
earlier in this document -- Blaze's `blaze_rt_args::get<...>()`, which would make a missing
argument a build error rather than a hang. Three hangs from one cause is the argument for
picking it back up.

What did work: `grep -rn "binary.cpp" *.py` found all three launchers in one go, and running
it BEFORE fixing anything is the difference between one edit and two hangs.

### Parallelising RoPE, and the first complete layer number

Same change as `binary.cpp`, for the same reason: the rotation is PER TILE -- the pairing never
crosses a 32-element boundary, which is the whole reason this op fits the model as a matmul with
`kt_dim` 1 -- so chunk c depends on nothing outside its own tiles of x, cos and sin and writes
only its own tiles of the output. A `chunk_begin`/`chunk_count` pair, and every core reads the
one 2KB rotation tile.

| RoPE on Q, [512,2048] = 1024 tiles in 128 chunks of 8 | | |
|---|---|---|
| 1 core | 2295.6 us | 1.0x |
| 2 | 1151.0 | 2.0x |
| 4 | 578.7 | 4.0x |
| 8 | 292.9 | 7.8x |
| 16 | 155.9 | 14.7x |
| 32 | 95.5 | 24.0x |
| **64** | **67.2** | **34.1x** |

Bit-identical to the single-core result at every core count, and the suite checks 12 chunks over
2, 5 and 12 cores exactly -- 5 into 12 being the uneven split.

**Every stage of the layer is now parallelised, which it never has been before:**

| stage, 64 cores, bfloat8_b | us |
|---|---|
| FFN gate + up x2 | 692.8 |
| flash attention, 32 heads | 336.3 |
| FFN down | 277.5 |
| silu(gate) * up | 264.2 |
| Q + output projections | 187.5 |
| rmsnorm x2 | 121.7 |
| RoPE on Q and K | 99.4 |
| K and V projections x2 | 67.2 |
| residual x2 | 67.0 |
| **whole layer** | **2113.5** |

**Against ttnn's TransformerBlock at 1820us: 1.16x.** That is the first number in this document
that is actually like for like -- same shapes, same grid, same weight format, and every stage
of the layer counted rather than the convenient ones. For scale, the same layer was 78.3ms on
one core when the breakdown was first taken, and 4424.0us as recently as two commits ago with
RoPE still serial.

The four stages that were serial went 13080.8us -> 430.5us together. The projection made
earlier put them at 204us if they scaled perfectly and was labelled optimistic; the outcome is
430.5, so **the optimistic end was off by 2.1x** and the reasons are specific and measured:
silu*up is SFPU-limited past 16 cores, the residual add is at 184GB/s and cannot go faster,
and RoPE tops out at 34.1x rather than 64x. Recording the projection separately from the
measurement is what makes that comparison possible at all.

Where the remaining 293us to ttnn sits, by the per-shape ratios already measured: the four
matmuls are 1.21x to 1.31x of ttnn except gate/up at 0.83x, and attention and the eltwise
stages have no ttnn counterpart measured on the same footing yet. Nothing here is a mystery
of the kind the NOC asymmetry was -- it is ordinary per-shape tuning from here.

## API misuse audit

Moved to its own file: `unified_api_hazards.md`. A list of ways the API can be misused
without the misuse being visible from one thread on one core -- CB page counts, projection
and core uniformity, and the host contract -- with what is already enforced recorded
alongside, so it is not re-derived. Two entries are verified live rather than hypothetical.

## Phase 11 -- Full block orchestration

Host-side and kernel-loop work, not model gaps.

- [x] GQA head mapping (n_heads != n_kv_heads) -- see below
- [x] Multi-core work partitioning across heads -- see below
- [x] Head concat + output projection -- see below (two launches; fusing is the item above)
- [x] End-to-end single-layer prefill against a reference -- see below

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
