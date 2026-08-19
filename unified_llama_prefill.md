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

The flag already exists as a defaulted parameter on `matmul_init`
(`tt/unified/math.hpp:397`) and reaches `matmul_block_init`. Nothing ever passes
non-zero and `MatmulGeometry` has no field for it. Confirmed as the right
mechanism: ttnn's SDPA does its QK matmul with `true /*transpose*/` and its PV
matmul with `false` (`compute_common.hpp:1707` and `:1870`). No separate transpose
op is needed.

- [ ] Decide: a `Transpose` template parameter on `MatmulGeometry`, or a
      `.transposed()` method on `MatmulNode`. Geometry is likely cleaner -- it is
      compile-time, and every restore site already has `G` in hand.
- [ ] `tt/unified/math.hpp` -- **five** sites hardcode transpose, and every one of
      them must carry the geometry's value. Miss a *restore* site and the error
      only shows on the second output block, not the first:
      - `:400` `matmul_init` -> `matmul_block_init` (already has the parameter)
      - `:515` `bias_finish`'s restore
      - `:565` `Dst`-mode reload restore
      - `:574` `matmul_block` itself, in the k loop
      - `:678` `L1`-mode restore
- [ ] Both `constexpr uint32_t kTranspose = 0;` declarations (`:486`, `:553`) go away
- [ ] Test against `torch.matmul(q, k.transpose(-1, -2))`

**Done when:** a transposed matmul matches torch; a fused-bias transposed matmul
still matches; and a **multi-block** (`num_blocks > 1`) transposed matmul matches
in both `Dst` and `L1` modes -- that last one is the only thing that exercises the
reload restores at `:565` and `:678`.

## Phase 4 -- Data-format reconfig between SFPU tree leaves

Latent, not blocking phase 7. `TileSource::emit` issues a bare `copy_tile` with no
`reconfig_data_format`. Two-CB `a + b` is correct today only because every operand
in every current test is bf16. The moment a mask or scaler arrives in a different
format it is silently wrong.

- [ ] Decide: reconfig per leaf, or a static assert that all leaves share a format
- [ ] Implement whichever, with a test that would fail under the other

**Done when:** a deliberately mixed-format tree either works or fails to compile --
never silently produces garbage.

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

Softmax's `x - rowmax` and `x * recip` are `sub_tiles_bcast_cols` /
`mul_tiles_bcast_cols`. These read **two CBs**, so they do not fit the SFPU
expression tree, which fuses over DST slots of one loaded block. They fit the
shape already built for the fused bias: `Strategy<FPUFusion>::bias_finish`
(`tt/unified/math.hpp:482`) is a tile loop of `add_tiles_bcast_rows(acc_cb,
bias_cb, t, t % ct_dim, t)` with a format reconfig on entry and a restore on exit.

So: a fourth kind, `BcastFusion`, that is `bias_finish` generalized.

```
Dim::Rows    vector 1 x Wt    tile t pairs with  t % Wt    <- what bias already does
Dim::Cols    vector Ht x 1    tile t pairs with  t / Wt
Dim::Scalar  vector 1 x 1     tile t pairs with  0         <- the 1/sqrt(d) scale
```

The pairing with reduce falls out and is worth encoding in the types:
`reduce<Rows>` produces exactly what `bcast_rows` consumes, `reduce<Cols>` what
`bcast_cols` consumes.

- [ ] `tt/unified/math.hpp` -- `BcastDim`, geometry (reuse `ReduceGeometry`?),
      `BcastNode<Geometry, Dim, Op>`, `Strategy<BcastFusion>`
- [ ] `tt/unified/api.h` -- entry points taking (block, vector operand)
- [ ] `tt/unified/impl_v1.hpp` -- definitions
- [ ] Refactor the fused bias onto it and prove the selftest trace is **byte
      identical** before and after
- [ ] Assert the vector operand's page count matches the geometry, the way
      `.bias()` does (`tt/unified/math.hpp:190`)
- [ ] Standalone bcast test on device (rows, cols, scalar)

**Done when:** bias is a special case of bcast with an unchanged trace, and all
three dims match torch.

**Known limitation, inherited:** like `.bias()`, the vector operand will be
duck-typed on `get_cb_id()` / `get_num_pages()`, so a loop-scoped `ComputeBlock`
passes the size assert and is then popped early. Only a distinct resident type
would catch it -- deliberately rejected before, worth revisiting if it bites.

## Phase 7 -- Attention kernel + test

No library change if 1-5 land.

- [ ] `unified_kernels/attention.cpp` -- Q@Kt (transpose) -> scale (bcast scalar)
      -> + mask (two-CB add) -> row max (`reduce_max<Cols>`) -> `- max`
      (bcast cols) -> `exp_` -> row sum (`reduce_sum<Cols>`) -> `recip` ->
      `* recip` (bcast cols) -> @V
- [ ] `test_unified_attention.py` -- against `F.scaled_dot_product_attention`
- [ ] Verify the resident-`ComputeBlock` idiom holds `x` alive across both the
      reduce and the bcast that consumes the same CB
- [ ] **PCC is not enough here.** Rows sum to 1 and every value is ~1/S, so a
      global scale error or a per-row offset sails through -- exactly like the
      bias and mean cases did. Add max-abs-error **and** an explicit
      row-sum-equals-1 check.
- [ ] Prove both new checks fail under deliberate sabotage
- [ ] Measure `exp_tile` default vs approx accuracy over a masked row before
      assuming the default is fine
- [ ] `git add -f` the test (`.gitignore:25` has `/test_*`)

**Done when:** matches torch within a stated tolerance, and every guard has been
shown to fail when it should.

## Phase 8 -- RMSNorm

Falls out of phases 1, 2 and 5 with no new library work:
`reduce_mean<Cols>` -> `rsqrt` -> `mul_bcast_cols` -> `mul` by the weight.

- [ ] `unified_kernels/rmsnorm.cpp` + test

## Phase 9 -- RoPE

Also no new library work once phase 2 lands. ttnn's
`rotary_embedding_llama.cpp` is 167 lines and uses only `mul_tiles`, `add_tiles`
and a matmul against a 32x32 `trans_mat` for the rotate-half.

- [ ] `unified_kernels/rope.cpp` + test, including the `trans_mat` construction

## Phase 10 -- Flash chunking / online softmax

The first genuinely new model concept. S=2048 scores are 8MB, far past L1, so real
prefill must stream K/V in chunks and rescale a running max and running sum across
them. Nothing in the model is stateful across blocks in that way today --
`Accumulator` holds a running total, but not a running *statistic* that
retroactively rescales what came before.

- [ ] Design the running-statistic idiom (an `OnlineSoftmax` alongside `Accumulator`?)
- [ ] Chunked K/V streaming in the kernel
- [ ] Test at S large enough that the non-flash path cannot fit

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
