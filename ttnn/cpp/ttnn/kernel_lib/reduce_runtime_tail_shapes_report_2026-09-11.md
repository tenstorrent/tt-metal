# Runtime valid shapes for reduction tail cores

This change adds an opt-in tail specialization to the reduction planner. The
factory assigns the static plan to full cores and the tail plan to tail cores.
Only tail kernels read a runtime shape. Several tail cores can share the same
compiled kernel while receiving different shapes.

The shape is **height, width, batches**, with height and width in elements and
leading dimensions flattened into batches. It replaces the planned logical work
within the planned bounds. It can shorten the reduced dimension, the non-reduced
dimension, and the number of batches. It is not a count of valid reduction lanes.

## Host API and integration

The public types and planner entry points are in
[host/reduce_host.hpp](host/reduce_host.hpp), with matching Python bindings in
[reduce_planner_nanobind.cpp](../operations/reduction/reduce_planner_nanobind.cpp).

| API | Meaning |
| --- | --- |
| `ReduceBlockSpec::tail` absent | Use the existing static shape; no runtime shape arguments. |
| `ReduceBlockSpec::tail = ReduceTailConfig{compute_offset, auxiliary_offset}` | Plan a tail specialization. The two offsets locate its three shape words in the respective kernels. |
| `ReduceValidShape{height, width, batches}` | Actual nonempty local work on one tail core. |
| `ReducePlan::get_runtime_shape_args(shape)` | Validate all three dimensions against the planned bounds and return `[height, width, batches]`. |

For example, a factory using resident input can start with:

```cpp
auto full = ReduceBlockSpec::tiled(256, 256, DataType::BFLOAT16, DataType::FLOAT32, 2);
full.resident_input_tiles = 128;
full.resident_output_tiles = 16;

auto tail = full;
tail.tail = ReduceTailConfig{
    .compute_runtime_arg_offset = static_cast<std::uint32_t>(compute_prefix.size()),
    .auxiliary_runtime_arg_offset = static_cast<std::uint32_t>(reader_prefix.size()),
};
auto plan = make_reduce_plan(
    tail, ReduceOpMath::SUM, ReduceOpDim::W, 1.0F, ReduceFp32Mode::Fast, hardware);
auto shape_words = plan.get_runtime_shape_args({65, 135, 2});
compute_prefix.insert(compute_prefix.end(), shape_words.begin(), shape_words.end());
reader_prefix.insert(reader_prefix.end(), shape_words.begin(), shape_words.end());
```

The factory supplies those runtime vectors to the compute and auxiliary-producing
kernels on that tail core. It supplies ordinary static plans to full cores. Core
assignment still belongs to the factory; the local reduction planner does not own
a `CoreRangeSet`. Runtime vector sizes remain fixed when a cached program is
updated; update the three values between dispatches.

The compute call remains `reduce<Call>()`. Kernels do not calculate partial element
counts or select masks themselves. `Call::is_tail` and the runtime offset are
decoded from the planner's compile-time record. Full specializations compile out
the runtime shape reads and output masking branch.

## Geometry, output and normalization

For resident inputs, shrinking the valid shape preserves the planned physical row
pitch and batch pitch. A shorter batch must not move the next batch's starting
address. The low-level `ReduceInputMemoryLayout` therefore also carries an optional
batch stride. Whole tiles outside the runtime shape are never reduced.

Outputs are packed compactly in batch order: `ceil(height/32)` tiles per batch for
W reduction, or `ceil(width/32)` for H reduction. Unused output pages are untouched.
Invalid lanes in the final non-reduced output tile are cleared **after** the
caller's post-reduction callback. The mask uses selection, so NaNs in those invalid
lanes become zero as well. Writers and subsequent fused work must use the actual
output shape. Accumulated calls must have matching runtime output shapes, just as
static accumulated calls must have matching static outputs.

SUM retains the caller's scalar, including a scalar that normalizes a larger,
distributed reduction. A standalone AVG preserves `scalar * nominal_reduced_extent`
as its multiplier and divides by the runtime reduced extent. For example, scalar
`1/256` with nominal width 256 produces a mean over runtime width 135. Accumulated
runtime-tail AVG is explicitly rejected: its normalization needs the union of
the calls' runtime extents. Such callers can accumulate SUM and normalize after
the final call.

## Auxiliary planning and memory

Marking a block as a tail reevaluates the recipe even if the nominal block is
tile-aligned:

* Native ReduceTile uses a full scaler, a runtime reduction-edge scaler and a
  runtime output-edge mask: three auxiliary tiles for an aligned nominal block.
* AccumulateViaAdd uses a runtime reduction-edge mask and a runtime output-edge
  mask: two auxiliary tiles for an aligned nominal block.
* An aligned runtime edge produces a full-width/full-height mask. The reduction
  edge recipe stays active, so its location does not depend on runtime parity.
* H planning reserves one DEST slot for the output mask and reduces its output
  group accordingly.
* Auxiliary recipe sharing compares runtime argument sources as well as tile
  patterns and values. Equal descriptors that read different shape arguments
  cannot share a mask.
* The planner counts the additional tiles in its L1 budget. If an additive plan
  falls back to the native algorithm, it recalculates input capacity against the
  larger scaler recipe. Sequence planning also checks the aggregate recipe's
  residency against each call's budget.
* Tail calls use the ordinary accumulator reload, avoiding a static parity-based
  zero-pair optimization when parity is only known at runtime.

Each serialized compute call gains three compile-time words: runtime shape offset
(or a sentinel for static calls), nominal logical height and nominal logical
width. Each auxiliary tile record gains its optional runtime extent offset.
Host and device share these layouts in
[reduce_plan_args_common.hpp](reduce_plan_args_common.hpp).

## Streaming contract

Tail FIFOs use `ChunkedWaitChunkedPop` with **fixed-size packets**. A packet always
contains `chunk.reduce_axis_tiles * chunk.output_tiles` pages, including the last
partial packet. Its unused tile slots contain arbitrary padding that compute
skips. The final valid reduction tile still receives its element-level mask.

For W reduction, the reader emits batch, output row, axis packet, tile within the
packet. For H reduction it emits batch, output-column group, axis packet, row
within the packet, column within the group. H packets keep the planned column
pitch even in the last, smaller output group.

Compute waits for and pops complete packets while evaluating only valid tiles.
The allocation holds an integer number of packets, preserving circular-buffer
wrap boundaries for every runtime shape. Readers ported to this option must
follow this packet contract; simply sending a shorter existing bulk stream is
insufficient. Resident inputs continue to use indexed access instead.

## Useful starting points

1. [The multicore tail test](../../../../tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py),
   `test_reduce_runtime_tail_cores`: one static core and two cores sharing a tail
   kernel, different per-core shapes, separate compute/reader prefixes, preserved
   physical strides and unused output pages. It also exercises two-call SUM/MAX
   accumulation. This is the clearest factory-side example.
2. [The planned compute overload](reduce_helpers_compute.inl), `reduce<Call>()`:
   reads the runtime shape only for tails, preserves resident pitches, corrects
   standalone AVG normalization and clears non-reduced output padding.
3. [Host auxiliary configuration](host/reduce_host.cpp),
   `configure_scalar_and_aux`: explains why a nominally aligned tail needs more
   auxiliary storage. `same_auxiliary_tile` shows how runtime sources affect
   sharing.
4. [Auxiliary materialization](reduce_helpers_dataflow.inl), `prepare_tile`:
   computes each edge mask from the runtime shape and the planned tile extent.
5. [The streaming test reader](../../../../tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_tail_stream_reader.cpp)
   and `test_reduce_runtime_tail_stream_wraps`: a real producer/consumer FIFO that
   repeatedly wraps a small allocation, with both partial axis packets and
   partial output groups.

## Scope and remaining constraints

Runtime tails currently support tiled W/H reductions with standard 32x32 input
and output tiles on the existing native and additive paths. The planner rejects
dense row-major, HW, INT32/SFPU and accurate FLOAT32/SFPU tail requests: these need
masking capabilities the current corresponding backends do not provide. Empty
cores should not issue a reduction. Accumulated AVG requires the explicit
normalization described above. Existing operations have not been automatically
opted into the new contract.

An existing Metal allocation-order issue surfaced when the test allocated a
shared scratch CB **after** different-sized auxiliary CBs on overlapping core
ranges. Tail-core accumulation was corrupted; using a tensor-backed accumulator,
or allocating the shared scratch CB first, made the same numerical test pass.
The test factory uses the latter order. The reduction change does not modify
Metal's allocator. The overlap handling in
[ProgramImpl::allocate_circular_buffers](../../../../tt_metal/impl/program/program.cpp)
is a separate improvement worth reviewing before factories rely on arbitrary CB
declaration order with differing per-group auxiliary sizes.

## Validation

Hardware validation uses the attached Wormhole N300. No performance claims are
made.

* Native C++ and Python bindings build passed with
  `CMAKE_BUILD_PARALLEL_LEVEL=8 PYTHONDONTWRITEBYTECODE=1 ./build_metal.sh --enable-ccache --build-ttnn-tests`.
  This used the installed toolchain directly.
* `build/test/ttnn/unit_tests_ttnn --gtest_filter='ReduceHostPlanner.*'`:
  **21 host tests passed**, including seven new tail planning, serialization and
  memory-budget tests.
* `bash scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py -q --maxfail=1`:
  **203 device tests passed**: 151 existing cases and 52 tail cases. The new cases
  comprise 32 multicore resident/accumulation configurations and 20 streaming
  configurations, each using partial, aligned and sub-face runtime shapes.
  Streaming covers both BFLOAT16 and FLOAT32 input/auxiliary formats; resident
  cases cover both BFLOAT16 and FLOAT32 destination accumulation.
* The smaller migration suite passed **all 63 cases**, with zero failed groups,
  using `scripts/run_reduce_migration_sanity.py --lane common --lane wormhole --lane wormhole-n300`.
  Its archived `--manifest` was
  `/localdev/malimpic/reviews/pr56063-local-spec-20260911-oer_2vst/sanity_test_suite.json`;
  results are in the log directory's `sanity-results/summary.json`. This covers
  the existing migrated operation kernels, including Moreh, LayerNorm, softmax,
  SDPA, examples and distributed post-allgather LayerNorm.
* Repository pre-commit hooks passed for the changed sources and report.

Logs are retained under
`/localdev/malimpic/reviews/pr56063-tail-shapes-20260911-T5gFfZ/`.
