# Reduction startup and auxiliary operand metadata

Reduction kernels no longer select the startup source-B buffer by inspecting
`Call::algorithm`. The
[Moreh height sum](../operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp),
[height mean](../operations/moreh/moreh_mean/device/kernels/moreh_mean_h.cpp) and
[height bias-gradient](../operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_multi_core_h.cpp)
kernels, the
[shared helper test kernel](../../../../tests/ttnn/unit_tests/kernel_lib/reduce/kernels/reduce_plan_sequence.cpp),
the [runnable Python example](../../../ttnn/operations/examples/reduce_block/program_descriptor_with_inline_kernels.py),
its [README](../../../ttnn/operations/examples/reduce_block/README.md),
and the [public helper example](reduce_helpers_compute.hpp) now use
the two-argument `compute_kernel_hw_startup(input, output)` overload. It
initializes both source register configurations from the input buffer. The
reduction helper configures its actual operands before executing them.

## Metadata contract

Startup looks up the buffer's storage format, unpacked register format and, on
Wormhole/Blackhole, tile size and face geometry. It does not read an input tile
or permanently bind a source register to that buffer's address. Tile operations
still receive the actual operand buffers.

For the standard 32x32 tiles used by the changed factories, the planner already
creates BF16 auxiliaries for BF16 input and FP32 auxiliaries for FP32 input.
Their formats, tile sizes and face geometry match. The factories obtain the
auxiliary format and page size from the planner and use the same default 32x32
tile metadata as the input, so their CB creation needs no change.

Identical storage metadata cannot be a universal requirement:

| Input | Auxiliary storage | Reason |
| --- | --- | --- |
| BF16 | BF16 | Matching floating-point representation. |
| FP32 | FP32 | Matching floating-point representation. |
| BFLOAT8_B / BFLOAT4_B | BF16 | The auxiliary producer writes floating-point scalers and masks, not compressed tiles. |
| INT32 | BF16 | This is an unused protocol tile on the SFPU path; SFPU scaling does not consume a scaler tile. |

Relabeling a BF16 mask as compressed input would give it the wrong encoding and
page size. Instead, the helper switches the hardware configuration when it
actually reads an auxiliary buffer. The existing conditional old/new
reconfiguration overload avoids format writes when the relevant metadata
matches.

The simple startup convention requires input reconfiguration to be enabled,
as it is by default in host-planned calls. An explicit caller that disables it
must configure the actual operands itself. Startup and subsequent operands must
also have compatible tile/face geometry: ordinary format reconfiguration does
not establish an arbitrary different geometry. This change verifies the existing
32x32 factories and does not expand support for other tile shapes.

## Additive masks and zero tiles

The additive helper normally reads input/input. It also reads input/auxiliary
when folding a partial tile through a mask, or pairing an odd leftover input
tile with the auxiliary zero tile during accumulation.

Those transitions previously omitted source-B reconfiguration. A new numerical
regression reproduced the partial-mask failure with BFLOAT8_B input and a BF16
mask: all 128 output elements differed, with a maximum absolute difference of
70. The failure occurred after changing startup to the simpler overload, but
the omitted transition also exists with the original additive startup, which
likewise configures input/input.

The helper now reconfigures source B to the auxiliary buffer before the shared
partial-mask fold and all three zero-pair paths, then restores input metadata
before normal addition resumes. The zero-pair paths cover grouped H streams,
W streams and resident indexed input. Both the operation initialization and the
tile operation continue to receive the real input and auxiliary buffer IDs.

## Quasar

The startup simplification has no identified Quasar-specific obstacle in the
source implementation. Quasar startup programs source register formats. The
operation initialization programs separate buffer descriptors containing shape,
format and L1 base address. Consequently, switching from input/input to
input/auxiliary still requires the operation's initialization even when their
formats match. Matching metadata does not make distinct buffer addresses
interchangeable.

The helper retains the relevant `reduce_init`, binary initialization and unpack
initialization calls. Format reconfiguration is not being used as a substitute
for descriptor initialization. Quasar's descriptor allocator and its existing
wrap/reinitialization behavior are unchanged.

The output side also needs descriptor initialization. On Quasar,
`pack_reconfig_data_format` updates the format gasket but does not bind the output
buffer descriptor. Native `reduce_init` configures the reduction's pack mask but
does not bind that descriptor either. Each helper call now runs `pack_init` for
its actual output buffer on Quasar. This covers sequences switching from an
accumulator CB to the final output CB, including when their formats match. The
initialization is independent of the optional format-reconfiguration mode and is
compiled out on Wormhole and Blackhole.

The audit also found that native cleanup called `reduce_uninit()` without an
operand, implicitly passing buffer 0. Quasar's MXFP4 column-reduction cleanup
uses that operand's metadata to restore its temporary `MxFp4_2x_B` source-format
override. The helper now passes its actual input buffer ID. Wormhole's wrapper
ignores this argument, and Blackhole's cleanup does not use it.

The host planner currently excludes `AccumulateViaAdd` on Quasar, so its planned
reductions continue to use `ReduceTile`. This change does not enable the additive
backend there. Quasar conclusions are based on source inspection and existing
host planner checks; no Quasar device execution is claimed.

Useful source entry points:

* [Startup](../../../../tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h).
* [Additive auxiliary transitions](reduce_helpers_compute.inl),
  `reduce_accumulate_via_add`, including `fold_partial_last` and
  `CopySeedZeroPair`.
* [Quasar startup and descriptor programming](../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_unpack_common_api.h),
  `llk_unpack_hw_configure` and `llk_unpack_program_bfd`.
* [Quasar native reduction initialization](../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_unpack_AB_reduce_api.h),
  `llk_unpack_AB_reduce_init`.
* [Quasar binary operand initialization](../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_unpack_AB_api.h),
  `llk_unpack_AB_init`.
* [Packer reconfiguration contract](../../../../tt_metal/hw/inc/api/compute/reconfig_data_format.h),
  `pack_reconfig_data_format`, and
  [Quasar pack initialization](../../../../tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_tile_api.h),
  `llk_pack_init`.

## Validation

The new `test_reduce_helpers_mixed_auxiliary_format` has 20 cases: BFLOAT8_B and
BFLOAT4_B input, W and H reduction, resident and chunked inputs, odd-tile
accumulation with and without a partial mask, and standalone native partial
reductions. Inputs are exactly representable small integers, and outputs are
checked with zero tolerance. Accumulated cases also assert that the plan actually
selected a zero-pair reload. Repeated outputs and two batches exercise restoration
of input metadata after auxiliary reads.

Validation used the local native toolchain and the attached Wormhole N300:

* Direct build: passed, using
  `CMAKE_BUILD_PARALLEL_LEVEL=8 PYTHONDONTWRITEBYTECODE=1 ./build_metal.sh --enable-ccache --build-ttnn-tests`.
* Host planner: 21 tests passed, using
  `build/test/ttnn/unit_tests_ttnn --gtest_filter='ReduceHostPlanner.*'`.
* Complete helper suite: 223 tests passed, using
  `bash scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py -q --maxfail=1`.
* Smaller migration suite: all 63 cases passed (50 Python, 13 C++), using
  `python_env/bin/python scripts/run_reduce_migration_sanity.py --manifest /localdev/malimpic/reviews/pr56063-local-spec-20260911-oer_2vst/sanity_test_suite.json --lane common --lane wormhole --lane wormhole-n300 --output-dir /localdev/malimpic/reviews/pr56063-reduce-startup-20260911-KAECx1/sanity-results`.
* The final Quasar-only pack initialization passed the native build. With that
  change and the updated example, all 20 mixed-format regressions and all three
  affected Moreh cases passed again.
  The Moreh cases were selected with `--lane common --kernel S063 --kernel S065 --kernel S078`.
* The runnable example's `test_reduce_block_correctness` and
  `test_reduce_block_accumulate_partial_zero_pair` both passed. They cover native
  and additive dispatch across multiple shapes, repeated kernel iterations and
  accumulated partial reductions. These were run through `run_safe_pytest.sh`
  with `--no-precompile`, using their node IDs in
  `tests/ttnn/unit_tests/operations/examples/test_reduce_block.py`.
* `pre-commit run --files` passed for all ten changed files. The `.inl` changes
  were also checked explicitly with `git-clang-format`; `git diff --check` passed.

Test commands used `PYTHONDONTWRITEBYTECODE=1` and `TT_METAL_HOME` set to the
checkout. No Quasar kernel build or device test was run. Logs are retained in
`/localdev/malimpic/reviews/pr56063-reduce-startup-20260911-KAECx1/`:
`build-quasar-descriptor.log`, `host-tests-final.log`, `helper-tests-final.log`,
`sanity-results/summary.json`, `mixed-final.log`, `moreh-descriptor-results/summary.json`
and `example-final.log`.
