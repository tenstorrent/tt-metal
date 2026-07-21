# Source files for ttnn_op_experimental_quasar.
# Module owners should update this file when adding/removing/renaming source files.
#
# This library holds standalone Quasar (metal 2.0) copies of selected ops. Each op is a
# near-identical clone of its original, isolated in the ttnn::prim::qsr (device) and
# ttnn::operations::experimental::quasar (host) namespaces so it can coexist with the original.
# (pool_generic uses ttnn::operations::pool::quasar for the host side.)
# Nanobind sources (*_nanobind.cpp) are compiled with the python module via ttnn/sources.cmake.

set(TTNN_OP_EXPERIMENTAL_QUASAR_API_HEADERS
    pad/pad.hpp
    tilize/tilize.hpp
    move/move.hpp
    untilize_with_unpadding/untilize_with_unpadding.hpp
    slice/slice.hpp
    transpose/transpose.hpp
    reshard/reshard.hpp
    halo/halo.hpp
    pool_generic/generic_pools.hpp
    conv2d/conv2d.hpp
    matmul/matmul.hpp
    binary_ng/types.hpp
    binary/binary.hpp
    binary/binary_composite.hpp
    fold/fold.hpp
    interleaved_to_sharded/interleaved_to_sharded.hpp
    sharded_to_interleaved/sharded_to_interleaved.hpp
    to_memory_config/to_memory_config_op.hpp
    reshape_view/reshape.hpp
    untilize/untilize.hpp
    tilize_with_val_padding/tilize_with_val_padding.hpp
    to_layout/to_layout_op.hpp
    reallocate/reallocate.hpp
    reduction/generic/generic_reductions.hpp
    to_device/to_device.hpp
    typecast/typecast.hpp
)

set(TTNN_OP_EXPERIMENTAL_QUASAR_SRCS
    # pad
    pad/pad.cpp
    pad/device/pad_device_operation.cpp
    pad/device/pad_rm_reader_writer_multi_core_program_factory.cpp
    pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp
    pad/device/pad_rm_reader_writer_program_factory.cpp
    pad/device/pad_rm_sharded_height_only_program_factory.cpp
    pad/device/pad_rm_sharded_width_only_program_factory.cpp
    pad/device/pad_tile_multicore_program_factory.cpp
    pad/device/pad_tile_program_factory.cpp
    # tilize
    tilize/tilize.cpp
    tilize/device/tilize_device_operation.cpp
    tilize/device/tilize_multi_core_block_program_factory.cpp
    tilize/device/tilize_multi_core_default_program_factory.cpp
    tilize/device/tilize_multi_core_sharded_program_factory.cpp
    tilize/device/tilize_multi_core_width_sharded_program_factory.cpp
    tilize/device/tilize_single_core_program_factory.cpp
    # untilize_with_unpadding
    untilize_with_unpadding/untilize_with_unpadding.cpp
    untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp
    untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp
    untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp
    untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp
    untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp
    untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp
    untilize_with_unpadding/device/factories/untilize_with_unpadding_single_core_program_factory.cpp
    # slice
    slice/slice.cpp
    slice/device/slice_device_operation.cpp
    slice/device/slice_program_factory_rm.cpp
    slice/device/slice_program_factory_rm_sharded.cpp
    slice/device/slice_program_factory_rm_stride.cpp
    slice/device/slice_program_factory_tile.cpp
    slice/device/slice_program_factory_tile_tensor_args.cpp
    # transpose
    transpose/transpose.cpp
    transpose/device/transpose_device_operation.cpp
    transpose/device/transpose_utils.cpp
    transpose/device/transpose_cn_program_factory.cpp
    transpose/device/transpose_hc_rm_program_factory.cpp
    transpose/device/transpose_hc_sharded_program_factory.cpp
    transpose/device/transpose_hc_tiled_interleaved_program_factory.cpp
    transpose/device/transpose_hc_tiled_program_factory.cpp
    transpose/device/transpose_wh_program_factory.cpp
    transpose/device/transpose_wh_sharded_program_factory.cpp
    transpose/device/transpose_wh_sharded_rm_program_factory.cpp
    # reshard
    reshard/reshard.cpp
    reshard/device/reshard_device_operation.cpp
    reshard/device/reshard_program_factory_generic.cpp
    reshard/device/reshard_program_factory_same_height.cpp
    reshard/device/reshard_program_factory_same_width.cpp
    reshard/device/nd_reshard_program_factory_copy_pages.cpp
    reshard/device/nd_reshard_program_factory_copy_local.cpp
    # move
    move/move.cpp
    move/device/move_device_operation.cpp
    move/device/move_overlap_program_factory.cpp
    move/device/move_program_factory.cpp
    move/device/move_sharded_program_factory.cpp
    # halo (no nanobind; internal op)
    halo/halo.cpp
    halo/device/halo_device_operation.cpp
    halo/device/untilize_with_halo_program_factory.cpp
    # pool_generic
    pool_generic/generic_pools.cpp
    pool_generic/device/pool_op.cpp
    pool_generic/device/pool_multi_core_program_factory.cpp
    # conv2d
    conv2d/conv2d.cpp
    conv2d/device/conv2d_device_operation.cpp
    conv2d/device/conv2d_op_sharded_program_factory.cpp
    conv2d/device/conv2d_op_width_sharded_program_factory.cpp
    # binary_ng (device backend; no host op / no nanobind)
    binary_ng/device/binary_ng_device_operation.cpp
    binary_ng/device/binary_ng_program_factory.cpp
    binary_ng/device/binary_ng_metal_v2_factory.cpp
    binary_ng/device/binary_ng_utils.cpp
    # binary (host front-end: add/subtract/multiply/... -> quasar binary_ng device op)
    binary/binary.cpp
    binary/common/binary_op_utils.cpp
    binary/device/binary_composite_op.cpp
    # fold (compositional: own device op + internal pad/transpose/slice/reshard -> quasar)
    fold/fold.cpp
    fold/device/fold_device_op.cpp
    fold/device/fold_multi_core_program_factory.cpp
    fold/device/fold_multi_core_dram_program_factory.cpp
    # to_memory_config trio (interleaved_to_sharded + sharded_to_interleaved + to_memory_config dispatcher)
    interleaved_to_sharded/interleaved_to_sharded.cpp
    interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
    sharded_to_interleaved/sharded_to_interleaved.cpp
    sharded_to_interleaved/device/sharded_to_interleaved_device_operation.cpp
    sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
    to_memory_config/to_memory_config_op.cpp
    # matmul
    matmul/matmul.cpp
    matmul/device/matmul_device_operation.cpp
    matmul/device/config/matmul_program_config.cpp
    matmul/device/utilities/matmul_utilities.cpp
    matmul/device/sparse/sparse_matmul_device_operation.cpp
    matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp
    matmul/device/factory/matmul_multicore_program_factory.cpp
    matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.cpp
    matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp
    matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp
    matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp
    matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp
    # reshape_view
    reshape_view/reshape.cpp
    reshape_view/reshape_common.cpp
    reshape_view/device/reshape_device_operation.cpp
    reshape_view/device/reshape_rm_program_factory.cpp
    reshape_view/device/reshape_tiled_program_factory.cpp
    # untilize
    untilize/untilize.cpp
    untilize/device/untilize_device_operation.cpp
    untilize/device/untilize_device_operation_types.cpp
    untilize/device/factories/untilize_multi_core_block_program_factory.cpp
    untilize/device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp
    untilize/device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp
    untilize/device/factories/untilize_multi_core_nd_shard_input_program_factory.cpp
    untilize/device/factories/untilize_multi_core_parallelize_column_program_factory.cpp
    untilize/device/factories/untilize_multi_core_program_factory.cpp
    untilize/device/factories/untilize_multi_core_sub_core_grids_program_factory.cpp
    untilize/device/factories/untilize_single_core_program_factory.cpp
    # tilize_with_val_padding
    tilize_with_val_padding/tilize_with_val_padding.cpp
    tilize_with_val_padding/device/tilize_with_val_padding_device_operation.cpp
    tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.cpp
    tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp
    tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_default_program_factory.cpp
    tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_sharded_program_factory.cpp
    tilize_with_val_padding/device/factories/tilize_with_val_padding_single_core_program_factory.cpp
    # to_layout (composite host op; no device op / kernels)
    to_layout/to_layout_op.cpp
    # reallocate (thin wrapper over quasar move; no device op / kernels)
    reallocate/reallocate.cpp
    # reduction/generic (internal op; pool_sum used by quasar avg_pool2d — no nanobind)
    reduction/generic/generic_reductions.cpp
    reduction/generic/device/common.cpp
    reduction/generic/device/reduce_op.cpp
    reduction/generic/device/reduce_op_device_operation.cpp
    reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp
    reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp
    reduction/generic/device/reduce_op_single_core_hw_program_factory.cpp
    reduction/generic/device/welford_reduce_device_operation.cpp
    reduction/generic/device/welford_reduce_program_factory.cpp
    # to_device (thin host->device transfer wrapper; no device op / kernels)
    to_device/to_device.cpp
    # typecast (public ttnn.experimental.quasar.typecast API and internal quasar pad dependency;
    # copy of operations/copy/typecast with a device op + 3 CB program factories.
    # DFB/metal2 kernel port is a follow-up.)
    typecast/typecast.cpp
    typecast/device/typecast_device_op.cpp
    typecast/device/typecast_program_factory.cpp
    typecast/device/typecast_rm_chunked_program_factory.cpp
    typecast/device/typecast_sharded_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/quasar/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_QUASAR_NANOBIND_SRCS
    quasar_nanobind.cpp
    pad/pad_nanobind.cpp
    tilize/tilize_nanobind.cpp
    move/move_nanobind.cpp
    untilize_with_unpadding/untilize_with_unpadding_nanobind.cpp
    slice/slice_nanobind.cpp
    transpose/transpose_nanobind.cpp
    reshard/reshard_nanobind.cpp
    pool_generic/generic_pools_nanobind.cpp
    conv2d/conv2d_nanobind.cpp
    matmul/matmul_nanobind.cpp
    binary/binary_nanobind.cpp
    fold/fold_nanobind.cpp
    to_memory_config/to_memory_config_nanobind.cpp
    reshape_view/reshape_nanobind.cpp
    untilize/untilize_nanobind.cpp
    tilize_with_val_padding/tilize_with_val_padding_nanobind.cpp
    to_layout/to_layout_nanobind.cpp
    reallocate/reallocate_nanobind.cpp
    to_device/to_device_nanobind.cpp
    typecast/typecast_nanobind.cpp
)
