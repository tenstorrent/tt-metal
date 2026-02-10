add_library(ttnn_op_data_movement ${LIB_TYPE})
add_library(TTNN::Ops::DataMovement ALIAS ttnn_op_data_movement)

target_precompile_headers(ttnn_op_data_movement REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_data_movement)
set_target_properties(
    ttnn_op_data_movement
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    bcast/device/kernels/*
    clone/device/kernels/*
    concat/device/kernels/*
    common/kernels/*
    fill_pad/device/kernels/*
    fill_rm/device/kernels/*
    fold/device/kernels/*
    gather/device/kernels/*
    copy/device/kernels/*
    indexed_fill/device/kernels/*
    moe_expert_token_remap/device/kernels/*
    moe_routing_remap/device/kernels/*
    non_zero_indices/device/kernels/*
    move/device/kernels/*
    pad/device/kernels/*
    permute/device/kernels/*
    reshape_on_device/device/kernels/*
    repeat/device/kernels/repeat_*_rm.cpp
    reshape_view/device/device/*
    scatter/device/kernels/*
    sharded/device/kernels/*
    sharded/reshard/device/kernels/*
    sort/device/kernels/*
    split/device/kernels/*
    slice/device/kernels/*
    tilize/device/kernels/*
    tilize_with_val_padding/device/kernels/*
    transpose/device/kernels/*
    untilize/device/kernels/*
    untilize_with_unpadding/device/kernels/*
)
target_sources(
    ttnn_op_data_movement
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            bcast/bcast.hpp
            bcast/bcast_types.hpp
            concat/concat.hpp
            copy/copy.hpp
            copy/device/copy_device_operation.hpp
            copy/device/copy_program_factory.hpp
            copy/device/copy_device_operation_types.hpp
            fill_pad/fill_pad.hpp
            fill_pad/device/fill_pad_device_operation.hpp
            fill_rm/fill_rm.hpp
            fill_rm/device/fill_rm_device_operation.hpp
            move/device/move_device_operation.hpp
            pad/pad.hpp
            permute/permute.hpp
            repeat/repeat.hpp
            reshape_view/reshape.hpp
            reshape_view/reshape_common.hpp
            sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.hpp
            sharded/interleaved_to_sharded/interleaved_to_sharded.hpp
            sharded/reshard/reshard.hpp
            sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp
            sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp
            sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp
            slice/slice.hpp
            tilize_with_val_padding/tilize_with_val_padding.hpp
            tilize_with_val_padding/device/tilize_with_val_padding_device_operation.hpp
            transpose/transpose.hpp
            untilize/untilize.hpp
            untilize_with_unpadding/untilize_with_unpadding.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            # Existing explicit kernel entries
            reshape_view/device/device/dataflow/reader_reshape_tiled.cpp
            reshape_view/device/device/dataflow/writer_reshape_tiled.cpp
            reshape_view/device/device/rm_reshape_interleaved.cpp
            reshape_view/device/hostdevcommon/common.hpp
            tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp
            untilize/device/kernels/dataflow/reader_unary_start_id.cpp
            untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp
            untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp
            tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp
            tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp
            untilize/device/kernels/compute/common.cpp
            untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp
            tilize/device/kernels/compute/tilize_wh.cpp
            untilize/device/kernels/compute/pack_untilize.cpp
            untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp
            untilize/device/kernels/compute/pack_untilize_wh.cpp
            # bcast/
            bcast/bcast_nanobind.hpp
            bcast/device/bcast_device_operation.hpp
            bcast/device/bcast_device_operation_types.hpp
            bcast/device/bcast_multi_core_h_program_factory.hpp
            bcast/device/bcast_multi_core_hw_program_factory.hpp
            bcast/device/bcast_multi_core_w_program_factory.hpp
            bcast/device/bcast_sharded_h_optimised_program_factory.hpp
            bcast/device/bcast_sharded_h_program_factory.hpp
            # chunk/
            chunk/chunk.hpp
            chunk/chunk_nanobind.hpp
            # clone/
            clone/clone.hpp
            clone/clone_nanobind.hpp
            clone/device/clone_device_operation.hpp
            # common/
            common/common.hpp
            # concat/
            concat/concat_nanobind.hpp
            concat/device/concat_device_operation.hpp
            concat/device/concat_device_operation_types.hpp
            concat/device/concat_program_factory.hpp
            concat/device/concat_s2i_program_factory.hpp
            concat/device/concat_s2s_multi_program_factory.hpp
            concat/device/concat_s2s_rm_program_factory.hpp
            concat/device/concat_s2s_tiled_program_factory.hpp
            # copy/
            copy/copy_nanobind.hpp
            # data_movement nanobind
            data_movement_nanobind.hpp
            # expand/
            expand/expand.hpp
            expand/expand_nanobind.hpp
            # fill_pad/
            fill_pad/fill_pad_nanobind.hpp
            fill_pad/device/fill_pad_device_operation_types.hpp
            fill_pad/device/fill_pad_program_factory.hpp
            # fill_rm/
            fill_rm/fill_rm_nanobind.hpp
            fill_rm/device/fill_rm_device_operation_types.hpp
            fill_rm/device/fill_rm_program_factory.hpp
            # fold/
            fold/fold.hpp
            fold/fold_nanobind.hpp
            fold/device/fold_device_op.hpp
            # gather/
            gather/gather.hpp
            gather/gather_nanobind.hpp
            gather/device/gather_device_operation.hpp
            gather/device/gather_device_operation_types.hpp
            gather/device/gather_program_factory.hpp
            gather/tosa/gather_tosa.hpp
            gather/tosa/gather_tosa_nanobind.hpp
            # indexed_fill/
            indexed_fill/indexed_fill.hpp
            indexed_fill/indexed_fill_nanobind.hpp
            indexed_fill/device/indexed_fill_device_operation.hpp
            indexed_fill/device/indexed_fill_device_operation_types.hpp
            indexed_fill/device/indexed_fill_program_factory.hpp
            # moe_expert_token_remap/
            moe_expert_token_remap/moe_expert_token_remap.hpp
            moe_expert_token_remap/moe_expert_token_remap_nanobind.hpp
            moe_expert_token_remap/device/moe_expert_token_remap_device_operation.hpp
            # moe_routing_remap/
            moe_routing_remap/moe_routing_remap.hpp
            moe_routing_remap/moe_routing_remap_nanobind.hpp
            moe_routing_remap/device/moe_routing_remap_device_operation.hpp
            # move/
            move/move.hpp
            move/move_nanobind.hpp
            move/device/move_device_operation_types.hpp
            move/device/move_overlap_program_factory.hpp
            move/device/move_program_factory.hpp
            move/device/move_sharded_program_factory.hpp
            # non_zero_indices/
            non_zero_indices/non_zero_indices.hpp
            non_zero_indices/non_zero_indices_nanobind.hpp
            non_zero_indices/device/non_zero_indices_device_operation.hpp
            non_zero_indices/device/non_zero_indices_device_operation_types.hpp
            non_zero_indices/device/non_zero_indices_program_factory.hpp
            # pad/
            pad/pad_nanobind.hpp
            pad/device/pad_device_operation.hpp
            pad/device/pad_device_operation_types.hpp
            pad/device/pad_rm_reader_writer_multi_core_program_factory.hpp
            pad/device/pad_rm_reader_writer_multi_core_v2_program_factory.hpp
            pad/device/pad_rm_reader_writer_program_factory.hpp
            pad/device/pad_rm_sharded_height_only_program_factory.hpp
            pad/device/pad_rm_sharded_width_only_program_factory.hpp
            pad/device/pad_tile_multicore_program_factory.hpp
            pad/device/pad_tile_program_factory.hpp
            # permute/
            permute/permute_nanobind.hpp
            permute/device/permute_device_operation.hpp
            # repeat/
            repeat/repeat_nanobind.hpp
            repeat/device/repeat_device_operation.hpp
            repeat/device/repeat_device_operation_types.hpp
            repeat/device/repeat_program_factory_common.hpp
            repeat/device/repeat_program_factory_higher_dim.hpp
            repeat/device/repeat_program_factory_last_dim.hpp
            # repeat_interleave/
            repeat_interleave/repeat_interleave.hpp
            repeat_interleave/repeat_interleave_nanobind.hpp
            # reshape_on_device/
            reshape_on_device/reshape.hpp
            reshape_on_device/reshape_nanobind.hpp
            reshape_on_device/device/reshape_device_operation_types.hpp
            reshape_on_device/device/reshape_op.hpp
            reshape_on_device/device/reshape_rm_program_factory.hpp
            reshape_on_device/device/reshape_tile_program_factory.hpp
            # reshape_view/
            reshape_view/reshape_nanobind.hpp
            reshape_view/reshape_common.hpp
            reshape_view/device/reshape_device_operation.hpp
            reshape_view/device/reshape_device_operation_types.hpp
            reshape_view/device/reshape_row_major_program_factory.hpp
            reshape_view/device/reshape_tiled_program_factory.hpp
            # roll/
            roll/roll.hpp
            roll/roll_nanobind.hpp
            # scatter/
            scatter/scatter.hpp
            scatter/scatter_nanobind.hpp
            scatter/scatter_enums.hpp
            scatter/tosa_scatter.hpp
            scatter/tosa_scatter_nanobind.hpp
            scatter/device/scatter_common.hpp
            scatter/device/scatter_device_operation.hpp
            scatter/device/scatter_device_operation_types.hpp
            scatter/device/scatter_program_factory.hpp
            scatter/device/scatter_reduce_bfloat16_program_factory.hpp
            # sharded/
            sharded/sharded_common.hpp
            sharded/interleaved_to_sharded/interleaved_to_sharded_nanobind.hpp
            sharded/interleaved_to_sharded/device/interleaved_to_sharded_op_types.hpp
            sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.hpp
            sharded/reshard/reshard_nanobind.hpp
            sharded/reshard/device/reshard_device_operation.hpp
            sharded/reshard/device/reshard_device_operation_types.hpp
            sharded/reshard/device/reshard_program_factory_generic.hpp
            sharded/reshard/device/reshard_program_factory_same_height.hpp
            sharded/reshard/device/reshard_program_factory_same_width.hpp
            sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp
            sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp
            sharded/sharded_to_interleaved/sharded_to_interleaved.hpp
            sharded/sharded_to_interleaved/sharded_to_interleaved_nanobind.hpp
            # sharded_partial/
            sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial.hpp
            sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial_nanobind.hpp
            sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.hpp
            sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op_types.hpp
            sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.hpp
            sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial.hpp
            sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial_nanobind.hpp
            sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation.hpp
            sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation_types.hpp
            sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.hpp
            # slice/
            slice/slice_nanobind.hpp
            slice/device/slice_device_operation.hpp
            slice/device/slice_device_operation_types.hpp
            slice/device/slice_program_factory_rm.hpp
            slice/device/slice_program_factory_rm_sharded.hpp
            slice/device/slice_program_factory_rm_stride.hpp
            slice/device/slice_program_factory_tile.hpp
            slice/device/slice_program_factory_tile_tensor_args.hpp
            # sort/
            sort/sort.hpp
            sort/sort_nanobind.hpp
            sort/device/sort_device_operation.hpp
            sort/device/sort_device_operation_types.hpp
            sort/device/sort_program_factory.hpp
            # split/
            split/split.hpp
            split/split_nanobind.hpp
            split/device/split_device_operation.hpp
            split/device/split_device_operation_types.hpp
            split/device/split_program_factory.hpp
            # squeeze/
            squeeze/squeeze.hpp
            squeeze/squeeze_nanobind.hpp
            # stack/
            stack/stack.hpp
            stack/stack_nanobind.hpp
            # tilize/
            tilize/tilize.hpp
            tilize/tilize_nanobind.hpp
            tilize/device/tilize_device_operation.hpp
            tilize/device/tilize_device_operation_types.hpp
            tilize/device/tilize_multi_core_block_program_factory.hpp
            tilize/device/tilize_multi_core_interleaved_program_factory.hpp
            tilize/device/tilize_multi_core_sharded_program_factory.hpp
            tilize/device/tilize_multi_core_width_sharded_program_factory.hpp
            tilize/device/tilize_single_core_program_factory.hpp
            # tilize_with_val_padding/
            tilize_with_val_padding/tilize_with_val_padding_nanobind.hpp
            tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp
            tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp
            tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp
            tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_interleaved_program_factory.hpp
            tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_sharded_program_factory.hpp
            tilize_with_val_padding/device/factories/tilize_with_val_padding_shared_variables.hpp
            tilize_with_val_padding/device/factories/tilize_with_val_padding_single_core_program_factory.hpp
            # transpose/
            transpose/transpose_nanobind.hpp
            transpose/device/transpose_cn_program_factory.hpp
            transpose/device/transpose_device_operation.hpp
            transpose/device/transpose_device_operation_types.hpp
            transpose/device/transpose_hc_rm_program_factory.hpp
            transpose/device/transpose_hc_sharded_program_factory.hpp
            transpose/device/transpose_hc_tiled_interleaved_program_factory.hpp
            transpose/device/transpose_hc_tiled_program_factory.hpp
            transpose/device/transpose_wh_program_factory.hpp
            transpose/device/transpose_wh_sharded_program_factory.hpp
            transpose/device/transpose_wh_sharded_rm_program_factory.hpp
            # unsqueeze/
            unsqueeze/unsqueeze.hpp
            unsqueeze/unsqueeze_nanobind.hpp
            # untilize/
            untilize/untilize_nanobind.hpp
            untilize/device/untilize_device_operation.hpp
            untilize/device/untilize_device_operation_types.hpp
            untilize/device/factories/untilize_multi_core_block_program_factory.hpp
            untilize/device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.hpp
            untilize/device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.hpp
            untilize/device/factories/untilize_multi_core_nd_shard_input_program_factory.hpp
            untilize/device/factories/untilize_multi_core_parallelize_column_program_factory.hpp
            untilize/device/factories/untilize_multi_core_program_factory.hpp
            untilize/device/factories/untilize_multi_core_sub_core_grids_program_factory.hpp
            untilize/device/factories/untilize_single_core_program_factory.hpp
            # untilize_with_unpadding/
            untilize_with_unpadding/untilize_with_unpadding_nanobind.hpp
            untilize_with_unpadding/device/untilize_with_unpadding_device_operation.hpp
            untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp
            untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp
            untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.hpp
            untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.hpp
            untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.hpp
            untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp
            untilize_with_unpadding/device/factories/untilize_with_unpadding_single_core_program_factory.hpp
            # view/
            view/view.hpp
            view/view_nanobind.hpp
    PRIVATE
        bcast/bcast.cpp
        bcast/bcast_types.cpp
        bcast/device/bcast_device_operation.cpp
        bcast/device/bcast_multi_core_h_program_factory.cpp
        bcast/device/bcast_sharded_h_program_factory.cpp
        bcast/device/bcast_sharded_h_optimised_program_factory.cpp
        bcast/device/bcast_multi_core_w_program_factory.cpp
        bcast/device/bcast_multi_core_hw_program_factory.cpp
        chunk/chunk.cpp
        clone/clone.cpp
        clone/device/clone_device_operation.cpp
        clone/device/clone_program_factory.cpp
        common/common.cpp
        concat/concat.cpp
        concat/device/concat_device_operation.cpp
        concat/device/concat_program_factory.cpp
        concat/device/concat_s2s_tiled_program_factory.cpp
        concat/device/concat_s2s_rm_program_factory.cpp
        concat/device/concat_s2s_multi_program_factory.cpp
        concat/device/concat_s2i_program_factory.cpp
        copy/copy.cpp
        copy/device/copy_device_operation.cpp
        copy/device/copy_program_factory.cpp
        expand/expand.cpp
        fill_pad/device/fill_pad_device_operation.cpp
        fill_pad/device/fill_pad_program_factory.cpp
        fill_pad/fill_pad.cpp
        fill_rm/device/fill_rm_device_operation.cpp
        fill_rm/device/fill_rm_program_factory.cpp
        fill_rm/fill_rm.cpp
        fold/device/fold_device_op.cpp
        fold/device/fold_multi_core_dram_program_factory.cpp
        fold/device/fold_multi_core_program_factory.cpp
        fold/device/fold_single_core_program_factory.cpp
        fold/fold.cpp
        indexed_fill/device/indexed_fill_device_operation.cpp
        indexed_fill/device/indexed_fill_program_factory.cpp
        indexed_fill/indexed_fill.cpp
        moe_expert_token_remap/moe_expert_token_remap.cpp
        moe_expert_token_remap/device/moe_expert_token_remap_device_operation.cpp
        moe_expert_token_remap/device/moe_expert_token_remap_program_factory.cpp
        moe_routing_remap/moe_routing_remap.cpp
        moe_routing_remap/device/moe_routing_remap_device_operation.cpp
        moe_routing_remap/device/moe_routing_remap_program_factory.cpp
        move/device/move_device_operation.cpp
        move/device/move_program_factory.cpp
        move/device/move_overlap_program_factory.cpp
        move/device/move_sharded_program_factory.cpp
        move/move.cpp
        non_zero_indices/device/non_zero_indices_device_operation.cpp
        non_zero_indices/device/non_zero_indices_program_factory.cpp
        non_zero_indices/non_zero_indices.cpp
        pad/device/pad_device_operation.cpp
        pad/device/pad_rm_reader_writer_multi_core_program_factory.cpp
        pad/device/pad_rm_reader_writer_multi_core_v2_program_factory.cpp
        pad/device/pad_rm_reader_writer_program_factory.cpp
        pad/device/pad_rm_sharded_height_only_program_factory.cpp
        pad/device/pad_rm_sharded_width_only_program_factory.cpp
        pad/device/pad_tile_multicore_program_factory.cpp
        pad/device/pad_tile_program_factory.cpp
        pad/pad.cpp
        permute/device/permute_device_operation.cpp
        permute/device/permute_rm_program_factory.cpp
        permute/device/permute_tiled_program_factory.cpp
        permute/permute.cpp
        repeat/device/repeat_program_factory_last_dim.cpp
        repeat/device/repeat_program_factory_higher_dim.cpp
        repeat/device/repeat_device_operation.cpp
        repeat/repeat.cpp
        repeat_interleave/repeat_interleave.cpp
        reshape_on_device/device/reshape_op.cpp
        reshape_on_device/device/reshape_tile_program_factory.cpp
        reshape_on_device/device/reshape_rm_program_factory.cpp
        reshape_on_device/reshape.cpp
        reshape_view/device/reshape_rm_program_factory.cpp
        reshape_view/device/reshape_tiled_program_factory.cpp
        reshape_view/device/reshape_device_operation.cpp
        reshape_view/reshape.cpp
        roll/roll.cpp
        scatter/scatter.cpp
        scatter/tosa_scatter.cpp
        scatter/device/scatter_device_operation.cpp
        scatter/device/scatter_reduce_bfloat16_program_factory.cpp
        scatter/device/scatter_program_factory.cpp
        sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
        sharded/interleaved_to_sharded/interleaved_to_sharded.cpp
        sharded/reshard/device/reshard_device_operation.cpp
        sharded/reshard/device/reshard_program_factory_same_width.cpp
        sharded/reshard/device/reshard_program_factory_same_height.cpp
        sharded/reshard/device/reshard_program_factory_generic.cpp
        sharded/reshard/device/nd_reshard_program_factory_copy_pages.cpp
        sharded/reshard/device/nd_reshard_program_factory_copy_local.cpp
        sharded/reshard/reshard.cpp
        sharded/sharded_common.cpp
        sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.cpp
        sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
        sharded/sharded_to_interleaved/sharded_to_interleaved.cpp
        sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
        sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
        sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial.cpp
        sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation.cpp
        sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.cpp
        sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial.cpp
        slice/device/slice_device_operation.cpp
        slice/device/slice_program_factory_rm.cpp
        slice/device/slice_program_factory_rm_sharded.cpp
        slice/device/slice_program_factory_rm_stride.cpp
        slice/device/slice_program_factory_tile.cpp
        slice/device/slice_program_factory_tile_tensor_args.cpp
        slice/slice.cpp
        split/device/split_device_operation.cpp
        split/device/split_program_factory.cpp
        split/split.cpp
        squeeze/squeeze.cpp
        stack/stack.cpp
        tilize/device/tilize_device_operation.cpp
        tilize/device/tilize_multi_core_interleaved_program_factory.cpp
        tilize/device/tilize_multi_core_block_program_factory.cpp
        tilize/device/tilize_single_core_program_factory.cpp
        tilize/device/tilize_multi_core_sharded_program_factory.cpp
        tilize/device/tilize_multi_core_width_sharded_program_factory.cpp
        tilize/tilize.cpp
        tilize_with_val_padding/device/tilize_with_val_padding_device_operation.cpp
        tilize_with_val_padding/device/factories/tilize_with_val_padding_single_core_program_factory.cpp
        tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp
        tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_interleaved_program_factory.cpp
        tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_sharded_program_factory.cpp
        tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.cpp
        tilize_with_val_padding/tilize_with_val_padding.cpp
        transpose/device/transpose_device_operation.cpp
        transpose/device/transpose_cn_program_factory.cpp
        transpose/device/transpose_hc_rm_program_factory.cpp
        transpose/device/transpose_hc_sharded_program_factory.cpp
        transpose/device/transpose_hc_tiled_interleaved_program_factory.cpp
        transpose/device/transpose_hc_tiled_program_factory.cpp
        transpose/device/transpose_wh_program_factory.cpp
        transpose/device/transpose_wh_sharded_program_factory.cpp
        transpose/device/transpose_wh_sharded_rm_program_factory.cpp
        transpose/transpose.cpp
        unsqueeze/unsqueeze.cpp
        untilize/device/untilize_device_operation.cpp
        untilize/device/untilize_device_operation_types.cpp
        untilize/device/factories/untilize_multi_core_parallelize_column_program_factory.cpp
        untilize/device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp
        untilize/device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp
        untilize/device/factories/untilize_multi_core_sub_core_grids_program_factory.cpp
        untilize/device/factories/untilize_single_core_program_factory.cpp
        untilize/device/factories/untilize_multi_core_block_program_factory.cpp
        untilize/device/factories/untilize_multi_core_program_factory.cpp
        untilize/device/factories/untilize_multi_core_nd_shard_input_program_factory.cpp
        untilize/untilize.cpp
        untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp
        untilize_with_unpadding/device/factories/untilize_with_unpadding_single_core_program_factory.cpp
        untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp
        untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp
        untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp
        untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp
        untilize_with_unpadding/untilize_with_unpadding.cpp
        view/view.cpp
        sort/sort.cpp
        sort/device/sort_device_operation.cpp
        sort/device/sort_program_factory.cpp
        gather/gather.cpp
        gather/device/gather_device_operation.cpp
        gather/device/gather_program_factory.cpp
        gather/tosa/gather_tosa.cpp
)

target_include_directories(ttnn_op_data_movement PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_data_movement PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_data_movement
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/data_movement
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_data_movement LIBRARY COMPONENT tar)
