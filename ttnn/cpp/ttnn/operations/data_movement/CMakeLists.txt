add_library(ttnn_op_data_movement ${LIB_TYPE})
add_library(TTNN::Ops::DataMovement ALIAS ttnn_op_data_movement)

target_precompile_headers(ttnn_op_data_movement REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_data_movement)
set_target_properties(
    ttnn_op_data_movement
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

target_sources(
    ttnn_op_data_movement
    PRIVATE
        bcast/bcast.cpp
        bcast/bcast_types.cpp
        bcast/device/bcast_device_operation.cpp
        bcast/device/multi_core_h/bcast_op_multi_core_h.cpp
        bcast/device/multi_core_h/bcast_op_sharded_h.cpp
        bcast/device/multi_core_h/bcast_op_sharded_h_optimised.cpp
        bcast/device/multi_core_hw/bcast_op_multi_core_hw.cpp
        bcast/device/multi_core_w/bcast_op_multi_core_w.cpp
        chunk/chunk.cpp
        clone/clone.cpp
        clone/device/clone_device_operation.cpp
        clone/device/clone_program_factory.cpp
        common/common.cpp
        concat/concat.cpp
        concat/device/concat_device_operation.cpp
        concat/device/concat_program_factory.cpp
        copy/copy.cpp
        copy/device/copy_device_operation.cpp
        copy/device/copy_program_factory.cpp
        expand/expand.cpp
        fill_pad/device/fill_pad_op.cpp
        fill_pad/device/fill_pad_program_factory.cpp
        fill_pad/fill_pad.cpp
        fill_rm/device/fill_rm_op.cpp
        fill_rm/fill_rm.cpp
        fold/device/fold_device_op.cpp
        fold/device/fold_multi_core_dram_program_factory.cpp
        fold/device/fold_multi_core_program_factory.cpp
        fold/device/fold_single_core_program_factory.cpp
        fold/fold.cpp
        indexed_fill/device/indexed_fill_op.cpp
        indexed_fill/device/indexed_fill_op_multi_core_program_factory.cpp
        indexed_fill/indexed_fill.cpp
        move/device/move_device_operation.cpp
        move/device/move_program_factory.cpp
        move/move.cpp
        non_zero_indices/device/non_zero_indices_op.cpp
        non_zero_indices/device/non_zero_indices_program_factory.cpp
        non_zero_indices/non_zero_indices.cpp
        pad/device/pad_op.cpp
        pad/device/pad_program_factory.cpp
        pad/pad.cpp
        permute/device/permute_device_operation.cpp
        permute/device/permute_rm_program_factory.cpp
        permute/device/permute_tiled_program_factory.cpp
        permute/permute.cpp
        repeat/device/host/repeat_program_factory.cpp
        repeat/device/repeat_device_operation.cpp
        repeat/repeat.cpp
        repeat_interleave/repeat_interleave.cpp
        reshape_on_device/device/reshape_op.cpp
        reshape_on_device/device/reshape_program_factory.cpp
        reshape_on_device/reshape.cpp
        reshape_view/device/host/reshape_rm_program_factory.cpp
        reshape_view/device/host/reshape_tiled_program_factory.cpp
        reshape_view/device/reshape_device_operation.cpp
        reshape_view/reshape.cpp
        roll/roll.cpp
        sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
        sharded/interleaved_to_sharded/interleaved_to_sharded.cpp
        sharded/reshard/device/reshard_op.cpp
        sharded/reshard/device/reshard_program_factory.cpp
        sharded/reshard/reshard.cpp
        sharded/sharded_common.cpp
        sharded/sharded_to_interleaved/device/sharded_to_interleaved_op.cpp
        sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
        sharded/sharded_to_interleaved/sharded_to_interleaved.cpp
        sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
        sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial.cpp
        sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_op.cpp
        sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial.cpp
        slice/device/slice_op.cpp
        slice/device/slice_program_factory.cpp
        slice/slice.cpp
        split/device/split_op.cpp
        split/device/split_program_factory.cpp
        split/split.cpp
        squeeze/squeeze.cpp
        stack/stack.cpp
        tilize/device/tilize_op.cpp
        tilize/device/tilize_program_factory.cpp
        tilize/tilize.cpp
        tilize_with_val_padding/device/tilize_with_val_padding_op.cpp
        tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp
        tilize_with_val_padding/tilize_with_val_padding.cpp
        transpose/device/transpose_op.cpp
        transpose/device/transpose_program_factory.cpp
        transpose/transpose.cpp
        unsqueeze/unsqueeze.cpp
        untilize/device/untilize_op.cpp
        untilize/device/untilize_program_factory.cpp
        untilize/untilize.cpp
        untilize_with_unpadding/device/untilize_with_unpadding_op.cpp
        untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp
        untilize_with_unpadding/untilize_with_unpadding.cpp
        view/view.cpp
)

file(
    GLOB_RECURSE kernels
    common/kernels/*
    move/device/kernels/*
    permute/device/kernels/*
    sharded/device/kernels/*
    transpose/device/kernels/*
)
target_sources(
    ttnn_op_data_movement
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            reshape_view/device/device/dataflow/reader_reshape_tiled.cpp
            reshape_view/device/device/dataflow/writer_reshape_tiled.cpp
            reshape_view/device/device/rm_reshape_interleaved.cpp
            reshape_view/device/hostdevcommon/common.hpp
)

target_include_directories(ttnn_op_data_movement PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_data_movement
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_data_movement
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/data_movement
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_data_movement LIBRARY COMPONENT tar)
