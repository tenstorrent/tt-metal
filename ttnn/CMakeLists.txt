set(ALL_TTNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operation_history.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/run_operation.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/all_gather_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/device/multi_core/all_gather_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/ccl_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/ccl_host_datastructures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/line_all_gather/line_all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/line_all_gather/line_all_gather_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/device/host/reduce_scatter_full_worker_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/conv_op_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/multi_core_optimized_conv/optimized_conv_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/multi_core_optimized_conv_sharded/optimized_conv_op_sharded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/multi_core_optimized_conv_sharded/optimized_conv_op_sharded_v2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/optimized_conv_op_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/core/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/data_transfer/data_transfer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_rm/device/fill_rm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_rm/fill_rm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_rm/fill_rm_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/non_zero_indices.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/non_zero_indices_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/device/pad_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/device/pad_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/permute/permute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/device/repeat_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/device/repeat_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/repeat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/repeat_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/device/slice_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/split/device/split_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/split/device/split_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/split/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize/device/tilize_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize/tilize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/transpose/device/transpose_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/transpose/device/transpose_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/transpose/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize/device/untilize_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize/device/untilize_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize/untilize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_halo_v2/device/untilize_with_halo_v2_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_halo_v2/device/untilize_with_halo_v2_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_halo_v2/untilize_with_halo_v2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/broadcast_width_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary_backward/device/binary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex/complex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_binary/device/complex_binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary/device/complex_unary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary_backward/device/complex_unary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary/ternary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary/where.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary_backward/device/ternary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/device/unary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/unary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary_backward/device/unary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/example_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/experimental_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/argmax/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/prefix_scan_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/prefix_scan_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/prefix_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/device/transformer_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/device/rotate_half_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/device/single_core/rotate_half_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/rotate_half.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/rotate_half_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/loss/loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/loss/loss_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_optimized_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/device/multi_core/groupnorm_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/device/multi_core/layernorm_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/softmax_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/downsample/device/downsample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/device/multi_core/groupnorm_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/device/transformer_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/paged_cache/device/paged_cache_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/avgpool/avg_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/downsample/device/downsample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/downsample/device/downsample_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/downsample/downsample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/downsample/downsample_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool2d_device_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool2d_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/max_pool2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/max_pool2d_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_multicore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_singlecore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/upsample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/upsample_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/argmax/device/argmax_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/moe/device/moe_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/moe/moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_all_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_nc_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_nc_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_op_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/prod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/topk/device/topk_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/attention_softmax/attention_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/attention_softmax/attention_softmax_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/concatenate_heads/concatenate_heads.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/concatenate_heads/concatenate_heads_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/transformer_pybind.cpp
)

# Split src and python bindings
# We only build python bindings optionally
set(TTNN_SRC)
set(PYBIND_SRC)
foreach(FILE ${ALL_TTNN_SRCS})
    if(FILE MATCHES "_pybind.cpp$")
        list(APPEND PYBIND_SRC ${FILE})
    else()
        list(APPEND TTNN_SRC ${FILE})
    endif()
endforeach()

### Setup TTNN as a shared library with optional Python bindings
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/tensor)
add_subdirectory(cpp/ttnn/deprecated)
set(TTNN_FINAL_SRC
    ${TTNN_SRC}
    ${QUEUE_SRCS}
    ${TENSOR_SRCS}
    ${TT_DNN_SRCS})

set(TTNN_PUBLIC_INCLUDE_DIRS
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${CMAKE_CURRENT_SOURCE_DIR} # ${PROJECT_SOURCE_DIR}/ttnn
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/deprecated # symlink to tt_eager; should become native folder once merge complete
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt)
set(TTNN_PUBLIC_LINK_LIBRARIES compiler_flags metal_header_directories metal_common_libs tt_metal linker_flags) # linker_flags = -rdynamic if tracy enabled
set(TTNN_PUBLIC_LINK_DIRS "")

set(TTNN_PRECOMPILED_HEADERS
    ${PROJECT_SOURCE_DIR}/tt_metal/tt_stl/reflection.hpp
    ${PROJECT_SOURCE_DIR}/ttnn/cpp/ttnn/operation.hpp
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/magic_enum/magic_enum.hpp
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy/public/tracy/Tracy.hpp
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt/fmt/core.h
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt/fmt/format.h
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/device_api_metal.h
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/tt_device.h
    <functional>
    <map>
    <memory>
    <unordered_map>
    <variant>
    <vector>)

if(WITH_PYTHON_BINDINGS)
    # TODO: should be using pybind11_add_module, but right now it introduces many build problems
    # pybinds will always be built as a shared library
    list(APPEND TTNN_FINAL_SRC
        ${PROJECT_SOURCE_DIR}/ttnn/cpp/pybind11/__init__.cpp
        ${PYBIND_SRC}
        ${TT_LIB_SRCS}) # TT_LIB_SRCS from tt_eager/tt_lib/CMakeLists.txt for python bindigns

    list(APPEND TTNN_PUBLIC_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/experimental/tt_lib
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include
        ${Python3_INCLUDE_DIRS})

    list(APPEND TTNN_PUBLIC_LINK_DIRS ${Python3_LIBRARY_DIRS})

    list(APPEND TTNN_PUBLIC_LINK_LIBRARIES pch_pybinds ${Python3_LIBRARIES})
endif()

add_library(ttnn SHARED ${TTNN_FINAL_SRC})
target_compile_options(ttnn PUBLIC -MP -Wno-int-to-pointer-cast -fno-var-tracking)
target_include_directories(ttnn PUBLIC ${TTNN_PUBLIC_INCLUDE_DIRS})
target_link_libraries(ttnn PUBLIC ${TTNN_PUBLIC_LINK_LIBRARIES})
target_link_directories(ttnn PUBLIC ${TTNN_PUBLIC_LINK_DIRS})
target_precompile_headers(ttnn PRIVATE ${TTNN_PRECOMPILED_HEADERS})


# Make sure library built is _ttnn.so and that it can find all it's linked libraries
# ttnn breaks if -fvisibility=hidden, so CXX_VISIBILITY_PRESET set to default
set_target_properties(ttnn PROPERTIES
    OUTPUT_NAME "_ttnn"
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
    INSTALL_RPATH "${PROJECT_BINARY_DIR}/lib"
    CXX_VISIBILITY_PRESET "default"
    ADDITIONAL_CLEAN_FILES "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_ttnn.so;${PROJECT_SOURCE_DIR}/ttnn/ttnn.egg-info"
)
