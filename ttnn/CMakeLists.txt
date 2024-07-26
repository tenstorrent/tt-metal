set(TTNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/run_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operation_history.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/conv_op_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/optimized_conv_op_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/multi_core_optimized_conv/optimized_conv_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/multi_core_optimized_conv_sharded/optimized_conv_op_sharded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d/device/multi_core_optimized_conv_sharded/optimized_conv_op_sharded_v2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/matmul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/single_core/bmm_op_single_core_tilize_untilize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core/bmm_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core_reuse/bmm_op_multi_core_reuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core_reuse_padding/bmm_op_multi_core_reuse_padding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core_reuse_mcast_1d_optimized/bmm_op_multi_core_reuse_mcast_1d_optimized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core_reuse_mcast_2d_optimized/bmm_op_multi_core_reuse_mcast_2d_optimized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core_reuse_mcast_dram_sharded_optimized/bmm_op_multi_core_reuse_dram_sharded_optimized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/device/multi_core_reuse_optimized/bmm_op_multi_core_reuse_optimized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/argmax/device/argmax_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/broadcast_width_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary_backward/device/binary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary_backward/device/ternary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_binary_backward/device/complex_binary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/upsample/upsample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/upsample/device/upsample_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/upsample/device/upsample_op_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/device/unary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary_backward/device/unary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary/device/complex_unary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_binary/device/complex_binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary/ternary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/ternary/where_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/complex_unary_backward/device/complex_unary_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/ccl_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/device/multi_core/all_gather_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/device/host/reduce_scatter_full_worker_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/example_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/argmax/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/topk/device/topk_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/device/slice_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/permute/permute_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/softmax_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/device/pad_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/device/multi_core/layernorm_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/downsample/device/downsample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/device/multi_core/groupnorm_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/device/transformer_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/avgpool/avg_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/maxpool/device/max_pool_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/prod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_nc_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_op_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_nc_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/prod/device/prod_all_program_factory.cpp
)

### Setup TTNN as a shared library with optional Python bindings
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/tensor)
add_subdirectory(cpp/ttnn/deprecated)
set(TTNN_FINAL_SRC
    ${TTNN_SRCS}
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
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/device_api.h
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
target_precompile_headers(ttnn PUBLIC ${TTNN_PRECOMPILED_HEADERS})


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
