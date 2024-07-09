set(TTNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/op_library/to_layout/to_layout_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/upsample/upsample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/upsample/device/upsample_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/upsample/device/upsample_op_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/unary/device/unary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/example_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/topk/device/topk_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding/embedding/device/embeddings_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/slice/device/slice_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/softmax_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/device/pad_op.cpp
)

add_library(ttnn_lib OBJECT ${TTNN_SRCS})
target_compile_options(ttnn_lib PUBLIC -MP -Wno-int-to-pointer-cast -fno-var-tracking)
target_link_libraries(ttnn_lib
    PUBLIC compiler_flags metal_header_directories metal_common_libs
)
target_include_directories(ttnn_lib PUBLIC
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_eager        # this is ... should be removed once we only have ttnn
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt
)
target_precompile_headers(ttnn_lib PRIVATE
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/magic_enum/magic_enum.hpp
    <functional>
    <map>
    <memory>
    <unordered_map>
    <variant>
    <vector>
)

set(TTNN_FINAL_SRC $<TARGET_OBJECTS:ttnn_lib>)
set(TTNN_PUBLIC_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set(TTNN_PUBLIC_LINK_LIBS compiler_flags linker_flags tt_eager) # linker_flags = -rdynamic if tracy enabled
if(WITH_PYTHON_BINDINGS)
    # TODO: should be using pybind11_add_module, but right now it introduces many build problems
    # pybinds will always be built as a shared library
    list(APPEND TTNN_FINAL_SRC ${PROJECT_SOURCE_DIR}/ttnn/cpp/pybind11/__init__.cpp)
    list(APPEND TTNN_PUBLIC_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include
        ${Python3_INCLUDE_DIRS})
    list(APPEND TTNN_PUBLIC_LINK_LIBS pch_pybinds)
endif()

add_library(ttnn SHARED ${TTNN_FINAL_SRC})
target_compile_options(ttnn PUBLIC -Wno-int-to-pointer-cast -fno-var-tracking)
target_link_libraries(ttnn PUBLIC ${TTNN_PUBLIC_LINK_LIBS})
target_include_directories(ttnn PUBLIC ${TTNN_PUBLIC_INCLUDE_DIRS})

# Make sure library built is _ttnn.so and that it can find all it's linked libraries
# ttnn breaks if -fvisibility=hidden, so CXX_VISIBILITY_PRESET set to default
set_target_properties(ttnn PROPERTIES
    OUTPUT_NAME "_ttnn"
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/tt_eager;${PROJECT_BINARY_DIR}/ttnn"
    INSTALL_RPATH "${PROJECT_BINARY_DIR}/lib"
    CXX_VISIBILITY_PRESET "default"
    ADDITIONAL_CLEAN_FILES "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_ttnn.so;${PROJECT_SOURCE_DIR}/ttnn/ttnn.egg-info"
)
