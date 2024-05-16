set(TTNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/op_library/binary/binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/op_library/to_layout/to_layout_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul.cpp

)
add_library(ttnn_lib OBJECT ${TTNN_SRCS})
target_compile_options(ttnn_lib PUBLIC -MP -Wno-int-to-pointer-cast -fno-var-tracking)
target_link_libraries(ttnn_lib PUBLIC compiler_flags metal_header_directories metal_common_libs)
target_include_directories(ttnn_lib PUBLIC
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tt_eager        # this is ... should be removed once we only have ttnn
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt
)
target_precompile_headers(ttnn_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/magic_enum/magic_enum.hpp
)


# TODO: should be using pybind11_add_module, but right now it introduces many build problems
# pybinds will always be built as a shared library
add_library(ttnn SHARED ${CMAKE_SOURCE_DIR}/ttnn/cpp/pybind11/__init__.cpp $<TARGET_OBJECTS:ttnn_lib>)
target_compile_options(ttnn PUBLIC -Wno-int-to-pointer-cast -fno-var-tracking)
target_link_libraries(ttnn PUBLIC compiler_flags linker_flags tt_eager pch_pybinds)     # linker_flags = -rdynamic if tracy enabled
target_include_directories(ttnn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/pybind11/include
    ${Python3_INCLUDE_DIRS}
)

# Make sure library built is _ttnn.so and that it can find all it's linked libraries
# ttnn breaks if -fvisibility=hidden, so CXX_VISIBILITY_PRESET set to default
set_target_properties(ttnn PROPERTIES
    OUTPUT_NAME "_ttnn"
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH "${CMAKE_BINARY_DIR}/tt_metal;${CMAKE_BINARY_DIR}/tt_eager;${CMAKE_BINARY_DIR}/ttnn"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
    CXX_VISIBILITY_PRESET "default"
    ADDITIONAL_CLEAN_FILES "${CMAKE_SOURCE_DIR}/ttnn/ttnn/_ttnn.so;${CMAKE_SOURCE_DIR}/ttnn/ttnn.egg-info"
)
