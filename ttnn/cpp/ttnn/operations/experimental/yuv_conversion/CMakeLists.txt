add_library(ttnn_op_experimental_yuv_conversion ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::YUVConversion ALIAS ttnn_op_experimental_yuv_conversion)

tt_reuse_precompile_headers(ttnn_op_experimental_yuv_conversion TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_yuv_conversion)

set_target_properties(
    ttnn_op_experimental_yuv_conversion
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_yuv_conversion
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES yuv_conversion.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/yuv_conversion_device_op.cpp
        device/yuv_conversion_program_factory.cpp
        yuv_conversion.cpp
)

target_include_directories(ttnn_op_experimental_yuv_conversion PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_yuv_conversion PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_experimental_yuv_conversion
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/yuv_conversion
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_yuv_conversion LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/CMakeLists.txt. `ttnn` only exists when WITH_PYTHON_BINDINGS is on,
# hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/yuv_conversion_nanobind.cpp)
endif()
