add_library(ttnn_op_experimental_slice_write ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::SliceWrite ALIAS ttnn_op_experimental_slice_write)

target_precompile_headers(ttnn_op_experimental_slice_write REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_slice_write)
set_target_properties(
    ttnn_op_experimental_slice_write
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            false
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_slice_write
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        slice_write.cpp
        device/slice_write_op.cpp
        device/slice_write_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_slice_write PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_slice_write
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_slice_write
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/slice_write
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_slice_write LIBRARY COMPONENT tar)
