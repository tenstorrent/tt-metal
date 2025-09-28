add_library(ttnn_op_sliding_window ${LIB_TYPE})
add_library(TTNN::Ops::SlidingWindow ALIAS ttnn_op_sliding_window)

target_precompile_headers(ttnn_op_sliding_window REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_sliding_window)
set_target_properties(
    ttnn_op_sliding_window
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels halo/device/kernels/*)
target_sources(
    ttnn_op_sliding_window
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        halo/device/halo_device_operation.cpp
        halo/device/untilize_with_halo_program_factory.cpp
        halo/halo.cpp
        sliding_window.cpp
)

target_include_directories(ttnn_op_sliding_window PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_sliding_window
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_sliding_window
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/sliding_window
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_sliding_window LIBRARY COMPONENT tar)
