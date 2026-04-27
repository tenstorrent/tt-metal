add_library(ttnn_op_pool ${LIB_TYPE})
add_library(TTNN::Ops::Pool ALIAS ttnn_op_pool)

tt_reuse_precompile_headers(ttnn_op_pool TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_pool)

set_target_properties(
    ttnn_op_pool
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    device/kernels/*
    grid_sample/device/kernels/*
    rotate/device/kernels/*
)

target_sources(
    ttnn_op_pool
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        global_avg_pool/global_avg_pool.cpp
        grid_sample/device/grid_sample_device_operation.cpp
        grid_sample/device/grid_sample_bilinear_program_factory.cpp
        grid_sample/device/grid_sample_nearest_program_factory.cpp
        grid_sample/device/grid_sample_utils.cpp
        grid_sample/grid_sample.cpp
        grid_sample/grid_sample_prepare_grid.cpp
        rotate/device/rotate_device_operation.cpp
        rotate/device/rotate_nearest_program_factory.cpp
        rotate/device/rotate_bilinear_program_factory.cpp
        rotate/rotate.cpp
        pool_utils.cpp
)

target_include_directories(ttnn_op_pool PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_pool
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_pool
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/pool
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_pool LIBRARY COMPONENT tar)
