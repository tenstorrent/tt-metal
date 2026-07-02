add_library(ttnn_op_experimental_neighbor_pad_halo ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::NeighborPadHalo ALIAS ttnn_op_experimental_neighbor_pad_halo)

tt_reuse_precompile_headers(ttnn_op_experimental_neighbor_pad_halo TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_neighbor_pad_halo)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_neighbor_pad_halo
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        neighbor_pad_halo.cpp
        device/neighbor_pad_halo_device_operation.cpp
        device/neighbor_pad_halo_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_neighbor_pad_halo PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_neighbor_pad_halo
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Experimental::CCL
)

install(
    TARGETS
        ttnn_op_experimental_neighbor_pad_halo
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_neighbor_pad_halo LIBRARY COMPONENT tar)
