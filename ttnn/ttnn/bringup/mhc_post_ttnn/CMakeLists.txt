include(sources.cmake)

add_library(ttnn_op_bringup_mhc_post_ttnn ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::MhcPostTtnn ALIAS ttnn_op_bringup_mhc_post_ttnn)

tt_reuse_precompile_headers(ttnn_op_bringup_mhc_post_ttnn TTNN::PCHFull)
target_include_directories(ttnn_op_bringup_mhc_post_ttnn PRIVATE ${FixmeOpAPIDir})

set_target_properties(
    ttnn_op_bringup_mhc_post_ttnn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The device kernels are the Python op's, unchanged (kernels/); installed beside the host code.
file(
    GLOB kernels
    kernels/*.cpp
    kernels/*.hpp
)

target_sources(
    ttnn_op_bringup_mhc_post_ttnn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_MHC_POST_TTNN_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_BRINGUP_MHC_POST_TTNN_SRCS}
)

target_link_libraries(ttnn_op_bringup_mhc_post_ttnn PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_bringup_mhc_post_ttnn
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/bringup/mhc_post_ttnn
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_bringup_mhc_post_ttnn FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings: registered on the shared `ttnn` Python module target (see offset_cumsum/CMakeLists.txt).
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_MHC_POST_TTNN_NANOBIND_SRCS})
endif()
