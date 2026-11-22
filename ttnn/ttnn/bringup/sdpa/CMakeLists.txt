include(sources.cmake)

add_library(ttnn_op_bringup_sdpa ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::Sdpa ALIAS ttnn_op_bringup_sdpa)

tt_reuse_precompile_headers(ttnn_op_bringup_sdpa TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_bringup_sdpa)
target_include_directories(
    ttnn_op_bringup_sdpa
    PRIVATE
        ${FixmeOpAPIDir}
        ${FixmeOpIncDirs}
)

set_target_properties(
    ttnn_op_bringup_sdpa
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
    device/kernels/*.h
)

target_sources(
    ttnn_op_bringup_sdpa
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_SDPA_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_BRINGUP_SDPA_SRCS}
)

target_link_libraries(ttnn_op_bringup_sdpa PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_bringup_sdpa
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/bringup/sdpa
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_bringup_sdpa FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_SDPA_NANOBIND_SRCS})
endif()
