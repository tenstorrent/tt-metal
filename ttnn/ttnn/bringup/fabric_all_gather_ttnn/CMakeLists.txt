include(sources.cmake)

add_library(ttnn_op_bringup_fabric_all_gather_ttnn ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::FabricAllGatherTtnn ALIAS ttnn_op_bringup_fabric_all_gather_ttnn)

tt_reuse_precompile_headers(ttnn_op_bringup_fabric_all_gather_ttnn TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_bringup_fabric_all_gather_ttnn)
target_include_directories(ttnn_op_bringup_fabric_all_gather_ttnn PRIVATE ${FixmeOpAPIDir})

set_target_properties(
    ttnn_op_bringup_fabric_all_gather_ttnn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)
target_sources(
    ttnn_op_bringup_fabric_all_gather_ttnn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_FABRIC_ALL_GATHER_TTNN_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_BRINGUP_FABRIC_ALL_GATHER_TTNN_SRCS}
)

target_link_libraries(ttnn_op_bringup_fabric_all_gather_ttnn PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_bringup_fabric_all_gather_ttnn
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/bringup/fabric_all_gather_ttnn
        COMPONENT ttnn-runtime
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_FABRIC_ALL_GATHER_TTNN_NANOBIND_SRCS})
endif()
