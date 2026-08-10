include(sources.cmake)

add_library(ttnn_op_qr ${LIB_TYPE})
add_library(TTNN::Ops::QR ALIAS ttnn_op_qr)

tt_reuse_precompile_headers(ttnn_op_qr TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_qr)

set_target_properties(
    ttnn_op_qr
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_qr
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_QR_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_QR_SRCS}
)

target_link_libraries(ttnn_op_qr PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_qr
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/qr
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_qr FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_QR_NANOBIND_SRCS})
endif()
