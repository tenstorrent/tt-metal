include(sources.cmake)

add_library(ttnn_op_bringup_prepared_program ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::PreparedProgram ALIAS ttnn_op_bringup_prepared_program)

tt_reuse_precompile_headers(ttnn_op_bringup_prepared_program TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_bringup_prepared_program)
target_include_directories(ttnn_op_bringup_prepared_program PRIVATE ${FixmeOpAPIDir})

set_target_properties(
    ttnn_op_bringup_prepared_program
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_bringup_prepared_program
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_PREPARED_PROGRAM_API_HEADERS}
    PRIVATE
        ${TTNN_OP_BRINGUP_PREPARED_PROGRAM_SRCS}
)

target_link_libraries(ttnn_op_bringup_prepared_program PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_bringup_prepared_program FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings: registered on the shared `ttnn` module target (see offset_cumsum/CMakeLists.txt).
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_PREPARED_PROGRAM_NANOBIND_SRCS})
endif()
