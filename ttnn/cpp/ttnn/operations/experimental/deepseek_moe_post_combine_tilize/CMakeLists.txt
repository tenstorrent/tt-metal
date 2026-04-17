add_library(ttnn_op_experimental_deepseek_moe_post_combine_tilize ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepseekMoEPostCombineTilize
    ALIAS ttnn_op_experimental_deepseek_moe_post_combine_tilize
)

target_precompile_headers(ttnn_op_experimental_deepseek_moe_post_combine_tilize REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_post_combine_tilize)

set_target_properties(
    ttnn_op_experimental_deepseek_moe_post_combine_tilize
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_moe_post_combine_tilize
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
        device/deepseek_moe_post_combine_tilize_device_operation.cpp
        device/deepseek_moe_post_combine_tilize_program_factory.cpp
        deepseek_moe_post_combine_tilize.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_moe_post_combine_tilize PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_moe_post_combine_tilize
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_moe_post_combine_tilize
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_moe_post_combine_tilize LIBRARY COMPONENT tar)
