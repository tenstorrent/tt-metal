add_library(ttnn_op_concat_new ${LIB_TYPE})
add_library(TTNN::Ops::ConcatNew ALIAS ttnn_op_concat_new)

tt_reuse_precompile_headers(ttnn_op_concat_new TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_concat_new)

set_target_properties(
    ttnn_op_concat_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_concat_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            concat_new.hpp
            device/concat_new_device_operation.hpp
            device/concat_new_device_operation_types.hpp
            device/concat_new_program_factory.hpp
            device/concat_new_s2i_program_factory.hpp
            device/concat_new_s2s_multi_program_factory.hpp
            device/concat_new_s2s_rm_program_factory.hpp
            device/concat_new_s2s_tiled_program_factory.hpp
    PRIVATE
        concat_new.cpp
        device/concat_new_device_operation.cpp
        device/concat_new_program_factory.cpp
        device/concat_new_s2i_program_factory.cpp
        device/concat_new_s2s_multi_program_factory.cpp
        device/concat_new_s2s_rm_program_factory.cpp
        device/concat_new_s2s_tiled_program_factory.cpp
)

target_include_directories(ttnn_op_concat_new PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_concat_new
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_concat_new LIBRARY COMPONENT tar)
