add_library(ttnn_op_experimental_kda_shared ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::Shared ALIAS ttnn_op_experimental_kda_shared)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_shared TT::CommonPCH)

target_sources(ttnn_op_experimental_kda_shared PRIVATE device/kda_factory_utils.cpp)
target_include_directories(ttnn_op_experimental_kda_shared PRIVATE ${FixmeOpAPIDir})
target_link_libraries(ttnn_op_experimental_kda_shared PRIVATE TT::Metalium PUBLIC TTNN::Core)

add_subdirectory(affine_exclusive_scan)
add_subdirectory(prepare_chunk_recurrence)
add_subdirectory(qkv_causal_conv1d_silu)
add_subdirectory(reduce_affine_transforms)
add_subdirectory(recurrent_chunk_scan)
add_subdirectory(sigmoid_gated_rms_norm)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE kda_nanobind.cpp)
endif()
