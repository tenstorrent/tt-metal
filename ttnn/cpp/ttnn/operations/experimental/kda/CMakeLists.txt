# KDA-domain build ownership. Shared utilities are compiled once in Common;
# operation leaves consume Common and are exported through one domain umbrella.
add_library(ttnn_op_experimental_kda_common ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::Common ALIAS ttnn_op_experimental_kda_common)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_common TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_common)

target_sources(ttnn_op_experimental_kda_common PRIVATE factory/kda_factory_utils.cpp)
target_include_directories(ttnn_op_experimental_kda_common PRIVATE ${FixmeOpAPIDir})
target_link_libraries(ttnn_op_experimental_kda_common PRIVATE TT::Metalium PUBLIC TTNN::Core)
install(TARGETS ttnn_op_experimental_kda_common LIBRARY COMPONENT tar)

add_subdirectory(sigmoid_gated_rms_norm)
add_subdirectory(qkv_causal_conv1d_silu)
add_subdirectory(reduce_affine_transforms)
add_subdirectory(affine_exclusive_scan)
add_subdirectory(prepare_chunk_recurrence)

add_library(ttnn_op_experimental_kda INTERFACE)
add_library(TTNN::Ops::Experimental::KDA ALIAS ttnn_op_experimental_kda)
target_link_libraries(
    ttnn_op_experimental_kda
    INTERFACE
        TTNN::Ops::Experimental::KDA::Common
        TTNN::Ops::Experimental::KDA::SigmoidGatedRMSNorm
        TTNN::Ops::Experimental::KDA::QkvCausalConv1dSilu
        TTNN::Ops::Experimental::KDA::ReduceAffineTransforms
        TTNN::Ops::Experimental::KDA::AffineExclusiveScan
        TTNN::Ops::Experimental::KDA::PrepareChunkRecurrence
)

# OBJECT libraries do not propagate their compiled objects through a nested
# INTERFACE link. Inject each KDA object set here exactly once; these generator
# expressions are empty when the corresponding target is a shared library.
target_sources(
    ttnn_op_experimental_kda
    INTERFACE
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::KDA::Common>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::KDA::SigmoidGatedRMSNorm>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::KDA::QkvCausalConv1dSilu>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::KDA::ReduceAffineTransforms>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::KDA::AffineExclusiveScan>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::KDA::PrepareChunkRecurrence>
)
