# experimental/ is one physical directory shared by many teams' ops, so this
# file is jointly owned by every team (or named individual, for ops without a
# team owner) with a subdirectory listed below (see CODEOWNERS) rather than
# by ttnn-core. Any of those owners can add/remove their own experimental ops
# here and self-approve, without ttnn-core or infra needing to be pinged, and
# without touching ttnn/CMakeLists.txt.
#
# Everything under experimental/ is migrated in here now except transformer/,
# which is one CMake target spanning multiple teams' code and needs an
# actual target split before ownership can split too.
#
# A number of the ops below have no team-specific CODEOWNERS rule at all and
# fall back to the generic ops-leads catch-all — that's the status quo, kept
# as-is here rather than guessed at. metalium-developers-ops-leads is listed
# on this file for exactly that reason, not because it's the deliberate,
# active owner of those ops.

# metalium-developers-convolutions
add_subdirectory(adaptive_pool)
add_subdirectory(conv3d)
add_subdirectory(cnn)

# metalium-developers-ops-data-movement
add_subdirectory(ccl)
add_subdirectory(ccl/moe_gpt)

# metalium-developers-mmfusedreduce
add_subdirectory(matmul)
add_subdirectory(quasar)
add_subdirectory(reduction)

# metalium-developers-sdpa
add_subdirectory(indexer_score)
add_subdirectory(topk_large_indices)

# metallium-maintainers-llama-models
add_subdirectory(paged_cache)

# metalium-developers-external-experience
add_subdirectory(isin)

# cglagovichTT, jonathansuTT (named individuals, not a team)
add_subdirectory(minimal_matmul)

# Jointly owned by convolutions + ops-data-movement — already covered by this
# file's CODEOWNERS union above, so no separate decision was needed to fold
# these in.
add_subdirectory(padded_slice)
add_subdirectory(slice_write)

# No team-specific CODEOWNERS rule — falls back to the ops-leads catch-all
# today. Kept as the status quo owner rather than guessed at.
add_subdirectory(bcast_to)
add_subdirectory(copy)
add_subdirectory(deepseek/mla/matmul_wo)
add_subdirectory(deepseek/moe/deepseek_moe_gate)
add_subdirectory(deepseek/moe/generalized_moe_gate)
add_subdirectory(deepseek/moe/moe_gate_mm)
add_subdirectory(deepseek_moe_post_combine_tilize)
add_subdirectory(dropout)
add_subdirectory(multi_scale_deformable_attn)
add_subdirectory(plusone)
add_subdirectory(reshape)
add_subdirectory(ssm)
add_subdirectory(tensor_prefetcher)
add_subdirectory(test)
add_subdirectory(topk_router_gpt)
add_subdirectory(unary_backward)

# Umbrella target for everything migrated into this file: ttnncpp links this
# one target instead of enumerating these op libraries itself. Add/remove
# entries here when adding/removing one of the ops above — never in
# ttnn/CMakeLists.txt.
add_library(ttnn_op_experimental_shared INTERFACE)
add_library(TTNN::Ops::Experimental ALIAS ttnn_op_experimental_shared)
target_link_libraries(
    ttnn_op_experimental_shared
    INTERFACE
        TTNN::Ops::Experimental::AdaptivePool
        TTNN::Ops::Experimental::Conv3d
        TTNN::Ops::Experimental::CNN
        TTNN::Ops::Experimental::CCL
        TTNN::Ops::Experimental::MoEGPT
        TTNN::Ops::Experimental::Matmul
        TTNN::Ops::Experimental::Quasar
        TTNN::Ops::Experimental::Reduction
        TTNN::Ops::Experimental::IndexerScore
        TTNN::Ops::Experimental::TopkLargeIndices
        TTNN::Ops::Experimental::PagedCache
        TTNN::Ops::Experimental::IsIn
        TTNN::Ops::Experimental::MinimalMatmul
        TTNN::Ops::Experimental::PaddedSlice
        TTNN::Ops::Experimental::SliceWrite
        TTNN::Ops::Experimental::BcastTo
        TTNN::Ops::Experimental::Copy
        TTNN::Ops::Experimental::Deepseek::MLA::Matmul_WO
        TTNN::Ops::Experimental::DeepSeek::MoE::DeepSeekMoeGate
        TTNN::Ops::Experimental::DeepSeek::MoE::GeneralizedMoeGate
        TTNN::Ops::Experimental::DeepSeek::MoE::MoEGateMM
        TTNN::Ops::Experimental::DeepseekMoEPostCombineTilize
        TTNN::Ops::Experimental::Dropout
        TTNN::Ops::Experimental::MultiScaleDeformableAttn
        TTNN::Ops::Experimental::PlusOne
        TTNN::Ops::Experimental::Reshape
        TTNN::Ops::Experimental::SSM
        TTNN::Ops::Experimental::TensorPrefetcher
        TTNN::Ops::Experimental::Test
        TTNN::Ops::Experimental::TopkRouterGpt
        TTNN::Ops::Experimental::UnaryBackward
)

# target_link_libraries() alone is not enough: per the CMake docs, an OBJECT
# library named in a target's INTERFACE_LINK_LIBRARIES (as these leaves are,
# via the INTERFACE keyword above) is treated as a plain Interface Library by
# further consumers — only usage requirements propagate, not its compiled
# objects. So with the default LIB_TYPE=OBJECT, ttnncpp would silently end up
# without these ops' code (undefined symbols at final link time). Splice the
# object files in directly via generator expression so they still land in
# whatever ultimately links this umbrella target. TARGET_OBJECTS resolves to
# nothing (and is harmless) for leaves that aren't OBJECT libraries — e.g.
# SHARED or STATIC, both of which ttnn can compile these as depending on
# build options — so this is safe unconditionally.
target_sources(
    ttnn_op_experimental_shared
    INTERFACE
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::AdaptivePool>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Conv3d>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::CNN>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::CCL>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::MoEGPT>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Matmul>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Quasar>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Reduction>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::IndexerScore>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::TopkLargeIndices>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::PagedCache>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::IsIn>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::MinimalMatmul>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::PaddedSlice>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::SliceWrite>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::BcastTo>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Copy>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Deepseek::MLA::Matmul_WO>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeek::MoE::DeepSeekMoeGate>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeek::MoE::GeneralizedMoeGate>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeek::MoE::MoEGateMM>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepseekMoEPostCombineTilize>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Dropout>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::MultiScaleDeformableAttn>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::PlusOne>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Reshape>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::SSM>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::TensorPrefetcher>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::Test>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::TopkRouterGpt>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::UnaryBackward>
)
