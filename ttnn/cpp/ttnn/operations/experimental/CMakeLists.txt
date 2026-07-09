# experimental/ is one physical directory shared by many teams' ops, so this
# file is jointly owned by every team (or named individual, for ops without a
# team owner) with a subdirectory listed below (see CODEOWNERS) rather than
# by ttnn-core. Any of those owners can add/remove their own experimental ops
# here and self-approve, without ttnn-core or infra needing to be pinged, and
# without touching ttnn/CMakeLists.txt.
#
# Only the ops below have been migrated in so far — everything else under
# experimental/ is still add_subdirectory()'d directly from
# ttnn/CMakeLists.txt, either because it has no team/individual owner in
# CODEOWNERS at all (a gap to close separately, not a CMake problem), or
# because it's one CMake target spanning multiple teams' code and needs an
# actual target split before ownership can split too (e.g. transformer/).
# Folding in more is later work, one owner at a time.

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

# Umbrella target for everything migrated into this file so far: ttnncpp
# links this one target instead of enumerating these 15 op libraries itself.
# Add/remove entries here when adding/removing one of the ops above — never
# in ttnn/CMakeLists.txt.
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
)
