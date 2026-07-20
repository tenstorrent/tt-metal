# operations/ is one physical directory shared by every op-owning team, so
# this file is jointly owned by every team with a subdirectory listed below
# (see CODEOWNERS) rather than by ops-leads/ttnn-core. Any of those teams can
# add/remove their own op here and self-approve, without ttnn-core or infra
# needing to be pinged, and without touching ttnn/CMakeLists.txt. Same idea
# as ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt, one level up.
#
# Everything ownerless is migrated in here too. Not yet migrated: eltwise,
# experimental (both have their own dedicated CMakeLists.txt already), moreh
# and kernel_helper_functions (already single add_subdirectory()/
# target_link_libraries() lines — no scattering to fix, so nothing gained by
# moving them), and transformer/ (one CMake target spanning multiple teams'
# code — needs an actual target split before ownership can split too).

# metalium-developers-convolutions
add_subdirectory(conv)
add_subdirectory(pool)
add_subdirectory(sliding_window)

# metalium-developers-ops-data-movement
add_subdirectory(ccl)
add_subdirectory(point_to_point)
add_subdirectory(data_movement)
add_subdirectory(embedding)
add_subdirectory(embedding_backward)

# metalium-developers-mmfusedreduce
add_subdirectory(matmul)
add_subdirectory(normalization)
add_subdirectory(reduction)

# No team-specific CODEOWNERS rule — falls back to the ops-leads catch-all
# today. Kept as the status quo owner rather than guessed at.
add_subdirectory(bernoulli)
add_subdirectory(creation)
add_subdirectory(core)
add_subdirectory(debug)
add_subdirectory(examples)
add_subdirectory(full)
add_subdirectory(full_like)
add_subdirectory(index_fill)
add_subdirectory(kv_cache)
add_subdirectory(loss)
add_subdirectory(prefetcher)
add_subdirectory(rand)
add_subdirectory(randn)
add_subdirectory(uniform)

# Umbrella target for everything migrated into this file: ttnncpp links this
# one target instead of enumerating these op libraries itself. Add/remove
# entries here when adding/removing one of the ops above — never in
# ttnn/CMakeLists.txt.
add_library(ttnn_op_operations_shared INTERFACE)
add_library(TTNN::Ops::OperationsShared ALIAS ttnn_op_operations_shared)
target_link_libraries(
    ttnn_op_operations_shared
    INTERFACE
        TTNN::Ops::Bernoulli
        TTNN::Ops::CCL
        TTNN::Ops::Conv
        TTNN::Ops::Core
        TTNN::Ops::Creation
        TTNN::Ops::DataMovement
        TTNN::Ops::Debug
        TTNN::Ops::Embedding
        TTNN::Ops::Embedding::Backward
        TTNN::Ops::Examples
        TTNN::Ops::Full
        TTNN::Ops::FullLike
        TTNN::Ops::IndexFill
        TTNN::Ops::KvCache
        TTNN::Ops::Loss
        TTNN::Ops::Matmul
        TTNN::Ops::Normalization
        TTNN::Ops::PointToPoint
        TTNN::Ops::Pool
        TTNN::Ops::Prefetcher
        TTNN::Ops::Rand
        TTNN::Ops::Randn
        TTNN::Ops::Reduction
        TTNN::Ops::SlidingWindow
        TTNN::Ops::Uniform
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
    ttnn_op_operations_shared
    INTERFACE
        $<TARGET_OBJECTS:TTNN::Ops::Bernoulli>
        $<TARGET_OBJECTS:TTNN::Ops::CCL>
        $<TARGET_OBJECTS:TTNN::Ops::Conv>
        $<TARGET_OBJECTS:TTNN::Ops::Core>
        $<TARGET_OBJECTS:TTNN::Ops::Creation>
        $<TARGET_OBJECTS:TTNN::Ops::DataMovement>
        $<TARGET_OBJECTS:TTNN::Ops::Debug>
        $<TARGET_OBJECTS:TTNN::Ops::Embedding>
        $<TARGET_OBJECTS:TTNN::Ops::Embedding::Backward>
        $<TARGET_OBJECTS:TTNN::Ops::Examples>
        $<TARGET_OBJECTS:TTNN::Ops::Full>
        $<TARGET_OBJECTS:TTNN::Ops::FullLike>
        $<TARGET_OBJECTS:TTNN::Ops::IndexFill>
        $<TARGET_OBJECTS:TTNN::Ops::KvCache>
        $<TARGET_OBJECTS:TTNN::Ops::Loss>
        $<TARGET_OBJECTS:TTNN::Ops::Matmul>
        $<TARGET_OBJECTS:TTNN::Ops::Normalization>
        $<TARGET_OBJECTS:TTNN::Ops::PointToPoint>
        $<TARGET_OBJECTS:TTNN::Ops::Pool>
        $<TARGET_OBJECTS:TTNN::Ops::Prefetcher>
        $<TARGET_OBJECTS:TTNN::Ops::Rand>
        $<TARGET_OBJECTS:TTNN::Ops::Randn>
        $<TARGET_OBJECTS:TTNN::Ops::Reduction>
        $<TARGET_OBJECTS:TTNN::Ops::SlidingWindow>
        $<TARGET_OBJECTS:TTNN::Ops::Uniform>
)
