# operations/ is one physical directory shared by every op-owning team, so
# this file is jointly owned by every team with a subdirectory listed below
# (see CODEOWNERS) rather than by ops-leads/ttnn-core. Any of those teams can
# add/remove their own op here and self-approve, without ttnn-core or infra
# needing to be pinged, and without touching ttnn/CMakeLists.txt. Same idea
# as ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt, one level up.
#
# Only the ops below have been migrated in so far — everything else under
# operations/ (eltwise, experimental, moreh, bernoulli, etc.) either already
# has its own dedicated CMakeLists.txt or is still add_subdirectory()'d
# directly from ttnn/CMakeLists.txt. Folding in more is later work, one team
# at a time.

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

# Umbrella target for everything migrated into this file so far: ttnncpp
# links this one target instead of enumerating these 11 op libraries itself.
# Add/remove entries here when adding/removing one of the ops above — never
# in ttnn/CMakeLists.txt.
add_library(ttnn_op_operations_shared INTERFACE)
add_library(TTNN::Ops::OperationsShared ALIAS ttnn_op_operations_shared)
target_link_libraries(
    ttnn_op_operations_shared
    INTERFACE
        TTNN::Ops::Conv
        TTNN::Ops::Pool
        TTNN::Ops::SlidingWindow
        TTNN::Ops::CCL
        TTNN::Ops::PointToPoint
        TTNN::Ops::DataMovement
        TTNN::Ops::Embedding
        TTNN::Ops::Embedding::Backward
        TTNN::Ops::Matmul
        TTNN::Ops::Normalization
        TTNN::Ops::Reduction
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
        $<TARGET_OBJECTS:TTNN::Ops::Conv>
        $<TARGET_OBJECTS:TTNN::Ops::Pool>
        $<TARGET_OBJECTS:TTNN::Ops::SlidingWindow>
        $<TARGET_OBJECTS:TTNN::Ops::CCL>
        $<TARGET_OBJECTS:TTNN::Ops::PointToPoint>
        $<TARGET_OBJECTS:TTNN::Ops::DataMovement>
        $<TARGET_OBJECTS:TTNN::Ops::Embedding>
        $<TARGET_OBJECTS:TTNN::Ops::Embedding::Backward>
        $<TARGET_OBJECTS:TTNN::Ops::Matmul>
        $<TARGET_OBJECTS:TTNN::Ops::Normalization>
        $<TARGET_OBJECTS:TTNN::Ops::Reduction>
)
