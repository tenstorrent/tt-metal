# Owned by @tenstorrent/metalium-developers-eltwise. Add/remove ops for this
# domain here without touching ttnn/CMakeLists.txt.
#
# eltwise/quantization is intentionally NOT included here — it's owned by
# named individuals rather than this team (see CODEOWNERS) and stays wired
# directly in ttnn/CMakeLists.txt.
add_subdirectory(binary)
add_subdirectory(binary_backward)
add_subdirectory(binary_ng)
add_subdirectory(complex)
add_subdirectory(complex_binary)
add_subdirectory(complex_unary)
add_subdirectory(complex_unary_backward)
add_subdirectory(ternary)
add_subdirectory(ternary_backward)
add_subdirectory(unary)
add_subdirectory(unary_backward)

# Domain-level umbrella target: ttnncpp links this one target instead of
# enumerating this team's individual op libraries. Add/remove entries here
# when adding/removing a module — never in ttnn/CMakeLists.txt.
add_library(ttnn_op_eltwise INTERFACE)
add_library(TTNN::Ops::Eltwise ALIAS ttnn_op_eltwise)
target_link_libraries(
    ttnn_op_eltwise
    INTERFACE
        TTNN::Ops::Eltwise::Binary
        TTNN::Ops::Eltwise::Binary::Backward
        TTNN::Ops::Eltwise::Binary::NG
        TTNN::Ops::Eltwise::Complex
        TTNN::Ops::Eltwise::Complex::Binary
        TTNN::Ops::Eltwise::Complex::Unary
        TTNN::Ops::Eltwise::Complex::Unary::Backward
        TTNN::Ops::Eltwise::Ternary
        TTNN::Ops::Eltwise::Ternary::Backward
        TTNN::Ops::Eltwise::Unary
        TTNN::Ops::Eltwise::Unary::Backward
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
    ttnn_op_eltwise
    INTERFACE
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Binary>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Binary::Backward>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Binary::NG>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Complex>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Complex::Binary>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Complex::Unary>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Complex::Unary::Backward>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Ternary>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Ternary::Backward>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Unary>
        $<TARGET_OBJECTS:TTNN::Ops::Eltwise::Unary::Backward>
)
