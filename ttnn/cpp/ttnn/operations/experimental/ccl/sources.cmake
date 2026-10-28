# Source files for the experimental CCL bindings.
# The experimental CCL ops were removed for agent evaluation; only the fabric_mux binding
# and the (empty) ttnn.experimental CCL submodule remain.
#
# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/ccl/CMakeLists.txt (see the
# `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_CCL_NANOBIND_SRCS
    ccl_experimental_nanobind.cpp
    fabric_mux/fabric_mux_nanobind.cpp
)
