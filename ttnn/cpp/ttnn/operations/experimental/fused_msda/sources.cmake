# Source files for ttnn_op_experimental_fused_msda.

set(TTNN_OP_EXPERIMENTAL_FUSED_MSDA_API_HEADERS fused_msda.hpp)

set(TTNN_OP_EXPERIMENTAL_FUSED_MSDA_SRCS
    fused_msda.cpp
    device/fused_msda_device_operation.cpp
    device/fused_msda_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/fused_msda/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that add/remove/rename
# doesn't touch a file with metalium-developers-infra as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_FUSED_MSDA_NANOBIND_SRCS fused_msda_nanobind.cpp)
