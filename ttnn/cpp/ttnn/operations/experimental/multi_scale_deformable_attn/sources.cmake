# Source files for ttnn_op_experimental_multi_scale_deformable_attn.

set(TTNN_OP_EXPERIMENTAL_MSDA_API_HEADERS multi_scale_deformable_attn.hpp)

set(TTNN_OP_EXPERIMENTAL_MSDA_SRCS
    multi_scale_deformable_attn.cpp
    device/multi_scale_deformable_attn_device_operation.cpp
    device/multi_scale_deformable_attn_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/multi_scale_deformable_attn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_MSDA_NANOBIND_SRCS multi_scale_deformable_attn_nanobind.cpp)
