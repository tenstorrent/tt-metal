set(TTNN_OP_EXPERIMENTAL_ATTN_RESIDUAL_API_HEADERS attn_res_gather_softmax.hpp)

set(TTNN_OP_EXPERIMENTAL_ATTN_RESIDUAL_SRCS
    attn_res_gather_softmax.cpp
    device/attn_res_gather_softmax_device_operation.cpp
    device/attn_res_gather_softmax_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/attn_residual/CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here rather than inline in
# CMakeLists.txt so that add/remove/rename doesn't touch a file with
# metalium-developers-infra as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_ATTN_RESIDUAL_NANOBIND_SRCS attn_res_gather_softmax_nanobind.cpp)
