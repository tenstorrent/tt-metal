# sparse_kv_push.hpp includes the kernel constants, so they are installed with it.
set(TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_API_HEADERS
    sparse_kv_push.hpp
    device/kernels/sparse_push_constants.hpp
)

set(TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_SRCS
    device/sparse_kv_push_common.cpp
    device/sparse_kv_push_device_operation.cpp
    device/sparse_kv_push_program_factory.cpp
    device/sparse_push_plan_device_operation.cpp
    sparse_kv_push.cpp
)

set(TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_NANOBIND_SRCS sparse_kv_push_nanobind.cpp)
