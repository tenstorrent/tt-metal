set(TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_API_HEADERS
    plan/sparse_push_plan.hpp
    push/sparse_kv_push.hpp
    push/sparse_kv_push_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_SRCS
    common/sparse_kv_push_common.cpp
    plan/device/sparse_push_plan_device_operation.cpp
    plan/device/sparse_push_plan_program_factory.cpp
    plan/sparse_push_plan.cpp
    push/device/sparse_kv_push_device_operation.cpp
    push/device/sparse_kv_push_program_factory.cpp
    push/sparse_kv_push.cpp
)

set(TTNN_OP_EXPERIMENTAL_SPARSE_KV_PUSH_NANOBIND_SRCS sparse_kv_push_nanobind.cpp)
