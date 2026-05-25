set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ITERATIVE_TOPK_API_HEADERS iterative_topk.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ITERATIVE_TOPK_SRCS
    device/iterative_topk_device_operation.cpp
    device/iterative_topk_program_factory.cpp
    device/iterative_topk_sharded_program_factory.cpp
    iterative_topk.cpp
)
