set(TTNN_OP_EXPERIMENTAL_GEMMA4_API_HEADERS
    rjsdpa.hpp
    update_cache.hpp
)
set(TTNN_OP_EXPERIMENTAL_GEMMA4_SRCS
    rjsdpa.cpp
    device/update_cache_device_operation.cpp
)
set(TTNN_OP_EXPERIMENTAL_GEMMA4_NANOBIND_SRCS rjsdpa_nanobind.cpp)
