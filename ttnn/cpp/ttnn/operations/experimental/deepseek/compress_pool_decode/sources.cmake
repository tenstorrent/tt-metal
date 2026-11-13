# Source files for ttnn_op_experimental_deepseek_compress_pool_decode.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_COMPRESS_POOL_DECODE_API_HEADERS compress_pool_decode.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_COMPRESS_POOL_DECODE_SRCS
    compress_pool_decode.cpp
    device/compress_pool_decode_device_operation.cpp
    device/compress_pool_decode_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/compress_pool_decode/CMakeLists.txt
# (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_COMPRESS_POOL_DECODE_NANOBIND_SRCS compress_pool_decode_nanobind.cpp)
