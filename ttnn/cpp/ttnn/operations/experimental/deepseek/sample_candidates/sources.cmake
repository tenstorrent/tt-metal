# Source files for ttnn_op_experimental_deepseek_sample_candidates.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_SAMPLE_CANDIDATES_API_HEADERS sample_candidates.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_SAMPLE_CANDIDATES_SRCS
    sample_candidates.cpp
    device/sample_candidates_device_operation.cpp
    device/sample_candidates_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/sample_candidates/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_SAMPLE_CANDIDATES_NANOBIND_SRCS sample_candidates_nanobind.cpp)
