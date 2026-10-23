# Source files for ttnn_op_experimental_isin.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_ISIN_API_HEADERS isin.hpp)

set(TTNN_OP_EXPERIMENTAL_ISIN_SRCS
    device/isin_device_operation.cpp
    device/isin_program_factory.cpp
    isin.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/isin/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_ISIN_NANOBIND_SRCS isin_nanobind.cpp)
