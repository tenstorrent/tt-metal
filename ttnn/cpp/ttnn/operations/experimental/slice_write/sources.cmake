# Source files for ttnn_op_experimental_slice_write.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_SLICE_WRITE_SRCS
    slice_write.cpp
    device/slice_write_device_operation.cpp
    device/slice_write_rm_sharded_input_program_factory.cpp
    device/slice_write_tiled_sharded_input_program_factory.cpp
    device/slice_write_rm_interleaved_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_SLICE_WRITE_API_HEADERS slice_write.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/slice_write/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_SLICE_WRITE_NANOBIND_SRCS slice_write_nanobind.cpp)
