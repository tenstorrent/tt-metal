# Source files for ttnn_op_experimental_padded_slice.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_PADDED_SLICE_API_HEADERS padded_slice.hpp)

set(TTNN_OP_EXPERIMENTAL_PADDED_SLICE_SRCS
    padded_slice.cpp
    device/padded_slice_device_operation.cpp
    device/padded_slice_utils.cpp
    device/padded_slice_rm_program_factory.cpp
    device/padded_slice_tile_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/padded_slice/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_PADDED_SLICE_NANOBIND_SRCS padded_slice_nanobind.cpp)
