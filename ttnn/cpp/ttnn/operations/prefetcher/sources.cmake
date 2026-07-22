# Source files for ttnn_op_prefetcher.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_PREFETCHER_API_HEADERS prefetcher/dram_prefetcher.hpp)

set(TTNN_OP_PREFETCHER_SRCS
    prefetcher/device/dram_prefetcher_device_operation.cpp
    prefetcher/device/dram_prefetcher_program_factory.cpp
    prefetcher/dram_prefetcher.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/prefetcher/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_PREFETCHER_NANOBIND_SRCS
    prefetcher/dram_prefetcher_nanobind.cpp
    prefetcher_nanobind.cpp
)
