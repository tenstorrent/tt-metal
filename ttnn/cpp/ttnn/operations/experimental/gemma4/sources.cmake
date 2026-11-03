# Source files for ttnn_op_experimental_gemma4.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_GEMMA4_API_HEADERS
    sliding_ring_sdpa/sliding_ring_sdpa.hpp
    sliding_ring_sdpa/device/sliding_halo_exchange.hpp
    sliding_ring_sdpa/device/sliding_ring_sdpa_device_operation.hpp
    sliding_ring_sdpa/device/sliding_ring_sdpa_device_operation_types.hpp
    sliding_ring_sdpa/device/sliding_ring_sdpa_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_GEMMA4_SRCS
    sliding_ring_sdpa/sliding_ring_sdpa.cpp
    sliding_ring_sdpa/device/sliding_halo_exchange.cpp
    sliding_ring_sdpa/device/sliding_ring_sdpa_device_operation.cpp
    sliding_ring_sdpa/device/sliding_ring_sdpa_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/gemma4/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_GEMMA4_NANOBIND_SRCS sliding_ring_sdpa/sliding_ring_sdpa_nanobind.cpp)
