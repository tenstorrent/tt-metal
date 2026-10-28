# Source files for ttnn_op_ccl.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CCL_SRCS
    ccl_op_fusion.cpp
    ccl_common.cpp
    ccl_host_datastructures.cpp
    common/types/ccl_types_args_emitters.cpp
    common/host/command_backend_runtime_args_overrider.cpp
    common/uops/ccl_command.cpp
    common/uops/command_lowering.cpp
    common/uops/ccl_host_commands.cpp
    common/host/ccl_worker_builder.cpp
    common/host/ccl_command_stream_builders.cpp
    common/host/moe_utils.cpp
    common/host/mesh_ring_plan.cpp
)

set(TTNN_OP_CCL_API_HEADERS
    ccl_common.hpp
    ccl_host_datastructures.hpp
    ccl_host_types.hpp
    ccl_op_fusion.hpp
    common/host/ccl_command_stream_builders.hpp
    common/host/moe_utils.hpp
    common/host/mesh_ring_plan.hpp
    common/types/ccl_types.hpp
    common/uops/ccl_command.hpp
    shared_with_host/hetergeneous_data_structs.hpp
    shared_with_host/snake_ring.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/ccl/CMakeLists.txt (see the `if(TARGET ttnn)`
# block there). Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra as a
# required co-owner.
set(TTNN_OP_CCL_NANOBIND_SRCS ccl_nanobind.cpp)
