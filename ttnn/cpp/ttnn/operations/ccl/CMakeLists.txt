set(CCL_TTNN_SRCS
    # Common
    ${CMAKE_CURRENT_SOURCE_DIR}/erisc_datamover_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl_op_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl_host_datastructures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/types/ccl_types_args_emitters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/uops/ccl_command.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/uops/ccl_host_commands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/host/ccl_worker_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/host/ccl_command_stream_builders.cpp
    # CCL Ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/all_gather_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/device/all_gather_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather/device/multi_core/all_gather_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/device/host/reduce_scatter_full_worker_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/device/reduce_scatter_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/reduce_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/reduce_scatter_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/host/reduce_scatter_worker_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/host/reduce_scatter_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/barrier/device/host/barrier_full_worker_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/barrier/device/barrier_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/barrier/barrier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/barrier/barrier_pybind.cpp
    CACHE INTERNAL
    "CCL sources to reuse in ttnn build"
)
