set(CCL_EXPERIMENTAL_TTNN_SRCS
    #Experimental Ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl_experimental_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_matmul/all_gather_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_matmul/all_gather_matmul_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_matmul/device/all_gather_matmul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_matmul/device/multi_core/all_gather_matmul_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/all_reduce_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce/device/all_reduce_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_async/device/reduce_scatter_async_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_async/device/reduce_scatter_async_program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_async/reduce_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_async/reduce_scatter_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_async/all_gather_async.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_async/all_gather_async_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_async/device/all_gather_async_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_async/device/all_gather_async_program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_async/device/all_gather_async_program_minimal_variants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_async/all_reduce_async.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_async/all_reduce_async_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_async/device/all_reduce_async_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_async/device/all_reduce_async_program_minimal_variants.cpp
    CACHE INTERNAL
    "CCL Experimental sources to reuse in ttnn build"
)
