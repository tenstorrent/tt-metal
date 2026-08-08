# KDA gated-RMS device kernel registration.
set(TTNN_KDA_GATED_RMS_KERNELS
    ${CMAKE_CURRENT_LIST_DIR}/device/kernels/compute/kda_gated_rms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kernels/dataflow/reader_kda_gated_rms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kernels/dataflow/writer_kda_gated_rms.cpp
)

set(TTNN_KDA_GATED_RMS_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/kda_gated_rms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_gated_rms_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_gated_rms_program_factory.cpp
)

set(TTNN_KDA_GATED_RMS_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/kda_gated_rms_nanobind.cpp)
