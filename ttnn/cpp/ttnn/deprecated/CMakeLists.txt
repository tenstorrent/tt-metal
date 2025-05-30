add_library(ttnn_deprecated INTERFACE)

set_target_properties(
    ttnn_deprecated
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
target_sources(
    ttnn_deprecated
    INTERFACE
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            tt_dnn/kernels/compute/moreh_common.hpp
            tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp
            tt_dnn/kernels/dataflow/generate_mm_scaler.hpp
            tt_dnn/kernels/dataflow/moreh_common.hpp
)

install(
    TARGETS
        ttnn_deprecated
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/deprecated
        COMPONENT ttnn-runtime
)
