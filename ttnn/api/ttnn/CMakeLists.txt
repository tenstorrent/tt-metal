add_library(ttnn_api INTERFACE)
add_library(TTNN::Api ALIAS ttnn_api)

target_sources(
    ttnn_api
    INTERFACE
        FILE_SET ttnn_api
        TYPE HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/ttnn/api/ttnn/
        FILES tensor/enum_types.hpp
)

install(
    TARGETS
        ttnn_api
    FILE_SET
    ttnn_api
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/api/ttnn/
        COMPONENT ttnn-runtime
)
