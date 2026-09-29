set(TTNN_OP_EXPERIMENTAL_KDA_SELECT_TILE_ROWS_API_HEADERS select_tile_rows.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_SELECT_TILE_ROWS_SRCS
    select_tile_rows.cpp
    device/select_tile_rows_device_operation.cpp
    device/select_tile_rows_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_SELECT_TILE_ROWS_NANOBIND_SRCS select_tile_rows_nanobind.cpp)
