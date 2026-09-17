set(TTNN_OP_EXPERIMENTAL_KDA_CHRONOLOGICAL_TOPOLOGY_API_HEADERS
    chronological_selections.hpp
    chronology.hpp
)
set(TTNN_OP_EXPERIMENTAL_KDA_CHRONOLOGICAL_TOPOLOGY_SRCS
    chronological_selections.cpp
    device/chronological_selections_device_operation.cpp
    device/chronological_selections_program_factory.cpp
)
set(TTNN_OP_EXPERIMENTAL_KDA_CHRONOLOGICAL_TOPOLOGY_NANOBIND_SRCS chronological_selections_nanobind.cpp)
