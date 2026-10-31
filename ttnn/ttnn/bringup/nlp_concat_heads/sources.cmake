# Source files for ttnn_op_bringup_nlp_concat_heads (written by fork_op.py: the source op had no target of its own).

set(TTNN_OP_BRINGUP_NLP_CONCAT_HEADS_API_HEADERS nlp_concat_heads.hpp)

set(TTNN_OP_BRINGUP_NLP_CONCAT_HEADS_SRCS
    device/nlp_concat_heads_device_operation.cpp
    device/nlp_concat_heads_program_factory.cpp
    nlp_concat_heads.cpp
)

set(TTNN_OP_BRINGUP_NLP_CONCAT_HEADS_NANOBIND_SRCS nlp_concat_heads_nanobind.cpp)
