import ttnn
from .muladd_test_program_descriptor import create_program_descriptor


def muladd_test(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    input_c: ttnn.Tensor,
):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_a.shape)), input_a.dtype, ttnn.TILE_LAYOUT, input_a.device(), ttnn.DRAM_MEMORY_CONFIG
    )

    program_descriptor = create_program_descriptor(input_a, input_b, input_c, output)

    return ttnn.generic_op([input_a, input_b, input_c, output], program_descriptor)
