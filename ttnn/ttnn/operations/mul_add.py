import ttnn


# For the golden function, use the same signature as the operation
# Keep in mind that all `ttnn.Tensor`s are converted to `torch.Tensor`s
# And arguments not needed by torch can be ignored using `*args` and `**kwargs`
def golden_function(
    input_tensor_a: "torch.Tensor", input_tensor_b: "torch.Tensor", input_tensor_c: "torch.Tensor", *args, **kwargs
):
    return (input_tensor_a * input_tensor_b) + input_tensor_c


ttnn.attach_golden_function(
    ttnn.mul_add,
    golden_function=golden_function,
)
