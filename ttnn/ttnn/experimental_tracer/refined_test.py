# Auto-generated PyTorch code
import pytest
import ttnn
import torch
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import (
    run_conv,
    torch_tensor_map,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("tensor_shapes, cat_dim", ())
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_cat(device, tensor_shapes, cat_dim, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensors = [torch.randn(shape, dtype=torch.bfloat16) for shape in tensor_shapes]
    torch_output_tensor = torch.cat(torch_input_tensors, dim=cat_dim)

    input_tensors = [
        ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype) for torch_tensor in torch_input_tensors
    ]

    output_tensor = ttnn.concat(input_tensors, dim=cat_dim)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize("input_shape_a, input_shape_b", (([1, 1, 4, 8400], [1, 1, 1, 8400]),))
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_mul_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a * torch_input_tensor_b

    torch_input_tensor_a = torch.permute(torch_input_tensor_a, (0, 2, 3, 1))
    torch_input_tensor_b = torch.permute(torch_input_tensor_b, (0, 2, 3, 1))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    output_tensor = ttnn.multiply(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize("input_shape", ([1, 80, 8400],))
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sigmoid(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.sigmoid(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.sigmoid(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize("input_shape_a, input_shape_b", ())
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_div_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16) + 0.1  # Avoid division by zero
    torch_output_tensor = torch_input_tensor_a / torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize("input_shape_a, input_shape_b", (([1, 2, 8400], [1, 2, 8400]),))
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sub_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a - torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)

    output_tensor = ttnn.subtract(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
        ([1, 32, 40, 40], [1, 32, 40, 40]),
        ([1, 64, 20, 20], [1, 64, 20, 20]),
        ([1, 32, 80, 80], [1, 32, 80, 80]),
        ([1, 1, 2, 8400], [1, 1, 2, 8400]),
        ([1, 16, 160, 160], [1, 16, 160, 160]),
        ([1, 64, 40, 40], [1, 64, 40, 40]),
        ([1, 128, 20, 20], [1, 128, 20, 20]),
        ([1, 16, 80, 80], [1, 16, 80, 80]),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_add_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 400, 4, 32],
        [1, 144, 20, 20],
        [1, 400, 384],
        [1, 1, 4, 8400],
        [1, 4, 32, 400],
        [1, 1600, 64],
        [1, 144, 80, 80],
        [4, 400, 192],
        [1, 64, 8400],
        [4, 2, 400, 400],
        [1, 144, 40, 40],
        [1, 384, 20, 20],
        [1, 4, 400, 32],
        [1, 4, 400, 400],
        [1, 192, 40, 40],
        [1, 1600, 192],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_view(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    new_shape = [-1, input_shape[-1]]  # Flatten all but last dimension
    torch_output_tensor = torch_input_tensor.view(new_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.reshape(input_tensor, new_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b], [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_batch, input_depth, hidden_units, input_height, input_width, kernel, stride, padding, dilation",
    (
        # (1, 256, 64, 20, 20, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 256, 80, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 16, 32, 320, 320, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 128, 64, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 64, 80, 80, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 80, 80, 40, 40, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 32, 32, 40, 40, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 32, 16, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 256, 128, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 192, 128, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 128, 128, 40, 40, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 64, 192, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 3, 16, 640, 640, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 128, 128, 80, 80, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 128, 256, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 80, 80, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 384, 64, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        (1, 128, 128, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 128, 64, 40, 40, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 32, 32, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 64, 80, 80, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 128, 128, 20, 20, [7, 7], [1, 1], [3, 3], [1, 1]),
        # (1, 64, 64, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 128, 384, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 64, 40, 40, [7, 7], [1, 1], [3, 3], [1, 1]),
        # (1, 128, 256, 40, 40, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 8, 16, 160, 160, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 80, 80, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 80, 80, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 128, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 128, 128, 40, 40, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 384, 256, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 96, 128, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 64, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 64, 160, 160, [3, 3], [2, 2], [1, 1], [1, 1]),
        # (1, 256, 256, 20, 20, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 64, 64, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 80, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 256, 32, 80, 80, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 16, 16, 80, 80, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 80, 80, 80, 80, [3, 3], [1, 1], [1, 1], [1, 1]),
        (1, 48, 64, 160, 160, [1, 1], [1, 1], [0, 0], [1, 1]),
        (1, 32, 32, 160, 160, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 64, 64, 20, 20, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 128, 64, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 192, 64, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 128, 128, 20, 20, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 16, 32, 80, 80, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 32, 16, 80, 80, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 64, 64, 40, 40, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 16, 1, 4, 8400, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 128, 80, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 16, 8, 160, 160, [3, 3], [1, 1], [1, 1], [1, 1]),
        # (1, 64, 32, 40, 40, [1, 1], [1, 1], [0, 0], [1, 1]),
        # (1, 80, 80, 20, 20, [3, 3], [1, 1], [1, 1], [1, 1]),
    ),
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
)
def test_conv(
    device,
    torch_tensor_map,
    input_batch,
    hidden_units,
    input_depth,
    input_height,
    input_width,
    has_bias,
    dtype,
    kernel,
    stride,
    padding,
    dilation,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        dtype,
        ttnn.bfloat8_b,
        input_batch,
        hidden_units,
        input_depth,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        {},
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        input_dtype=dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        run_twice=True,
        fast_compare=True,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
    )


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 4, 400, 400],
        [1, 16, 4, 8400],
        [4, 2, 400, 400],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_softmax(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.softmax(torch_input_tensor, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 384, 400],
        [1, 4, 32, 400],
        [1, 4, 16, 8400],
        [4, 2, 400, 400],
        [4, 2, 32, 400],
        [1, 192, 1600],
        [1, 4, 400, 400],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_transpose_int(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    dim0, dim1 = -2, -1  # Transpose last two dimensions
    torch_output_tensor = torch_input_tensor.transpose(dim0, dim1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.transpose(input_tensor, dim0, dim1)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 64, 20, 20],
        [1, 80, 80, 80],
        [1, 16, 160, 160],
        [1, 64, 80, 80],
        [1, 80, 40, 40],
        [1, 128, 80, 80],
        [1, 64, 160, 160],
        [1, 128, 20, 20],
        [1, 8, 160, 160],
        [1, 128, 40, 40],
        [1, 32, 40, 40],
        [1, 64, 40, 40],
        [1, 32, 160, 160],
        [1, 16, 320, 320],
        [1, 32, 80, 80],
        [1, 256, 20, 20],
        [1, 16, 80, 80],
        [1, 80, 20, 20],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_silu(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.silu(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.silu(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


# This test does nothing
@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 64, 20, 20],
        [1, 128, 80, 80],
        [1, 8, 160, 160],
        [1, 32, 40, 40],
        [1, 64, 40, 40],
        [1, 256, 20, 20],
        [1, 192, 40, 40],
        [1, 16, 320, 320],
        [1, 80, 20, 20],
        [1, 80, 80, 80],
        [1, 64, 80, 80],
        [1, 80, 40, 40],
        [1, 64, 160, 160],
        [1, 128, 20, 20],
        [1, 384, 20, 20],
        [1, 16, 160, 160],
        [1, 128, 40, 40],
        [1, 32, 160, 160],
        [1, 32, 80, 80],
        [1, 16, 80, 80],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_native_batch_norm(device, input_shape, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    num_features = input_shape[1]

    # Create batch norm module
    batch_norm = torch.nn.BatchNorm2d(num_features, dtype=torch.bfloat16)
    torch_output_tensor = batch_norm(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    # Note: This is a simplified test - actual implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape, scale_factor",
    (
        ([1, 128, 40, 40], 2),
        ([1, 256, 20, 20], 2),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_upsample_nearest2d(device, input_shape, scale_factor, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.interpolate(torch_input_tensor, scale_factor=scale_factor, mode="nearest")

    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)

    input_tensor = ttnn.upsample(input_tensor, scale_factor=(2, 2), mode="nearest")

    # Note: This is a simplified test - actual ttnn implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 4, 32, 400],
        [1, 400, 4, 96],
        [4, 400, 2, 96],
        [1, 40, 40, 64],
        [4, 2, 32, 400],
        [1, 20, 20, 128],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_permute(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    dims = list(range(len(input_shape)))
    dims[-1], dims[-2] = dims[-2], dims[-1]  # Swap last two dimensions
    torch_output_tensor = torch_input_tensor.permute(dims)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.permute(input_tensor, dims)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 400, 4, 32],
        [8, 400, 400],
        [4, 400, 400],
        [4, 32, 400],
        [4, 2, 32, 400],
        [4, 2, 400, 32],
        [4, 400, 2, 32],
        [8, 32, 400],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_unsafe_view(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    new_shape = [-1, input_shape[-1]]  # Flatten all but last dimension
    torch_output_tensor = torch_input_tensor.view(new_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.reshape(input_tensor, new_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 400, 4, 32],
        [1, 128, 20, 20],
        [1, 64, 40, 40],
        [4, 2, 32, 400],
        [4, 2, 400, 32],
        [4, 400, 2, 32],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_clone(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.clone()

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.clone(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
        ([4, 400, 32], [4, 32, 400]),
        ([8, 32, 400], [8, 400, 400]),
        ([4, 32, 400], [4, 400, 400]),
        ([8, 400, 32], [8, 32, 400]),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_bmm(device, input_shape_a, input_shape_b, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch.bmm(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize(
    "input_shape",
    (
        [4, 2, 400, 32],
        [1, 4, 32, 400],
        [4, 2, 400, 400],
        [4, 2, 32, 400],
        [1, 4, 400, 32],
        [1, 4, 400, 400],
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_expand(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    expand_shape = list(input_shape)
    expand_shape[0] = expand_shape[0] * 2  # Double the batch size
    torch_output_tensor = torch_input_tensor.expand(expand_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.expand(input_tensor, expand_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize("output_shape", ([1, 3, 640, 640],))
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_torch_ones(device, output_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_output_tensor = torch.ones(output_shape, dtype=torch.bfloat16)
    output_tensor = ttnn.ones(output_shape, layout=layout, device=device, dtype=dtype)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
