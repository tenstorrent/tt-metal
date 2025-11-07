import ttnn
import torch

# torch.set_printoptions(precision=20,threshold=1024)

device_id = 0
device = ttnn.open_device(device_id=device_id)
dtype = ttnn.float32
w = 32 * 32
h = 32 * 32

try:
    a_torch = torch.full((1, 1, w, h), 1.0, dtype=torch.float64)
    b_torch = torch.arange(0, w * h, dtype=torch.float64).reshape(1, 1, w, h)
    # b_torch = torch.full((1, 1, w, h), 11.0, dtype=torch.float64)
    c_torch = torch.full((1, 1, w, h), 33.0, dtype=torch.float64)

    condition = a_torch != 0.0

    a = ttnn.from_torch(a_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # c = ttnn.from_torch(c_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    c = 33.0  # causes failure!

    d = ttnn.where(a, b, c)
    d_torch = ttnn.to_torch(d)
    d_torch_gold = torch.where(condition, b_torch, c_torch)

    print(d_torch, d_torch_gold)
    mask = d_torch != d_torch_gold
    print("masked", d_torch[mask], d_torch_gold[mask])

    assert torch.all(d_torch == d_torch_gold)

finally:
    ttnn.close_device(device)
