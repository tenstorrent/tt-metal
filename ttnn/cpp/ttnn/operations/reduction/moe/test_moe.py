import torch
import ttnn
import time


def test_moe(device):
    print("start test_moe")
    torch_gates = torch.randn(1, 1, 32, 64)
    torch_gates[:, :, :, 8:] = 0  # float('-inf')
    for i in range(2):
        mask = torch.zeros(1, 1, 1, 32)
        mask[:, :, :, 2:] = float("-inf")
        mask_2 = torch.zeros(1, 1, 1, 64)
        mask_2[:, :, :, 8:] = float("-inf")

        gate_logits_1SBE = ttnn.from_torch(
            torch_gates,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
        )
        top2_mask_11BB = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
        )
        expert_mask_11BB = ttnn.from_torch(
            mask_2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
        )

        start_time = time.time()
        weights_1SB1 = ttnn.moe(gate_logits_1SBE, expert_mask_11BB, top2_mask_11BB, 32)

        vals, ind = torch.topk(torch_gates + mask_2, 32, dim=-1)
        # print(ind==0)
        # print(torch.sum((torch.softmax(vals + mask, dim=-1)*((ind)==0))[:, :, :, :2], dim=-1, keepdim=True))
        print(ttnn.to_torch(weights_1SB1)[:, :, :, :1])
        # print("time", time.time()- start_time)

        # gate_logits_1SBE.deallocate()
        # top2_mask_11BB.deallocate()
        # expert_mask_11BB.deallocate()
        # weights_1SB1.deallocate()
