# Topology probe for tiny_placement: worker NoC ids and the DRAM bank endpoints the accessor uses.
import json
from pathlib import Path
import pytest, ttnn

torch = __import__("torch")  # experiment script; ttnn bans a global torch import

HERE = Path(__file__).parent


def test_topo(device):
    grid = device.compute_with_storage_grid_size()
    n = grid.x * grid.y
    out = ttnn.from_torch(
        torch.zeros(n, 32, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cores = ttnn.num_cores_to_corerangeset(n, grid, row_wise=True)
    clist = ttnn.corerange_to_cores(cores, None, True)
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(clist):
        rt[c.x][c.y] = [i]
    acc = ttnn.TensorAccessorArgs(out)
    k = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "kernels/topo_probe.cpp"),
        core_ranges=cores,
        compile_time_args=[0, *acc.get_compile_time_args()],
        runtime_args=rt,
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=128,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=128)],
    )
    ttnn.generic_op([out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))
    t = ttnn.to_torch(out).to(torch.int64) & 0xFFFFFFFF
    dec = lambda hi: ((hi >> 4) & 0x3F, (hi >> 10) & 0x3F)  # addr>>32: x at bit 36, y at bit 42
    res = {"grid": [grid.x, grid.y], "cores": []}
    for i, c in enumerate(clist):
        r = t[i].tolist()
        res["cores"].append(
            {
                "logical": [c.x, c.y],
                "noc0_id": r[0:2],
                "noc1_id": r[2:4],
                "bank_noc0": [dec(v) for v in r[4:12]],
                "bank_noc1": [dec(v) for v in r[12:20]],
            }
        )
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        a = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
        res[f"optimal_{noc}"] = [[c.x, c.y] for c in a]
    res["dram_grid"] = [device.dram_grid_size().x, device.dram_grid_size().y]
    for c in [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9)]:
        try:
            v = device.worker_core_from_logical_core(c)
            res[f"virt_{c.x}_{c.y}"] = [v.x, v.y]
        except Exception as e:
            res["virt_err"] = str(e)[:200]
    (HERE / "topo.json").write_text(json.dumps(res, indent=1))
    print("TOPO", json.dumps({k: v for k, v in res.items() if k != "cores"}))
    for e in res["cores"][:12] + res["cores"][-3:]:
        print("TOPO_CORE", e)
