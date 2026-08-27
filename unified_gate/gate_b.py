# SPDX-License-Identifier: Apache-2.0
#
# GATE B, from Python -- the same thing gate_host_b.cpp does, through the ttnn.program_spec
# shim, with ttnn tensors and a torch comparison.
#
# This is the checkpoint the port needs: if this passes, unified_harness.py has a route to
# Metal 2.0 and the suites can start moving over one at a time.

import torch
import ttnn

ps = ttnn.program_spec

KERNEL = "unified_gate/gate_b.cpp"
TILES_PER_BLOCK = 4
NUM_BLOCKS = 4
NUM_TILES = TILES_PER_BLOCK * NUM_BLOCKS  # 16 tiles == 128 x 128
TILE_BYTES = 32 * 32 * 2  # bfloat16

# The kernel's DM thread numbering is adaptor_v1's, not the host's: COMPILE_FOR_BRISC is
# thread 0 and COMPILE_FOR_NCRISC is thread 1. gate_b.cpp does noc_load<0> and
# noc_store<1>, so dm0 (RISCV_0) produces `in` and dm1 (RISCV_1) consumes `out`. The DFB
# bindings below have to agree with that, and nothing but this comment says so.
SLOT_IN = 0
SLOT_OUT = 1


def build_spec(tensor_spec, repo_root):
    cta = {
        "cb_in": SLOT_IN,
        "cb_out": SLOT_OUT,
        "num_blocks": NUM_BLOCKS,
        "tiles_per_block": TILES_PER_BLOCK,
    }

    def kernel(name, hw_config):
        k = ps.KernelSpec()
        k.unique_id = name
        k.source = KERNEL
        k.hw_config = hw_config
        k.compile_time_args = cta
        k.compiler_options.include_paths = [repo_root]
        # Every projection names both accessors, so every projection binds both tensors.
        # Legal because a tensor binding carries no exclusive role -- unlike a DFB binding.
        k.tensor_bindings = [
            ps.KernelSpec.TensorBinding(),
            ps.KernelSpec.TensorBinding(),
        ]
        k.tensor_bindings[0].tensor_parameter_name = "in"
        k.tensor_bindings[0].accessor_name = "in"
        k.tensor_bindings[1].tensor_parameter_name = "out"
        k.tensor_bindings[1].accessor_name = "out"
        return k

    dm0_cfg = ps.DataMovementGen1Config()
    dm0_cfg.processor = ttnn.DataMovementProcessor.RISCV_0
    dm0_cfg.noc = ttnn.NOC.NOC_0
    dm1_cfg = ps.DataMovementGen1Config()
    dm1_cfg.processor = ttnn.DataMovementProcessor.RISCV_1
    dm1_cfg.noc = ttnn.NOC.NOC_1

    dm0 = kernel("dm0", dm0_cfg)
    dm1 = kernel("dm1", dm1_cfg)
    compute = kernel("compute", ps.ComputeGen1Config())

    dm0.dfb_bindings = [ps.producer_of("in", "in")]
    compute.dfb_bindings = [ps.consumer_of("in", "in"), ps.producer_of("out", "out")]
    dm1.dfb_bindings = [ps.consumer_of("out", "out")]

    def dfb(name):
        d = ps.DataflowBufferSpec()
        d.unique_id = name
        d.entry_size = TILE_BYTES
        d.num_entries = 2 * TILES_PER_BLOCK
        d.data_format_metadata = ttnn.bfloat16
        return d

    wu = ps.WorkUnitSpec()
    wu.name = "wu0"
    wu.kernels = ["dm0", "dm1", "compute"]
    wu.target_nodes = ttnn.CoreCoord(0, 0)

    spec = ps.ProgramSpec()
    spec.name = "unified_gate_b_py"
    spec.kernels = [dm0, dm1, compute]
    spec.dataflow_buffers = [dfb("in"), dfb("out")]
    spec.tensor_parameters = [
        ps.TensorParameter("in", tensor_spec),
        ps.TensorParameter("out", tensor_spec),
    ]
    spec.work_units = [wu]
    return spec


def main():
    import os

    repo_root = os.environ.get("TT_METAL_HOME", os.getcwd())
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        # Away from zero, so recip is well conditioned.
        src = torch.rand(1, 1, 128, 128, dtype=torch.float32) * 4.0 + 0.5

        t_in = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t_out = ttnn.from_torch(torch.zeros_like(src), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        spec = build_spec(t_in.spec, repo_root)
        run_args = ps.ProgramRunArgs()
        ps.run_program_spec(device, spec, run_args, [("in", t_in), ("out", t_out)])

        got = ttnn.to_torch(t_out).float()
        want = 1.0 / src.to(torch.bfloat16).float()

        rel = ((got - want).abs() / want.abs()).max().item()
        pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
        ok = rel < 0.02 and pcc > 0.999
        print(f"max rel err = {rel:.5f}   PCC = {pcc:.6f}   {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    raise SystemExit(main())
