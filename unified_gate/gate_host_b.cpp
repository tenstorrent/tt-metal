// SPDX-License-Identifier: Apache-2.0
//
// Gate B host: runs the unified library's own kernel shape (unary.cpp) under a
// Metal 2.0 ProgramSpec, with tensor parameters and named compile-time args.
//
// Thread numbering is the KERNEL's, not the host's: tt/unified/adaptor.hpp maps
// COMPILE_FOR_BRISC -> DM thread 0 and COMPILE_FOR_NCRISC -> DM thread 1. The kernel
// does noc_load<0> (so thread 0 / BRISC / RISCV_0 produces `in`) and noc_store<1>
// (so thread 1 / NCRISC / RISCV_1 consumes `out`). The DFB endpoint bindings below
// must agree with that, and nothing but this comment says so -- which is the new
// two-places-must-agree contract the port introduces.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr uint32_t kTileBytes = 32 * 32 * 2;  // bfloat16
constexpr uint32_t kTilesPerBlock = 4;
constexpr uint32_t kNumBlocks = 4;
constexpr uint32_t kNumTiles = kTilesPerBlock * kNumBlocks;  // 16 tiles == 128 x 128
constexpr uint32_t kTotalBytes = kTileBytes * kNumTiles;

uint16_t to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return static_cast<uint16_t>(bits >> 16);
}
float from_bf16(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string kernel_source = argc > 1 ? argv[1] : "unified_gate/gate_b.cpp";
    const std::string repo_root = std::getenv("TT_METAL_HOME") ? std::getenv("TT_METAL_HOME") : ".";

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    auto tensor_layout = TensorLayout(
        DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    auto tensor_spec = TensorSpec(Shape{1, 1, 128, 128}, tensor_layout);

    MeshTensor in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);
    MeshTensor out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);

    const NodeCoord node{0, 0};
    constexpr uint32_t kSlotIn = 0;
    constexpr uint32_t kSlotOut = 1;

    ProgramSpec spec;
    spec.name = "unified_gate_b";

    // dm0 == BRISC == the kernel's DM thread 0 == noc_load<0>
    KernelSpec dm0{
        .unique_id = KernelSpecName{"dm0"},
        .source = std::filesystem::path(kernel_source),
        .num_threads = 1,
        .hw_config = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
    };
    // dm1 == NCRISC == the kernel's DM thread 1 == noc_store<1>
    KernelSpec dm1{
        .unique_id = KernelSpecName{"dm1"},
        .source = std::filesystem::path(kernel_source),
        .num_threads = 1,
        .hw_config = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1},
    };
    KernelSpec compute{
        .unique_id = KernelSpecName{"compute"},
        .source = std::filesystem::path(kernel_source),
        .num_threads = 1,
        .hw_config = ComputeGen1Config{},
    };

    const KernelSpec::CompileTimeArgs cta{
        {"cb_in", kSlotIn},
        {"cb_out", kSlotOut},
        {"num_blocks", kNumBlocks},
        {"tiles_per_block", kTilesPerBlock},
    };
    const KernelSpec::CompilerOptions::IncludePaths includes{std::filesystem::path(repo_root)};
    for (KernelSpec* k : {&dm0, &dm1, &compute}) {
        k->compile_time_args = cta;
        k->compiler_options.include_paths = includes;
        // Every projection names both accessors, so every projection binds both tensors.
        // Unlike a DFB binding, a tensor binding has no exclusive role, so this is legal.
        k->tensor_bindings.push_back(TensorBinding{TensorParamName{"in"}, "in"});
        k->tensor_bindings.push_back(TensorBinding{TensorParamName{"out"}, "out"});
    }

    DataflowBufferSpec dfb_in{
        .unique_id = DFBSpecName{"in"},
        .entry_size = kTileBytes,
        .num_entries = 2 * kTilesPerBlock,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    DataflowBufferSpec dfb_out{
        .unique_id = DFBSpecName{"out"},
        .entry_size = kTileBytes,
        .num_entries = 2 * kTilesPerBlock,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    dm0.dfb_bindings.push_back(ProducerOf(DFBSpecName{"in"}, "in"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"in"}, "in"));
    // GATE_OMIT_PRODUCER drops compute's producer binding of `out`, so the kernel still
    // pushes to slot 1 but no kernel claims that endpoint. Probes whether the host catches
    // hazard D20 (a Storage naming a buffer the host never declared for it).
    if (!std::getenv("GATE_OMIT_PRODUCER")) {
        compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out"}, "out"));
    }
    dm1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out"}, "out"));

    spec.kernels = {dm0, dm1, compute};
    spec.dataflow_buffers = {dfb_in, dfb_out};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{"in"}, .spec = tensor_spec},
        TensorParameter{.unique_id = TensorParamName{"out"}, .spec = tensor_spec},
    };
    spec.work_units = std::vector<WorkUnitSpec>{WorkUnitSpec{
        .name = "wu0",
        .kernels = {KernelSpecName{"dm0"}, KernelSpecName{"dm1"}, KernelSpecName{"compute"}},
        .target_nodes = node,
    }};

    printf("building program...\n");
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device, spec);
    Program& program = workload.get_programs().begin()->second;
    printf("program built\n");
    if (std::getenv("GATE_BUILD_ONLY")) {
        printf("BUILD-ONLY: spec accepted\n");
        mesh_device->close();
        return 0;
    }

    ProgramRunArgs params;
    params.tensor_args = {
        {TensorParamName{"in"}, TensorArgument{in_tensor}},
        {TensorParamName{"out"}, TensorArgument{out_tensor}},
    };
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> host_in(kTotalBytes / 4);
    for (uint32_t i = 0; i < host_in.size(); ++i) {
        const float a = 1.0f + static_cast<float>((2 * i) % 11) * 0.5f;
        const float b = 1.0f + static_cast<float>((2 * i + 1) % 11) * 0.5f;
        host_in[i] = (static_cast<uint32_t>(to_bf16(b)) << 16) | to_bf16(a);
    }
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), host_in);
    std::vector<uint32_t> zero(kTotalBytes / 4, 0);
    detail::WriteToBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), zero);

    printf("enqueueing...\n");
    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    printf("done\n");

    std::vector<uint32_t> host_out;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), host_out);

    size_t bad = 0;
    for (uint32_t i = 0; i < host_in.size(); ++i) {
        for (int half = 0; half < 2; ++half) {
            const float src = from_bf16(static_cast<uint16_t>(host_in[i] >> (16 * half)));
            const float got = from_bf16(static_cast<uint16_t>(host_out[i] >> (16 * half)));
            const float want = 1.0f / src;
            if (!(got >= want - 0.02f * want && got <= want + 0.02f * want)) {
                if (bad < 8) {
                    printf("  MISMATCH word %u half %d: src=%g want=%g got=%g\n", i, half, src, want, got);
                }
                ++bad;
            }
        }
    }
    printf("%s: %zu / %zu values wrong\n", bad ? "FAIL" : "PASS", bad, host_in.size() * 2);

    mesh_device->close();
    return bad ? 1 : 0;
}
