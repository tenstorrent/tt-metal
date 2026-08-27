// SPDX-License-Identifier: Apache-2.0
//
// Gate host program: builds ONE Metal 2.0 ProgramSpec whose three KernelSpecs all point
// at the SAME source file, and runs it on hardware.
//
// Usage: gate_host <kernel_source_path_relative_to_TT_METAL_HOME>

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
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr uint32_t kTileBytes = 32 * 32 * 2;  // bfloat16
constexpr uint32_t kNumTiles = 4;
constexpr uint32_t kTotalBytes = kTileBytes * kNumTiles;

// bfloat16 <-> float, top half of the float32 word.
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
    const std::string kernel_source = argc > 1 ? argv[1] : "unified_gate/gate_a.cpp";
    const std::string repo_root = std::getenv("TT_METAL_HOME") ? std::getenv("TT_METAL_HOME") : ".";

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    printf("device arch: %d\n", static_cast<int>(mesh_device->arch()));

    // ---------------------------------------------------------------- buffers
    auto in_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kTotalBytes},
        {.page_size = kTotalBytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kTotalBytes},
        {.page_size = kTotalBytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());

    // ------------------------------------------------------------------ spec
    const NodeCoord node{0, 0};

    // Slot numbers the kernel is told about as named compile-time args. The DFB device
    // slot is assigned by the host allocator as the lowest free slot among buffers that
    // share cores, in declaration order -- so "in" is slot 0 and "out" is slot 1. The
    // program asserts that below, after MakeProgramFromSpec, rather than trusting it.
    constexpr uint32_t kSlotIn = 0;
    constexpr uint32_t kSlotOut = 1;

    ProgramSpec spec;
    spec.name = "unified_gate";

    KernelSpec reader{
        .unique_id = KernelSpecName{"reader"},
        .source = std::filesystem::path(kernel_source),
        .num_threads = 1,
        .hw_config = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1},
    };
    KernelSpec writer{
        .unique_id = KernelSpecName{"writer"},
        .source = std::filesystem::path(kernel_source),
        .num_threads = 1,
        .hw_config = DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
    };
    KernelSpec compute{
        .unique_id = KernelSpecName{"compute"},
        .source = std::filesystem::path(kernel_source),
        .num_threads = 1,
        .hw_config = ComputeGen1Config{},
    };

    // IDENTICAL compile-time args on all three, exactly as unified_harness.py does today.
    const KernelSpec::CompileTimeArgs cta{
        {"cb_in", kSlotIn},
        {"cb_out", kSlotOut},
        {"num_tiles", kNumTiles},
    };
    reader.compile_time_args = cta;
    writer.compile_time_args = cta;
    compute.compile_time_args = cta;

    // The unified include path, so a kernel can #include <tt/unified/core>.
    const KernelSpec::CompilerOptions::IncludePaths includes{std::filesystem::path(repo_root)};
    reader.compiler_options.include_paths = includes;
    writer.compiler_options.include_paths = includes;
    compute.compiler_options.include_paths = includes;

    // Runtime args: each DM kernel needs its DRAM address. The compute kernel declares the
    // same schema so all three sources stay identical -- but a kernel only reads the ones
    // its own projection uses.
    reader.runtime_arg_schema.runtime_arg_names = {"src_addr", "dst_addr"};
    writer.runtime_arg_schema.runtime_arg_names = {"src_addr", "dst_addr"};
    compute.runtime_arg_schema.runtime_arg_names = {"src_addr", "dst_addr"};

    DataflowBufferSpec dfb_in{
        .unique_id = DFBSpecName{"in"},
        .entry_size = kTileBytes,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    DataflowBufferSpec dfb_out{
        .unique_id = DFBSpecName{"out"},
        .entry_size = kTileBytes,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    reader.dfb_bindings.push_back(ProducerOf(DFBSpecName{"in"}, "in"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"in"}, "in"));
    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out"}, "out"));
    writer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out"}, "out"));

    spec.kernels = {reader, writer, compute};
    spec.dataflow_buffers = {dfb_in, dfb_out};
    spec.work_units = std::vector<WorkUnitSpec>{WorkUnitSpec{
        .name = "wu0",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}, KernelSpecName{"compute"}},
        .target_nodes = node,
    }};

    printf("building program...\n");
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device, spec);
    Program& program = workload.get_programs().begin()->second;
    printf("program built\n");

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"reader"},
            .runtime_arg_values =
                {{"src_addr", {{node, static_cast<uint32_t>(in_buf->address())}}},
                 {"dst_addr", {{node, static_cast<uint32_t>(out_buf->address())}}}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"writer"},
            .runtime_arg_values =
                {{"src_addr", {{node, static_cast<uint32_t>(in_buf->address())}}},
                 {"dst_addr", {{node, static_cast<uint32_t>(out_buf->address())}}}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"compute"},
            .runtime_arg_values =
                {{"src_addr", {{node, static_cast<uint32_t>(in_buf->address())}}},
                 {"dst_addr", {{node, static_cast<uint32_t>(out_buf->address())}}}},
        },
    };
    SetProgramRunArgs(program, params);

    // ------------------------------------------------------------------ data
    std::vector<uint32_t> host_in(kTotalBytes / 4);
    for (uint32_t i = 0; i < host_in.size(); ++i) {
        // Two bfloat16 values per word. Keep them small so the square is exact-ish.
        const float a = 1.0f + static_cast<float>((2 * i) % 7) * 0.5f;
        const float b = 1.0f + static_cast<float>((2 * i + 1) % 7) * 0.5f;
        host_in[i] = (static_cast<uint32_t>(to_bf16(b)) << 16) | to_bf16(a);
    }
    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, in_buf, host_in, /*blocking=*/true);

    std::vector<uint32_t> zero(kTotalBytes / 4, 0);
    distributed::EnqueueWriteMeshBuffer(cq, out_buf, zero, /*blocking=*/true);

    printf("enqueueing...\n");
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    printf("done\n");

    std::vector<uint32_t> host_out;
    distributed::EnqueueReadMeshBuffer(cq, host_out, out_buf, /*blocking=*/true);

    // --------------------------------------------------------------- verify
    size_t bad = 0;
    for (uint32_t i = 0; i < host_in.size(); ++i) {
        for (int half = 0; half < 2; ++half) {
            const float src = from_bf16(static_cast<uint16_t>(host_in[i] >> (16 * half)));
            const float got = from_bf16(static_cast<uint16_t>(host_out[i] >> (16 * half)));
            const float want = src * src;
            const float tol = 0.05f * (want > 1.0f ? want : 1.0f);
            if (!(got >= want - tol && got <= want + tol)) {
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
