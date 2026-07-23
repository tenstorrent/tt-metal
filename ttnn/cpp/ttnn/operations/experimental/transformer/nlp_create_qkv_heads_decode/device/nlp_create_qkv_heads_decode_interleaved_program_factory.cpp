// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_interleaved_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

namespace {

// Metal 2.0 named resource handles for the interleaved ProgramSpec.
// (Names prefixed to avoid Unity-build collisions with the sibling sharded factories.)
const TensorParamName IL_INPUT{"input"};
const TensorParamName IL_Q_OUT{"q_out"};
const TensorParamName IL_K_OUT{"k_out"};
const TensorParamName IL_V_OUT{"v_out"};

const DFBSpecName IL_Q_DFB{"q_out"};
const DFBSpecName IL_K_DFB{"k_out"};
const DFBSpecName IL_V_DFB{"v_out"};
const DFBSpecName IL_SCRATCH_READER_DFB{"il_scratch_reader"};
const DFBSpecName IL_SCRATCH_WRITER_DFB{"il_scratch_writer"};

const KernelSpecName IL_READER{"q_reader"};
const KernelSpecName IL_WRITER{"q_writer"};

constexpr const char* kInterleavedKernel =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
    "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts NLPCreateQKVHeadsDecodeInterleavedProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_mesh = input_tensor.mesh_tensor();
    const auto& q_mesh = output[0].mesh_tensor();
    const auto& k_mesh = output[1].mesh_tensor();
    const auto& v_mesh = output[2].mesh_tensor();

    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    auto k_shard_spec = output[1].shard_spec().value();
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    auto v_shard_spec = output[2].shard_spec().value();
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    Buffer* in_buffer = input_tensor.buffer();

    // The reader kernel reads each face row as a single 16-element noc_async_read transaction
    // (`16 * element_size` bytes). When the input is DRAM-interleaved and that read size is below
    // the device DRAM read alignment (Blackhole bf16: 32 < 64), the NOC alignment rule
    // ((src & (alignment-1)) == (dst & (alignment-1))) is violated for half the (batch, head)
    // parities and the read silently returns wrong data (issue #43270). When that condition
    // holds, switch the kernel to a DRAM-aligned scratch+memcpy path; otherwise the original
    // direct-read fast path runs unchanged. Sharded inputs do not go through this factory.
    const bool is_dram = in_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const bool use_aligned_path = is_dram && (sub_tile_line_bytes < dram_alignment);

    // Per-RISC scratch DFB sized for one DRAM-aligned chunk per tile in a single head. The two
    // RISCs read different phases concurrently, so they need independent scratch slots — assign
    // distinct DFBs (the reader binds IL_SCRATCH_READER_DFB, the writer IL_SCRATCH_WRITER_DFB,
    // both under the kernel-side accessor name "scratch"). The kernel reads DRAM-aligned chunks
    // into this DFB; the NOC requires (src & (alignment-1)) == (dst & (alignment-1)). Since the
    // source addresses are aligned to dram_alignment, the destination addresses inside the scratch
    // must also be aligned to dram_alignment. DFBs in L1 are only allocated at L1 alignment (16 B
    // on BH), so oversize the DFB by one dram_alignment chunk (num_entries = head_tiles + 1) and
    // have the kernel round its base up.

    // ----------------------------------------------------------------------------
    // Dataflow buffers. Output DFBs (q/k/v) are borrowed-memory over the resident
    // output shards (the reader/writer writes via get_write_ptr()); the scratch DFBs
    // are allocated only on the DRAM-aligned path and each self-looped on its one
    // touching kernel. NOTE: this factory is single-work-unit (non-sharded input is
    // always overlap), so borrowed output DFBs are safe here — the multi-work-unit
    // borrowed-DFB framework bug that forces the sharded/subcoregrid factories onto
    // TensorParameter outputs does not apply. See METAL2_PORT_REPORT.md.
    // ----------------------------------------------------------------------------
    DataflowBufferSpec q_out_dfb{
        .unique_id = IL_Q_DFB,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = IL_Q_OUT,
    };
    DataflowBufferSpec k_out_dfb{
        .unique_id = IL_K_DFB,
        .entry_size = single_tile_size,
        .num_entries = k_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = IL_K_OUT,
    };
    DataflowBufferSpec v_out_dfb{
        .unique_id = IL_V_DFB,
        .entry_size = single_tile_size,
        .num_entries = v_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = IL_V_OUT,
    };

    // Float16_b is just a placeholder DataFormat for the scratch DFBs — the kernel only treats
    // them as raw L1 storage and copies bytes via tt_memmove.
    DataflowBufferSpec scratch_reader_dfb{
        .unique_id = IL_SCRATCH_READER_DFB,
        .entry_size = dram_alignment,
        .num_entries = head_tiles + 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    DataflowBufferSpec scratch_writer_dfb{
        .unique_id = IL_SCRATCH_WRITER_DFB,
        .entry_size = dram_alignment,
        .num_entries = head_tiles + 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // ----------------------------------------------------------------------------
    // Reader / writer kernels: one source, two KernelSpecs over the same q_cores,
    // differing by the PHASES_TO_READ CTA (1 vs 2) and their scratch DFB. Both raw-
    // write every output DFB, so the pair is a two-toucher work-split → 1P+1C
    // (reader PRODUCER, writer CONSUMER; cosmetic on Gen1). The scratch DFB is a
    // per-kernel self-loop, bound only on the DRAM-aligned path.
    // ----------------------------------------------------------------------------
    auto make_kernel = [&](const KernelSpecName& unique_id,
                           uint32_t phases_to_read,
                           const DFBSpecName& scratch_dfb,
                           DataMovementHardwareConfig hw_config,
                           DFBEndpointType out_endpoint) {
        KernelSpec k{
            .unique_id = unique_id,
            .source = std::filesystem::path(kInterleavedKernel),
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IL_Q_DFB, .accessor_name = "q_out", .endpoint_type = out_endpoint},
                 DFBBinding{.dfb_spec_name = IL_K_DFB, .accessor_name = "k_out", .endpoint_type = out_endpoint},
                 DFBBinding{.dfb_spec_name = IL_V_DFB, .accessor_name = "v_out", .endpoint_type = out_endpoint}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = IL_INPUT, .accessor_name = "input"}},
            .compile_time_args =
                {{"element_size", element_size},
                 {"sub_tile_line_bytes", sub_tile_line_bytes},
                 {"head_size", head_size},
                 {"num_q_heads", num_q_heads},
                 {"num_kv_heads", num_kv_heads},
                 {"head_size_num_tiles", head_tiles},
                 {"phases_to_read", phases_to_read},
                 {"dram_align_bytes", dram_alignment}},
            .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_batch"}},
            .hw_config = std::move(hw_config),
        };
        if (use_aligned_path) {
            k.compiler_options.defines = {{"USE_ALIGNED_PATH", "1"}};
            // Sync-free scratch: one DM kernel both fills and drains it → self-loop (PRODUCER + CONSUMER).
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = scratch_dfb, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER});
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = scratch_dfb, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::CONSUMER});
        }
        return k;
    };

    IDevice* device = input_tensor.device();
    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input
    // (phase1 and phase2 of a tile respectively).
    KernelSpec reader = make_kernel(
        IL_READER,
        /*phases_to_read=*/1,
        IL_SCRATCH_READER_DFB,
        ttnn::create_reader_datamovement_config(device->arch()),
        DFBEndpointType::PRODUCER);
    KernelSpec writer = make_kernel(
        IL_WRITER,
        /*phases_to_read=*/2,
        IL_SCRATCH_WRITER_DFB,
        ttnn::create_writer_datamovement_config(device->arch()),
        DFBEndpointType::CONSUMER);

    // ----------------------------------------------------------------------------
    // Per-node runtime args: in_tile_offset_by_batch (same value on reader + writer).
    // ----------------------------------------------------------------------------
    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    KernelRunArgs reader_run{.kernel = IL_READER};
    KernelRunArgs writer_run{.kernel = IL_WRITER};
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : ((i - 16) * sub_tile_line_bytes) + (512 * element_size);

        const auto& core = cores[i];
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"in_tile_offset_by_batch", in_tile_offset_by_batch}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"in_tile_offset_by_batch", in_tile_offset_by_batch}});
    }

    // ----------------------------------------------------------------------------
    // Assemble spec + run-args.
    // ----------------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "nlp_create_qkv_heads_decode_interleaved";
    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {q_out_dfb, k_out_dfb, v_out_dfb};
    if (use_aligned_path) {
        spec.dataflow_buffers.push_back(scratch_reader_dfb);
        spec.dataflow_buffers.push_back(scratch_writer_dfb);
    }
    spec.tensor_parameters = {
        TensorParameter{.unique_id = IL_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = IL_Q_OUT, .spec = output[0].tensor_spec()},
        TensorParameter{.unique_id = IL_K_OUT, .spec = output[1].tensor_spec()},
        TensorParameter{.unique_id = IL_V_OUT, .spec = output[2].tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{
        .name = "main",
        .kernels = {IL_READER, IL_WRITER},
        .target_nodes = q_cores,
    }};

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_params.tensor_args = {
        {IL_INPUT, TensorArgument{input_mesh}},
        {IL_Q_OUT, TensorArgument{q_mesh}},
        {IL_K_OUT, TensorArgument{k_mesh}},
        {IL_V_OUT, TensorArgument{v_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::experimental::prim
