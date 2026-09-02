// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_interleaved_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

ttnn::device_operation::ProgramArtifacts NLPCreateQKVHeadsDecodeInterleavedProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& q_mesh_tensor = output[0].mesh_tensor();
    const auto& k_mesh_tensor = output[1].mesh_tensor();
    const auto& v_mesh_tensor = output[2].mesh_tensor();

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    // Per-instance DRAM-aligned staging regions (aligned path only). Ported first as self-looped
    // DFBs (each instance bound PRODUCER + CONSUMER), a shape Gen2 rejects on DM kernels; now
    // Scratchpads (ai/post_port/style/sync_free_dfbs.md) — each instance only ever grabbed the
    // base address once (no FIFO calls), so no synchronization is lost.
    const ScratchpadSpecName READER_SCRATCH{"reader_scratch"};
    const ScratchpadSpecName WRITER_SCRATCH{"writer_scratch"};
    const TensorParamName QKV_IN{"qkv_in"};
    const TensorParamName Q_OUT_TENSOR{"q_out_tensor"};
    const TensorParamName K_OUT_TENSOR{"k_out_tensor"};
    const TensorParamName V_OUT_TENSOR{"v_out_tensor"};

    tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;

    // The reader kernel reads each face row as a single 16-element noc_async_read transaction
    // (`16 * element_size` bytes). When the input is DRAM-interleaved and that read size is below
    // the device DRAM read alignment (Blackhole bf16: 32 < 64), the NOC alignment rule
    // ((src & (alignment-1)) == (dst & (alignment-1))) is violated for half the (batch, head)
    // parities and the read silently returns wrong data (issue #43270). When that condition
    // holds, switch the kernel to a DRAM-aligned scratch+memcpy path; otherwise the original
    // direct-read fast path runs unchanged. Sharded inputs do not go through this factory.
    const bool is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const bool use_aligned_path = is_dram && (sub_tile_line_bytes < dram_alignment);

    // NOTE(Quasar uplift): the q/k/v outputs were first ported as borrowed DFBs
    // (borrowed_from = *_OUT_TENSOR) whose PRODUCER/CONSUMER endpoints were cosmetic — both kernel
    // instances only ever grabbed get_write_ptr() (the shard base) and raw-wrote disjoint sub-tile
    // regions via NOC reads; no FIFO call ever touched them. Per
    // ai/post_port/style/sync_free_dfbs.md (sync-free + borrowed → LocalTensorAccessor) the kernels
    // now bind the output tensors directly. This is also required on Quasar: DFB
    // get_write_ptr()/get_read_ptr() return the UNCACHED L1 alias on Gen2 DM, and NOC APIs do not
    // accept uncached addresses, so the old pointer-extraction pattern would hand the NOC an
    // illegal destination address. LocalTensorAccessor::get_bank_base_address() returns the
    // tensor's plain (cached) L1 shard base on both generations — on Gen1 it is the identical
    // address the borrowed DFB returned.

    // Per-RISC scratch region sized for one DRAM-aligned chunk per tile in a single head. The two
    // RISCs read different phases concurrently, so they need independent scratch slots — each
    // instance binds its own Scratchpad under one shared accessor name. The kernel reads
    // DRAM-aligned chunks into this region; the NOC requires
    // (src & (alignment-1)) == (dst & (alignment-1)). Since the source addresses are aligned to
    // dram_alignment, the destination addresses inside the scratch must also be aligned to
    // dram_alignment. L1 allocations are only L1-aligned (16 B on BH), so oversize the region by
    // one dram_alignment chunk and have the kernel round its base up.
    // (Formerly a self-looped DFB — each instance bound PRODUCER + CONSUMER — a shape Gen2 rejects
    // for DM kernels. The kernel never made a FIFO call on it, so it is a Scratchpad by nature;
    // its placeholder Float16_b data_format_metadata had no consumer and drops with the spec.)
    Group<ScratchpadSpec> scratchpads;
    if (use_aligned_path) {
        const uint32_t scratch_size_per_node = dram_alignment * (head_tiles + 1);
        scratchpads.push_back(ScratchpadSpec{
            .unique_id = READER_SCRATCH,
            .size_per_node = scratch_size_per_node,
        });
        scratchpads.push_back(ScratchpadSpec{
            .unique_id = WRITER_SCRATCH,
            .size_per_node = scratch_size_per_node,
        });
    }

    KernelSpec::CompilerOptions::Defines defines;
    if (use_aligned_path) {
        defines.emplace("USE_ALIGNED_PATH", "1");
    }

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input
    // (phase1 and phase2 of a tile respectively). Both instances share one kernel source and
    // differ only by PHASES_TO_READ, their Scratchpad, and their DM hardware config.
    const std::filesystem::path kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";

    auto make_kernel = [&](const KernelSpecName& unique_id,
                           const ScratchpadSpecName& scratch_spad,
                           uint32_t phases_to_read,
                           DataMovementHardwareConfig hw_config) {
        // Sync-free single-toucher scratch: each instance binds its own Scratchpad under the
        // shared accessor name (aligned path only).
        Group<ScratchpadBinding> scratchpad_bindings;
        if (use_aligned_path) {
            scratchpad_bindings.push_back(ScratchpadBinding{
                .scratchpad_spec_name = scratch_spad,
                .accessor_name = "aligned_scratch",
            });
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = kernel_source,
            .compiler_options = {.defines = defines},
            .scratchpad_bindings = std::move(scratchpad_bindings),
            // Both instances raw-write disjoint sub-tile regions of the resident q/k/v output
            // shards via LocalTensorAccessor (see the uplift note above).
            .tensor_bindings =
                {{
                     .tensor_parameter_name = QKV_IN,
                     .accessor_name = "qkv_in",
                 },
                 {
                     .tensor_parameter_name = Q_OUT_TENSOR,
                     .accessor_name = "q_out",
                 },
                 {
                     .tensor_parameter_name = K_OUT_TENSOR,
                     .accessor_name = "k_out",
                 },
                 {
                     .tensor_parameter_name = V_OUT_TENSOR,
                     .accessor_name = "v_out",
                 }},
            .compile_time_args =
                {
                    {"ELEMENT_SIZE", element_size},
                    {"SUBTILE_LINE_BYTES", sub_tile_line_bytes},
                    {"head_size", head_size},
                    {"num_q_heads", num_q_heads},
                    {"num_kv_heads", num_kv_heads},
                    {"head_size_num_tiles", head_tiles},
                    {"PHASES_TO_READ", phases_to_read},
                    {"DRAM_ALIGN_BYTES", dram_alignment},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_batch"}},
            .hw_config = std::move(hw_config),
        };
    };

    const auto arch = input_tensor.device()->arch();
    // phase 1 on the reader instance, phase 2 on the writer instance
    KernelSpec reader = make_kernel(READER, READER_SCRATCH, 1, create_reader_datamovement_config(arch));
    KernelSpec writer = make_kernel(WRITER, WRITER_SCRATCH, 2, create_writer_datamovement_config(arch));

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_decode_interleaved",
        .kernels = {std::move(reader), std::move(writer)},
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = QKV_IN, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = Q_OUT_TENSOR, .spec = output[0].tensor_spec()},
                TensorParameter{.unique_id = K_OUT_TENSOR, .spec = output[1].tensor_spec()},
                TensorParameter{.unique_id = V_OUT_TENSOR, .spec = output[2].tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = q_cores,
        }},
    };

    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : ((i - 16) * sub_tile_line_bytes) + (512 * element_size);

        const auto& core = cores[i];
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"in_tile_offset_by_batch", in_tile_offset_by_batch}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"in_tile_offset_by_batch", in_tile_offset_by_batch}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {QKV_IN, input_mesh_tensor},
        {Q_OUT_TENSOR, q_mesh_tensor},
        {K_OUT_TENSOR, k_mesh_tensor},
        {V_OUT_TENSOR, v_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
