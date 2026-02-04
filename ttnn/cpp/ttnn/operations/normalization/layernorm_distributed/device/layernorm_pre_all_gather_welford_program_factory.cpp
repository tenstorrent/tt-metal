// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_welford_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"

#include <optional>
#include <string>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

inline uint16_t bfloat16(float float_num) {
    uint32_t uint32_data;
    TT_FATAL(
        sizeof float_num == sizeof uint32_data,
        "Float size ({}) must equal uint32 size ({})",
        sizeof float_num,
        sizeof uint32_data);

    uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
    // just move upper 16 to lower 16 (truncate)
    uint32_data = (uint32_data >> 16);

    // store lower 16 as 16-bit uint
    return (uint16_t)uint32_data;
}
inline uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}

std::pair<std::optional<Tensor>, uint32_t> create_reciprocal_tensor_if_needed(
    IDevice* device, uint32_t W, const CoreRangeSet& cores, const bool use_welford) {
    const auto num_cores = cores.num_cores();
    std::optional<Tensor> recip_tensor = std::nullopt;
    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        const auto recip_dtype = tt::tt_metal::DataType::FLOAT32;
        const tt::tt_metal::ShardSpec shard_spec(cores, {1, W}, ShardOrientation::ROW_MAJOR);
        const MemoryConfig mem_config =
            MemoryConfig{tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec};
        const tt::tt_metal::TensorLayout tensor_layout(
            tt::tt_metal::TensorLayout(recip_dtype, Layout::ROW_MAJOR, mem_config));
        const Shape tensor_shape{num_cores, W};
        const TensorSpec tensor_spec(tensor_shape, tensor_layout);
        // Compute the reciprocals of an ascending sequence of integers
        std::vector<float> reciprocals(num_cores * W);
        for (uint32_t i = 0; i < W; i++) {
            // Compute for first row
            reciprocals[i] = 1.0f / (i + 1);
        }
        for (uint32_t i = 1; i < num_cores; i++) {
            // Copy to other rows
            std::copy(reciprocals.begin(), reciprocals.begin() + W, reciprocals.begin() + i * W);
        }

        if (auto* p_mesh_device = dynamic_cast<distributed::MeshDevice*>(device)) {
            recip_tensor = Tensor::from_vector(std::move(reciprocals), tensor_spec, p_mesh_device);
        } else {
            TT_THROW("Cannot cast to MeshDevice");
        }

        reciprocal_CB_size_bytes = recip_tensor->buffer()->aligned_size_per_bank();
    }

    return std::make_pair(recip_tensor, reciprocal_CB_size_bytes);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

LayerNormPreAllGatherWelfordProgramFactory::cached_program_t LayerNormPreAllGatherWelfordProgramFactory::create(
    const LayerNormPreAllGatherParams& operation_attributes, const Tensor& tensor_args, Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args;
    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    TT_FATAL(!is_rmsnorm, "rms_norm is not compatible with welford, please disable welford flag to use rms norm");

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    const uint32_t in0_tiles = 2;

    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)writer_block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> compute_defines;

    auto reader_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {Wt, W};

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_welford.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args,
        .defines = compute_defines};
    auto compute_kernels_id = tt::tt_metal::CreateKernel(program, compute_kernel_file, all_cores, compute_config);

    // Create circular buffers
    // c_in0 -> a
    auto cb_src0_config =
        tt::tt_metal::CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // LN and RMS shared intermediates
    // c_intermed0 -> x^2
    auto cb_intermed0_config =
        tt::tt_metal::CircularBufferConfig(in0_tiles * single_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    auto cb_out0_config =
        tt::tt_metal::CircularBufferConfig(in0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out0_config);

    auto [recip_tensor, reciprocal_CB_size_bytes] =
        CMAKE_UNIQUE_NAMESPACE::create_reciprocal_tensor_if_needed(device, W, all_cores, true);

    constexpr tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;
    auto c_recip_config =
        tt::tt_metal::CircularBufferConfig(reciprocal_CB_size_bytes, {{tt::CBIndex::c_2, reciprocal_cb_data_format}})
            .set_page_size(tt::CBIndex::c_2, reciprocal_CB_size_bytes)
            .set_globally_allocated_address(*recip_tensor.value().buffer());
    tt::tt_metal::CreateCircularBuffer(program, all_cores, c_recip_config);

    // Log all circular buffers
    for (const auto& cb : program.circular_buffers()) {
        for ([[maybe_unused]] const auto index : cb->buffer_indices()) {
            log_debug(tt::LogOp, "cb_id {}", index);
            log_debug(tt::LogOp, "page_size: {}", cb->page_size(index));
            log_debug(tt::LogOp, "num_pages: {}", cb->num_pages(index));
            log_debug(tt::LogOp, "data_format: {}", cb->data_format(index));
        }
    }

    uint32_t curr_row = 0;
    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernels_id, core, {a_addr, num_tile_rows_per_core, Wt, in_tile_offset, packed_winv_value});
        tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset});
        curr_row += num_tile_rows_per_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernels_id,
         .writer_kernel_id = writer_kernels_id,
         .num_cores = num_cores,
         .grid_size = grid_size}};
}

void LayerNormPreAllGatherWelfordProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormPreAllGatherParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto input_addr = tensor_args.buffer()->address();
    const auto output_addr = output.buffer()->address();

    auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
        const CoreCoord core = {i % shared_vars.grid_size.x, i / shared_vars.grid_size.x};

        {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
            reader_args[0] = input_addr;
        }

        {
            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::prim
