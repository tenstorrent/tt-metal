// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_rmsnorm_program_factory.hpp"

#include <bit>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm {

namespace {

constexpr uint32_t kCbX = tt::CBIndex::c_0;
constexpr uint32_t kCbGamma = tt::CBIndex::c_1;
constexpr uint32_t kCbLocalStat = tt::CBIndex::c_2;
constexpr uint32_t kCbGather = tt::CBIndex::c_3;
constexpr uint32_t kCbScaleSrc = tt::CBIndex::c_4;
constexpr uint32_t kCbScale = tt::CBIndex::c_5;
constexpr uint32_t kCbOut = tt::CBIndex::c_6;
constexpr uint32_t kCbGammaRaw = tt::CBIndex::c_7;
constexpr uint32_t kCbIn = tt::CBIndex::c_8;
constexpr uint32_t kCbTileOut = tt::CBIndex::c_9;

constexpr char kReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/sharded_rmsnorm/device/kernels/dataflow/"
    "reader_sharded_rmsnorm.cpp";
constexpr char kWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/sharded_rmsnorm/device/kernels/dataflow/"
    "writer_sharded_rmsnorm.cpp";
constexpr char kComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/sharded_rmsnorm/device/kernels/compute/"
    "compute_sharded_rmsnorm.cpp";

}  // namespace

ShardedRmsNormProgramFactory::cached_program_t ShardedRmsNormProgramFactory::create(
    const ShardedRmsNormParams& operation_attributes,
    const ShardedRmsNormInputs& tensor_args,
    ShardedRmsNormTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight;
    auto& output = tensor_return_value;

    const auto& in_shard = input.shard_spec().value();
    const CoreRangeSet& all_cores = in_shard.grid;
    const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    const uint32_t num_cores = static_cast<uint32_t>(cores.size());
    const uint32_t shard_w = in_shard.shape[1];
    const uint32_t w_tiles = shard_w / constants::TILE_WIDTH;
    const uint32_t d = static_cast<uint32_t>(input.logical_shape()[3]);

    const bool input_is_rm = input.layout() == Layout::ROW_MAJOR;
    const bool output_is_rm = operation_attributes.output_layout == Layout::ROW_MAJOR;
    const bool output_is_sharded = output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool scatter_out = output_is_sharded && !output_is_rm;

    IDevice* device = input.device();
    Program program = CreateProgram();

    const tt::DataFormat bf16 = DataFormat::Float16_b;
    const tt::DataFormat fp32 = DataFormat::Float32;
    const tt::tt_metal::Tile row_tile({1, constants::TILE_WIDTH});
    const tt::tt_metal::Tile full_tile({constants::TILE_HEIGHT, constants::TILE_WIDTH});
    const uint32_t row_tile_bytes = row_tile.get_tile_size(bf16);
    const uint32_t stat_tile_bytes = row_tile.get_tile_size(fp32);
    const uint32_t full_tile_bytes = full_tile.get_tile_size(bf16);

    auto make_row_cb = [&](uint32_t idx, uint32_t num_pages, DataFormat df, uint32_t page_bytes) {
        auto cfg = CircularBufferConfig(num_pages * page_bytes, {{idx, df}})
                       .set_page_size(idx, page_bytes)
                       .set_tile_dims(idx, row_tile);
        return CreateCircularBuffer(program, all_cores, cfg);
    };

    CBHandle cb_x_handle = 0;
    CBHandle cb_in_handle = 0;
    if (input_is_rm) {
        auto cfg = CircularBufferConfig(w_tiles * row_tile_bytes, {{kCbX, bf16}})
                       .set_page_size(kCbX, row_tile_bytes)
                       .set_tile_dims(kCbX, row_tile)
                       .set_globally_allocated_address(*input.buffer());
        cb_x_handle = CreateCircularBuffer(program, all_cores, cfg);
    } else {
        make_row_cb(kCbX, w_tiles, bf16, row_tile_bytes);
        auto in_cfg = CircularBufferConfig(w_tiles * full_tile_bytes, {{kCbIn, bf16}})
                          .set_page_size(kCbIn, full_tile_bytes)
                          .set_tile_dims(kCbIn, full_tile)
                          .set_globally_allocated_address(*input.buffer());
        cb_in_handle = CreateCircularBuffer(program, all_cores, in_cfg);
    }

    make_row_cb(kCbGamma, w_tiles, bf16, row_tile_bytes);
    make_row_cb(kCbGammaRaw, 2 * w_tiles, bf16, row_tile_bytes);
    make_row_cb(kCbLocalStat, 1, fp32, stat_tile_bytes);
    make_row_cb(kCbGather, num_cores, fp32, stat_tile_bytes);
    make_row_cb(kCbScaleSrc, 1, fp32, stat_tile_bytes);
    make_row_cb(kCbScale, 1, fp32, stat_tile_bytes);

    CBHandle cb_out_handle = 0;
    CBHandle cb_tile_out_handle = 0;
    if (output_is_sharded && output_is_rm) {
        auto cfg = CircularBufferConfig(w_tiles * row_tile_bytes, {{kCbOut, bf16}})
                       .set_page_size(kCbOut, row_tile_bytes)
                       .set_tile_dims(kCbOut, row_tile)
                       .set_globally_allocated_address(*output.buffer());
        cb_out_handle = CreateCircularBuffer(program, all_cores, cfg);
    } else {
        make_row_cb(kCbOut, w_tiles, bf16, row_tile_bytes);
        if (!output_is_rm) {
            auto tile_cfg = CircularBufferConfig(w_tiles * full_tile_bytes, {{kCbTileOut, bf16}})
                                .set_page_size(kCbTileOut, full_tile_bytes)
                                .set_tile_dims(kCbTileOut, full_tile);
            if (output_is_sharded) {
                tile_cfg.set_globally_allocated_address(*output.buffer());
            }
            cb_tile_out_handle = CreateCircularBuffer(program, all_cores, tile_cfg);
        }
    }

    const uint32_t arrive_sem_id = CreateSemaphore(program, all_cores, 0);
    const uint32_t ready_sem_id = CreateSemaphore(program, all_cores, 0);

    const CoreCoord hub_phys = device->worker_core_from_logical_core(cores[0]);
    const bool is_rect = all_cores.ranges().size() == 1;
    CoreCoord mcast_start_phys{0, 0};
    CoreCoord mcast_end_phys{0, 0};
    if (is_rect && num_cores > 1) {
        const auto& r = all_cores.ranges()[0];
        mcast_start_phys = device->worker_core_from_logical_core(r.start_coord);
        mcast_end_phys = device->worker_core_from_logical_core(r.end_coord);
    }

    std::vector<uint32_t> reader_ct_args = {
        kCbX,
        kCbGamma,
        kCbGammaRaw,
        kCbIn,
        kCbOut,
        kCbTileOut,
        w_tiles,
        input_is_rm ? 1u : 0u,
        scatter_out ? 1u : 0u,
    };
    TensorAccessorArgs(weight.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        kCbLocalStat,
        kCbGather,
        kCbScaleSrc,
        kCbScale,
        kCbOut,
        kCbTileOut,
        w_tiles,
        num_cores,
        stat_tile_bytes,
        row_tile_bytes,
        arrive_sem_id,
        ready_sem_id,
        static_cast<uint32_t>(hub_phys.x),
        static_cast<uint32_t>(hub_phys.y),
        is_rect ? 1u : 0u,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        static_cast<uint32_t>(mcast_end_phys.x),
        static_cast<uint32_t>(mcast_end_phys.y),
        output_is_sharded ? 1u : 0u,
        output_is_rm ? 1u : 0u,
    };
    TensorAccessorArgs(output_is_sharded ? static_cast<const Buffer*>(nullptr) : output.buffer())
        .append_to(writer_ct_args);

    const uint32_t inv_d_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(d));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.epsilon);
    const std::vector<uint32_t> compute_ct_args = {
        kCbX,
        kCbGamma,
        kCbLocalStat,
        kCbGather,
        kCbScaleSrc,
        kCbScale,
        kCbOut,
        w_tiles,
        num_cores,
        inv_d_bits,
        eps_bits,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_ct_args));
    const KernelHandle writer_kernel_id =
        CreateKernel(program, kWriterKernelPath, all_cores, WriterDataMovementConfig(writer_ct_args));
    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });

    const uint32_t weight_addr = weight.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        SetRuntimeArgs(program, reader_kernel_id, core, {weight_addr, i});
        SetRuntimeArgs(program, compute_kernel_id, core, {i == 0 ? 1u : 0u});

        std::vector<uint32_t> writer_rt_args = {out_addr, i};
        if (i == 0 && !is_rect && num_cores > 1) {
            writer_rt_args.reserve(2 + 2 * (num_cores - 1));
            for (uint32_t j = 1; j < num_cores; ++j) {
                const auto phys = device->worker_core_from_logical_core(cores[j]);
                writer_rt_args.push_back(static_cast<uint32_t>(phys.x));
                writer_rt_args.push_back(static_cast<uint32_t>(phys.y));
            }
        }
        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cb_in = cb_in_handle,
         .cb_x = cb_x_handle,
         .cb_out = cb_out_handle,
         .cb_tile_out = cb_tile_out_handle,
         .input_is_rm = input_is_rm,
         .output_is_sharded = output_is_sharded,
         .output_is_rm = output_is_rm,
         .cores = std::move(cores)}};
}

void ShardedRmsNormProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ShardedRmsNormParams& /*operation_attributes*/,
    const ShardedRmsNormInputs& tensor_args,
    ShardedRmsNormTensorReturn& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    if (shared.input_is_rm) {
        UpdateDynamicCircularBufferAddress(program, shared.cb_x, *tensor_args.input_tensor.buffer());
    } else {
        UpdateDynamicCircularBufferAddress(program, shared.cb_in, *tensor_args.input_tensor.buffer());
    }

    if (shared.output_is_sharded) {
        if (shared.output_is_rm) {
            UpdateDynamicCircularBufferAddress(program, shared.cb_out, *tensor_return_value.buffer());
        } else {
            UpdateDynamicCircularBufferAddress(program, shared.cb_tile_out, *tensor_return_value.buffer());
        }
    }

    const uint32_t weight_addr = tensor_args.weight.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();
    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        reader_args_by_core[core.x][core.y][0] = weight_addr;
        writer_args_by_core[core.x][core.y][0] = out_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm
