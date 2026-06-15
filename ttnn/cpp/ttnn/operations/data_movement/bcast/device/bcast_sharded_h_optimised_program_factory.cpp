// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_sharded_h_optimised_program_factory.hpp"

#include <cmath>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include <tt-metalium/tilize_utils.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Per-core runtime args for one dispatch, derived purely from (operation_attributes, inputs, output).
// Reader arg0 is src1 (b)'s ADDRESS, left as placeholder 0 here (the factory binds it as a patchable
// Buffer* via buffer_bindings on a cache miss; get_dynamic writes the live address on a cache hit).
// All other slots are shard-geometry-derived. SINGLE SOURCE OF TRUTH shared by both create_descriptor()
// (cache miss) and get_dynamic_runtime_args() (cache hit re-apply). Every core in `cores` is a work core.
struct BcastShardedHOptPerCoreArgs {
    std::vector<CoreCoord> cores;
    std::vector<KernelDescriptor::CoreRuntimeArgs> reader_args;
    std::vector<KernelDescriptor::CoreRuntimeArgs> compute_args;
};

BcastShardedHOptPerCoreArgs compute_bcast_sharded_h_opt_per_core_args(
    const BcastParams& /*operation_attributes*/, const Tensor& a, const Tensor& b, const Tensor& output) {
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t NC = N * C;

    IDevice* device = a.device();

    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();

    uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    uint32_t Wt = 0, Ht = 0;
    if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.y + 1;
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
    }

    const uint32_t h_blk = std::min(Ht, 8u);
    const uint32_t w_blk = std::min(Wt, 8u);

    (void)output;

    const uint32_t ncores_y = ncores / ncores_x;
    TT_FATAL((NC * H / TILE_HEIGHT) % bN == 0, "N*C*H of input0 must be divisible by batch size of input1");
    const uint32_t Ht_per_batch_b = std::min((NC * H / TILE_HEIGHT) / bN, Ht);
    const uint32_t batch_b = Ht / Ht_per_batch_b;

    BcastShardedHOptPerCoreArgs result;
    result.cores.resize(ncores);
    result.reader_args.resize(ncores);
    result.compute_args.resize(ncores);

    for (uint32_t i = 0; i < ncores; i++) {
        CoreCoord core;
        uint32_t offset = 0;
        if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            core = {i / ncores_x, i % ncores_x};
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (i / ncores_x) + Wt * ncores_y * ((i % ncores_x) / (ncores_x / bN));
            } else {
                offset = Wt * (i % ncores_x) + Wt * ncores_x * ((i / ncores_x) / (ncores_y / bN));
            }
        } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            core = {i % ncores_x, i / ncores_x};
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (core.x + core.y * ncores_x);
            } else {
                offset = Wt * (ncores_y * core.x + core.y);
                if (core.y == ncores_y) {
                    offset = Wt * (ncores_y * ncores_x + core.x);
                }
            }
        }
        const uint32_t tile_offset = Wt * ncores;  // used in multi batch weight for block sharded

        result.cores[i] = core;
        result.reader_args[i] = KernelDescriptor::CoreRuntimeArgs{
            0u,           // (0) src1_addr
            Ht,           // (1) Ht
            Wt,           // (2) Wt
            offset,       // (3) read offset in1
            tile_offset,  // (4) in1 offset between batches
            w_blk,        // (5) block size in w
            batch_b,      // (6) in1 batch size
        };
        result.compute_args[i] = KernelDescriptor::CoreRuntimeArgs{
            NC,              // (0) B
            Ht,              // (1) Hbatch  for block sharded
            Wt,              // (2) Wt
            h_blk,           // (3) h block size
            batch_b,         // (4) in1 batch size
            Ht_per_batch_b,  // (5) Ht per in1 batch size (bN)
        };
    }

    return result;
}

}  // namespace

tt::tt_metal::ProgramDescriptor BcastShardedHOptimisedProgramFactory::create_descriptor(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& bshape = b.padded_shape();
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;

    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();

    const auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    const auto act_df = datatype_to_dataformat_converter(a.dtype());
    const auto b_df = datatype_to_dataformat_converter(b.dtype());
    const auto out_df = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tt::tile_size(act_df);
    const uint32_t input1_tile_size = tt::tile_size(b_df);
    const uint32_t output_tile_size = tt::tile_size(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    const uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)TILE_WIDTH);
    const uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)TILE_HEIGHT);
    const uint32_t num_tile_per_core = ntiles_along_width * ntiles_along_height;

    uint32_t Wt = 0;
    if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        Wt = shard_spec.shape[1] / TILE_WIDTH;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        TT_ASSERT(
            (shard_spec.shape[0] % (bN * TILE_HEIGHT) == 0),
            "Shard height per batch must be divisible by TILE_HEIGHT {} {} {} ",
            shard_spec.shape[0],
            bN,
            TILE_HEIGHT);
    } else {
        TT_THROW("Unsupported memory layout");
    }

    TT_ASSERT(
        (shard_spec.shape[0] % TILE_HEIGHT == 0) && (shard_spec.shape[0] % TILE_WIDTH == 0),
        "Shard shapes must be multiple of TILE_HEIGHT ");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    const uint32_t in_cb_pagesize = aligned_input_tile_nbytes;

    const uint32_t output_cb_index = tt::CBIndex::c_16;

    const uint32_t w_blk = std::min(Wt, 8u);

    const uint32_t num_input_tiles = w_blk;
    const uint32_t src1_cb_index = tt::CBIndex::c_1;

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_tile_nbytes * num_tile_per_core,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = act_df,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = src0_buffer,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input1_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = b_df,
            .page_size = input1_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_tile_nbytes * num_tile_per_core,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = out_df,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = dst_buffer,
    });

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    const bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};
    (void)dst_is_dram;
    (void)writer_compile_time_args;

    const std::map<std::string, std::string> bcast_defines =
        bcast_op_utils::get_defines(BcastOpDim::H, operation_attributes.math_op);

    static constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_h_sharded_optimised.cpp";
    static constexpr const char* BCAST_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_h_sharded_optimised.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = BCAST_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.defines = {bcast_defines.begin(), bcast_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{};

    // ---- Per-core runtime args ----
    // Single source of truth: compute_bcast_sharded_h_opt_per_core_args() derives every per-core arg the
    // same way for both this cache-miss build and the cache-hit re-apply in get_dynamic_runtime_args().
    // Reader arg0 is src1 (b)'s address; src0/output are CB `.buffer`-bound, but src1's CB (c_1) is a
    // staging buffer, so its address is rewritten in get_dynamic. Avoids the #46506 slow-path rebuild on
    // a cache hit; re-applying every core also covers work-core-set changes across a shared cache entry.
    const auto per_core = compute_bcast_sharded_h_opt_per_core_args(operation_attributes, a, b, output);

    for (uint32_t i = 0; i < per_core.cores.size(); i++) {
        const CoreCoord& core = per_core.cores[i];
        auto reader_args = per_core.reader_args[i];
        reader_args[0] = b.buffer()->address();  // baked address; binding below re-patches on a cache hit
        reader_desc.runtime_args.emplace_back(core, reader_args);
        // src1 (b) is read via TensorAccessor using arg 0 (its base address). Bind it as a patchable
        // Buffer* rt-arg so the descriptor fast cache-hit path re-patches it each dispatch.
        reader_desc.buffer_bindings.push_back({core, 0u, src1_buffer});

        compute_desc.runtime_args.emplace_back(core, per_core.compute_args[i]);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

std::vector<tt::tt_metal::DynamicRuntimeArg> BcastShardedHOptimisedProgramFactory::get_dynamic_runtime_args(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    // Kernel order matches create_descriptor(): reader(0), compute(1). There is no writer kernel.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kComputeKernelIdx = 1;

    const auto per_core = compute_bcast_sharded_h_opt_per_core_args(operation_attributes, a, b, output);

    const uint32_t src1_addr =
        b.buffer() != nullptr ? static_cast<uint32_t>(b.buffer()->address()) : 0u;

    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    for (uint32_t i = 0; i < per_core.cores.size(); i++) {
        const CoreCoord& core = per_core.cores[i];
        const auto& r = per_core.reader_args[i];
        const auto& c = per_core.compute_args[i];

        // reader arg0 is src1 (b)'s address; re-apply it plus all other shard-geometry slots.
        dynamic_args.push_back({kReaderKernelIdx, core, 0u, src1_addr});
        for (uint32_t aIdx = 1; aIdx < r.size(); ++aIdx) {
            dynamic_args.push_back({kReaderKernelIdx, core, aIdx, r[aIdx]});
        }
        for (uint32_t aIdx = 0; aIdx < c.size(); ++aIdx) {
            dynamic_args.push_back({kComputeKernelIdx, core, aIdx, c[aIdx]});
        }
    }
    return dynamic_args;
}

}  // namespace ttnn::prim
