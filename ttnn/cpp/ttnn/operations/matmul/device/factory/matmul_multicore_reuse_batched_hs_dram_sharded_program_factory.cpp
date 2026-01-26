// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim {
namespace reuse_batched_hs_dram_sharded_optimized_helpers {

tt::tt_metal::IDevice* get_device_for_dram_banks(const ttnn::Tensor& a, const ttnn::MeshCoordinate& coord) {
    ttnn::distributed::MeshDevice* device = a.device();
    const ttnn::distributed::MeshDeviceView& view = device->get_view();
    if (!view.contains(coord) || !view.is_local(coord)) {
        return device;
    }
    return a.device()->get_device(coord);
}

void get_max_page_size_and_num_pages(
    tt::tt_metal::IDevice* device, uint32_t num_tiles, uint32_t tile_size, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * tile_size;

    uint32_t noc_max_page_size;
    if (device->arch() == tt::ARCH::WORMHOLE_B0) {
        noc_max_page_size = 8192;
    } else if (device->arch() == tt::ARCH::BLACKHOLE) {
        noc_max_page_size = 16384;
    } else {
        TT_FATAL(false, "Unsupported architecture for DRAM sharded matmul. Only Wormhole and Blackhole are supported.");
    }

    page_size = (noc_max_page_size / tile_size) * tile_size;
    while (total_size % page_size != 0 && page_size >= tile_size) {
        page_size -= tile_size;
    }
    num_pages = total_size / page_size;
}

void get_optimal_dram_bank_to_reader_assignment(
    tt::tt_metal::IDevice* device,
    std::vector<CoreCoord>& all_worker_cores_ordered,
    CoreRangeSet& all_worker_cores,
    tt_metal::NOC noc) {
    all_worker_cores_ordered = device->get_optimal_dram_bank_to_logical_worker_assignment(noc);
    std::set<CoreRange> all_cores_set;
    for (const auto& worker_core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(worker_core));
    }
    all_worker_cores = CoreRangeSet(all_cores_set);
}

// Batch-sharded DRAM matmul
// For batched matmul: [1, B, M, N] x [1, B, N, K] = [1, B, M, K]
// Sharded by batch dimension - each worker handles B/num_workers complete matmuls
std::pair<tt::tt_metal::Program, MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::shared_variables_t>
create_program_batch_sharded(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& /* output_all_storage_cores */,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,            // Total batch size
    uint32_t M,            // M dimension (rows of A, rows of output)
    uint32_t N,            // N dimension (cols of A, rows of B - contracted dimension)
    uint32_t K,            // K dimension (cols of B, cols of output)
    uint32_t in0_block_w,  // Block width for inner loop over N
    uint32_t per_core_M,   // M tiles per core (should equal M for batch sharding)
    uint32_t per_core_K,   // K tiles per core (should equal K for batch sharding)
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    bool skip_compute,
    bool skip_write_back) {
    log_debug(tt::LogOp, "Batch-sharded DRAM matmul");
    log_debug(tt::LogOp, "B: {}, M: {}, N: {}, K: {}", B, M, N, K);
    log_debug(tt::LogOp, "per_core_M: {}, per_core_K: {}, in0_block_w: {}", per_core_M, per_core_K, in0_block_w);

    tt_metal::Program program{};

    // For batch sharding, workers run on the L1 shard grid (input_all_storage_cores)
    // because the sharded CBs are tied to those core locations.
    // Use row_major=true for core enumeration (works correctly with 1D grids).
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // Get cores in row-major order (x varies first, then y)
    std::vector<CoreCoord> all_worker_cores_ordered = corerange_to_cores(input_all_storage_cores, std::nullopt, true);
    CoreRangeSet all_worker_cores = input_all_storage_cores;

    uint32_t num_cores = all_worker_cores_ordered.size();
    uint32_t num_dram_banks = device->num_dram_channels();
    uint32_t batches_per_core = (B + num_cores - 1) / num_cores;

    TT_FATAL(
        num_cores <= num_dram_banks,
        "Number of worker cores ({}) cannot exceed number of DRAM banks ({})",
        num_cores,
        num_dram_banks);

    log_debug(
        tt::LogOp,
        "num_cores: {}, num_dram_banks: {}, batches_per_core: {}",
        num_cores,
        num_dram_banks,
        batches_per_core);

    // Subblock parameters
    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_K, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t num_blocks = N / in0_block_w;  // Number of inner loop iterations
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    // Tile sizes
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // CB sizes
    // in0: M x in0_block_w tiles per block
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    // in1: in0_block_w x K tiles per block
    uint32_t in1_block_tiles = in0_block_w * per_core_K;
    uint32_t in1_CB_tiles = in1_block_tiles * 3;  // triple buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    // output: M x K tiles
    uint32_t out_block_tiles = per_core_M * per_core_K;
    uint32_t interm0_CB_size = out_block_tiles * interm0_single_tile_size;

    // Sharded input buffer (in0 in L1)
    uint32_t in0_shard_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0] *
                               in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_CB_size = in0_shard_tiles * in0_single_tile_size;

    // Bias CB
    uint32_t in3_block_tiles = per_core_K;
    uint32_t in3_CB_size = in3_block_tiles * bias_single_tile_size;

    // Output reshard buffer
    uint32_t out_shard_tiles = out_buffer->shard_spec().shape()[0] / output_tile.get_tile_shape()[0] *
                               out_buffer->shard_spec().shape()[1] / output_tile.get_tile_shape()[1];
    uint32_t out_reshard_CB_size = out_shard_tiles * output_single_tile_size;

    // Page sizes for DRAM reads
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    // Tensor stride calculations (bytes per batch)
    // M, N, K are already in tiles, so stride = num_tiles * tile_size
    uint32_t in0_batch_stride_bytes = per_core_M * N * in0_single_tile_size;
    uint32_t in1_batch_stride_bytes = N * per_core_K * in1_single_tile_size;
    uint32_t out_batch_stride_bytes = per_core_M * per_core_K * output_single_tile_size;

    log_debug(
        tt::LogOp,
        "in0_batch_stride_bytes: {}, in1_batch_stride_bytes: {}, out_batch_stride_bytes: {}",
        in0_batch_stride_bytes,
        in1_batch_stride_bytes,
        out_batch_stride_bytes);

    // Use input shard grid as worker cores (CBs are tied to these cores)
    CoreRangeSet worker_cores_crs = all_worker_cores;

    // CB 0: in0 (activations)
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, worker_cores_crs, src0_cb_config);

    // CB 1: in1 (weights)
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    tt_metal::CreateCircularBuffer(program, worker_cores_crs, src1_cb_config);

    // CB 2: sharded in0 buffer
    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in0_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, worker_cores_crs, src2_cb_config);

    // CB 3: bias (if fused)
    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
        tt_metal::CreateCircularBuffer(program, worker_cores_crs, cb_src3_config);
    }

    // CB 4 & 5: output and intermediate
    // CB 4 is directly mapped to the output buffer (no separate reshard step)
    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CBHandle cb_output;
    if (interm0_data_format != output_data_format) {
        // Need separate CBs for output and intermediate
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(out_reshard_CB_size, {{output_cb_index, output_data_format}})
                .set_page_size(output_cb_index, output_single_tile_size)
                .set_tile_dims(output_cb_index, output_tile)
                .set_globally_allocated_address(*out_buffer);
        cb_output = tt_metal::CreateCircularBuffer(program, worker_cores_crs, output_cb_config);

        tt_metal::CircularBufferConfig interm0_cb_config =
            tt_metal::CircularBufferConfig(interm0_CB_size, {{interm0_cb_index, interm0_data_format}})
                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                .set_tile_dims(interm0_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, worker_cores_crs, interm0_cb_config);
    } else {
        // Output and intermediate share the same buffer - directly map to output buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(out_reshard_CB_size, output_cb_data_format_spec)
                .set_page_size(output_cb_index, output_single_tile_size)
                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                .set_tile_dims(output_cb_index, output_tile)
                .set_tile_dims(interm0_cb_index, output_tile)
                .set_globally_allocated_address(*out_buffer);
        cb_output = tt_metal::CreateCircularBuffer(program, worker_cores_crs, output_cb_config);
    }

    // CB 6 is no longer needed - we write directly to CB 4 which is backed by the output buffer

    // Kernel defines
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            using ttnn::operations::unary::utils::get_defines;
            mm_kernel_defines.merge(get_defines(
                fused_activation.value().op_type,
                fused_activation.value().params,
                "ACTIVATION",
                "i",
                tt_metal::dataformat_to_datatype_converter(output_data_format)));
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    // Note: We intentionally do NOT define MATMUL_DRAM_SHARDED here.
    // When defined, compute writes to CB5 instead of CB4, which doesn't match our writer.
    // The 1D DRAM sharded factory also doesn't define it.

    // in0 reader compile time args
    std::vector<uint32_t> in0_reader_compile_args = {
        (uint32_t)in0_block_tiles,                         // in0_block_num_tiles
        (uint32_t)in0_block_tiles * in0_single_tile_size,  // in0_block_size_bytes
        (uint32_t)num_blocks,                              // num_blocks (N / in0_block_w)
        (uint32_t)batches_per_core,                        // num_batches_per_core
        (uint32_t)in0_batch_stride_bytes,                  // in0_tensor_stride_batch_bytes
    };

    // in1 reader/writer compile time args
    std::vector<uint32_t> in1_writer_compile_args = {
        (uint32_t)in1_buffer_page_size,
        (uint32_t)in1_buffer_num_pages,
        (uint32_t)per_core_K,              // in1_block_w (K tiles)
        (uint32_t)in1_block_tiles,         // in1_block_num_tiles
        (uint32_t)num_blocks,              // num_blocks (N / in0_block_w)
        (uint32_t)out_block_tiles,         // out_block_num_tiles
        (uint32_t)batches_per_core,        // num_batches_per_core
        (uint32_t)in1_batch_stride_bytes,  // in1_tensor_stride_batch_bytes
        (uint32_t)out_batch_stride_bytes,  // out_tensor_stride_batch_bytes
    };
    if (bias_buffer != nullptr) {
        in1_writer_compile_args.push_back(bias_buffer_page_size);
        in1_writer_compile_args.push_back(bias_buffer_num_pages);
        in1_writer_compile_args.push_back(in3_block_tiles);
    }

    // Compute kernel compile time args
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_K / out_subblock_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_tiles,         // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles
        in1_num_subblocks,       // in1_num_subblocks
        in1_block_tiles,         // in1_block_num_tiles
        per_core_K,              // in1_per_core_w
        num_blocks,              // num_blocks
        1,                       // out_num_blocks_x
        1,                       // out_num_blocks_y
        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        batches_per_core,        // batch (batches per core)
        out_block_tiles,         // out_block_num_tiles
        untilize_out ? 1u : 0u,  // untilize_out
        0u,                      // get_batch_from_reader
        0u,                      // in0_transpose_tile
    };

    // Create kernels
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    for (auto core_range : worker_cores_crs.ranges()) {
        for (auto core : core_range) {
            std::cout << "CoreX: " << core.x << " CoreY: " << core.y << std::endl;
        }
    }
    auto mm_kernel_in0_reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp",
        worker_cores_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_reader_compile_args,
            .defines = reader_defines});

    auto mm_kernel_in1_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp",
        worker_cores_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_writer_compile_args,
            .defines = writer_defines});

    auto mm_kernel_compute_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        worker_cores_crs,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    // Set runtime args for each worker
    // Worker i (in corerange_to_cores order) has L1 shard i, so read from DRAM bank i
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;

    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];
        uint32_t bank_id = i;  // Worker i reads from bank i (linear mapping)

        // in0 reader runtime args
        std::vector<uint32_t> in0_reader_runtime_args = {
            1u,  // worker_core_type (1 = active worker)
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_reader_id, core, in0_reader_runtime_args);

        // in1 writer runtime args
        std::vector<uint32_t> in1_writer_runtime_args = {
            1u,  // is_worker_core
            in1_buffer->address(),
            bias_buffer != nullptr ? bias_buffer->address() : 0u,
            bank_id,
            bank_id & 0x3,  // vc
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_writer_id, core, in1_writer_runtime_args);
        writer_kernel_ids.push_back(mm_kernel_in1_writer_id);

        // Compute runtime args
        std::vector<uint32_t> compute_runtime_args = {
            1u,  // is_worker_core
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_compute_id, core, compute_runtime_args);
    }

    return {
        std::move(program),
        MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::shared_variables_t{
            writer_kernel_ids, all_worker_cores_ordered, cb_src2, cb_output}};
}

}  // namespace reuse_batched_hs_dram_sharded_optimized_helpers

std::pair<tt::tt_metal::Program, MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::shared_variables_t>
matmul_multi_core_reuse_batched_hs_dram_sharded_optimized_(
    const ttnn::MeshCoordinate& mesh_coord,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::prim::MatmulParams& operation_attributes) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    const auto& output = output_tensors.at(0);
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Bias tensor must be on device");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
        bias_buffer = c.buffer();
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt::tt_metal::IDevice* device =
        reuse_batched_hs_dram_sharded_optimized_helpers::get_device_for_dram_banks(a, mesh_coord);

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>(
            operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& fused_activation = program_config.fused_activation;
    const auto& untilize_out = operation_attributes.untilize_out;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // For batch sharding: [1, B, M, N] x [1, B, N, K] = [1, B, M, K]
    // ashape = [1, B, M, N], bshape = [1, B, N, K]
    uint32_t B = ashape[1];
    uint32_t M = ashape[-2] / in0_tile_shape[0];  // M in tiles
    uint32_t N = ashape[-1] / in0_tile_shape[1];  // N in tiles (contracted dimension)
    uint32_t K = bshape[-1] / in1_tile_shape[1];  // K in tiles

    TT_FATAL(N % in0_block_w == 0, "N ({}) must be divisible by in0_block_w ({})", N, in0_block_w);

    return reuse_batched_hs_dram_sharded_optimized_helpers::create_program_batch_sharded(
        device,
        input_all_cores_storage,
        output_all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        B,
        M,
        N,
        K,
        in0_block_w,
        per_core_M,
        per_core_N,  // This is per_core_K for batch sharding
        fused_activation,
        in0_buffer,
        in1_buffer,
        bias_buffer,
        out_buffer,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        false,   // skip_compute
        false);  // skip_write_back
}

MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::cached_mesh_workload_t
MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto [program, shared_variable] = matmul_multi_core_reuse_batched_hs_dram_sharded_optimized_(
                mesh_coord, tensor_args, tensor_return_value, operation_attributes);
            shared_variables[single_coord_range] = shared_variable;
            workload.add_program(single_coord_range, std::move(program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto* src_buffer_a = input_tensors.at(0).buffer();
        auto* src_buffer_b = input_tensors.at(1).buffer();
        const auto& bias_tensor = optional_input_tensors.at(0);
        auto* dst_buffer = output_tensors.at(0).buffer();
        auto shared_variables = cached_workload.shared_variables.at(mesh_coord_range);

        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_src2, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output_reshard, *dst_buffer);

        const auto& all_worker_cores_ordered = shared_variables.all_worker_cores_ordered;
        const auto& writer_kernel_ids = shared_variables.writer_kernel_ids;

        for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
            auto core = all_worker_cores_ordered[i];
            auto writer_kernel_id = writer_kernel_ids[i];
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[1] = src_buffer_b->address();
            if (bias_tensor.has_value()) {
                writer_runtime_args[2] = bias_tensor.value().buffer()->address();
            } else {
                writer_runtime_args[2] = 0;
            }
        }
    }
}

}  // namespace ttnn::prim
