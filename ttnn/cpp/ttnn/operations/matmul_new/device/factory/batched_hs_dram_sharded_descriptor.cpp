// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "batched_hs_dram_sharded_descriptor.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim::matmul_new_detail {

using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;

tt::tt_metal::ProgramDescriptor BatchedHSDRAMShardedDescriptorFactory::create_descriptor(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;

    // ========================================================================
    // Extract tensor information
    // ========================================================================
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
    tt::DataFormat in0_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());

    Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    tt::tt_metal::Tile bias_tile = output_tile;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Bias tensor must be on device");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
        bias_buffer = c.buffer();
        bias_data_format = datatype_to_dataformat_converter(c.dtype());
        bias_tile = c.tensor_spec().tile();
    }

    IDevice* device = a.device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_storage_cores = a.shard_spec().value().grid;
    CoreRangeSet output_all_storage_cores = output.shard_spec().value().grid;

    Buffer* in0_buffer = a.buffer();
    Buffer* in1_buffer = b.buffer();
    Buffer* out_buffer = output.buffer();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);

    TT_FATAL(
        in0_buffer->size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_buffer->size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_buffer->size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_buffer->size(),
        in1_single_tile_size);
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] = {}, B.shape[-2] = {}) must match for matmul",
        ashape[-1],
        bshape[-2]);
    TT_FATAL(
        ashape[-2] % in0_tile_shape[0] == 0,
        "A.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        ashape[-2],
        in0_tile_shape[0]);
    TT_FATAL(
        ashape[-1] % in0_tile_shape[1] == 0,
        "A.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        ashape[-1],
        in0_tile_shape[1]);
    TT_FATAL(
        bshape[-2] % in1_tile_shape[0] == 0,
        "B.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        bshape[-2],
        in1_tile_shape[0]);
    TT_FATAL(
        bshape[-1] % in1_tile_shape[1] == 0,
        "B.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        bshape[-1],
        in1_tile_shape[1]);

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

    // For batch sharding: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
    uint32_t B = ashape[1];
    uint32_t M = ashape[-2] / in0_tile_shape[0];  // M in tiles
    uint32_t K = ashape[-1] / in0_tile_shape[1];  // K in tiles
    uint32_t N = bshape[-1] / in1_tile_shape[1];  // N in tiles

    TT_FATAL(
        per_core_M == M,
        "For batch sharding, per_core_M ({}) must equal M ({}) - each core handles complete matmuls",
        per_core_M,
        M);
    TT_FATAL(
        per_core_N == N,
        "For batch sharding, per_core_N ({}) must equal N ({}) - each core handles complete matmuls",
        per_core_N,
        N);
    TT_FATAL(K % in0_block_w == 0, "K ({}) must be divisible by in0_block_w ({})", K, in0_block_w);

    // ========================================================================
    // Core setup and parameter computation
    // ========================================================================
    NOC in1_noc = detail::preferred_noc_for_dram_read(device->arch());
    NOC in0_noc = detail::preferred_noc_for_dram_write(device->arch());

    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores, in1_noc);

    // Input/output storage core ordering
    std::vector<CoreCoord> input_storage_cores_ordered =
        corerange_to_cores(input_all_storage_cores, std::nullopt, true);
    std::vector<CoreCoord> output_storage_cores_ordered =
        corerange_to_cores(output_all_storage_cores, std::nullopt, true);

    uint32_t num_workers = all_worker_cores_ordered.size();
    uint32_t num_input_storage_cores = input_storage_cores_ordered.size();
    uint32_t num_output_storage_cores = output_storage_cores_ordered.size();

    TT_FATAL(
        num_input_storage_cores == num_workers,
        "Input storage cores ({}) must match number of workers/DRAM banks ({}). "
        "Use an L1 shard grid with {} cores.",
        num_input_storage_cores,
        num_workers,
        num_workers);
    TT_FATAL(
        num_output_storage_cores == num_workers,
        "Output storage cores ({}) must match number of workers/DRAM banks ({}). "
        "Use an L1 shard grid with {} cores.",
        num_output_storage_cores,
        num_workers,
        num_workers);

    for (uint32_t i = 0; i < num_workers; ++i) {
        TT_FATAL(
            input_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Input storage core ordering mismatch at index {}! "
            "Storage core ({}, {}) != Worker core ({}, {}). "
            "The L1 shard grid must use the same core ordering as "
            "device.get_optimal_dram_bank_to_logical_worker_assignment(). "
            "Using e.g. a simple rectangular CoreRange will cause data misrouting.",
            i,
            input_storage_cores_ordered[i].x,
            input_storage_cores_ordered[i].y,
            all_worker_cores_ordered[i].x,
            all_worker_cores_ordered[i].y);

        TT_FATAL(
            output_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Output storage core ordering mismatch at index {}! "
            "Storage core ({}, {}) != Worker core ({}, {}). "
            "The L1 shard grid must use the same core ordering as "
            "device.get_optimal_dram_bank_to_logical_worker_assignment(). "
            "Using e.g. a simple rectangular CoreRange will cause data misrouting.",
            i,
            output_storage_cores_ordered[i].x,
            output_storage_cores_ordered[i].y,
            all_worker_cores_ordered[i].x,
            all_worker_cores_ordered[i].y);
    }

    // Build NOC coordinate vectors for input and output storage cores
    std::vector<uint32_t> input_storage_noc_x, input_storage_noc_y;
    std::vector<uint32_t> output_storage_noc_x, output_storage_noc_y;
    for (const auto& core : input_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        input_storage_noc_x.push_back(phys_core.x);
        input_storage_noc_y.push_back(phys_core.y);
    }
    for (const auto& core : output_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        output_storage_noc_x.push_back(phys_core.x);
        output_storage_noc_y.push_back(phys_core.y);
    }

    // Compute bounding box
    std::set<CoreRange> all_cores_set;
    for (const auto& core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : input_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : output_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    CoreRangeSet all_cores(all_cores_set);
    CoreRange bounding_box = all_cores.bounding_box();
    CoreRangeSet all_cores_in_rect_grid({bounding_box});

    uint32_t num_cores = num_workers;
    uint32_t num_dram_banks = device->num_dram_channels();
    uint32_t batches_per_core = (B + num_cores - 1) / num_cores;

    TT_FATAL(
        num_cores <= num_dram_banks,
        "Number of worker cores ({}) cannot exceed number of DRAM banks ({})",
        num_cores,
        num_dram_banks);

    // Subblock parameters
    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_N, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    // Tile sizes
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // CB sizes
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = in0_block_w * per_core_N;
    uint32_t in1_CB_tiles = in1_block_tiles * 3;  // triple buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t interm0_CB_size = out_block_tiles * interm0_single_tile_size;

    // Sharded input buffer (in0 in L1)
    uint32_t in0_shard_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0] *
                               in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_CB_size = in0_shard_tiles * in0_single_tile_size;

    // Bias CB
    uint32_t in3_block_tiles = per_core_N;
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
    uint32_t in0_batch_stride_bytes = per_core_M * K * in0_single_tile_size;
    uint32_t in1_batch_stride_bytes = K * per_core_N * in1_single_tile_size;
    uint32_t out_batch_stride_bytes = per_core_M * per_core_N * output_single_tile_size;

    // ========================================================================
    // Build ProgramDescriptor
    // ========================================================================
    ProgramDescriptor desc;

    // -- Circular Buffers --
    // CB0: in0 (activations) - on all cores in bounding box
    {
        CBDescriptor cb;
        cb.total_size = in0_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = TileDescriptor(in0_tile),
        });
        desc.cbs.push_back(std::move(cb));
    }
    // CB1: in1 (weights) - on all cores in bounding box
    {
        CBDescriptor cb;
        cb.total_size = in1_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = TileDescriptor(in1_tile),
        });
        desc.cbs.push_back(std::move(cb));
    }
    // CB2: sharded in0 buffer - on INPUT storage cores, backed by in0_buffer
    {
        CBDescriptor cb;
        cb.total_size = in2_CB_size;
        cb.core_ranges = input_all_storage_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = TileDescriptor(in0_tile),
        });
        cb.buffer = in0_buffer;
        desc.cbs.push_back(std::move(cb));
    }
    // CB3: bias (if fused) - on all cores in bounding box
    if (bias_buffer != nullptr) {
        CBDescriptor cb;
        cb.total_size = in3_CB_size;
        cb.core_ranges = all_cores_in_rect_grid;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
            .data_format = bias_data_format,
            .page_size = bias_single_tile_size,
            .tile = TileDescriptor(bias_tile),
        });
        desc.cbs.push_back(std::move(cb));
    }
    // CB4 & CB5: output and intermediate - on worker cores
    {
        uint32_t output_cb_index = tt::CBIndex::c_4;
        uint32_t interm0_cb_index = tt::CBIndex::c_5;

        if (interm0_data_format != output_data_format) {
            // Separate output and interm0 CBs
            {
                CBDescriptor cb;
                cb.total_size = out_reshard_CB_size;
                cb.core_ranges = all_worker_cores;
                cb.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(output_cb_index),
                    .data_format = output_data_format,
                    .page_size = output_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                });
                desc.cbs.push_back(std::move(cb));
            }
            {
                CBDescriptor cb;
                cb.total_size = interm0_CB_size;
                cb.core_ranges = all_worker_cores;
                cb.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                });
                desc.cbs.push_back(std::move(cb));
            }
        } else {
            // Shared buffer for output and interm0
            CBDescriptor cb;
            cb.total_size = out_reshard_CB_size;
            cb.core_ranges = all_worker_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            desc.cbs.push_back(std::move(cb));
        }
    }
    // CB6: output reshard buffer - on OUTPUT storage cores, backed by out_buffer
    {
        CBDescriptor cb;
        cb.total_size = out_reshard_CB_size;
        cb.core_ranges = output_all_storage_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = TileDescriptor(output_tile),
        });
        cb.buffer = out_buffer;
        desc.cbs.push_back(std::move(cb));
    }

    // -- Kernel defines --
    KernelDescriptor::Defines compute_defines;
    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;

    if (bias_buffer != nullptr) {
        compute_defines.emplace_back("FUSE_BIAS", "1");
        writer_defines.emplace_back("FUSE_BIAS", "1");
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            compute_defines.emplace_back("PACK_RELU", "1");
        } else {
            using ttnn::operations::unary::utils::get_defines;
            auto activation_defines = get_defines(
                fused_activation.value().op_type,
                fused_activation.value().params,
                "ACTIVATION",
                "i",
                dataformat_to_datatype_converter(output_data_format));
            for (const auto& [k, v] : activation_defines) {
                compute_defines.emplace_back(k, v);
            }
        }
    }
    if (packer_l1_acc_en) {
        compute_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    compute_defines.emplace_back("MATMUL_DRAM_SHARDED", "1");
    writer_defines.emplace_back("OUT_SHARDED", "1");

    // -- Compile-time args --
    // in0 reader
    std::vector<uint32_t> in0_reader_compile_args = {
        (uint32_t)in0_block_tiles,                         // in0_block_num_tiles
        (uint32_t)in0_block_tiles * in0_single_tile_size,  // in0_block_size_bytes
        (uint32_t)num_blocks,                              // num_blocks (K / in0_block_w)
        (uint32_t)batches_per_core,                        // num_batches_per_core
        (uint32_t)in0_batch_stride_bytes,                  // in0_tensor_stride_batch_bytes
        (uint32_t)in2_CB_size,                             // in0_shard_size_bytes (full shard)
    };

    // in1 reader/writer
    std::vector<uint32_t> in1_writer_compile_args = {
        (uint32_t)in1_buffer_page_size,
        (uint32_t)in1_buffer_num_pages,
        (uint32_t)per_core_N,              // in1_block_w (N tiles)
        (uint32_t)in1_block_tiles,         // in1_block_num_tiles
        (uint32_t)num_blocks,              // num_blocks (K / in0_block_w)
        (uint32_t)out_block_tiles,         // out_block_num_tiles
        (uint32_t)batches_per_core,        // num_batches_per_core
        (uint32_t)in1_batch_stride_bytes,  // in1_tensor_stride_batch_bytes
        (uint32_t)out_batch_stride_bytes,  // out_tensor_stride_batch_bytes (for NOC write)
        (uint32_t)out_reshard_CB_size,     // out_shard_size_bytes (full shard)
    };
    if (bias_buffer != nullptr) {
        in1_writer_compile_args.push_back(bias_buffer_page_size);
        in1_writer_compile_args.push_back(bias_buffer_num_pages);
        in1_writer_compile_args.push_back(in3_block_tiles);
    }

    // Compute kernel
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_tiles,         // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles
        in1_num_subblocks,       // in1_num_subblocks
        in1_block_tiles,         // in1_block_num_tiles
        per_core_N,              // in1_per_core_w
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

    // -- Kernel Descriptors --
    KernelDescriptor in0_reader_desc;
    in0_reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp";
    in0_reader_desc.core_ranges = all_cores_in_rect_grid;
    in0_reader_desc.compile_time_args = in0_reader_compile_args;
    in0_reader_desc.defines = reader_defines;
    in0_reader_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = in0_noc,
    };

    KernelDescriptor in1_writer_desc;
    in1_writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp";
    in1_writer_desc.core_ranges = all_cores_in_rect_grid;
    in1_writer_desc.compile_time_args = in1_writer_compile_args;
    in1_writer_desc.defines = writer_defines;
    in1_writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = in1_noc,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation.cpp";
    compute_desc.core_ranges = all_cores_in_rect_grid;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = compute_defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // ========================================================================
    // Runtime args
    // ========================================================================
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);
    std::set<CoreCoord> worker_cores_set(all_worker_cores_ordered.begin(), all_worker_cores_ordered.end());

    // Set idle args for non-worker cores in the bounding box
    for (const auto& core : all_cores_in_rect_grid_vec) {
        bool is_worker = worker_cores_set.contains(core);

        if (!is_worker) {
            in0_reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{0u});
            in1_writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{0u});
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{0u});
        }
    }

    // Set worker args for active cores
    std::vector<uint32_t> bank_ids;
    for (uint32_t worker_idx = 0; worker_idx < all_worker_cores_ordered.size(); ++worker_idx) {
        auto core = all_worker_cores_ordered[worker_idx];

        uint32_t bank_id = worker_idx;

        // Calculate VC (virtual channel) to avoid conflicts
        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < worker_idx; ++j) {
            auto core_prev = all_worker_cores_ordered[j];
            if (core_prev.y == core.y && ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        // in0 reader runtime args
        in0_reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                1u,                               // worker_core_type (1 = active worker)
                input_storage_noc_x[worker_idx],  // input storage core NOC x
                input_storage_noc_y[worker_idx],  // input storage core NOC y
                in0_buffer->address(),            // L1 address of in0 shard on storage core
            });

        // in1 writer runtime args
        in1_writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                1u,  // is_worker_core
                in1_buffer->address(),
                bias_buffer != nullptr ? bias_buffer->address() : 0u,
                bank_id,
                vc,
                output_storage_noc_x[worker_idx],  // output storage core NOC x
                output_storage_noc_y[worker_idx],  // output storage core NOC y
                out_buffer->address(),             // L1 address of output shard on storage core
            });

        // Compute runtime args
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{1u});
    }

    // Push kernels to descriptor
    desc.kernels.push_back(std::move(in0_reader_desc));
    desc.kernels.push_back(std::move(in1_writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim::matmul_new_detail
