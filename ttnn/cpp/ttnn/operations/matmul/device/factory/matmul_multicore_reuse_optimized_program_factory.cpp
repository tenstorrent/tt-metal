// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::Tensor;
using tt::tt_metal::WriterConfigDescriptor;

namespace ttnn::prim {

// Program is constructed from the ProgramDescriptor, then shared_variables_t
// is populated with kernel/CB handles for override_runtime_arguments().
MatmulMultiCoreReuseOptimizedProgramFactory::cached_program_t MatmulMultiCoreReuseOptimizedProgramFactory::create(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    ProgramDescriptor descriptor = create_descriptor(operation_attributes, tensor_args, tensor_return_value);

    tt_metal::Program program{descriptor};

    // Kernel handles are assigned sequentially in descriptor order:
    // reader=0, writer=1, compute kernels follow
    constexpr tt_metal::KernelHandle reader_id = 0;
    constexpr tt_metal::KernelHandle writer_id = 1;

    // Look up CB handles by buffer index
    tt_metal::CBHandle cb_src0 = 0, cb_src1 = 0, cb_output = 0;
    for (const auto& cb : program.circular_buffers()) {
        if (cb->buffer_indices().count(tt::CBIndex::c_0)) {
            cb_src0 = cb->id();
        }
        if (cb->buffer_indices().count(tt::CBIndex::c_1)) {
            cb_src1 = cb->id();
        }
        if (cb->buffer_indices().count(tt::CBIndex::c_4)) {
            cb_output = cb->id();
        }
    }

    // Compute num_cores and cores vector for shared_variables_t
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(operation_attributes.program_config.value());
    CoreCoord compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);

    bool in0_is_sharded = a.is_sharded();
    bool in1_is_sharded = b.is_sharded();
    bool output_is_sharded = output.is_sharded();

    std::optional<tt::tt_metal::ShardSpec> shard_spec = std::nullopt;
    if (in0_is_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (in1_is_sharded) {
        shard_spec = b.shard_spec().value();
    } else if (output_is_sharded) {
        shard_spec = output.shard_spec().value();
    }

    bool transpose_a = operation_attributes.transpose_a;
    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    uint32_t B = get_batch_size(ashape);
    uint32_t M = operations::matmul::utilities::get_M_dim(ashape, in0_tile, false);

    bool transpose_b = operation_attributes.transpose_b;
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);
    uint32_t N = operations::matmul::utilities::get_N_dim(bshape, in1_tile);

    uint32_t per_core_M = program_config.per_core_M;
    uint32_t per_core_N = program_config.per_core_N;
    uint32_t num_output_blocks_total = (B * M / per_core_M) * (N / per_core_N);

    uint32_t num_cores = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreCoord core_range = compute_with_storage_grid_size;

    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) = tt::tt_metal::split_work_to_cores(core_range, num_output_blocks_total);
    }

    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    }
    const auto cores = grid_to_cores(num_cores, core_range.x, core_range.y, row_major);

    return {std::move(program), {reader_id, writer_id, cb_src0, cb_src1, cb_output, num_cores, cores}};
}

void MatmulMultiCoreReuseOptimizedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto mm_kernel_in0_reader_id = shared_variables.mm_kernel_in0_reader_id;
    auto mm_kernel_in1_reader_writer_id = shared_variables.mm_kernel_in1_reader_writer_id;
    auto cb_src0 = shared_variables.cb_src0;
    auto cb_src1 = shared_variables.cb_src1;
    auto cb_output = shared_variables.cb_output;
    auto cores = shared_variables.cores;

    const auto& input_tensors = tensor_args.input_tensors;
    const auto& output_tensors = tensor_return_value;

    auto* src_buffer_a = input_tensors.at(0).buffer();
    auto* src_buffer_b = input_tensors.at(1).buffer();

    auto* dst_buffer = output_tensors.at(0).buffer();

    const bool src0_sharded = input_tensors[0].memory_config().is_sharded();
    const bool src1_sharded = input_tensors[1].memory_config().is_sharded();
    const bool out_sharded = output_tensors[0].memory_config().is_sharded();

    const bool update_reader_args = !src0_sharded;

    const bool update_writer_args = !(src1_sharded and out_sharded);

    if (update_reader_args || update_writer_args) {
        auto& reader_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in0_reader_id);

        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_reader_writer_id);

        for (const auto& core : cores) {
            if (update_reader_args) {
                auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer_a->address();  // in0_tensor_addr
            }

            if (update_writer_args) {
                auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer_b->address();  // in1_tensor_addr
                runtime_args[3] = dst_buffer->address();    // out_tensor_addr
            }
        }
    }
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
    }

    if (src1_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer_b);
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    }
}

CoreRangeSet MatmulMultiCoreReuseOptimizedProgramFactory::default_core_range(IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    return CoreRangeSet({CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
}

tt::tt_metal::ProgramDescriptor MatmulMultiCoreReuseOptimizedProgramFactory::create_descriptor(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<CoreRangeSet>& core_range_set) {
    // core_range_set reserved for future use; matmul derives cores from program_config
    (void)core_range_set;

    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(operation_attributes.program_config.value());

    TT_FATAL(operation_attributes.output_dtype.has_value(), "Output dtype should have been provided");
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Bcast batch should have been provided");

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);

    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;
    bool untilize_out = operation_attributes.untilize_out;

    CoreCoord compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
    uint32_t in0_block_w = program_config.in0_block_w;
    uint32_t out_subblock_h = program_config.out_subblock_h;
    uint32_t out_subblock_w = program_config.out_subblock_w;
    uint32_t per_core_M = program_config.per_core_M;
    uint32_t per_core_N = program_config.per_core_N;

    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format =
        tt_metal::datatype_to_dataformat_converter(operation_attributes.output_dtype.value());

    tt_metal::IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    if (fp32_dest_acc_en) {
        TT_FATAL(
            out_subblock_h * out_subblock_w <= 4,
            "Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode");
    }

    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = operations::matmul::utilities::get_M_dim(ashape, in0_tile, false);
    uint32_t Kt = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    uint32_t Nt = operations::matmul::utilities::get_N_dim(bshape, in1_tile);
    uint32_t M = Mt;
    uint32_t N = Nt;
    uint32_t K = Kt;

    const auto ashape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = ashape_logical[-1] % in0_tile.get_width();

    // Derived parameters
    uint32_t batch_scale_factor = per_core_M > M ? per_core_M / M : 1;
    uint32_t per_core_M_per_batch = per_core_M > M ? M : per_core_M;
    uint32_t num_blocks = (K / in0_block_w);
    bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();
    bool in0_is_sharded = a.is_sharded();
    bool in1_is_sharded = b.is_sharded();
    bool output_is_sharded = output.is_sharded();

    // CB sizes
    uint32_t in0_block_num_tiles = per_core_M_per_batch * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_num_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = per_core_M * K;
    } else {
        in0_CB_tiles *= 2;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_num_tiles;
    if (in1_is_sharded) {
        in1_CB_tiles *= num_blocks * batch_scale_factor;
    } else {
        in1_CB_tiles *= 2;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    // Compute kernel args
    uint32_t in0_num_subblocks = (per_core_M_per_batch / out_subblock_h);
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t out_num_subblocks_h = per_core_M_per_batch / out_subblock_h;
    uint32_t out_num_subblocks_w = in1_num_subblocks;
    uint32_t num_tiles_per_block_out = per_core_M_per_batch * per_core_N;
    uint32_t num_output_blocks_total = (B * M / per_core_M) * (N / per_core_N);

    std::optional<tt::tt_metal::ShardSpec> shard_spec = std::nullopt;
    if (in0_is_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (in1_is_sharded) {
        shard_spec = b.shard_spec().value();
    } else if (output_is_sharded) {
        shard_spec = output.shard_spec().value();
    }

    // Core splitting
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    CoreCoord core_range = compute_with_storage_grid_size;

    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = num_output_blocks_total / num_cores * batch_scale_factor;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) = tt::tt_metal::split_work_to_cores(core_range, num_output_blocks_total);
        num_blocks_per_core_group_1 *= batch_scale_factor;
        num_blocks_per_core_group_2 *= batch_scale_factor;
    }
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;
    TT_FATAL(num_evenly_divided_output_blocks > 0, "Not all cores from core_range was used!");

    const auto in0_tensor_stride_w = transpose_a ? M : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : K;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;

    // Compile time args for reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,
        (std::uint32_t)in0_block_w,
        (std::uint32_t)per_core_M_per_batch,
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)in0_last_ktile_w,
        (std::uint32_t)num_blocks,
        (std::uint32_t)bcast_batch,
        (std::uint32_t)M * K,
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    // Compile time args for reader/writer
    std::vector<uint32_t> reader_writer_compile_time_args = {
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)per_core_N,
        (std::uint32_t)in0_block_w,
        (std::uint32_t)in1_block_num_tiles,
        (std::uint32_t)num_blocks,
        (std::uint32_t)bcast_batch,
        (std::uint32_t)K * N,
        (std::uint32_t)1,
        (std::uint32_t)N,
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h * N,
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h,
        (std::uint32_t)(out_subblock_w * out_subblock_h),
        (std::uint32_t)out_num_subblocks_w,
        (std::uint32_t)out_num_subblocks_h,
        (std::uint32_t)M * N,
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(reader_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(reader_writer_compile_time_args);

    // Reader defines
    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines reader_writer_defines;
    if (in0_is_sharded) {
        reader_defines.emplace_back("IN0_SHARDED", "1");
    }
    if (in1_is_sharded) {
        reader_writer_defines.emplace_back("IN1_SHARDED", "1");
    }
    if (output_is_sharded) {
        reader_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    // Blackhole intermediate CB read workaround
    bool in0_needs_intermediate_cb_read = false;
    bool in1_needs_intermediate_cb_read = false;
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
        if (in0_needs_intermediate_cb_read) {
            reader_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
        }
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
        if (in1_needs_intermediate_cb_read) {
            reader_writer_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
        }
    }

    // Named compile-time args for CB indices (enables fusion/chaining)
    KernelDescriptor::NamedCompileTimeArgs cb_named_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", tt::CBIndex::c_5},
    };

    // Compute kernel compile time args (group 1)
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        1,  // out_num_blocks_x
        1,  // out_num_blocks_y
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        num_blocks_per_core_group_1,
        out_block_tiles,
        untilize_out,
        false,  // get_batch_from_reader
        in0_transpose_tile,
    };

    // Compute defines
    KernelDescriptor::Defines compute_defines;
    if (packer_l1_acc_en) {
        compute_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    if (in1_transpose_tile) {
        compute_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }

    // Build per-core runtime args
    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    }
    const auto cores = grid_to_cores(num_cores, core_range.x, core_range.y, row_major);

    uint32_t m_blocks_per_batch = M / per_core_M_per_batch;
    uint32_t n_blocks_per_batch = N / per_core_N;
    uint32_t blocks_per_batch = m_blocks_per_batch * n_blocks_per_batch;
    uint32_t in0_batch_stride = M * K;
    uint32_t in1_batch_stride = K * N;
    uint32_t in0_m_block_stride = per_core_M_per_batch * (transpose_a ? 1 : K);
    uint32_t in1_n_block_stride = per_core_N * (transpose_b ? K : 1);

    KernelDescriptor::RuntimeArgs reader_runtime_args;
    KernelDescriptor::RuntimeArgs reader_writer_runtime_args;
    KernelDescriptor::RuntimeArgs compute_runtime_args_g1;
    KernelDescriptor::RuntimeArgs compute_runtime_args_g2;
    reader_runtime_args.reserve(num_cores);
    reader_writer_runtime_args.reserve(num_cores);
    compute_runtime_args_g1.reserve(g1_numcores);

    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_output_blocks_per_core =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        uint32_t start_batch = num_blocks_written / blocks_per_batch;
        uint32_t block_within_batch = num_blocks_written % blocks_per_batch;
        uint32_t start_m_block = block_within_batch / n_blocks_per_batch;
        uint32_t start_n_block = block_within_batch % n_blocks_per_batch;

        uint32_t in0_start_tile_id = (start_batch * in0_batch_stride) + (start_m_block * in0_m_block_stride);
        uint32_t in1_start_tile_id =
            (bcast_batch ? 0 : (start_batch * in1_batch_stride)) + (start_n_block * in1_n_block_stride);

        reader_runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                (uint32_t)in0_buffer->address(),
                in0_start_tile_id,
                num_output_blocks_per_core,
            });

        reader_writer_runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                (uint32_t)in1_buffer->address(),
                in1_start_tile_id,
                num_output_blocks_per_core,
                (uint32_t)out_buffer->address(),
                num_blocks_written * num_tiles_per_block_out,
            });

        // Compute kernels have no per-core runtime args
        if (i < g1_numcores) {
            compute_runtime_args_g1.emplace_back(core, std::vector<uint32_t>{});
        } else {
            compute_runtime_args_g2.emplace_back(core, std::vector<uint32_t>{});
        }

        num_blocks_written += num_output_blocks_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel descriptor
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.named_compile_time_args = cb_named_args;
    reader_kernel_desc.defines = reader_defines;
    reader_kernel_desc.runtime_args = std::move(reader_runtime_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // Reader/Writer kernel descriptor (reads in1, writes output)
    KernelDescriptor reader_writer_kernel_desc;
    reader_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp";
    reader_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_writer_kernel_desc.core_ranges = all_cores;
    reader_writer_kernel_desc.compile_time_args = reader_writer_compile_time_args;
    reader_writer_kernel_desc.named_compile_time_args = cb_named_args;
    reader_writer_kernel_desc.defines = reader_writer_defines;
    reader_writer_kernel_desc.runtime_args = std::move(reader_writer_runtime_args);
    reader_writer_kernel_desc.config = WriterConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_writer_kernel_desc));

    // Compute kernel descriptor (core group 1)
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = core_group_1;
    compute_kernel_desc.compile_time_args = compute_kernel_args_group_1;
    compute_kernel_desc.named_compile_time_args = cb_named_args;
    compute_kernel_desc.defines = compute_defines;
    compute_kernel_desc.runtime_args = std::move(compute_runtime_args_g1);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode};
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    // Core group 2 compute kernel (if needed)
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles,
            in1_per_core_w,
            num_blocks,
            1,  // out_num_blocks_x
            1,  // out_num_blocks_y
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            num_blocks_per_core_group_2,
            out_block_tiles,
            untilize_out,
            false,  // get_batch_from_reader
            in0_transpose_tile,
        };

        KernelDescriptor compute_kernel_desc_g2;
        compute_kernel_desc_g2.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation.cpp";
        compute_kernel_desc_g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_kernel_desc_g2.core_ranges = core_group_2;
        compute_kernel_desc_g2.compile_time_args = compute_kernel_args_group_2;
        compute_kernel_desc_g2.named_compile_time_args = cb_named_args;
        compute_kernel_desc_g2.defines = compute_defines;
        compute_kernel_desc_g2.runtime_args = std::move(compute_runtime_args_g2);
        compute_kernel_desc_g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode};
        program_descriptor.kernels.push_back(std::move(compute_kernel_desc_g2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    auto make_cb_descriptor = [&all_cores](
                                  uint32_t total_size,
                                  uint8_t buffer_index,
                                  tt::DataFormat data_format,
                                  uint32_t page_size,
                                  const tt::tt_metal::Tile& tile,
                                  Buffer* buffer = nullptr) {
        CBDescriptor cb_desc;
        cb_desc.total_size = total_size;
        cb_desc.core_ranges = all_cores;
        tt::tt_metal::TileDescriptor tile_desc{tile};
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = buffer_index, .data_format = data_format, .page_size = page_size, .tile = tile_desc});
        cb_desc.buffer = buffer;
        return cb_desc;
    };

    // CB 0: Input A
    program_descriptor.cbs.push_back(make_cb_descriptor(
        in0_CB_size,
        tt::CBIndex::c_0,
        in0_data_format,
        in0_single_tile_size,
        in0_tile,
        in0_is_sharded ? in0_buffer : nullptr));

    // CB 1: Input B
    program_descriptor.cbs.push_back(make_cb_descriptor(
        in1_CB_size,
        tt::CBIndex::c_1,
        in1_data_format,
        in1_single_tile_size,
        in1_tile,
        in1_is_sharded ? in1_buffer : nullptr));

    // CB 4 and CB 5: Output and intermediate accumulator
    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // Separate output and intermediate CBs
        program_descriptor.cbs.push_back(make_cb_descriptor(

            out_CB_size,
            tt::CBIndex::c_4,
            output_data_format,
            output_single_tile_size,
            output_tile,
            output_is_sharded ? out_buffer : nullptr));
        program_descriptor.cbs.push_back(make_cb_descriptor(
            interm0_CB_size, tt::CBIndex::c_5, interm0_data_format, interm0_single_tile_size, output_tile));
    } else {
        // Shared output+intermediate CB
        CBDescriptor output_cb_desc;
        output_cb_desc.total_size = out_CB_size;
        output_cb_desc.core_ranges = all_cores;
        tt::tt_metal::TileDescriptor output_tile_desc{output_tile};
        output_cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        output_cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        output_cb_desc.buffer = output_is_sharded ? out_buffer : nullptr;
        program_descriptor.cbs.push_back(std::move(output_cb_desc));
    }

    // Optional CBs for Blackhole intermediate reads
    if (in1_needs_intermediate_cb_read) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            in1_single_tile_size, tt::CBIndex::c_9, in1_data_format, in1_single_tile_size, in1_tile));
    }
    if (in0_needs_intermediate_cb_read) {
        program_descriptor.cbs.push_back(make_cb_descriptor(
            in0_single_tile_size, tt::CBIndex::c_8, in0_data_format, in0_single_tile_size, in0_tile));
    }

    // Optional transpose CB
    if (in0_transpose_tile) {
        program_descriptor.cbs.push_back(
            make_cb_descriptor(in0_CB_size, tt::CBIndex::c_10, in0_data_format, in0_single_tile_size, in0_tile));
    }

    return program_descriptor;
}

MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory::cached_mesh_workload_t
MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange mesh_coord_range{mesh_coord, mesh_coord};
            auto single_device_program =
                MatmulMultiCoreReuseOptimizedProgramFactory::create(attributes, tensor_args, tensor_return_value);
            shared_variables[mesh_coord_range] = single_device_program.shared_variables;
            workload.add_program(mesh_coord_range, std::move(single_device_program.program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = MatmulMultiCoreReuseOptimizedProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        MatmulMultiCoreReuseOptimizedProgramFactory::override_runtime_arguments(
            cached_program_proxy, attributes, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::prim
