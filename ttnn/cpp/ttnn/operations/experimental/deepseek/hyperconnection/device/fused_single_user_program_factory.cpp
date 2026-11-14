// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_program_factory.hpp"

#include <bit>
#include <cstdlib>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace fused_single_user_detail {

constexpr uint32_t kRoleCollapse = 0;
constexpr uint32_t kRolePost = 1;
constexpr uint32_t kRoleComb = 2;
constexpr uint32_t kCollapseCoreCount = 8;

constexpr uint32_t kFsCbFusedW = tt::CBIndex::c_0;
// The reader's parameter tile (see kernels/compute/hc_single_user_sfpu.h) and the DRAM bias rows
// it is assembled from.
constexpr uint32_t kFsCbParams = tt::CBIndex::c_1;
constexpr uint32_t kFsCbStage = tt::CBIndex::c_2;
constexpr uint32_t kFsCbHidden = tt::CBIndex::c_3;
constexpr uint32_t kFsCbPre = tt::CBIndex::c_4;
// ROW_MAJOR streams: the collapsed row as h x 32 tiles, before the writer spreads it into row 0
// of the 32x32 output tiles.
constexpr uint32_t kFsCbCollapsedRow = tt::CBIndex::c_5;
constexpr uint32_t kFsCbCollapsedOut = tt::CBIndex::c_6;
constexpr uint32_t kFsCbPostOut = tt::CBIndex::c_7;
constexpr uint32_t kFsCbCombOut = tt::CBIndex::c_8;
constexpr uint32_t kFsCbPreOut = tt::CBIndex::c_9;

constexpr char kFsReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "reader_fused_single_user.cpp";
constexpr char kFsComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/compute/"
    "compute_fused_single_user.cpp";
constexpr char kFsWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "writer_fused_single_user.cpp";

CoreRangeSet single_core_set(const CoreCoord& core) { return CoreRangeSet({CoreRange(core, core)}); }

CoreRangeSet contiguous_cores(uint32_t first, uint32_t count) {
    return CoreRangeSet({CoreRange(CoreCoord(first, 0), CoreCoord(first + count - 1, 0))});
}

// Height of the h x 32 tiles a ROW_MAJOR stream row is read as: the largest h <= 16 (a tile's
// first two faces) with a whole number of tiles per row.
uint32_t row_tile_height(uint32_t row_tiles) {
    if (const char* forced = std::getenv("TT_HC_SINGLE_USER_ROW_TILE_H")) {
        const uint32_t h = static_cast<uint32_t>(std::atoi(forced));
        TT_FATAL(h >= 1 && h <= 16 && row_tiles % h == 0, "bad TT_HC_SINGLE_USER_ROW_TILE_H={}", h);
        return h;
    }
    for (uint32_t h = 16; h > 1; h /= 2) {
        if (row_tiles % h == 0) {
            return h;
        }
    }
    return 1;
}

}  // namespace fused_single_user_detail

using namespace fused_single_user_detail;

FusedSingleUserSharedVariables build_single_user_hc(
    tt::tt_metal::Program& program,
    const FusedSingleUserParams& operation_attributes,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value,
    const FusedWSource& fused_w_source) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& comb_bias = tensor_args.comb_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;
    auto& post_out = tensor_return_value[0];
    auto& comb_out = tensor_return_value[1];
    auto& collapsed_out = tensor_return_value[2];
    const bool use_pre_mix = tensor_args.pre_mix.has_value();
    const Buffer* pre_mix_buffer = use_pre_mix ? tensor_args.pre_mix->buffer() : nullptr;
    const uint32_t pre_mix_addr = use_pre_mix ? pre_mix_buffer->address() : 0u;
    const Buffer* pre_out_buffer = use_pre_mix ? tensor_return_value[3].buffer() : nullptr;
    const uint32_t pre_out_addr = use_pre_mix ? pre_out_buffer->address() : 0u;

    const CoreRangeSet collapse_cores = contiguous_cores(0, kCollapseCoreCount);
    const CoreCoord post_core{8, 0};
    const CoreCoord comb_core{9, 0};
    const CoreRangeSet post_cores = single_core_set(post_core);
    const CoreRangeSet comb_cores = single_core_set(comb_core);
    const CoreRangeSet all_cores = collapse_cores.merge(post_cores).merge(comb_cores);
    // pre is computed on the collapse cores, or -- given pre_mix, which the collapse cores take
    // into kFsCbPre instead -- on the post core, which returns it.
    const CoreRangeSet params_cores = use_pre_mix ? post_cores.merge(comb_cores) : all_cores;

    // fused_w, the biases and every output are BFLOAT16, like the streams.
    const auto param_format = datatype_to_dataformat_converter(hidden_streams.dtype());
    const uint32_t tile_size_bytes = tile_size(param_format);
    // The fn dot's root pushes its ROW_MAJOR [1, N] result.
    const bool fused_w_row_major = fused_w_source.pushed || tensor_args.fused_w->layout() == Layout::ROW_MAJOR;
    const uint32_t fused_w_cb = fused_w_source.pushed ? fused_w_source.cb_index : kFsCbFusedW;
    const bool hidden_row_major = hidden_streams.layout() == Layout::ROW_MAJOR;
    const uint32_t d_tiles = static_cast<uint32_t>(hidden_streams.padded_shape()[-1]) / constants::TILE_WIDTH;
    const uint32_t d_tiles_per_core = d_tiles / kCollapseCoreCount;
    const uint32_t num_streams = operation_attributes.num_streams;
    const uint32_t row_tile_h = hidden_row_major ? row_tile_height(d_tiles_per_core) : constants::TILE_HEIGHT;
    const uint32_t row_chunks = hidden_row_major ? d_tiles_per_core / row_tile_h : 0;
    const uint32_t hidden_pages = hidden_row_major ? num_streams * row_chunks : d_tiles_per_core;
    const uint32_t collapse_tiles = hidden_row_major ? row_chunks : d_tiles_per_core;

    auto* device = hidden_streams.device();
    const auto fused_w_noc = device->worker_core_from_logical_core(CoreCoord{0, 0});

    auto make_cb = [&](uint32_t index, const CoreRangeSet& cores) {
        CreateCircularBuffer(
            program,
            cores,
            CircularBufferConfig(tile_size_bytes, {{index, param_format}}).set_page_size(index, tile_size_bytes));
    };
    // A pushed fused_w lands in a CB the caller creates over these cores and the root alike.
    if (!fused_w_source.pushed) {
        make_cb(kFsCbFusedW, all_cores);
    }
    make_cb(kFsCbParams, post_cores.merge(comb_cores));
    make_cb(kFsCbStage, params_cores);
    make_cb(kFsCbPostOut, post_cores);
    make_cb(kFsCbPreOut, post_cores);
    make_cb(kFsCbCombOut, comb_cores);

    // The hidden shard and the collapsed output shard are consumed / produced in place.
    const auto hidden_format = datatype_to_dataformat_converter(hidden_streams.dtype());
    const auto collapsed_format = datatype_to_dataformat_converter(collapsed_out.dtype());
    const tt::tt_metal::Tile row_tile({row_tile_h, constants::TILE_WIDTH});
    // The row-major collapse keeps every compute tile h x 32 (see the compute kernel), its own-pre
    // parameters and pre included; pre_mix needs only row 0 of the pre tile.
    auto make_collapse_cb = [&](uint32_t index) {
        if (!hidden_row_major) {
            make_cb(index, collapse_cores);
            return;
        }
        const uint32_t page_bytes = row_tile.get_tile_size(param_format);
        CreateCircularBuffer(
            program,
            collapse_cores,
            CircularBufferConfig(page_bytes, {{index, param_format}})
                .set_page_size(index, page_bytes)
                .set_tile_dims(index, row_tile));
    };
    if (!use_pre_mix) {
        make_collapse_cb(kFsCbParams);
    }
    make_collapse_cb(kFsCbPre);
    CBHandle hidden_cb = 0;
    {
        const tt::tt_metal::Tile hidden_tile = hidden_row_major ? row_tile : hidden_streams.tensor_spec().tile();
        const uint32_t page_bytes = hidden_tile.get_tile_size(hidden_format);
        auto config = CircularBufferConfig(hidden_pages * page_bytes, {{kFsCbHidden, hidden_format}})
                          .set_page_size(kFsCbHidden, page_bytes)
                          .set_tile_dims(kFsCbHidden, hidden_tile)
                          .set_globally_allocated_address(*hidden_streams.buffer());
        hidden_cb = CreateCircularBuffer(program, collapse_cores, config);
    }
    // A ROW_MAJOR collapsed output is the collapsed row itself: its h x 32 chunks are packed straight
    // into the output shard, with no 32x32 tiles to scatter it into.
    const bool collapsed_row_major = collapsed_out.layout() == Layout::ROW_MAJOR;
    CBHandle collapsed_output_cb = 0;
    if (!collapsed_row_major) {
        auto config = CircularBufferConfig(
                          d_tiles_per_core * tile_size(collapsed_format), {{kFsCbCollapsedOut, collapsed_format}})
                          .set_page_size(kFsCbCollapsedOut, tile_size(collapsed_format))
                          .set_tile_dims(kFsCbCollapsedOut, collapsed_out.tensor_spec().tile())
                          .set_globally_allocated_address(*collapsed_out.buffer());
        collapsed_output_cb = CreateCircularBuffer(program, collapse_cores, config);
    }
    if (hidden_row_major) {
        const uint32_t page_bytes = row_tile.get_tile_size(collapsed_format);
        auto config = CircularBufferConfig(row_chunks * page_bytes, {{kFsCbCollapsedRow, collapsed_format}})
                          .set_page_size(kFsCbCollapsedRow, page_bytes)
                          .set_tile_dims(kFsCbCollapsedRow, row_tile);
        if (collapsed_row_major) {
            config.set_globally_allocated_address(*collapsed_out.buffer());
            collapsed_output_cb = CreateCircularBuffer(program, collapse_cores, config);
        } else {
            CreateCircularBuffer(program, collapse_cores, config);
        }
    }

    const uint32_t pre_scale_bits = std::bit_cast<uint32_t>(operation_attributes.pre_scale);
    const uint32_t post_scale_bits = std::bit_cast<uint32_t>(operation_attributes.post_scale);
    const uint32_t comb_scale_bits = std::bit_cast<uint32_t>(operation_attributes.comb_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);

    auto reader_compile_args = [&](uint32_t role) {
        std::vector<uint32_t> args = {
            role,
            fused_w_cb,
            kFsCbHidden,
            kFsCbParams,
            kFsCbStage,
            kFsCbPre,
            num_streams,
            fused_w_row_major ? 1u : 0u,
            use_pre_mix ? 1u : 0u,
            hidden_row_major ? 1u : 0u,
            hidden_pages,
            static_cast<uint32_t>(fused_w_noc.x),
            static_cast<uint32_t>(fused_w_noc.y),
            comb_bias.logical_shape()[-2] == 1 ? 1u : 0u,
            kFsCbCollapsedRow,
            row_chunks,
            kFsCbCollapsedOut,
            d_tiles_per_core,
            collapsed_row_major ? 1u : 0u,
            fused_w_source.pushed ? 1u : 0u,
            fused_w_source.semaphore_id,
        };
        TensorAccessorArgs(pre_bias.buffer()).append_to(args);
        TensorAccessorArgs(post_bias.buffer()).append_to(args);
        TensorAccessorArgs(comb_bias.buffer()).append_to(args);
        TensorAccessorArgs(pre_mix_buffer).append_to(args);
        return args;
    };

    auto compute_config = [&](uint32_t role) {
        return ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = {
                role,
                kFsCbHidden,
                kFsCbParams,
                kFsCbPre,
                hidden_row_major ? kFsCbCollapsedRow : kFsCbCollapsedOut,
                kFsCbPostOut,
                kFsCbCombOut,
                kFsCbPreOut,
                pre_scale_bits,
                post_scale_bits,
                comb_scale_bits,
                eps_bits,
                num_streams,
                operation_attributes.sinkhorn_iters,
                use_pre_mix ? 1u : 0u,
                hidden_row_major ? 1u : 0u,
                collapse_tiles,
            }};
    };

    auto writer_compile_args =
        [&](uint32_t role, uint32_t cb_out, uint32_t cb_out2, bool has_out2, const Buffer* out, const Buffer* out2) {
            std::vector<uint32_t> args = {role, cb_out, cb_out2, has_out2 ? 1u : 0u, d_tiles_per_core, row_chunks};
            TensorAccessorArgs(out).append_to(args);
            TensorAccessorArgs(out2).append_to(args);
            return args;
        };

    FusedSingleUserSharedVariables shared;
    shared.collapse_cores = corerange_to_cores(collapse_cores, collapse_cores.num_cores(), true);
    shared.post_core = post_core;
    shared.comb_core = comb_core;
    shared.has_collapse_writer = hidden_row_major && !collapsed_row_major;
    shared.hidden_cb = hidden_cb;
    shared.collapsed_output_cb = collapsed_output_cb;

    // Kernel groups start in creation order, ~0.2 us apart: comb, the longest chain (DRAM bias read,
    // Sinkhorn, DRAM write), goes first.
    shared.comb_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, comb_cores, ReaderDataMovementConfig(reader_compile_args(kRoleComb)));
    shared.comb_compute_kernel_id = CreateKernel(program, kFsComputeKernelPath, comb_cores, compute_config(kRoleComb));
    shared.comb_writer_kernel_id = CreateKernel(
        program,
        kFsWriterKernelPath,
        comb_cores,
        WriterDataMovementConfig(
            writer_compile_args(kRoleComb, kFsCbCombOut, kFsCbCombOut, false, comb_out.buffer(), comb_out.buffer())));

    shared.post_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, post_cores, ReaderDataMovementConfig(reader_compile_args(kRolePost)));
    shared.post_compute_kernel_id = CreateKernel(program, kFsComputeKernelPath, post_cores, compute_config(kRolePost));
    shared.post_writer_kernel_id = CreateKernel(
        program,
        kFsWriterKernelPath,
        post_cores,
        WriterDataMovementConfig(
            writer_compile_args(kRolePost, kFsCbPostOut, kFsCbPreOut, use_pre_mix, post_out.buffer(), pre_out_buffer)));

    shared.collapse_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, collapse_cores, ReaderDataMovementConfig(reader_compile_args(kRoleCollapse)));
    shared.collapse_compute_kernel_id =
        CreateKernel(program, kFsComputeKernelPath, collapse_cores, compute_config(kRoleCollapse));
    if (shared.has_collapse_writer) {
        shared.collapse_writer_kernel_id = CreateKernel(
            program,
            kFsWriterKernelPath,
            collapse_cores,
            WriterDataMovementConfig(writer_compile_args(
                kRoleCollapse,
                kFsCbCollapsedRow,
                kFsCbCollapsedOut,
                false,
                collapsed_out.buffer(),
                collapsed_out.buffer())));
    }

    const std::vector<uint32_t> reader_args = {
        tensor_args.fused_w.has_value() ? tensor_args.fused_w->buffer()->address() : 0u,
        pre_bias.buffer()->address(),
        post_bias.buffer()->address(),
        comb_bias.buffer()->address(),
        pre_mix_addr,
        collapsed_out.buffer()->address()};
    for (const auto& core : shared.collapse_cores) {
        SetRuntimeArgs(program, shared.collapse_reader_kernel_id, core, reader_args);
        if (shared.has_collapse_writer) {
            SetRuntimeArgs(program, shared.collapse_writer_kernel_id, core, {collapsed_out.buffer()->address(), 0u});
        }
    }
    SetRuntimeArgs(program, shared.post_reader_kernel_id, post_core, reader_args);
    SetRuntimeArgs(program, shared.comb_reader_kernel_id, comb_core, reader_args);
    SetRuntimeArgs(program, shared.post_writer_kernel_id, post_core, {post_out.buffer()->address(), pre_out_addr});
    SetRuntimeArgs(program, shared.comb_writer_kernel_id, comb_core, {comb_out.buffer()->address(), 0u});

    return shared;
}

void override_single_user_hc(
    tt::tt_metal::Program& program,
    const FusedSingleUserSharedVariables& shared,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    using namespace tt::tt_metal;

    const uint32_t fused_w_addr = tensor_args.fused_w.has_value() ? tensor_args.fused_w->buffer()->address() : 0u;
    const uint32_t pre_bias_addr = tensor_args.pre_bias.buffer()->address();
    const uint32_t post_bias_addr = tensor_args.post_bias.buffer()->address();
    const uint32_t comb_bias_addr = tensor_args.comb_bias.buffer()->address();
    const uint32_t pre_mix_addr = tensor_args.pre_mix.has_value() ? tensor_args.pre_mix->buffer()->address() : 0u;
    const uint32_t post_addr = tensor_return_value[0].buffer()->address();
    const uint32_t comb_addr = tensor_return_value[1].buffer()->address();
    const uint32_t collapsed_addr = tensor_return_value[2].buffer()->address();
    const uint32_t pre_out_addr = tensor_args.pre_mix.has_value() ? tensor_return_value[3].buffer()->address() : 0u;
    UpdateDynamicCircularBufferAddress(program, shared.hidden_cb, *tensor_args.hidden_streams.buffer());
    UpdateDynamicCircularBufferAddress(program, shared.collapsed_output_cb, *tensor_return_value[2].buffer());

    auto set_reader_args = [&](auto& args) {
        args[0] = fused_w_addr;
        args[1] = pre_bias_addr;
        args[2] = post_bias_addr;
        args[3] = comb_bias_addr;
        args[4] = pre_mix_addr;
        args[5] = collapsed_addr;
    };
    auto& collapse_reader_args = GetRuntimeArgs(program, shared.collapse_reader_kernel_id);
    for (const auto& core : shared.collapse_cores) {
        set_reader_args(collapse_reader_args[core.x][core.y]);
    }
    if (shared.has_collapse_writer) {
        auto& collapse_writer_args = GetRuntimeArgs(program, shared.collapse_writer_kernel_id);
        for (const auto& core : shared.collapse_cores) {
            collapse_writer_args[core.x][core.y][0] = collapsed_addr;
        }
    }
    set_reader_args(GetRuntimeArgs(program, shared.post_reader_kernel_id)[shared.post_core.x][shared.post_core.y]);
    set_reader_args(GetRuntimeArgs(program, shared.comb_reader_kernel_id)[shared.comb_core.x][shared.comb_core.y]);

    auto& post_writer_args =
        GetRuntimeArgs(program, shared.post_writer_kernel_id)[shared.post_core.x][shared.post_core.y];
    post_writer_args[0] = post_addr;
    post_writer_args[1] = pre_out_addr;
    GetRuntimeArgs(program, shared.comb_writer_kernel_id)[shared.comb_core.x][shared.comb_core.y][0] = comb_addr;
}

FusedSingleUserProgramFactory::cached_program_t FusedSingleUserProgramFactory::create(
    const FusedSingleUserParams& operation_attributes,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto shared = build_single_user_hc(program, operation_attributes, tensor_args, tensor_return_value, FusedWSource{});
    return cached_program_t{std::move(program), std::move(shared)};
}

void FusedSingleUserProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedSingleUserParams& /*operation_attributes*/,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    override_single_user_hc(cached_program.program, cached_program.shared_variables, tensor_args, tensor_return_value);
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
