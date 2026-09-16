// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace fused_single_user_detail {

constexpr uint32_t kRoleCollapse = 0;
constexpr uint32_t kRolePost = 1;
constexpr uint32_t kRoleComb = 2;
constexpr uint32_t kCollapseCoreCount = 8;
constexpr uint32_t kTotalCoreCount = 10;

constexpr uint32_t kFsCbFusedW = tt::CBIndex::c_0;
constexpr uint32_t kFsCbPreW = tt::CBIndex::c_1;
constexpr uint32_t kFsCbPreBias = tt::CBIndex::c_2;
constexpr uint32_t kFsCbHidden = tt::CBIndex::c_3;
constexpr uint32_t kFsCbPre = tt::CBIndex::c_4;
constexpr uint32_t kFsCbScratch = tt::CBIndex::c_5;
constexpr uint32_t kFsCbCollapsedOut = tt::CBIndex::c_6;
constexpr uint32_t kFsCbPostW = tt::CBIndex::c_7;
constexpr uint32_t kFsCbPostBias = tt::CBIndex::c_8;
constexpr uint32_t kFsCbPostOut = tt::CBIndex::c_9;
constexpr uint32_t kFsCbPostCol = tt::CBIndex::c_10;
constexpr uint32_t kFsCbCombW = tt::CBIndex::c_11;
constexpr uint32_t kFsCbCombBias = tt::CBIndex::c_12;
constexpr uint32_t kFsCbScaler = tt::CBIndex::c_13;
constexpr uint32_t kFsCbMask = tt::CBIndex::c_14;
constexpr uint32_t kFsCbComb = tt::CBIndex::c_15;
constexpr uint32_t kFsCbReduce = tt::CBIndex::c_16;
constexpr uint32_t kFsCbEpsMask = tt::CBIndex::c_17;
constexpr uint32_t kFsCbCombOut = tt::CBIndex::c_18;
constexpr uint32_t kFsCbBiasSrc = tt::CBIndex::c_19;
constexpr uint32_t kFsCbHiddenSrc = tt::CBIndex::c_20;

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

CoreCoord pick_free_core(
    const CoreRangeSet& occupied, const CoreCoord& grid, const CoreCoord& preferred, const CoreCoord& also_skip) {
    auto is_free = [&](const CoreCoord& core) {
        return core.x < grid.x && core.y < grid.y && !occupied.contains(core) && core != also_skip;
    };
    if (is_free(preferred)) {
        return preferred;
    }
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            const CoreCoord core{x, y};
            if (is_free(core)) {
                return core;
            }
        }
    }
    TT_THROW("fused_hyperconnection: no free core for post/comb outside the hidden shard grid");
}

}  // namespace fused_single_user_detail

using namespace fused_single_user_detail;

FusedSingleUserProgramFactory::cached_program_t FusedSingleUserProgramFactory::create(
    const FusedSingleUserParams& operation_attributes,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& fused_w = tensor_args.fused_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& comb_bias = tensor_args.comb_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;
    auto& post_out = tensor_return_value[0];
    auto& comb_out = tensor_return_value[1];
    auto& collapsed_out = tensor_return_value[2];

    const CoreRangeSet collapse_cores = hidden_streams.shard_spec()->grid;
    auto* device = fused_w.device();
    const CoreCoord device_grid = device->compute_with_storage_grid_size();
    const CoreCoord post_core = pick_free_core(collapse_cores, device_grid, CoreCoord{8, 0}, CoreCoord{9, 0});
    const CoreCoord comb_core = pick_free_core(collapse_cores, device_grid, CoreCoord{9, 0}, post_core);
    const CoreRangeSet post_cores = single_core_set(post_core);
    const CoreRangeSet comb_cores = single_core_set(comb_core);
    const CoreRangeSet all_cores = collapse_cores.merge(post_cores).merge(comb_cores);

    const uint32_t tile_size_bytes = tile_size(datatype_to_dataformat_converter(fused_w.dtype()));
    const bool fused_w_rm_1x32 = fused_w.layout() == Layout::ROW_MAJOR;
    const bool hidden_rm = hidden_streams.layout() == Layout::ROW_MAJOR;
    const bool collapsed_rm = fused_w_rm_1x32 || hidden_rm;
    const tt::tt_metal::Tile tiny_tile({1, constants::TILE_WIDTH}, false);
    const tt::tt_metal::Tile hidden_4x32_tile({4, constants::TILE_WIDTH}, false);
    const uint32_t tiny_tile_bytes = tiny_tile.get_tile_size(datatype_to_dataformat_converter(fused_w.dtype()));
    const uint32_t shard_w = static_cast<uint32_t>(hidden_streams.shard_spec()->shape[1]);
    const uint32_t shard_h = static_cast<uint32_t>(hidden_streams.shard_spec()->shape[0]);
    const uint32_t d_tiles_per_core = shard_w / constants::TILE_WIDTH;
    const uint32_t num_streams = operation_attributes.num_streams;

    const auto bbox = collapse_cores.bounding_box();
    const bool legacy_1d_mcast = collapse_cores.num_cores() == kCollapseCoreCount &&
                                 bbox.start_coord == CoreCoord{0, 0} &&
                                 bbox.end_coord == CoreCoord{kCollapseCoreCount - 1, 0} &&
                                 post_core == CoreCoord{8, 0} && comb_core == CoreCoord{9, 0};
    const bool unicast_fused_w = !legacy_1d_mcast;

    uint32_t hidden_pages_per_core = shard_h;
    uint32_t row_stride_elems = shard_w;
    if (hidden_rm) {
        const uint32_t page = static_cast<uint32_t>(hidden_streams.buffer()->page_size());
        const uint32_t aligned = static_cast<uint32_t>(hidden_streams.buffer()->aligned_page_size());
        if (page >= shard_h * shard_w * 2) {
            hidden_pages_per_core = 1;
            row_stride_elems = shard_w;
        } else {
            hidden_pages_per_core = shard_h;
            row_stride_elems = aligned / 2;
        }
    }

    Program program = CreateProgram();
    const uint32_t data_ready_sem_id = CreateSemaphore(program, all_cores, 0);
    const uint32_t receiver_ready_sem_id = CreateSemaphore(program, all_cores, 0);

    const auto sender_noc = device->worker_core_from_logical_core(CoreCoord{0, 0});
    uint32_t mcast_start_x = 0;
    uint32_t mcast_start_y = 0;
    uint32_t mcast_end_x = 0;
    uint32_t mcast_end_y = 0;
    if (legacy_1d_mcast) {
        const auto mcast_a = device->worker_core_from_logical_core(CoreCoord{1, 0});
        const auto mcast_b = device->worker_core_from_logical_core(CoreCoord{kTotalCoreCount - 1, 0});
        mcast_start_x = std::min<uint32_t>(mcast_a.x, mcast_b.x);
        mcast_start_y = std::min<uint32_t>(mcast_a.y, mcast_b.y);
        mcast_end_x = std::max<uint32_t>(mcast_a.x, mcast_b.x);
        mcast_end_y = std::max<uint32_t>(mcast_a.y, mcast_b.y);
    }
    const uint32_t num_receivers = static_cast<uint32_t>(all_cores.num_cores()) - 1;

    auto make_cb = [&](uint32_t index,
                       uint32_t num_pages,
                       const CoreRangeSet& cores,
                       const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
        const uint32_t page =
            tile.has_value() ? tile->get_tile_size(datatype_to_dataformat_converter(fused_w.dtype())) : tile_size_bytes;
        CircularBufferConfig config =
            CircularBufferConfig(num_pages * page, {{index, datatype_to_dataformat_converter(fused_w.dtype())}})
                .set_page_size(index, page);
        if (tile.has_value()) {
            config.set_tile_dims(index, *tile);
        }
        CreateCircularBuffer(program, cores, config);
    };

    const std::optional<tt::tt_metal::Tile> row_tile =
        fused_w_rm_1x32 ? std::optional<tt::tt_metal::Tile>(tiny_tile) : std::nullopt;
    const std::optional<tt::tt_metal::Tile> collapse_pre_tile =
        hidden_rm ? std::optional<tt::tt_metal::Tile>(hidden_4x32_tile) : row_tile;
    const std::optional<tt::tt_metal::Tile> collapse_scratch_tile =
        hidden_rm ? std::optional<tt::tt_metal::Tile>(hidden_4x32_tile) : row_tile;

    // This CB is a private receive buffer on every participating core. Core 0
    // fills its copy from the one-core width-sharded input and broadcasts it.
    make_cb(kFsCbFusedW, 1, all_cores, row_tile);

    CBHandle hidden_cb = 0;
    CBHandle hidden_src_cb = 0;
    CBHandle collapsed_output_cb = 0;
    if (hidden_rm) {
        const uint32_t raw_page = static_cast<uint32_t>(hidden_streams.buffer()->page_size());
        auto src_config = CircularBufferConfig(
                              hidden_pages_per_core * raw_page,
                              {{kFsCbHiddenSrc, datatype_to_dataformat_converter(hidden_streams.dtype())}})
                              .set_page_size(kFsCbHiddenSrc, raw_page)
                              .set_globally_allocated_address(*hidden_streams.buffer());
        hidden_src_cb = CreateCircularBuffer(program, collapse_cores, src_config);

        const uint32_t compute_page =
            hidden_4x32_tile.get_tile_size(datatype_to_dataformat_converter(hidden_streams.dtype()));
        auto compute_config = CircularBufferConfig(
                                  d_tiles_per_core * compute_page,
                                  {{kFsCbHidden, datatype_to_dataformat_converter(hidden_streams.dtype())}})
                                  .set_page_size(kFsCbHidden, compute_page)
                                  .set_tile_dims(kFsCbHidden, hidden_4x32_tile);
        hidden_cb = CreateCircularBuffer(program, collapse_cores, compute_config);
    } else {
        auto config = CircularBufferConfig(
                          d_tiles_per_core * tile_size_bytes,
                          {{kFsCbHidden, datatype_to_dataformat_converter(hidden_streams.dtype())}})
                          .set_page_size(kFsCbHidden, tile_size_bytes)
                          .set_tile_dims(kFsCbHidden, hidden_streams.tensor_spec().tile())
                          .set_globally_allocated_address(*hidden_streams.buffer());
        hidden_cb = CreateCircularBuffer(program, collapse_cores, config);
    }
    {
        const uint32_t collapsed_page = collapsed_rm ? tiny_tile_bytes : tile_size_bytes;
        auto config = CircularBufferConfig(
                          d_tiles_per_core * collapsed_page,
                          {{kFsCbCollapsedOut, datatype_to_dataformat_converter(collapsed_out.dtype())}})
                          .set_page_size(kFsCbCollapsedOut, collapsed_page)
                          .set_globally_allocated_address(*collapsed_out.buffer());
        if (collapsed_rm) {
            config.set_tile_dims(kFsCbCollapsedOut, tiny_tile);
        } else {
            config.set_tile_dims(kFsCbCollapsedOut, collapsed_out.tensor_spec().tile());
        }
        collapsed_output_cb = CreateCircularBuffer(program, collapse_cores, config);
    }
    make_cb(kFsCbPreW, 1, collapse_cores, collapse_pre_tile);
    make_cb(kFsCbPreBias, 1, collapse_cores, collapse_pre_tile);
    make_cb(kFsCbPre, 1, collapse_cores, collapse_pre_tile);
    make_cb(kFsCbScratch, 2, collapse_cores, collapse_scratch_tile);
    make_cb(kFsCbScratch, 2, post_cores, row_tile);
    if (hidden_rm) {
        make_cb(kFsCbScaler, 1, collapse_cores, hidden_4x32_tile);
    }
    if (fused_w_rm_1x32 || hidden_rm) {
        make_cb(kFsCbBiasSrc, 1, collapse_cores);
    }
    if (fused_w_rm_1x32) {
        make_cb(kFsCbBiasSrc, 1, post_cores);
    }

    // Post branch.
    make_cb(kFsCbPostW, 1, post_cores, row_tile);
    make_cb(kFsCbPostBias, 1, post_cores, row_tile);
    make_cb(kFsCbPostOut, 1, post_cores, row_tile);
    make_cb(kFsCbPostCol, 1, post_cores);

    // Comb/Sinkhorn branch.
    make_cb(kFsCbCombW, 1, comb_cores);
    make_cb(kFsCbCombBias, 1, comb_cores);
    make_cb(kFsCbScaler, 1, comb_cores);
    make_cb(kFsCbMask, 1, comb_cores);
    make_cb(kFsCbComb, 2, comb_cores);
    make_cb(kFsCbReduce, 2, comb_cores);
    make_cb(kFsCbEpsMask, 1, comb_cores);
    make_cb(kFsCbCombOut, 1, comb_cores);

    const uint32_t scaler_bits = std::bit_cast<uint32_t>(1.0f);
    const uint32_t pre_scale_bits = std::bit_cast<uint32_t>(operation_attributes.pre_scale);
    const uint32_t post_scale_bits = std::bit_cast<uint32_t>(operation_attributes.post_scale);
    const uint32_t comb_scale_bits = std::bit_cast<uint32_t>(operation_attributes.comb_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);
    const uint32_t two_bits = std::bit_cast<uint32_t>(2.0f);

    auto reader_compile_args = [&](uint32_t role) {
        std::vector<uint32_t> args = {
            role,
            kFsCbFusedW,
            data_ready_sem_id,
            kFsCbPreW,
            kFsCbPreBias,
            kFsCbHidden,
            kFsCbPostW,
            kFsCbPostBias,
            kFsCbCombW,
            kFsCbCombBias,
            kFsCbScaler,
            kFsCbMask,
            kFsCbEpsMask,
            d_tiles_per_core,
            num_streams,
            scaler_bits,
            eps_bits,
            mcast_start_x,
            mcast_start_y,
            mcast_end_x,
            mcast_end_y,
            num_receivers,
            receiver_ready_sem_id,
            static_cast<uint32_t>(sender_noc.x),
            static_cast<uint32_t>(sender_noc.y),
            kFsCbBiasSrc,
            static_cast<uint32_t>(hidden_rm),
            shard_w,
            row_stride_elems,
            hidden_rm ? kFsCbHiddenSrc : kFsCbHidden,
            hidden_pages_per_core,
        };
        TensorAccessorArgs(fused_w.buffer()).append_to(args);
        TensorAccessorArgs(pre_bias.buffer()).append_to(args);
        TensorAccessorArgs(post_bias.buffer()).append_to(args);
        TensorAccessorArgs(hidden_streams.buffer()).append_to(args);
        TensorAccessorArgs(comb_bias.buffer()).append_to(args);
        return args;
    };

    auto compute_compile_args = [&](uint32_t role) {
        return std::vector<uint32_t>{
            role,
            kFsCbPreW,
            kFsCbPostW,
            kFsCbPreBias,
            kFsCbPostBias,
            kFsCbHidden,
            kFsCbPostOut,
            kFsCbCollapsedOut,
            kFsCbScratch,
            kFsCbPre,
            kFsCbCombW,
            kFsCbCombBias,
            kFsCbScaler,
            kFsCbMask,
            kFsCbComb,
            kFsCbReduce,
            kFsCbEpsMask,
            kFsCbCombOut,
            pre_scale_bits,
            post_scale_bits,
            eps_bits,
            two_bits,
            num_streams,
            operation_attributes.sinkhorn_iters,
            comb_scale_bits,
        };
    };

    auto writer_compile_args = [&](uint32_t role) {
        std::vector<uint32_t> args = {role, kFsCbPostOut, kFsCbPostCol, kFsCbCombOut};
        TensorAccessorArgs(post_out.buffer()).append_to(args);
        TensorAccessorArgs(comb_out.buffer()).append_to(args);
        return args;
    };

    FusedSingleUserSharedVariables shared;
    shared.collapse_cores = corerange_to_cores(collapse_cores, collapse_cores.num_cores(), true);
    shared.post_core = post_core;
    shared.comb_core = comb_core;
    shared.hidden_cb = hidden_cb;
    shared.hidden_src_cb = hidden_src_cb;
    shared.collapsed_output_cb = collapsed_output_cb;
    shared.hidden_is_rm = hidden_rm;

    std::map<std::string, std::string> rm_defines;
    if (fused_w_rm_1x32) {
        rm_defines["FUSED_W_1x32"] = "1";
    }
    if (unicast_fused_w) {
        rm_defines["UNICAST_FUSED_W"] = "1";
    }
    std::map<std::string, std::string> collapse_reader_defines = rm_defines;
    std::map<std::string, std::string> collapse_compute_defines = rm_defines;
    if (hidden_rm) {
        collapse_reader_defines["HIDDEN_4x32"] = "1";
        collapse_compute_defines["HIDDEN_4x32"] = "1";
    }

    shared.collapse_reader_kernel_id = CreateKernel(
        program,
        kFsReaderKernelPath,
        collapse_cores,
        ReaderDataMovementConfig(reader_compile_args(kRoleCollapse), collapse_reader_defines));
    shared.post_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, post_cores, ReaderDataMovementConfig(reader_compile_args(kRolePost), rm_defines));
    shared.comb_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, comb_cores, ReaderDataMovementConfig(reader_compile_args(kRoleComb), rm_defines));

    auto compute_config = [&](uint32_t role, const std::map<std::string, std::string>& defines) {
        return ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_args(role),
            .defines = defines};
    };
    shared.collapse_compute_kernel_id = CreateKernel(
        program, kFsComputeKernelPath, collapse_cores, compute_config(kRoleCollapse, collapse_compute_defines));
    shared.post_compute_kernel_id =
        CreateKernel(program, kFsComputeKernelPath, post_cores, compute_config(kRolePost, rm_defines));
    shared.comb_compute_kernel_id =
        CreateKernel(program, kFsComputeKernelPath, comb_cores, compute_config(kRoleComb, rm_defines));

    shared.post_writer_kernel_id = CreateKernel(
        program, kFsWriterKernelPath, post_cores, WriterDataMovementConfig(writer_compile_args(kRolePost), rm_defines));
    shared.comb_writer_kernel_id = CreateKernel(
        program, kFsWriterKernelPath, comb_cores, WriterDataMovementConfig(writer_compile_args(kRoleComb)));

    std::vector<uint32_t> unicast_dests;
    if (unicast_fused_w) {
        for (const auto& core : corerange_to_cores(all_cores, all_cores.num_cores(), true)) {
            if (core == CoreCoord{0, 0}) {
                continue;
            }
            const auto phys = device->worker_core_from_logical_core(core);
            unicast_dests.push_back(static_cast<uint32_t>(phys.x));
            unicast_dests.push_back(static_cast<uint32_t>(phys.y));
        }
    }

    auto make_reader_runtime_args = [&](bool is_source) {
        std::vector<uint32_t> args = {
            fused_w.buffer()->address(),
            pre_bias.buffer()->address(),
            post_bias.buffer()->address(),
            hidden_streams.buffer()->address(),
            comb_bias.buffer()->address(),
            is_source ? 1u : 0u};
        if (is_source && unicast_fused_w) {
            args.insert(args.end(), unicast_dests.begin(), unicast_dests.end());
        }
        return args;
    };

    for (const auto& core : shared.collapse_cores) {
        SetRuntimeArgs(
            program, shared.collapse_reader_kernel_id, core, make_reader_runtime_args(core == CoreCoord{0, 0}));
        SetRuntimeArgs(program, shared.collapse_compute_kernel_id, core, {d_tiles_per_core});
    }
    SetRuntimeArgs(program, shared.post_reader_kernel_id, post_core, make_reader_runtime_args(false));
    SetRuntimeArgs(program, shared.post_compute_kernel_id, post_core, {});
    SetRuntimeArgs(
        program, shared.post_writer_kernel_id, post_core, {post_out.buffer()->address(), comb_out.buffer()->address()});

    SetRuntimeArgs(program, shared.comb_reader_kernel_id, comb_core, make_reader_runtime_args(false));
    SetRuntimeArgs(program, shared.comb_compute_kernel_id, comb_core, {});
    SetRuntimeArgs(
        program, shared.comb_writer_kernel_id, comb_core, {post_out.buffer()->address(), comb_out.buffer()->address()});

    return cached_program_t{std::move(program), std::move(shared)};
}

void FusedSingleUserProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedSingleUserParams& /*operation_attributes*/,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t fused_w_addr = tensor_args.fused_w.buffer()->address();
    const uint32_t pre_bias_addr = tensor_args.pre_bias.buffer()->address();
    const uint32_t post_bias_addr = tensor_args.post_bias.buffer()->address();
    const uint32_t hidden_addr = tensor_args.hidden_streams.buffer()->address();
    const uint32_t comb_bias_addr = tensor_args.comb_bias.buffer()->address();
    const uint32_t post_addr = tensor_return_value[0].buffer()->address();
    const uint32_t comb_addr = tensor_return_value[1].buffer()->address();
    if (shared.hidden_is_rm) {
        UpdateDynamicCircularBufferAddress(program, shared.hidden_src_cb, *tensor_args.hidden_streams.buffer());
    } else {
        UpdateDynamicCircularBufferAddress(program, shared.hidden_cb, *tensor_args.hidden_streams.buffer());
    }
    UpdateDynamicCircularBufferAddress(program, shared.collapsed_output_cb, *tensor_return_value[2].buffer());

    auto& collapse_reader_args = GetRuntimeArgs(program, shared.collapse_reader_kernel_id);
    for (const auto& core : shared.collapse_cores) {
        auto& args = collapse_reader_args[core.x][core.y];
        args[0] = fused_w_addr;
        args[1] = pre_bias_addr;
        args[2] = post_bias_addr;
        args[3] = hidden_addr;
        args[4] = comb_bias_addr;
    }

    auto& post_reader_args =
        GetRuntimeArgs(program, shared.post_reader_kernel_id)[shared.post_core.x][shared.post_core.y];
    post_reader_args[0] = fused_w_addr;
    post_reader_args[1] = pre_bias_addr;
    post_reader_args[2] = post_bias_addr;
    post_reader_args[3] = hidden_addr;
    post_reader_args[4] = comb_bias_addr;

    auto& comb_reader_args =
        GetRuntimeArgs(program, shared.comb_reader_kernel_id)[shared.comb_core.x][shared.comb_core.y];
    comb_reader_args[0] = fused_w_addr;
    comb_reader_args[1] = pre_bias_addr;
    comb_reader_args[2] = post_bias_addr;
    comb_reader_args[3] = hidden_addr;
    comb_reader_args[4] = comb_bias_addr;

    auto& post_writer_args =
        GetRuntimeArgs(program, shared.post_writer_kernel_id)[shared.post_core.x][shared.post_core.y];
    post_writer_args[0] = post_addr;
    post_writer_args[1] = comb_addr;
    auto& comb_writer_args =
        GetRuntimeArgs(program, shared.comb_writer_kernel_id)[shared.comb_core.x][shared.comb_core.y];
    comb_writer_args[0] = post_addr;
    comb_writer_args[1] = comb_addr;
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
