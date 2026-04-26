// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal forked unary program factory scoped to the sharded-TILE path used by
// the gate routed_matmul's post-activation. Kernels are copies of unary_ng's
// that early-return via guard.h when
//   expert_token_counts[global_expert_idx_table[local_expert_idx]]
//       <= curr_expert_iter * expert_iter_length.

#include "routed_unary_program_factory.hpp"

#include "ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/circular_buffer_constants.h>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

using tt::tt_metal::CircularBufferConfig;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;

namespace {

// Guard scratch CB: 512 bytes (two 256-byte halves, one per table). Same size and
// role as in the matmul fork. Must not collide with c_0 / c_2.
constexpr uint32_t kGuardCbIndex = tt::CBIndex::c_11;
constexpr uint32_t kGuardCbBytes = 512;

}  // namespace

RoutedUnaryProgramFactory::cached_program_t RoutedUnaryProgramFactory::create(
    const RoutedUnaryParams& operation_attributes,
    const RoutedUnaryInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    auto* global_table_buffer = tensor_args.global_expert_idx_table.buffer();
    auto* counts_buffer = tensor_args.expert_token_counts.buffer();

    // Scope guardrails — this factory is intentionally narrow.
    TT_FATAL(input.layout() == Layout::TILE, "routed_unary: only TILE layout is supported");
    TT_FATAL(output.layout() == Layout::TILE, "routed_unary: only TILE output layout is supported");
    TT_FATAL(input.is_sharded(), "routed_unary: only sharded input is supported");
    TT_FATAL(output.is_sharded(), "routed_unary: only sharded output is supported");
    TT_FATAL(
        input.memory_config().memory_layout() == output.memory_config().memory_layout(),
        "routed_unary: input and output must share the same sharded layout");

    const auto& in_shard_opt = input.memory_config().shard_spec();
    const auto& out_shard_opt = output.memory_config().shard_spec();
    TT_FATAL(in_shard_opt.has_value(), "routed_unary: input must have a shard_spec");
    TT_FATAL(out_shard_opt.has_value(), "routed_unary: output must have a shard_spec");

    const auto& in_shard = in_shard_opt.value();
    const auto& out_shard = out_shard_opt.value();
    TT_FATAL(in_shard.grid == out_shard.grid, "routed_unary: input and output shard grids must match");
    TT_FATAL(
        in_shard.shape[0] == out_shard.shape[0] && in_shard.shape[1] == out_shard.shape[1],
        "routed_unary: input and output shard shapes must match");

    const auto in_tile_shape = input.tensor_spec().tile();
    const auto out_tile_shape = output.tensor_spec().tile();
    const uint32_t in_tile_h = in_tile_shape.get_height();
    const uint32_t in_tile_w = in_tile_shape.get_width();

    TT_FATAL(
        in_shard.shape[0] % in_tile_h == 0 && in_shard.shape[1] % in_tile_w == 0,
        "routed_unary: shard shape must be a multiple of tile shape");

    const uint32_t num_tiles_per_core = (in_shard.shape[0] / in_tile_h) * (in_shard.shape[1] / in_tile_w);
    TT_FATAL(num_tiles_per_core > 0, "routed_unary: zero tiles per core");

    const DataFormat in_data_format = datatype_to_dataformat_converter(input.dtype());
    const DataFormat out_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t in_single_tile_size = tile_size(in_data_format);
    const uint32_t out_single_tile_size = tile_size(out_data_format);

    Program program{};

    const CoreRangeSet all_cores = in_shard.grid;

    // --- CBs ---
    // cb_src (c_0): globally allocated to the input shard buffer.
    constexpr uint32_t src_cb_index = tt::CBIndex::c_0;
    auto src_cb_config =
        CircularBufferConfig(num_tiles_per_core * in_single_tile_size, {{src_cb_index, in_data_format}})
            .set_page_size(src_cb_index, in_single_tile_size)
            .set_globally_allocated_address(*src_buffer)
            .set_tile_dims(src_cb_index, in_tile_shape);
    auto cb_src = tt_metal::CreateCircularBuffer(program, all_cores, src_cb_config);

    // cb_out (c_2): globally allocated to the output shard buffer.
    constexpr uint32_t out_cb_index = tt::CBIndex::c_2;
    auto out_cb_config =
        CircularBufferConfig(num_tiles_per_core * out_single_tile_size, {{out_cb_index, out_data_format}})
            .set_page_size(out_cb_index, out_single_tile_size)
            .set_globally_allocated_address(*dst_buffer)
            .set_tile_dims(out_cb_index, out_tile_shape);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    // cb_guard (c_11): scratch for DRAM reads of the two guard tables. 512 bytes
    // split into two 256-byte halves (one per table), mirroring the matmul fork.
    auto guard_cb_config = CircularBufferConfig(kGuardCbBytes, {{kGuardCbIndex, DataFormat::Float16_b}})
                               .set_page_size(kGuardCbIndex, kGuardCbBytes);
    auto cb_guard = tt_metal::CreateCircularBuffer(program, all_cores, guard_cb_config);

    // --- Compile-time args ---
    // Dataflow kernels receive the two guard TensorAccessorArgs at the end of
    // their compile_args vector. Both tables are small DRAM ROW_MAJOR uint32
    // buffers; the kernel uses a TensorAccessor to resolve bank+offset per page.
    std::vector<uint32_t> reader_compile_time_args;
    const uint32_t reader_global_cta_off = reader_compile_time_args.size();
    tt::tt_metal::TensorAccessorArgs(*global_table_buffer).append_to(reader_compile_time_args);
    const uint32_t reader_counts_cta_off = reader_compile_time_args.size();
    tt::tt_metal::TensorAccessorArgs(*counts_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    const uint32_t writer_global_cta_off = writer_compile_time_args.size();
    tt::tt_metal::TensorAccessorArgs(*global_table_buffer).append_to(writer_compile_time_args);
    const uint32_t writer_counts_cta_off = writer_compile_time_args.size();
    tt::tt_metal::TensorAccessorArgs(*counts_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args;  // empty; compute kernel only needs named args

    // --- Defines ---
    // Wire the single unary op into SFPU_OP_CHAIN_0 via get_block_defines (the
    // same helper unary_ng's factory uses).
    TT_FATAL(
        operation_attributes.op_chain.size() == 1,
        "routed_unary: op_chain must contain exactly one op (got {})",
        operation_attributes.op_chain.size());
    auto compute_defines =
        ttnn::operations::unary_ng::get_block_defines(operation_attributes.op_chain, "0", "0", input.dtype());

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    reader_defines["ROUTED_GUARD_ENABLED"] = "1";
    writer_defines["ROUTED_GUARD_ENABLED"] = "1";
    // Compute kernel: GUARD_COMPUTE_KERNEL is #defined directly in the kernel
    // source (before `#include "../../guard.h"`), matching bmm_routed.cpp's
    // pattern — keeping it out of defines avoids a redefinition warning.
    compute_defines["ROUTED_GUARD_ENABLED"] = "1";

    // --- Kernels ---
    const auto compute_with_storage_grid = input.device()->compute_with_storage_grid_size();
    bool noc_set_by_factory = false;  // rely on default NOC
    (void)noc_set_by_factory;
    (void)compute_with_storage_grid;

    // Reader = RISCV_0 (BRISC). Publishes skip to TRISC mailboxes via guard_check_wait().
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/routed_unary/"
        "dataflow/reader_routed_unary.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines,
            .named_compile_args = {
                {"GUARD_CB_ID", kGuardCbIndex},
                {"GUARD_ARG_BASE", 3u},
                {"GUARD_GLOBAL_TABLE_CTA_OFFSET", reader_global_cta_off},
                {"GUARD_COUNTS_CTA_OFFSET", reader_counts_cta_off},
            }});

    // Writer = RISCV_1 (NCRISC). Direct DRAM reads via guard_check_brisc().
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/routed_unary/"
        "dataflow/writer_routed_unary.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines,
            .named_compile_args = {
                {"GUARD_CB_ID", kGuardCbIndex},
                {"GUARD_ARG_BASE", 3u},
                {"GUARD_GLOBAL_TABLE_CTA_OFFSET", writer_global_cta_off},
                {"GUARD_COUNTS_CTA_OFFSET", writer_counts_cta_off},
            }});

    // Compute = TRISC. Reads mailbox published by BRISC via guard_check_wait()
    // (GUARD_COMPUTE_KERNEL selects the mailbox-read implementation).
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
    (void)packer_l1_acc;
    (void)dst_full_sync_en;

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (operation_attributes.preserve_fp32_precision) {
        unpack_to_dest_mode[src_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/routed_unary/"
        "compute/eltwise_sfpu_routed.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = operation_attributes.bfp8_pack_precise,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines,
            .named_compile_args = {
                {"GUARD_CB_ID", kGuardCbIndex},
                {"GUARD_ARG_BASE", 3u},
            }});

    // --- Per-core runtime args ---
    // All cores process the same number of tiles (block-sharded with evenly
    // divisible grid). Layout:
    //   reader: [src_addr, num_pages, start_id, guard5]
    //   writer: [dst_addr, num_pages, start_id, guard5]
    //   compute: [num_tiles, scalar1, scalar2, guard5]
    // Only guard5 and the buffer addresses change per dispatch; num_pages /
    // num_tiles are fixed for a cached program (shard shape is invariant).
    const uint32_t global_table_addr = global_table_buffer->address();
    const uint32_t token_counts_addr = counts_buffer->address();
    const uint32_t local_expert_idx = operation_attributes.local_expert_idx;
    const uint32_t curr_expert_iter = operation_attributes.curr_expert_iter;
    const uint32_t expert_iter_length = operation_attributes.expert_iter_length;

    auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_major=*/true);
    for (const auto& core : cores) {
        std::vector<uint32_t> reader_args = {
            src_buffer->address(),
            num_tiles_per_core,
            /*start_id=*/0u,
            global_table_addr,
            token_counts_addr,
            local_expert_idx,
            curr_expert_iter,
            expert_iter_length,
        };
        std::vector<uint32_t> writer_args = {
            dst_buffer->address(),
            num_tiles_per_core,
            /*start_id=*/0u,
            global_table_addr,
            token_counts_addr,
            local_expert_idx,
            curr_expert_iter,
            expert_iter_length,
        };
        std::vector<uint32_t> compute_args = {
            num_tiles_per_core,
            /*scalar1=*/0u,
            /*scalar2=*/0u,
            global_table_addr,
            token_counts_addr,
            local_expert_idx,
            curr_expert_iter,
            expert_iter_length,
        };
        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_args);
        tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_args);
    }

    return {
        std::move(program),
        {reader_kernel_id, writer_kernel_id, compute_kernel_id, cb_src, cb_out, cb_guard, std::move(cores)}};
}

void RoutedUnaryProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RoutedUnaryParams& operation_attributes,
    const RoutedUnaryInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& shared = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    auto* global_table_buffer = tensor_args.global_expert_idx_table.buffer();
    auto* counts_buffer = tensor_args.expert_token_counts.buffer();

    // Update globally-allocated CB addresses so a program-cache hit uses the
    // current input/output shard buffers (which may differ between dispatches).
    UpdateDynamicCircularBufferAddress(program, shared.cb_src, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared.cb_out, *dst_buffer);

    const uint32_t global_table_addr = global_table_buffer->address();
    const uint32_t token_counts_addr = counts_buffer->address();
    const uint32_t local_expert_idx = operation_attributes.local_expert_idx;
    const uint32_t curr_expert_iter = operation_attributes.curr_expert_iter;
    const uint32_t expert_iter_length = operation_attributes.expert_iter_length;

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    auto& compute_args_by_core = GetRuntimeArgs(program, shared.compute_kernel_id);

    // Guard args live at [GUARD_ARG_BASE = 3 .. 7] in every kernel.
    auto write_guard = [&](auto& args) {
        args[3] = global_table_addr;
        args[4] = token_counts_addr;
        args[5] = local_expert_idx;
        args[6] = curr_expert_iter;
        args[7] = expert_iter_length;
    };

    for (const auto& core : shared.cores) {
        auto& r = reader_args_by_core[core.x][core.y];
        r[0] = src_buffer->address();
        write_guard(r);

        auto& w = writer_args_by_core[core.x][core.y];
        w[0] = dst_buffer->address();
        write_guard(w);

        auto& c = compute_args_by_core[core.x][core.y];
        // c[0] (num_tiles) and c[1]/c[2] (scalars) are program-cache invariants.
        write_guard(c);
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device
