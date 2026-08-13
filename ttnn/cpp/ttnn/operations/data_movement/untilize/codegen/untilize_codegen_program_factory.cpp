// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/untilize/codegen/untilize_codegen_program_factory.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// SEQ_IDENTITY, see codegen/kernels/sequencers.h.
constexpr uint32_t kSeqIdentity = 0;

// ops/untilize/builder.py::_compute_block_ct_dim -- max pack_untilize block-ct-dim
// (4 for 32-bit DEST accumulation dtypes, 8 otherwise), then the largest divisor
// of Wt not exceeding it.
uint32_t compute_block_ct_dim(uint32_t Wt, bool fp32) {
    uint32_t max_bct = fp32 ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (Wt % bct == 0) {
            return bct;
        }
    }
    return 1;
}

// ops/untilize/spec.py::_choose_2d_ncol.
uint32_t choose_2d_ncol(uint32_t total_tile_rows, uint32_t Wt, uint32_t valid_cores) {
    if (total_tile_rows >= valid_cores || Wt < 2) {
        return 1;
    }
    uint32_t max_ncol = std::min(valid_cores / total_tile_rows, Wt);
    uint32_t best = 1;
    for (uint32_t d = 2; d <= max_ncol; ++d) {
        if (Wt % d == 0) {
            best = d;
        }
    }
    return best;
}

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

// Mirrors ops/untilize/builder.py::_split_cores (device split_cores_rect path):
// a rectangular CoreRangeSet over the full compute grid, split by tile-rows
// (or by units, for the 2D/column-parallel branches). row_wise=false, matching
// the generator's ttnn.split_work_to_cores / corerange_to_cores default.
CoreSplit split_work(IDevice* device, uint32_t total_work) {
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return CoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t work_for_core(const CoreSplit& split, const CoreCoord& core) {
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

}  // namespace

// Mirrors common/codegen_common/factory/cb_policy.py::plan_cb_depths' 4-tier
// selection, budget defaulting to the same USABLE_L1 = 1_400_000 constant
// (builder_utils.USABLE_L1 / cb_policy._DEFAULT_USABLE_L1) both this factory
// and supported_by_codegen()'s CB-fit gate must agree on.
namespace {

struct CbPlan {
    bool ok = false;
    uint32_t cb_in_depth = 0;
    uint32_t cb_out_depth = 0;
    uint32_t read_batch = 0;
};

CbPlan plan_cb_depths(uint32_t pages_per_unit, uint32_t page_size, uint32_t block_units, uint32_t budget) {
    const uint32_t p = pages_per_unit;
    const uint64_t double_both = (2ull * p + 2ull * p) * page_size;
    const uint64_t double_in = (2ull * p + p) * page_size;
    const uint64_t single_both = (1ull * p + p) * page_size;
    if (double_both <= budget) {
        return CbPlan{.ok = true, .cb_in_depth = 2 * p, .cb_out_depth = 2 * p, .read_batch = p};
    }
    if (double_in <= budget) {
        return CbPlan{.ok = true, .cb_in_depth = 2 * p, .cb_out_depth = p, .read_batch = p};
    }
    if (single_both <= budget) {
        return CbPlan{.ok = true, .cb_in_depth = p, .cb_out_depth = p, .read_batch = block_units};
    }
    return CbPlan{.ok = false};
}
}  // namespace

bool untilize_cb_plan_fits(const UntilizeDispatchPlan& plan, uint32_t budget_bytes) {
    const uint32_t block_units = compute_block_ct_dim(plan.pages_per_unit, plan.fp32_dest_acc);
    return plan_cb_depths(plan.pages_per_unit, plan.tile_size_bytes, block_units, budget_bytes).ok;
}

UntilizeDispatchPlan plan_untilize_dispatch(const Tensor& input, bool with_unpadding) {
    UntilizeDispatchPlan plan;
    IDevice* device = input.device();

    const auto& logical_shape = input.logical_shape();
    const auto& padded_shape = input.padded_shape();
    const bool fp32 = (input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::INT32 ||
                        input.dtype() == DataType::UINT32);
    tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataType out_dtype = input.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input.dtype();
    tt::DataFormat out_df = datatype_to_dataformat_converter(out_dtype);
    const uint32_t tile_size_bytes = std::max(tt::tile_size(in_df), tt::tile_size(out_df));

    if (with_unpadding) {
        // ops/untilize/spec.py::build_untilize_with_unpadding host section
        // (ceil-divided physical tile geometry over the padded shape).
        const uint32_t H_padded = padded_shape[-2];
        const uint32_t W_padded = padded_shape[-1];
        const uint32_t Wt = (W_padded + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
        const uint32_t Ht = (H_padded + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        uint32_t NC = 1;
        for (int i = 0; i + 2 < static_cast<int>(padded_shape.rank()); ++i) {
            NC *= padded_shape[i];
        }
        plan.kind = UntilizeDispatchKind::WithUnpadding;
        plan.total_tile_rows = NC * Ht;
        plan.Wt = Wt;
        plan.pages_per_unit = Wt;
        plan.fp32_dest_acc = fp32;
        plan.tile_size_bytes = tile_size_bytes;
        return plan;
    }

    const uint32_t W = logical_shape[-1];
    const uint32_t H = logical_shape[-2];
    uint32_t NC = 1;
    for (int i = 0; i + 2 < static_cast<int>(logical_shape.rank()); ++i) {
        NC *= logical_shape[i];
    }
    const uint32_t Wt = W / tt::constants::TILE_WIDTH;
    const uint32_t Ht = H / tt::constants::TILE_HEIGHT;
    const uint32_t total_tile_rows = NC * Ht;

    plan.fp32_dest_acc = fp32;
    plan.tile_size_bytes = tile_size_bytes;
    plan.Wt = Wt;
    plan.total_tile_rows = total_tile_rows;

    // Column-parallel: single tile-row, many tile-columns.
    if (total_tile_rows == 1 && Wt > 1) {
        plan.kind = UntilizeDispatchKind::ColumnParallel;
        plan.pages_per_unit = Wt;  // conservative: all Wt tiles could land on one core group
        return plan;
    }

    // 2D (tile-row x column-block) split -- every dtype in scope (bf16/bf8_b).
    if (Wt > 1) {
        auto grid_size = device->compute_with_storage_grid_size();
        const uint32_t core_budget = grid_size.x * grid_size.y;
        const uint32_t ncol = choose_2d_ncol(total_tile_rows, Wt, core_budget);
        if (ncol >= 2) {
            plan.kind = UntilizeDispatchKind::TwoDColumn;
            plan.ncol = ncol;
            plan.pages_per_unit = Wt / ncol;
            return plan;
        }
    }

    // Per-tile-row split (single or cliff): whether it's a cliff only affects
    // kernel count, not CB sizing (both compute groups share the same Wt).
    auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t core_budget = grid_size.x * grid_size.y;
    plan.kind = (total_tile_rows > core_budget) ? UntilizeDispatchKind::Cliff : UntilizeDispatchKind::Single;
    plan.pages_per_unit = Wt;
    return plan;
}

namespace {

// The trailing pair of `writer_untilize_interleaved.cpp`'s compile-time argument
// contract, which the kernel reads as DST_PAGES_PER_ROW and
// DST_LOGICAL_PAGE_SIZE. spec.py appends
// `shard_split_writer_ct_args(out_t)` to that writer's args at both of its call
// sites (build_untilize_tile and build_untilize_with_unpadding) and to neither
// column-parallel writer, which reads no such pair.
//
// `(0, 0)` is not a placeholder: it is the value
// common/codegen_common/rm_shard_pages.py's NO_SHARD_SPLIT carries for any
// destination that is not ROW_MAJOR width- or block-sharded, and the kernel's
// `if constexpr (PAGES_PER_ROW <= 1)` collapses to the single unsplit
// `noc_async_write` it would have emitted without the header. A sharded
// destination is served by build_untilize_sharded / build_untilize_i2s, which
// are separate builder entry points this port does not transliterate, so it is
// refused in ttnn::untilize's routing rather than approximated here. The
// TT_FATAL is unreachable while that gate holds and is what makes widening the
// scope fail loudly instead of scattering the output tensor.
void append_shard_split_ct_args(std::vector<uint32_t>& ct_args, const Tensor& output) {
    TT_FATAL(
        !output.memory_config().is_sharded(),
        "UntilizeCodegen: a sharded destination belongs to build_untilize_sharded, which this port does not "
        "implement; ttnn::untilize must route it to native");
    ct_args.push_back(0);
    ct_args.push_back(0);
}

}  // namespace

ProgramDescriptor UntilizeCodegenProgramFactory::create_descriptor(
    const UntilizeCodegenParams& /*operation_attributes*/,
    const UntilizeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "UntilizeCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "UntilizeCodegen output must be allocated on device!");

    IDevice* device = input.device();
    const DataType in_dtype = input.dtype();
    const DataType out_dtype = tensor_return_value.dtype();
    tt::DataFormat in_df = datatype_to_dataformat_converter(in_dtype);
    tt::DataFormat out_df = datatype_to_dataformat_converter(out_dtype);
    const uint32_t elem_size_bytes = tt::tile_size(out_df) / (tt::constants::TILE_HW);
    const uint32_t in_tile_size = tt::tile_size(in_df);
    const uint32_t out_tile_size = tt::tile_size(out_df);
    const uint32_t tile_size_bytes = std::max(in_tile_size, out_tile_size);

    const bool with_unpadding = input.logical_shape() != input.padded_shape();
    UntilizeDispatchPlan plan = plan_untilize_dispatch(input, with_unpadding);

    const uint32_t max_bct = plan.fp32_dest_acc ? 4 : 8;
    ComputeConfigDescriptor compute_config{
        .fp32_dest_acc_en = plan.fp32_dest_acc,
    };
    if (in_dtype == DataType::FLOAT32) {
        // _utd_modes: FLOAT32 input CB must unpack straight to 32-bit DEST or
        // pack_untilize silently rounds through bf16 (see spec.py::_utd_modes).
        compute_config.unpack_to_dest_mode.assign(kUntilizeCbInId + 1, UnpackToDestMode::Default);
        compute_config.unpack_to_dest_mode[kUntilizeCbInId] = UnpackToDestMode::UnpackToDestFp32;
    }

    ProgramDescriptor desc;

    auto make_cb = [&](uint32_t cb_id, tt::DataFormat df, uint32_t page_size, uint32_t depth, const CoreRangeSet& cr) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = depth * page_size,
            .core_ranges = cr,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_id,
                .data_format = df,
                .page_size = page_size,
            }}},
        });
    };

    if (plan.kind == UntilizeDispatchKind::WithUnpadding) {
        // ops/untilize/spec.py::build_untilize_with_unpadding.
        const uint32_t Wt = plan.Wt;
        const uint32_t total_tile_rows = plan.total_tile_rows;
        const uint32_t block_ct_dim = compute_block_ct_dim(Wt, plan.fp32_dest_acc);
        CbPlan cb_plan = plan_cb_depths(Wt, tile_size_bytes, block_ct_dim, kUntilizeUsableL1);
        TT_FATAL(cb_plan.ok, "UntilizeCodegen: CB plan does not fit in L1 for this input");

        const auto& padded_shape = input.padded_shape();
        const auto& logical_shape = input.logical_shape();
        const uint32_t H_unpadded_per_batch = logical_shape[-2];
        const uint32_t W_unpadded = logical_shape[-1];
        const uint32_t padded_batch_h = plan.total_tile_rows == 0
                                             ? 0
                                             : (padded_shape[-2] + tt::constants::TILE_HEIGHT - 1) /
                                                   tt::constants::TILE_HEIGHT * tt::constants::TILE_HEIGHT;
        const uint32_t unpadded_row_bytes = W_unpadded * elem_size_bytes;
        const uint32_t padded_row_bytes = Wt * tt::constants::TILE_WIDTH * elem_size_bytes;

        CoreSplit split = split_work(device, total_tile_rows);

        make_cb(kUntilizeCbInId, in_df, in_tile_size, cb_plan.cb_in_depth, split.all_cores);
        make_cb(kUntilizeCbOutId, in_df, out_tile_size, cb_plan.cb_out_depth, split.all_cores);

        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/reader_tile_interleaved_unified.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqIdentity},
            {"cb_id", kUntilizeCbInId},
            {"batch", cb_plan.read_batch},
            {"src_page_pitch", 0},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {kUntilizeCbOutId, unpadded_row_bytes};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_ct_args.push_back(Wt);
        append_shard_split_ct_args(writer_ct_args, output);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/writer_untilize_interleaved.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        KernelDescriptor compute_desc;
        compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/compute_untilize.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = split.all_cores;
        compute_desc.compile_time_args = {0, Wt, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
        compute_desc.config = compute_config;

        // Stateful writer accumulator: out_page_offset accumulates in core
        // iteration order, matching ops/untilize/spec.py::_writer_rt / _count_valid_sticks.
        uint32_t out_page_offset = 0;
        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            reader_desc.emplace_runtime_args(core, {src_buffer, n * Wt, start * Wt});
            const uint32_t start_stick = start * tt::constants::TILE_HEIGHT;
            const uint32_t n_phys = n * tt::constants::TILE_HEIGHT;
            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,
                 n,
                 start_stick,
                 padded_row_bytes,
                 H_unpadded_per_batch,
                 padded_batch_h,
                 out_page_offset});
            // _count_valid_sticks: count sticks with (pos % batch_h) < H_unpadded.
            if (padded_batch_h != 0) {
                const uint32_t full_batches = n_phys / padded_batch_h;
                const uint32_t remainder = n_phys % padded_batch_h;
                const uint32_t pos0 = start_stick % padded_batch_h;
                uint32_t valid = full_batches * H_unpadded_per_batch;
                for (uint32_t i = 0; i < remainder; ++i) {
                    if ((pos0 + i) % padded_batch_h < H_unpadded_per_batch) {
                        valid++;
                    }
                }
                out_page_offset += valid;
            } else {
                out_page_offset += n_phys;
            }
            start += n;
        }
        // per-core-group tile-row counts (tpc1/tpc2): split into two compute
        // kernels only when both groups are non-empty (cliff), matching
        // ops/untilize/spec.py::_build_untilize_tile_cliff.
        std::vector<KernelDescriptor> compute_kds;
        if (split.work_per_core_1 > 0 && !split.core_group_1.empty()) {
            KernelDescriptor cg1 = compute_desc;
            cg1.core_ranges = split.core_group_1;
            cg1.compile_time_args = {split.work_per_core_1, Wt, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
            compute_kds.push_back(std::move(cg1));
        }
        if (split.work_per_core_2 > 0 && !split.core_group_2.empty()) {
            KernelDescriptor cg2 = compute_desc;
            cg2.core_ranges = split.core_group_2;
            cg2.compile_time_args = {split.work_per_core_2, Wt, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
            compute_kds.push_back(std::move(cg2));
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        for (auto& kd : compute_kds) {
            desc.kernels.push_back(std::move(kd));
        }
        return desc;
    }

    if (plan.kind == UntilizeDispatchKind::ColumnParallel) {
        // ops/untilize/spec.py::_build_untilize_column_parallel.
        const uint32_t Wt = plan.Wt;
        CoreSplit split = split_work(device, Wt);
        const uint32_t max_tpc = std::max(split.work_per_core_1, split.work_per_core_2);
        const uint32_t block_ct_dim = compute_block_ct_dim(max_tpc, plan.fp32_dest_acc);
        CbPlan cb_plan = plan_cb_depths(max_tpc, tile_size_bytes, block_ct_dim, kUntilizeUsableL1);
        TT_FATAL(cb_plan.ok, "UntilizeCodegen: CB plan does not fit in L1 for this input (column-parallel)");

        const uint32_t W = input.logical_shape()[-1];
        const uint32_t full_stick_size = W * elem_size_bytes;

        make_cb(kUntilizeCbInId, in_df, in_tile_size, cb_plan.cb_in_depth, split.all_cores);
        make_cb(kUntilizeCbOutId, in_df, out_tile_size, cb_plan.cb_out_depth, split.all_cores);

        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/reader_tile_interleaved_unified.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqIdentity},
            {"cb_id", kUntilizeCbInId},
            {"batch", cb_plan.read_batch},
            {"src_page_pitch", 0},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {kUntilizeCbOutId, full_stick_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/writer_untilize_col_parallel.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        std::vector<KernelDescriptor> compute_kds;
        if (split.work_per_core_1 > 0 && !split.core_group_1.empty()) {
            KernelDescriptor cg1 = KernelDescriptor{};
            cg1.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/compute_untilize.cpp";
            cg1.source_type = KernelDescriptor::SourceType::FILE_PATH;
            cg1.core_ranges = split.core_group_1;
            cg1.compile_time_args = {1, split.work_per_core_1, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
            cg1.config = compute_config;
            compute_kds.push_back(std::move(cg1));
        }
        if (split.work_per_core_2 > 0 && !split.core_group_2.empty()) {
            KernelDescriptor cg2 = KernelDescriptor{};
            cg2.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/compute_untilize.cpp";
            cg2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            cg2.core_ranges = split.core_group_2;
            cg2.compile_time_args = {1, split.work_per_core_2, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
            cg2.config = compute_config;
            compute_kds.push_back(std::move(cg2));
        }

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            reader_desc.emplace_runtime_args(core, {src_buffer, n, start});
            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,
                 tt::constants::TILE_HEIGHT,
                 start * tt::constants::TILE_WIDTH * elem_size_bytes,
                 n * tt::constants::TILE_WIDTH * elem_size_bytes,
                 n,
                 0u});
            start += n;
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        for (auto& kd : compute_kds) {
            desc.kernels.push_back(std::move(kd));
        }
        return desc;
    }

    if (plan.kind == UntilizeDispatchKind::TwoDColumn) {
        // ops/untilize/spec.py::_build_untilize_2d_column.
        const uint32_t Wt = plan.Wt;
        const uint32_t total_tile_rows = plan.total_tile_rows;
        const uint32_t ncol = plan.ncol;
        const uint32_t tpc = Wt / ncol;
        const uint32_t num_units = total_tile_rows * ncol;

        auto grid_size = device->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, wpc1, wpc2] =
            tt::tt_metal::split_work_to_cores(grid_size, num_units, /*row_wise=*/true);
        // Every core owns exactly one (tile-row, column-block) unit; iteration
        // order must be row_wise=true so index i maps to
        // (tile_row=i/ncol, col_block=i%ncol), matching spec.py's
        // ttnn.corerange_to_cores(core_grid, row_wise=True) enumeration.
        std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);

        const uint32_t block_ct_dim = compute_block_ct_dim(tpc, plan.fp32_dest_acc);
        CbPlan cb_plan = plan_cb_depths(tpc, tile_size_bytes, block_ct_dim, kUntilizeUsableL1);
        TT_FATAL(cb_plan.ok, "UntilizeCodegen: CB plan does not fit in L1 for this input (2D column)");

        const uint32_t W = input.logical_shape()[-1];
        const uint32_t full_stick_size = W * elem_size_bytes;
        const uint32_t col_chunk_bytes = tpc * tt::constants::TILE_WIDTH * elem_size_bytes;

        make_cb(kUntilizeCbInId, in_df, in_tile_size, cb_plan.cb_in_depth, all_cores);
        make_cb(kUntilizeCbOutId, in_df, out_tile_size, cb_plan.cb_out_depth, all_cores);

        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/reader_tile_interleaved_unified.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqIdentity},
            {"cb_id", kUntilizeCbInId},
            {"batch", cb_plan.read_batch},
            {"src_page_pitch", 0},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {kUntilizeCbOutId, full_stick_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/writer_untilize_col_parallel.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        KernelDescriptor compute_desc;
        compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/compute_untilize.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = {1, tpc, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
        compute_desc.config = compute_config;

        for (uint32_t i = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores[i];
            const uint32_t tile_row = i / ncol;
            const uint32_t col_block = i % ncol;
            const uint32_t start_tile = tile_row * Wt + col_block * tpc;
            reader_desc.emplace_runtime_args(core, {src_buffer, tpc, start_tile});
            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,
                 tt::constants::TILE_HEIGHT,
                 col_block * col_chunk_bytes,
                 col_chunk_bytes,
                 tpc,
                 tile_row * tt::constants::TILE_HEIGHT});
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        desc.kernels.push_back(std::move(compute_desc));
        return desc;
    }

    // Single or Cliff (per-tile-row split, no unpadding).
    {
        const uint32_t Wt = plan.Wt;
        const uint32_t total_tile_rows = plan.total_tile_rows;
        const uint32_t block_ct_dim = compute_block_ct_dim(Wt, plan.fp32_dest_acc);
        CbPlan cb_plan = plan_cb_depths(Wt, tile_size_bytes, block_ct_dim, kUntilizeUsableL1);
        TT_FATAL(cb_plan.ok, "UntilizeCodegen: CB plan does not fit in L1 for this input");

        const uint32_t W = input.logical_shape()[-1];
        const uint32_t unpadded_row_size_bytes = W * elem_size_bytes;
        const uint32_t padded_row_size_bytes = Wt * tt::constants::TILE_WIDTH * elem_size_bytes;

        CoreSplit split = split_work(device, total_tile_rows);

        make_cb(kUntilizeCbInId, in_df, in_tile_size, cb_plan.cb_in_depth, split.all_cores);
        make_cb(kUntilizeCbOutId, in_df, out_tile_size, cb_plan.cb_out_depth, split.all_cores);

        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/reader_tile_interleaved_unified.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqIdentity},
            {"cb_id", kUntilizeCbInId},
            {"batch", cb_plan.read_batch},
            {"src_page_pitch", 0},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {kUntilizeCbOutId, unpadded_row_size_bytes};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_ct_args.push_back(Wt);
        append_shard_split_ct_args(writer_ct_args, output);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/writer_untilize_interleaved.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        std::vector<KernelDescriptor> compute_kds;
        if (split.work_per_core_1 > 0 && !split.core_group_1.empty()) {
            KernelDescriptor cg1 = KernelDescriptor{};
            cg1.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/compute_untilize.cpp";
            cg1.source_type = KernelDescriptor::SourceType::FILE_PATH;
            cg1.core_ranges = split.core_group_1;
            cg1.compile_time_args = {split.work_per_core_1, Wt, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
            cg1.config = compute_config;
            compute_kds.push_back(std::move(cg1));
        }
        if (split.work_per_core_2 > 0 && !split.core_group_2.empty()) {
            KernelDescriptor cg2 = KernelDescriptor{};
            cg2.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels/compute_untilize.cpp";
            cg2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            cg2.core_ranges = split.core_group_2;
            cg2.compile_time_args = {split.work_per_core_2, Wt, kUntilizeCbInId, kUntilizeCbOutId, max_bct};
            cg2.config = compute_config;
            compute_kds.push_back(std::move(cg2));
        }

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            reader_desc.emplace_runtime_args(core, {src_buffer, n * Wt, start * Wt});
            writer_desc.emplace_runtime_args(
                core, {dst_buffer, n, start * tt::constants::TILE_HEIGHT, padded_row_size_bytes, 0u, 0u, 0u});
            start += n;
        }

        // Single-compute path (no cliff): the shared assemble reorders kernels
        // to [reader, writer, compute] like the legacy builder, so emit them
        // in that order directly here (we build our own descriptor, not via
        // the Python factory's assemble/assemble_custom split).
        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        for (auto& kd : compute_kds) {
            desc.kernels.push_back(std::move(kd));
        }
        return desc;
    }
}

}  // namespace ttnn::prim
