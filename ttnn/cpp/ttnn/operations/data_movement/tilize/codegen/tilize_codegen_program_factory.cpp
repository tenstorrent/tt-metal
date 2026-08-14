// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

#include "tilize_codegen_device_operation.hpp"
#include "tilize_codegen_supported.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Kernels were copied verbatim from the codegen builder into codegen/kernels/.
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels";
constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kModeTilerow = 1;  // reader_stick_interleaved_unified.cpp MODE_TILEROW
constexpr uint32_t kDefaultWriteBatch = 4;
// spec.py's _L1_RESERVE = 128 * 1024 (builder.py:251 "reserve for code/stacks"); the only
// reserve the row CB planner subtracts before admitting a depth.
constexpr uint32_t kL1Reserve = 128 * 1024;

std::string kernel_path(const char* name) { return std::string(kKernelDir) + "/" + name; }

// Mirrors spec.py's _utd_modes: FLOAT32 input needs UnpackToDestFp32 on cb_in (c_0), everything
// else (bf16/uint32/int32/uint16) unpacks through the native path unmodified.
bool needs_unpack_to_dest_fp32(DataType in_dtype) { return in_dtype == DataType::FLOAT32; }

// Mirrors spec.py's largest_divisor_le / _largest_divisor_le: largest divisor of n that is <=
// limit. O(sqrt(n)), matching the Python source exactly (not an approximation).
uint32_t largest_divisor_le(uint32_t n, uint32_t limit) {
    if (n <= limit) {
        return n;
    }
    uint32_t best = 1;
    for (uint32_t d = 1; static_cast<uint64_t>(d) * d <= n; ++d) {
        if (n % d == 0) {
            uint32_t q = n / d;
            if (d <= limit) {
                best = std::max(best, d);
            }
            if (q <= limit) {
                best = std::max(best, q);
            }
        }
    }
    return best;
}

// spec.py's _tilize_required_cb_out: minimum output pages for compute plus the pipelined
// writer. write_batch<=1 needs only one compute chunk resident; a batched writer must also
// retain the previous batch, rounded up to a whole number of compute_chunk groups so compute
// never blocks reserving a partial group while the writer waits on pages only that group
// supplies.
uint32_t tilize_required_cb_out(uint32_t write_batch, uint32_t compute_chunk) {
    if (write_batch <= 1) {
        return compute_chunk;
    }
    uint32_t numerator = 2 * write_batch;
    return ((numerator + compute_chunk - 1) / compute_chunk) * compute_chunk;
}

uint32_t aligned_tile_page_size(uint32_t tile_bytes) {
    uint32_t align = tt::tt_metal::hal::get_dram_alignment();
    return ((tile_bytes + align - 1) / align) * align;
}

struct RowCbPlan {
    uint32_t num_cores;
    uint32_t chunk_wt;
    uint32_t num_col_chunks;
    uint32_t write_batch;
    uint32_t cb_in_depth;
    uint32_t cb_out_depth;
};

// Mirrors spec.py's scale_cb_depths_to_l1 (cb_policy.py:251+): scales cb_in_depth/cb_out_depth
// down together (floored, minimum 1 each) until in_depth*in_page + out_depth*out_page fits the
// reserve-adjusted budget. supported_by_codegen() has already rejected any shape whose
// single-tile floor cannot fit, so this only ever shrinks a plan that started oversized, never
// pushes a floor below 1.
void scale_cb_depths_to_l1(uint32_t& in_depth, uint32_t& out_depth, uint32_t in_page, uint32_t out_page, uint32_t l1) {
    uint32_t available = (l1 > kL1Reserve) ? (l1 - kL1Reserve) : 0;
    while (in_depth > 1 || out_depth > 1) {
        uint64_t used = static_cast<uint64_t>(in_depth) * in_page + static_cast<uint64_t>(out_depth) * out_page;
        if (used <= available) {
            break;
        }
        if (in_depth > 1) {
            --in_depth;
        }
        if (out_depth > 1) {
            --out_depth;
        }
    }
}

// Direct transliteration of spec.py's _tilize_row_cb_plan, minus the env-override knobs (no
// analog needed in the C++ port) and use_low_perf (not part of this port's scope; the row
// builder is only ever reached for the default multicore path -- use_low_perf routes to native
// via supported_execution_controls). `l1` is the arch's raw per-core L1 total (ArchConfig.
// l1_size_bytes equivalent), matching the quantity spec.py's chunk-sizing math uses; the
// code/stack reserve is applied only in the final scale_cb_depths_to_l1 step, exactly as
// _select_tilize_cb_depths does.
RowCbPlan tilize_row_cb_plan(
    uint32_t total_ht, uint32_t wt, uint32_t ts, uint32_t out_ts, uint32_t l1, const CoreCoord& grid) {
    uint32_t cb_depth;
    uint32_t chunk_wt;
    uint32_t num_col_chunks;
    if (2ull * 2 * wt * ts <= l1) {
        cb_depth = 2;
        chunk_wt = wt;
        num_col_chunks = 1;
    } else if (2ull * wt * ts <= l1) {
        cb_depth = 1;
        chunk_wt = wt;
        num_col_chunks = 1;
    } else {
        uint32_t max_chunk = l1 / (2 * ts);
        chunk_wt = largest_divisor_le(wt, max_chunk);
        cb_depth = (2ull * 2 * chunk_wt * ts <= l1) ? 2 : 1;
        num_col_chunks = wt / chunk_wt;
    }

    auto [num_cores, core_range, cg1, cg2, tpc1, tpc2] =
        tt::tt_metal::split_work_to_cores(grid, total_ht, /*row_wise=*/true);
    (void)core_range;
    (void)cg1;
    (void)cg2;
    (void)tpc1;
    (void)tpc2;

    bool minimal_work = (num_col_chunks == 1 && chunk_wt == 1 && total_ht <= num_cores);
    if (minimal_work) {
        cb_depth = 1;
    }
    bool force_single = (total_ht > num_cores || minimal_work);
    uint32_t write_batch = force_single ? 1 : kDefaultWriteBatch;

    uint32_t cb_in_depth = cb_depth * chunk_wt;
    uint32_t cb_out_depth = std::max(cb_depth * chunk_wt, tilize_required_cb_out(write_batch, chunk_wt));
    if (minimal_work) {
        cb_out_depth = 1;
    }
    uint32_t in_page = chunk_wt * ts;
    uint32_t out_page = chunk_wt * out_ts;
    scale_cb_depths_to_l1(cb_in_depth, cb_out_depth, in_page, out_page, l1);
    return RowCbPlan{num_cores, chunk_wt, num_col_chunks, write_batch, cb_in_depth, cb_out_depth};
}

}  // namespace

ProgramDescriptor TilizeCodegenProgramFactory::create_descriptor(
    const TilizeCodegenOperationAttributes& operation_attributes,
    const TilizeCodegenTensorArgs& tensor_args,
    const Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;
    (void)operation_attributes;

    IDevice* device = input.device();
    Buffer* in_buf = input.buffer();
    Buffer* out_buf = output.buffer();

    DataType in_dtype = input.dtype();
    DataType out_dtype = output.dtype();
    tt::DataFormat in_fmt = tt::tt_metal::datatype_to_dataformat_converter(in_dtype);
    tt::DataFormat out_fmt = tt::tt_metal::datatype_to_dataformat_converter(out_dtype);
    uint32_t ts = tt::tile_size(in_fmt);
    uint32_t out_ts = tt::tile_size(out_fmt);

    const auto& logical_shape = input.logical_shape();
    uint32_t rank = logical_shape.rank();
    uint32_t h = logical_shape[-2];
    uint32_t w = logical_shape[-1];
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= logical_shape[i];
    }
    uint32_t ht = (h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (w + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t total_ht = nc * ht;

    uint32_t elem_size = input.element_size();
    uint32_t stick_size_bytes = w * elem_size;
    uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t aligned_ps = ((stick_size_bytes + dram_alignment - 1) / dram_alignment) * dram_alignment;

    auto grid = device->compute_with_storage_grid_size();
    RowCbPlan plan = tilize_row_cb_plan(total_ht, wt, ts, out_ts, device->l1_size_per_core(), grid);
    uint32_t out_page = aligned_tile_page_size(out_ts);
    uint32_t elem_w_bytes = TILE_WIDTH * elem_size;

    auto [num_cores, core_range, cg1, cg2, tpc1, tpc2] =
        tt::tt_metal::split_work_to_cores(grid, total_ht, /*row_wise=*/true);
    (void)num_cores;

    ProgramDescriptor desc;

    uint32_t cb_in_page = aligned_tile_page_size(ts);
    uint32_t cb_out_page = aligned_tile_page_size(out_ts);
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan.cb_in_depth * cb_in_page,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(kCbIn), .data_format = in_fmt, .page_size = cb_in_page}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan.cb_out_depth * cb_out_page,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(kCbOut), .data_format = out_fmt, .page_size = cb_out_page}}},
    });

    // Reader: reader_stick_interleaved_unified.cpp MODE_TILEROW. Named CT args first, then the
    // TensorAccessorArgs positional block (matches spec.py's named_ct + reader_ct_positional
    // ordering exactly).
    KernelDescriptor reader;
    reader.kernel_source = kernel_path("reader_stick_interleaved_unified.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = core_range;
    reader.named_compile_time_args = {
        {"mode", kModeTilerow},
        {"cb_id", kCbIn},
        {"stick_bytes", stick_size_bytes},
        {"aligned_page_size", aligned_ps},
        {"seq_id", 0u},   // unused by MODE_TILEROW
        {"batch", 1u},    // unused by MODE_TILEROW
        {"nabatch", 4u},  // batched-read optimization
        // Benign MODE_TILEROW_PAD defaults: kernel_main() is not templated, so the
        // get_named_compile_time_arg_val(...) reads inside the `if constexpr (MODE ==
        // MODE_TILEROW_PAD)` branch are still compiled (and their names resolved) for every
        // mode that uses this shared kernel, including MODE_TILEROW. Mirrors
        // builder_utils.py's _merge_tilerow_pad_defaults, which every non-PAD stick-reader
        // builder picks up automatically.
        {"elem_size", 2u},
        {"tile_height", 32u},
        {"tile_row_shift_bits", 0u},
        {"num_pages_in_row", 1u},
        {"unpadded_X_bytes", 0u},
        {"valid_last_page_bytes", 0u},
        {"page_size", 32u},
    };
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_tilize_interleaved.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, out_page};
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.compile_time_args.push_back(plan.write_batch);
    writer.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (needs_unpack_to_dest_fp32(in_dtype)) {
        unpack_to_dest_mode[kCbIn] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute;
    compute.kernel_source = kernel_path("compute_tilize.cpp");
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = core_range;
    compute.compile_time_args = {kCbIn, kCbOut, plan.num_col_chunks, plan.chunk_wt};
    compute.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = (in_dtype == DataType::FLOAT32 || in_dtype == DataType::INT32 ||
                              in_dtype == DataType::UINT32 || out_dtype == DataType::FLOAT32 ||
                              out_dtype == DataType::INT32 || out_dtype == DataType::UINT32),
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    uint32_t assigned = 0;
    auto emit_group = [&](const CoreRangeSet& group, uint32_t wpc) {
        if (group.empty()) {
            return;
        }
        for (const auto& core : corerange_to_cores(group, std::nullopt, true)) {
            uint32_t n = std::min(wpc, total_ht - assigned);
            reader.emplace_runtime_args(
                core,
                {in_buf,
                 n,
                 assigned * TILE_HEIGHT,
                 TILE_HEIGHT,
                 plan.chunk_wt,
                 plan.num_col_chunks,
                 elem_w_bytes});
            writer.emplace_runtime_args(core, {out_buf, n * wt, assigned * wt});
            compute.emplace_runtime_args(core, {n});
            assigned += n;
        }
    };
    emit_group(cg1, tpc1);
    emit_group(cg2, tpc2);

    // Kernel order [reader, writer, compute] mirrors spec.py's build_tilize_row (via
    // ProgramFactory.assemble, which reorders to legacy [reader, writer, compute] order).
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
