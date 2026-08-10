// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

std::pair<uint32_t, uint32_t> pad_rm_batches_for_l1(
    uint32_t input_page, uint32_t output_page, uint32_t budget, uint32_t read_batch, uint32_t write_batch) {
    auto footprint = [&]() -> uint64_t {
        const uint64_t depth = static_cast<uint64_t>(std::max(read_batch, write_batch)) * 2;
        return depth * output_page + output_page + input_page;
    };
    const uint64_t safe_budget = budget > kPadL1SafetyMargin ? budget - kPadL1SafetyMargin : 0;
    while (footprint() > safe_budget && (read_batch > 1 || write_batch > 1)) {
        if (read_batch >= write_batch && read_batch > 1) {
            read_batch = std::max(1u, read_batch / 2);
        } else if (write_batch > 1) {
            write_batch = std::max(1u, write_batch / 2);
        }
    }
    return {read_batch, write_batch};
}

uint32_t pack_pad_value(DataType dtype, float value) {
    if (value == 0.0f) {
        return 0;
    }
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        // Match Python's int(value) & 0xFFFFFFFF: truncate toward zero, then
        // reinterpret the low 32 bits.
        int64_t iv = static_cast<int64_t>(value);
        return static_cast<uint32_t>(iv & 0xFFFFFFFFLL);
    }
    if (dtype == DataType::UINT16) {
        uint32_t v;
        if (std::isnan(value) || value <= 0.0f) {
            v = 0;
        } else if (std::isinf(value) || value >= 65535.0f) {
            v = 0xFFFF;
        } else {
            v = static_cast<uint32_t>(std::floor(value + 0.5f));
        }
        return (v << 16) | v;
    }
    float fv = value;
    if (std::isfinite(fv) && std::fabs(fv) > 3.4028234663852886e38f) {
        fv = std::copysign(std::numeric_limits<float>::infinity(), fv);
    }
    uint32_t bits;
    std::memcpy(&bits, &fv, sizeof(bits));
    if (dtype == DataType::FLOAT32) {
        return bits;
    }
    // BFLOAT16: RNE round of the float32 bit pattern, canonicalizing NaN to
    // 0x7fc0 first (matches tt-metal's bfloat16 constructor), then duplicate
    // the 16-bit result into both halves of the uint32 word.
    uint32_t bf16;
    if (std::isnan(fv)) {
        bf16 = 0x7FC0;
    } else {
        bf16 = ((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) & 0xFFFFu;
    }
    return (bf16 << 16) | bf16;
}

namespace {

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

// ops/pad/builder.py::_auto_max_cores never actually clamps (it returns
// total_items itself for any total_items >= 1); the real device-grid clamp
// happens inside ttnn.split_work_to_cores / this C++ split_work_to_cores, so
// we call it directly against total_work with no separate "tuned cores" step,
// matching the repeat port's pattern.
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

uint32_t round_up_u32(uint32_t value, uint32_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

// SEQ_PAD, see codegen/kernels/sequencers.h.
constexpr uint32_t kSeqPad = 5;

}  // namespace

ProgramDescriptor PadCodegenProgramFactory::create_descriptor(
    const PadCodegenParams& operation_attributes,
    const PadCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "PadCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "PadCodegen output must be allocated on device!");

    IDevice* device = input.device();
    const auto& allocator = device->allocator();
    const uint32_t dram_alignment = allocator->get_alignment(BufferType::DRAM);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    ProgramDescriptor desc;

    if (input.layout() == ttnn::TILE_LAYOUT) {
        // TILE-interleaved back(+front on N/C)-padding via the shared
        // reader_tile_interleaved_unified.cpp sequencer reader (seq_id=SEQ_PAD)
        // + writer_pad_tiled_interleaved.cpp. Mirrors ops/pad/spec.py::build_pad_tiled.
        const auto& in_shape = input.padded_shape();
        const uint32_t N = in_shape[0];
        const uint32_t C = in_shape[1];
        const uint32_t Ht_in = in_shape[2] / tt::constants::TILE_HEIGHT;
        const uint32_t Wt_in = in_shape[3] / tt::constants::TILE_WIDTH;
        const uint32_t N_out = operation_attributes.N_out;
        const uint32_t C_out = operation_attributes.C_out;
        const uint32_t Ht_out = operation_attributes.H_out;
        const uint32_t Wt_out = operation_attributes.W_out;
        const uint32_t front_wt = operation_attributes.front_w;
        const uint32_t front_ht = operation_attributes.front_h;
        const uint32_t front_c = operation_attributes.front_c;
        const uint32_t front_n = operation_attributes.front_n;

        uint32_t tile_bytes = tt::tile_size(cb_data_format);
        tile_bytes = round_up_u32(tile_bytes, dram_alignment);
        const uint32_t pad_buf_size = round_up_u32(tile_bytes, dram_alignment);
        const uint32_t total_out_tiles = N_out * C_out * Ht_out * Wt_out;
        const uint32_t read_batch = operation_attributes.read_batch;
        const uint32_t write_batch = operation_attributes.write_batch;
        const uint32_t cb_depth = std::max(read_batch, write_batch) * 2;
        constexpr uint32_t cb_id = 0;
        constexpr uint32_t cb_pad_id = 1;

        CoreSplit split = split_work(device, total_out_tiles);

        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_depth * tile_bytes,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_id,
                .data_format = cb_data_format,
                .page_size = tile_bytes,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = pad_buf_size,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_pad_id,
                .data_format = cb_data_format,
                .page_size = pad_buf_size,
            }}},
        });

        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/reader_tile_interleaved_unified.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqPad},
            {"cb_id", cb_id},
            {"batch", read_batch},
            // reader_tile_interleaved_unified.cpp unconditionally reads this
            // named arg (NAMED-CT-arg trap; see repeat's port).
            {"src_page_pitch", 0},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {cb_id, tile_bytes};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_ct_args.push_back(write_batch);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/writer_pad_tiled_interleaved.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        // Stateful per-core walk mirroring ops/pad/spec.py::_tiled_reader_rt /
        // sequencers.h::seq_pad_next: emit the current (src_tile, tile-coord)
        // then advance through this core's n_tiles pages.
        uint32_t st_src = 0, st_wt = 0, st_ht = 0, st_c = 0, st_n = 0;
        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n_tiles = work_for_core(split, core);
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 n_tiles,
                 st_src,
                 st_wt,
                 st_ht,
                 st_c,
                 st_n,
                 Wt_in,
                 Ht_in,
                 C,
                 N,
                 Wt_out,
                 Ht_out,
                 C_out,
                 N_out,
                 tile_bytes,
                 operation_attributes.packed_pad_value,
                 cb_pad_id,
                 front_wt,
                 front_ht,
                 front_c,
                 front_n});
            writer_desc.emplace_runtime_args(core, {dst_buffer, n_tiles, start});
            for (uint32_t t = 0; t < n_tiles; ++t) {
                const bool is_data = (st_wt >= front_wt && st_wt < front_wt + Wt_in) && (st_ht >= front_ht) &&
                                      (st_ht < front_ht + Ht_in) && (st_c >= front_c) && (st_c < front_c + C) &&
                                      (st_n >= front_n) && (st_n < front_n + N);
                if (is_data) {
                    st_src++;
                }
                st_wt++;
                if (st_wt == Wt_out) {
                    st_wt = 0;
                    st_ht++;
                    if (st_ht == Ht_out) {
                        st_ht = 0;
                        st_c++;
                        if (st_c == C_out) {
                            st_c = 0;
                            st_n++;
                        }
                    }
                }
            }
            start += n_tiles;
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        return desc;
    }

    // RM-interleaved front+back padding via reader_pad_rm_interleaved.cpp +
    // writer_pad_rm_interleaved.cpp. Mirrors ops/pad/spec.py::build_pad_rm.
    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t N_out = operation_attributes.N_out;
    const uint32_t C_out = operation_attributes.C_out;
    const uint32_t H_out = operation_attributes.H_out;
    const uint32_t W_out = operation_attributes.W_out;
    const uint32_t front_w = operation_attributes.front_w;
    const uint32_t front_h = operation_attributes.front_h;
    const uint32_t front_c = operation_attributes.front_c;
    const uint32_t front_n = operation_attributes.front_n;

    const uint32_t elem_size = input.element_size();
    const uint32_t stick_size = W * elem_size;
    const uint32_t stick_size_out = W_out * elem_size;
    const uint32_t front_pad_w_bytes = front_w * elem_size;
    const uint32_t back_pad_w_bytes = (W_out - W - front_w) * elem_size;
    // cb_out's page pitch must be a dram_alignment (NOT just 16B) multiple: a
    // DRAM->L1 NOC read requires l1_addr % A == dram_addr % A, and slot k sits
    // at k*stick_size_out_aligned. See reader_pad_rm_interleaved.cpp's header
    // comment / docs/PAD_RM_RANK4_CORRECTNESS_BUG.md for the historical bug
    // this guards against.
    const uint32_t stick_size_out_aligned = round_up_u32(stick_size_out, std::max(16u, dram_alignment));

    const uint32_t total_out_sticks = N_out * C_out * H_out;
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t cb_pad_id = 1;
    constexpr uint32_t cb_stage_id = 2;
    const uint32_t pad_buf_size = round_up_u32(stick_size_out, dram_alignment);
    const uint32_t stage_buf_size = round_up_u32(stick_size, dram_alignment);
    auto [read_batch, write_batch] =
        pad_rm_batches_for_l1(stage_buf_size, stick_size_out_aligned, ttnn::operations::data_movement::get_max_l1_space(input));
    const uint32_t cb_depth = std::max(read_batch, write_batch) * 2;
    const uint32_t in_read_size = stage_buf_size;

    CoreSplit split = split_work(device, total_out_sticks);

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_depth * stick_size_out_aligned,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = cb_data_format,
            .page_size = stick_size_out_aligned,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = pad_buf_size,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_pad_id,
            .data_format = cb_data_format,
            .page_size = pad_buf_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = stage_buf_size,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_stage_id,
            .data_format = cb_data_format,
            .page_size = stage_buf_size,
        }}},
    });

    std::vector<uint32_t> reader_ct_args = {
        H,
        C,
        N,
        H_out,
        C_out,
        N_out,
        stick_size,
        stick_size_out,
        stick_size_out_aligned,
        back_pad_w_bytes,
        operation_attributes.packed_pad_value,
        read_batch,
        cb_id,
        cb_pad_id,
        front_pad_w_bytes,
        front_h,
        front_c,
        front_n,
        cb_stage_id,
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    reader_ct_args.push_back(in_read_size);
    reader_ct_args.push_back(dram_alignment);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/reader_pad_rm_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {cb_id, stick_size_out, stick_size_out_aligned};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(write_batch);
    // Trailing (pages_per_row, logical_page_size) pair for a ROW_MAJOR
    // width/block-sharded destination; codegen only ever targets an
    // interleaved output (supported_by_codegen rejects sharded output), so
    // this collapses to the historical single write per stick.
    writer_ct_args.push_back(0);
    writer_ct_args.push_back(0);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/writer_pad_rm_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Stateful per-core walk mirroring ops/pad/spec.py::_rm_reader_rt.
    uint32_t st_src = 0, st_h = 0, st_c = 0, st_n = 0;
    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n_sticks = work_for_core(split, core);
        reader_desc.emplace_runtime_args(core, {src_buffer, n_sticks, st_src, st_h, st_c, st_n});
        writer_desc.emplace_runtime_args(core, {dst_buffer, n_sticks, start});
        for (uint32_t t = 0; t < n_sticks; ++t) {
            const bool is_data = (st_h >= front_h) && (st_h < front_h + H) && (st_c >= front_c) &&
                                  (st_c < front_c + C) && (st_n >= front_n) && (st_n < front_n + N);
            if (is_data) {
                st_src++;
            }
            st_h++;
            if (st_h == H_out) {
                st_h = 0;
                st_c++;
                if (st_c == C_out) {
                    st_c = 0;
                    st_n++;
                }
            }
        }
        start += n_sticks;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
