// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "tt_stl/assert.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// codegen/kernels/sequencers.h: constexpr uint32_t SEQ_PAD = 5;
constexpr uint32_t kSeqPad = 5;

// ops/pad/spec.py: _L1_ALIGN / _L1_SAFETY_MARGIN.
constexpr uint32_t kL1Align = 16;
constexpr uint32_t kL1SafetyMargin = 64 * 1024;

}  // namespace

uint32_t pack_pad_value(DataType dtype, float value) {
    if (value == 0.0f) {
        return 0;
    }
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        return static_cast<uint32_t>(static_cast<int32_t>(value));
    }
    float fv = value;
    // Saturate to signed infinity outside the float32 dynamic range (bfloat16 represents
    // infinity exactly), matching ops/pad/builder.py's overflow guard.
    if (std::isfinite(fv) && std::abs(fv) > std::numeric_limits<float>::max()) {
        fv = std::copysign(std::numeric_limits<float>::infinity(), fv);
    }
    uint32_t bits;
    std::memcpy(&bits, &fv, sizeof(bits));
    if (dtype == DataType::FLOAT32) {
        return bits;
    }
    // BFLOAT16: tt-metal's bfloat16(float) constructor canonicalizes every NaN payload to
    // 0x7fc0, otherwise round-to-nearest-even into the upper 16 bits, then duplicate into
    // both halves of the word (one bf16 scalar packs the full uint32_t CB fill word).
    uint32_t bf16 = std::isnan(fv) ? 0x7FC0u : (((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) & 0xFFFFu);
    return (bf16 << 16) | bf16;
}

std::pair<uint32_t, uint32_t> rm_pad_batches_for_l1(
    uint32_t input_page_bytes, uint32_t output_page_bytes, uint32_t budget, uint32_t read_batch, uint32_t write_batch) {
    auto footprint = [&]() {
        uint32_t depth = std::max(read_batch, write_batch) * 2;
        return depth * output_page_bytes + output_page_bytes + input_page_bytes;
    };
    uint32_t safe_budget = budget > kL1SafetyMargin ? budget - kL1SafetyMargin : 0;
    while (footprint() > safe_budget && (read_batch > 1 || write_batch > 1)) {
        if (read_batch >= write_batch && read_batch > 1) {
            read_batch = std::max<uint32_t>(1, read_batch / 2);
        } else if (write_batch > 1) {
            write_batch = std::max<uint32_t>(1, write_batch / 2);
        }
    }
    return {read_batch, write_batch};
}

PadCodegenParams build_pad_codegen_params(
    const Tensor& input_4d,
    uint32_t front_n,
    uint32_t front_c,
    uint32_t front_h,
    uint32_t front_w,
    uint32_t back_n,
    uint32_t back_c,
    uint32_t back_h,
    uint32_t back_w,
    float pad_value,
    const MemoryConfig& output_mem_config) {
    const auto& shape = input_4d.logical_shape();
    const uint32_t N = shape[0];
    const uint32_t C = shape[1];
    const uint32_t H = shape[2];
    const uint32_t W = shape[3];

    PadCodegenParams attrs;
    attrs.N_out = N + front_n + back_n;
    attrs.C_out = C + front_c + back_c;
    attrs.H_out = H + front_h + back_h;
    attrs.W_out = W + front_w + back_w;
    attrs.front_n = front_n;
    attrs.front_c = front_c;
    attrs.front_h = front_h;
    attrs.front_w = front_w;
    attrs.packed_pad_value = pack_pad_value(input_4d.dtype(), pad_value);
    attrs.output_mem_config = output_mem_config;

    if (input_4d.layout() == Layout::TILE) {
        // ops/pad/spec.py: TILE always uses the fixed READ_BATCH/WRITE_BATCH defaults.
        attrs.read_batch = kPadCodegenReadBatchDefault;
        attrs.write_batch = kPadCodegenWriteBatchDefault;
    } else {
        // ops/pad/spec.py: RM batches are L1-clamped via _rm_pad_batches_for_l1 at the
        // wide-stick L1 cliff; usable_l1 mirrors that helper's device-derived budget as the
        // repo-wide convention (device->l1_size_per_core() - allocator's L1 base address).
        const uint32_t elem_size = input_4d.element_size();
        const uint32_t dram_alignment = input_4d.buffer()->alignment();
        const uint32_t input_page = tt::align(W * elem_size, dram_alignment);
        const uint32_t output_page = tt::align(attrs.W_out * elem_size, dram_alignment);
        IDevice* device = input_4d.device();
        const uint32_t budget =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
        const auto [rb, wb] = rm_pad_batches_for_l1(
            input_page, output_page, budget, kPadCodegenReadBatchDefault, kPadCodegenWriteBatchDefault);
        attrs.read_batch = rb;
        attrs.write_batch = wb;
    }
    return attrs;
}

namespace {

// TILE-interleaved back+front padding, byte-identical to ops/pad/spec.py's build_pad_tiled.
ProgramDescriptor create_descriptor_tiled(
    const PadCodegenParams& attrs, const PadCodegenInputs& tensor_args, Tensor& output) {
    const Tensor& input = tensor_args.input;
    IDevice* device = input.device();

    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];

    const uint32_t Ht_in = tt::div_up(H, TILE_HEIGHT);
    const uint32_t Wt_in = tt::div_up(W, TILE_WIDTH);
    const uint32_t Ht_out = tt::div_up(attrs.H_out, TILE_HEIGHT);
    const uint32_t Wt_out = tt::div_up(attrs.W_out, TILE_WIDTH);
    const uint32_t front_ht = attrs.front_h / TILE_HEIGHT;
    const uint32_t front_wt = attrs.front_w / TILE_WIDTH;
    const uint32_t N_out = attrs.N_out;
    const uint32_t C_out = attrs.C_out;
    const uint32_t front_c = attrs.front_c;
    const uint32_t front_n = attrs.front_n;

    const uint32_t dram_alignment = input.buffer()->alignment();
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    // DRAM page pitch is placement-specific: the pad reader (fill size), CB page, and writer
    // must ALL step at the same aligned pitch or multi-page tensors accrue a page skew.
    const uint32_t tile_bytes = tt::align(tt::tile_size(data_format), dram_alignment);
    const uint32_t pad_buf_size = tt::align(tile_bytes, dram_alignment);

    const uint32_t total_out_tiles = N_out * C_out * Ht_out * Wt_out;
    TT_FATAL(total_out_tiles > 0, "pad_codegen (TILE): zero-volume output");

    const uint32_t read_batch = attrs.read_batch;
    const uint32_t write_batch = attrs.write_batch;
    const uint32_t cb_depth = std::max(read_batch, write_batch) * 2;
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t cb_pad_id = 1;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, total_out_tiles);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_depth * tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = data_format,
            .page_size = tile_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = pad_buf_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_pad_id,
            .data_format = data_format,
            .page_size = pad_buf_size,
        }}},
    });

    Buffer* in_buffer = input.buffer();
    Buffer* out_buffer = output.buffer();

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/reader_tile_interleaved_unified.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    TensorAccessorArgs(*in_buffer).append_to(reader_desc.compile_time_args);
    // "src_page_pitch" (0 => absent) is unconditionally read by
    // reader_tile_interleaved_unified.cpp; every named-CT-arg lookup for an undeclared name
    // is a kernel-compile-time failure (get_named_ct_arg's __builtin_unreachable), so this
    // pad port -- which never needs the pitch override -- must still supply the sentinel.
    reader_desc.named_compile_time_args = {
        {"seq_id", kSeqPad}, {"cb_id", cb_id}, {"batch", read_batch}, {"src_page_pitch", 0}};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/writer_pad_tiled_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {cb_id, tile_bytes};
    TensorAccessorArgs(*out_buffer).append_to(writer_desc.compile_time_args);
    writer_desc.compile_time_args.push_back(write_batch);
    writer_desc.config = WriterConfigDescriptor{};

    // Full stateful per-core reader-RT walk, mirroring ops/pad/spec.py's _tiled_reader_rt:
    // called once per core, in order, threading (src_tile, wt, ht, c, n) across the split.
    struct WalkState {
        uint32_t src = 0, wt = 0, ht = 0, c = 0, n = 0;
    } st;

    const auto grid = compute_with_storage_grid_size;
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    const uint32_t g1_cores = core_group_1.num_cores();
    uint32_t start = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t n_tiles = (i < g1_cores) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;

        reader_desc.emplace_runtime_args(
            core,
            {in_buffer,
             n_tiles,
             st.src,
             st.wt,
             st.ht,
             st.c,
             st.n,
             Wt_in,
             Ht_in,
             C,
             N,
             Wt_out,
             Ht_out,
             C_out,
             N_out,
             tile_bytes,
             attrs.packed_pad_value,
             cb_pad_id,
             front_wt,
             front_ht,
             front_c,
             front_n});
        for (uint32_t t = 0; t < n_tiles; ++t) {
            const bool is_data = (st.wt >= front_wt && st.wt < front_wt + Wt_in) &&
                                 (st.ht >= front_ht && st.ht < front_ht + Ht_in) &&
                                 (st.c >= front_c && st.c < front_c + C) && (st.n >= front_n && st.n < front_n + N);
            if (is_data) {
                st.src++;
            }
            st.wt++;
            if (st.wt == Wt_out) {
                st.wt = 0;
                st.ht++;
                if (st.ht == Ht_out) {
                    st.ht = 0;
                    st.c++;
                    if (st.c == C_out) {
                        st.c = 0;
                        st.n++;
                    }
                }
            }
        }

        writer_desc.emplace_runtime_args(core, {out_buffer, n_tiles, start});
        start += n_tiles;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

// RM-interleaved front+back padding, byte-identical to ops/pad/spec.py's build_pad_rm.
ProgramDescriptor create_descriptor_rm(
    const PadCodegenParams& attrs, const PadCodegenInputs& tensor_args, Tensor& output) {
    const Tensor& input = tensor_args.input;
    IDevice* device = input.device();

    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t N_out = attrs.N_out;
    const uint32_t C_out = attrs.C_out;
    const uint32_t H_out = attrs.H_out;
    const uint32_t W_out = attrs.W_out;
    const uint32_t front_h = attrs.front_h;
    const uint32_t front_c = attrs.front_c;
    const uint32_t front_n = attrs.front_n;
    const uint32_t front_w = attrs.front_w;

    const uint32_t dram_alignment = input.buffer()->alignment();
    const uint32_t elem_size = input.element_size();
    const uint32_t stick_size = W * elem_size;
    const uint32_t stick_size_out = W_out * elem_size;
    const uint32_t front_pad_w_bytes = front_w * elem_size;
    const uint32_t back_pad_w_bytes = (W_out - W - front_w) * elem_size;
    const uint32_t stick_size_out_aligned = tt::align(stick_size_out, kL1Align);

    const uint32_t total_out_sticks = N_out * C_out * H_out;
    TT_FATAL(total_out_sticks > 0, "pad_codegen (RM): zero-volume output");

    const uint32_t read_batch = attrs.read_batch;
    const uint32_t write_batch = attrs.write_batch;
    const uint32_t cb_depth = std::max(read_batch, write_batch) * 2;
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t cb_pad_id = 1;
    constexpr uint32_t cb_stage_id = 2;
    const uint32_t pad_buf_size = tt::align(stick_size_out, dram_alignment);
    const uint32_t stage_buf_size = tt::align(stick_size, dram_alignment);
    // in_read_size: DRAM-aligned input page bytes -- the reader's staging path reads exactly
    // this many bytes into the aligned staging CB, then RISC-memmoves stick_size real bytes.
    const uint32_t in_read_size = stage_buf_size;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, total_out_sticks);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_depth * stick_size_out_aligned,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = stick_size_out_aligned,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = pad_buf_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_pad_id,
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = pad_buf_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = stage_buf_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_stage_id,
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = stage_buf_size,
        }}},
    });

    Buffer* in_buffer = input.buffer();
    Buffer* out_buffer = output.buffer();

    // CT layout is byte-identical to ops/pad/spec.py's build_pad_rm reader_ct: fixed header,
    // then accessor args, then in_read_size + dram_alignment appended AFTER the accessor so
    // the reader's fixed-index ABI (reader_pad_rm_interleaved.cpp) never shifts.
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/reader_pad_rm_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = {
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
        attrs.packed_pad_value,
        read_batch,
        cb_id,
        cb_pad_id,
        front_pad_w_bytes,
        front_h,
        front_c,
        front_n,
        cb_stage_id,
    };
    TensorAccessorArgs(*in_buffer).append_to(reader_desc.compile_time_args);
    reader_desc.compile_time_args.push_back(in_read_size);
    reader_desc.compile_time_args.push_back(dram_alignment);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/writer_pad_rm_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {cb_id, stick_size_out, stick_size_out_aligned};
    TensorAccessorArgs(*out_buffer).append_to(writer_desc.compile_time_args);
    writer_desc.compile_time_args.push_back(write_batch);
    writer_desc.config = WriterConfigDescriptor{};

    // Full stateful per-core reader-RT walk, mirroring ops/pad/spec.py's _rm_reader_rt: called
    // once per core, in order, threading (src_stick, h, c, n) across the split.
    struct WalkState {
        uint32_t src = 0, h = 0, c = 0, n = 0;
    } st;

    const auto grid = compute_with_storage_grid_size;
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    const uint32_t g1_cores = core_group_1.num_cores();
    uint32_t start = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t n_sticks = (i < g1_cores) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        reader_desc.emplace_runtime_args(core, {in_buffer, n_sticks, st.src, st.h, st.c, st.n});
        for (uint32_t t = 0; t < n_sticks; ++t) {
            const bool is_data = (st.h >= front_h && st.h < front_h + H) && (st.c >= front_c && st.c < front_c + C) &&
                                 (st.n >= front_n && st.n < front_n + N);
            if (is_data) {
                st.src++;
            }
            st.h++;
            if (st.h == H_out) {
                st.h = 0;
                st.c++;
                if (st.c == C_out) {
                    st.c = 0;
                    st.n++;
                }
            }
        }

        writer_desc.emplace_runtime_args(core, {out_buffer, n_sticks, start});
        start += n_sticks;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor PadCodegenProgramFactory::create_descriptor(
    const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args, Tensor& output_tensor) {
    if (tensor_args.input.layout() == Layout::TILE) {
        return create_descriptor_tiled(operation_attributes, tensor_args, output_tensor);
    }
    return create_descriptor_rm(operation_attributes, tensor_args, output_tensor);
}

}  // namespace ttnn::prim
