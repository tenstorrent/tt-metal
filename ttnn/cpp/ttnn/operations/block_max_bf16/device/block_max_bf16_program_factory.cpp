// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_max_bf16_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/block_max_bf16_args.hpp"

namespace ttnn::operations::block_max_bf16 {

using namespace tt::tt_metal;
namespace args = ::block_max_bf16;

namespace {

constexpr const char* kBlockMaxKernelDir = "ttnn/cpp/ttnn/operations/block_max_bf16/device/kernels/";

// The Python _grid(): the device grid clamped to max_grid, each side at least 1.
std::pair<uint32_t, uint32_t> block_max_grid(const BlockMaxBf16Params& attrs, const Tensor& input) {
    const CoreCoord g = input.device()->compute_with_storage_grid_size();
    auto gx = static_cast<uint32_t>(g.x);
    auto gy = static_cast<uint32_t>(g.y);
    if (attrs.max_grid.has_value()) {
        gx = std::max(1u, std::min(gx, (*attrs.max_grid)[0]));
        gy = std::max(1u, std::min(gy, (*attrs.max_grid)[1]));
    }
    return {gx, gy};
}

// The Python _rows(): the product of every dim but the last.
uint32_t block_max_rows(const Tensor& t) {
    const auto& shape = t.logical_shape();
    return static_cast<uint32_t>(shape.volume() / shape[-1]);
}

Buffer* len_buffer(const BlockMaxBf16Inputs& t) { return t.n_tensor.has_value() ? t.n_tensor->buffer() : nullptr; }

CBDescriptor scratch_cb(uint32_t scratch, const CoreRangeSet& crs) {
    TT_FATAL(
        args::cb::SCRATCH < hal::get_num_dataflow_buffers(),
        "block_max_bf16: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        args::cb::SCRATCH,
        hal::get_num_dataflow_buffers());
    return CBDescriptor{
        .total_size = scratch,
        .core_ranges = crs,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(args::cb::SCRATCH),
            .data_format = tt::DataFormat::UInt8,
            .page_size = scratch,
        }},
    };
}

// Accessors of in, out and len (out again without n_tensor, so the kernel's layout matches the Python's).
void append_accessors(std::vector<uint32_t>& ct, const BlockMaxBf16Inputs& t, const Tensor& output) {
    TensorAccessorArgs(*t.input.buffer()).append_to(ct);
    TensorAccessorArgs(*output.buffer()).append_to(ct);
    TensorAccessorArgs(t.n_tensor.has_value() ? *t.n_tensor->buffer() : *output.buffer()).append_to(ct);
}

// Both kernels (READER then WRITER, the kernel_index order), the same args but HALF.
ProgramDescriptor build_descriptor(
    const char* kernel_name,
    const std::vector<uint32_t>& ct,
    uint32_t half_slot,
    uint32_t rows,
    uint32_t scratch,
    uint32_t gx,
    uint32_t gy,
    const BlockMaxBf16Params& attrs,
    const BlockMaxBf16Inputs& t,
    const Tensor& output) {
    const CoreRangeSet crs(CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1)));
    const bool has_len = t.n_tensor.has_value();

    std::vector<std::variant<uint32_t, Buffer*>> common(args::common_arg::COUNT, 0u);
    common[args::common_arg::IN_ADDR] = t.input.buffer();
    common[args::common_arg::OUT_ADDR] = output.buffer();
    common[args::common_arg::LEN_ADDR] = len_buffer(t);
    common[args::common_arg::N] = has_len ? 0u : attrs.n;
    common[args::common_arg::ROWS] = rows;

    ProgramDescriptor desc;
    desc.cbs.push_back(scratch_cb(scratch, crs));
    for (const uint32_t half : {args::half::READER, args::half::WRITER}) {
        KernelDescriptor k;
        k.kernel_source = std::string(kBlockMaxKernelDir) + kernel_name;
        k.core_ranges = crs;
        k.compile_time_args = ct;
        k.compile_time_args[half_slot] = half;
        append_accessors(k.compile_time_args, t, output);
        KernelDescriptor::RTArgList common_list;
        for (const auto& arg : common) {
            std::visit([&common_list](auto value) { common_list.push_back(value); }, arg);
        }
        k.emplace_common_runtime_args(common_list);
        if (half == args::half::READER) {
            k.config = ReaderConfigDescriptor{};
        } else {
            k.config = WriterConfigDescriptor{};
        }
        desc.kernels.push_back(std::move(k));
    }
    TT_FATAL(desc.kernels.size() == args::kernel_index::COUNT, "block_max_bf16: kernel order out of sync");
    return desc;
}

// The per-call common args, in common_arg order: what the key leaves out (the three addresses and n). ROWS and every
// compile-time arg are pinned by the key.
struct CallArgs {
    uint32_t in_addr;
    uint32_t out_addr;
    uint32_t len_addr;
    uint32_t n;
};

// Writes the call's values into both kernels' common args. The program's own args are the record of what the last
// completed hit (or the miss) wrote: when they already hold the call's values, nothing is written.
void write_common_args(Program& program, const CallArgs& call) {
    const auto holds_call = [&program, &call](uint32_t kernel) {
        const auto& common = GetCommonRuntimeArgs(program, kernel);
        return common[args::common_arg::IN_ADDR] == call.in_addr &&
               common[args::common_arg::OUT_ADDR] == call.out_addr &&
               common[args::common_arg::LEN_ADDR] == call.len_addr && common[args::common_arg::N] == call.n;
    };
    if (holds_call(args::kernel_index::READER) && holds_call(args::kernel_index::WRITER)) {
        return;
    }
    for (const uint32_t kernel : {args::kernel_index::READER, args::kernel_index::WRITER}) {
        auto& common = GetCommonRuntimeArgs(program, kernel);
        common[args::common_arg::IN_ADDR] = call.in_addr;
        common[args::common_arg::OUT_ADDR] = call.out_addr;
        common[args::common_arg::LEN_ADDR] = call.len_addr;
        common[args::common_arg::N] = call.n;
    }
}

}  // namespace

ProgramDescriptor BlockMaxReduceProgramFactory::create_descriptor(
    const BlockMaxBf16Params& attrs, const BlockMaxBf16Inputs& t, Tensor& output) {
    const uint32_t factor = attrs.factor;
    TT_FATAL(factor >= 1 && factor <= 16 && (factor & (factor - 1)) == 0, "block_max_bf16: factor {}", factor);
    const auto ni = static_cast<uint32_t>(t.input.logical_shape()[-1]);
    const auto [gx, gy] = block_max_grid(attrs, t.input);
    const uint32_t span = tt::round_up(
        (args::CAST_CHUNK * 4) + (args::CAST_CHUNK / factor * 2) + args::SCRATCH_ALIGN, args::SCRATCH_ALIGN);
    const uint32_t scratch = args::SCRATCH_ALIGN + (4 * span);  // two RISCs x two buffers

    std::vector<uint32_t> ct(args::reduce_ct::COUNT);
    ct[args::reduce_ct::CB] = args::cb::SCRATCH;
    ct[args::reduce_ct::FACTOR] = factor;
    ct[args::reduce_ct::NI] = ni;
    ct[args::reduce_ct::GX] = gx;
    ct[args::reduce_ct::NCORES] = gx * gy;
    ct[args::reduce_ct::CHUNK] = args::CAST_CHUNK;
    ct[args::reduce_ct::HAS_LEN] = t.n_tensor.has_value() ? 1u : 0u;
    return build_descriptor(
        "block_max_bf16.cpp", ct, args::reduce_ct::HALF, block_max_rows(t.input), scratch, gx, gy, attrs, t, output);
}

void BlockMaxReduceProgramFactory::override_runtime_arguments(
    Program& program,
    const BlockMaxBf16Params& attrs,
    const BlockMaxBf16Inputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const bool has_len = t.n_tensor.has_value();
    write_common_args(
        program,
        CallArgs{
            .in_addr = t.input.buffer()->address(),
            .out_addr = output.buffer()->address(),
            .len_addr = has_len ? t.n_tensor->buffer()->address() : 0u,
            .n = has_len ? 0u : attrs.n,  // n is out of the key: re-applied here
        });
}

ProgramDescriptor BlockMaxStripProgramFactory::create_descriptor(
    const BlockMaxBf16Params& attrs, const BlockMaxBf16Inputs& t, Tensor& output) {
    const auto nb = static_cast<uint32_t>(output.logical_shape()[-1]);
    const auto [gx, gy] = block_max_grid(attrs, t.input);
    const uint32_t scratch =
        args::SCRATCH_ALIGN + (2 * 4 * args::STRIP_CHUNK * args::STRIP_RUN_BYTES);  // two RISCs x 2 in + 2 out
    const uint32_t strips = block_max_rows(output) / tt::constants::TILE_HEIGHT;

    std::vector<uint32_t> ct(args::strip_ct::COUNT);
    ct[args::strip_ct::CB] = args::cb::SCRATCH;
    ct[args::strip_ct::NB] = nb;
    ct[args::strip_ct::GX] = gx;
    ct[args::strip_ct::NCORES] = gx * gy;
    ct[args::strip_ct::CHUNK_BLOCKS] = args::STRIP_CHUNK;
    ct[args::strip_ct::HAS_LEN] = t.n_tensor.has_value() ? 1u : 0u;
    return build_descriptor("block_max_strip.cpp", ct, args::strip_ct::HALF, strips, scratch, gx, gy, attrs, t, output);
}

void BlockMaxStripProgramFactory::override_runtime_arguments(
    Program& program,
    const BlockMaxBf16Params& attrs,
    const BlockMaxBf16Inputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const bool has_len = t.n_tensor.has_value();
    write_common_args(
        program,
        CallArgs{
            .in_addr = t.input.buffer()->address(),
            .out_addr = output.buffer()->address(),
            .len_addr = has_len ? t.n_tensor->buffer()->address() : 0u,
            .n = has_len ? 0u : attrs.n,  // n is out of the key: re-applied here
        });
}

}  // namespace ttnn::operations::block_max_bf16
