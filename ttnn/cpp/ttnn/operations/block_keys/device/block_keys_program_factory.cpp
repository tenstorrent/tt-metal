// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_keys_program_factory.hpp"

#include <algorithm>
#include <utility>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "kernels/block_keys_args.hpp"

namespace ttnn::operations::block_keys {

using namespace tt::tt_metal;

namespace {

constexpr const char* kBlockKeysKernel = "ttnn/cpp/ttnn/operations/block_keys/device/kernels/block_keys.cpp";

uint32_t leading_rows(const Tensor& t) {
    const auto& shape = t.logical_shape();
    uint32_t rows = 1;
    for (uint32_t i = 0; i + 1 < shape.rank(); ++i) {
        rows *= shape[i];
    }
    return rows;
}

}  // namespace

ProgramDescriptor BlockKeysProgramFactory::create_descriptor(
    const BlockKeysParams& attrs, const BlockKeysInputs& t, Tensor& output) {
    TT_FATAL(
        ::block_keys::cb::SCRATCH < hal::get_num_dataflow_buffers(),
        "block_keys: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        ::block_keys::cb::SCRATCH,
        hal::get_num_dataflow_buffers());

    const uint32_t rows = leading_rows(t.pos);
    const uint32_t kp = t.pos.logical_shape()[-1];
    const uint32_t k = t.ids.logical_shape()[-1];
    const uint32_t gx = attrs.grid_x;
    const uint32_t ncores = std::max(1u, std::min(gx * attrs.grid_y, rows));

    // The first ncores cores filled row by row (core i at (i % gx, i / gx), the kernel's c = y * GX + x).
    const CoreRangeSet cores = num_cores_to_corerangeset(ncores, CoreCoord(gx, attrs.grid_y), /*row_wise=*/true);

    // The kernel aligns its base up (up to SCRATCH_ALIGN - 1 bytes lost), then the pos row and the ids row, each
    // rounded up to SCRATCH_ALIGN.
    constexpr uint32_t align = ::block_keys::SCRATCH_ALIGN;
    const uint32_t word = sizeof(uint32_t);
    const uint32_t scratch = align + tt::round_up(word * kp, align) + tt::round_up(word * k, align);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(::block_keys::cb::SCRATCH),
            .data_format = tt::DataFormat::UInt8,
            .page_size = scratch,
        }},
    });

    KernelDescriptor kernel;
    kernel.kernel_source = kBlockKeysKernel;
    kernel.core_ranges = cores;
    kernel.compile_time_args.assign(::block_keys::ct_arg::COUNT, 0u);
    kernel.compile_time_args[::block_keys::ct_arg::CB] = ::block_keys::cb::SCRATCH;
    kernel.compile_time_args[::block_keys::ct_arg::BLOCK] = attrs.block;
    kernel.compile_time_args[::block_keys::ct_arg::K] = k;
    kernel.compile_time_args[::block_keys::ct_arg::KP] = kp;
    kernel.compile_time_args[::block_keys::ct_arg::NBW] = attrs.nbw;
    kernel.compile_time_args[::block_keys::ct_arg::GX] = gx;
    TensorAccessorArgs(*t.pos.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(*t.ids.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);

    // Pushed in common_arg order: POS_ADDR, IDS_ADDR, OUT_ADDR, PER, EXTRA.
    KernelDescriptor::RTArgList common;
    common.push_back(t.pos.buffer());
    common.push_back(t.ids.buffer());
    common.push_back(output.buffer());
    common.push_back(rows / ncores);
    common.push_back(rows % ncores);
    kernel.emplace_common_runtime_args(common);
    TT_FATAL(
        kernel.common_runtime_args.size() == ::block_keys::common_arg::COUNT,
        "block_keys: common runtime argument layout is out of sync with block_keys_args.hpp");
    kernel.config = ReaderConfigDescriptor{};

    desc.kernels.push_back(std::move(kernel));
    return desc;
}

void BlockKeysProgramFactory::override_runtime_arguments(
    Program& program,
    const BlockKeysParams& /*attrs*/,
    const BlockKeysInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // PER / EXTRA and every compile-time arg are pinned by the key; only the three addresses vary.
    auto& args = GetCommonRuntimeArgs(program, ::block_keys::kernel_index::KEYS);
    args[::block_keys::common_arg::POS_ADDR] = t.pos.buffer()->address();
    args[::block_keys::common_arg::IDS_ADDR] = t.ids.buffer()->address();
    args[::block_keys::common_arg::OUT_ADDR] = output.buffer()->address();
}

}  // namespace ttnn::operations::block_keys
