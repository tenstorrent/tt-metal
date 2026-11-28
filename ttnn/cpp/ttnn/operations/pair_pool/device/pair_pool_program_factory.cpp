// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pair_pool_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "kernels/pair_pool_args.hpp"

namespace ttnn::operations::pair_pool {

using namespace tt::tt_metal;
using namespace ::pair_pool;

namespace {

constexpr const char* kPairPoolKernelDir = "ttnn/cpp/ttnn/operations/pair_pool/device/kernels/";

std::string pair_pool_kernel_path(const char* name) { return std::string(kPairPoolKernelDir) + name; }

// Output tile-rows pool two input tile-rows each.
constexpr uint32_t kRowsPerUnit = 2 * tt::constants::TILE_HEIGHT;

}  // namespace

ProgramDescriptor PairPoolProgramFactory::create_descriptor(
    const PairPoolParams& attrs, const PairPoolInputs& t, Tensor& output) {
    TT_FATAL(
        cb::OUT < hal::get_num_dataflow_buffers(),
        "pair_pool: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::OUT,
        hal::get_num_dataflow_buffers());

    // plan(): units = rows / 64 * HT, ncores = max(1, min(units, gx * gy)), per, extra = divmod(units, ncores)
    const auto& padded = t.input.padded_shape();
    const auto rows = static_cast<uint32_t>(padded.volume() / padded[-1]);
    const uint32_t ht = t.input.logical_shape()[-1] / 2 / tt::constants::TILE_WIDTH;
    const uint32_t units = rows / kRowsPerUnit * ht;
    const uint32_t gx = attrs.grid_x;
    const uint32_t ncores = std::max(1u, std::min(units, gx * attrs.grid_y));
    const uint32_t per = units / ncores;
    const uint32_t extra = units % ncores;

    // Core i = (i % gx, i / gx): whole rows plus the remainder of the last row.
    const CoreRangeSet cores =
        num_cores_to_corerangeset(ncores, CoreCoord(gx, attrs.grid_y), /*row_wise=*/true).merge_ranges();

    const uint32_t page = tile_size(DataType::FLOAT32);
    const auto cb_format = cb_dataformat_for(DataType::FLOAT32);
    auto make_cb = [&](uint32_t idx, uint32_t pages) {
        return CBDescriptor{
            .total_size = page * pages,
            .core_ranges = cores,
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = cb_format, .page_size = page}},
        };
    };

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(cb::STAGE, cb::STAGE_TILES));
    desc.cbs.push_back(make_cb(cb::IN, cb::IN_TILES));
    desc.cbs.push_back(make_cb(cb::OUT, cb::OUT_TILES));

    const KernelDescriptor::Defines defines = {{"PAIR_POOL_GX", std::to_string(gx)}};

    KernelDescriptor reader;
    reader.kernel_source = pair_pool_kernel_path("pair_pool_reader.cpp");
    reader.core_ranges = cores;
    reader.compile_time_args.resize(reader_ct::COUNT);
    reader.compile_time_args[reader_ct::CB_STAGE] = cb::STAGE;
    reader.compile_time_args[reader_ct::CB_IN] = cb::IN;
    reader.compile_time_args[reader_ct::HT] = ht;
    TensorAccessorArgs(*t.input.buffer()).append_to(reader.compile_time_args);
    reader.defines = defines;
    {
        static_assert(reader_common::SRC_ADDR == 0 && reader_common::PER == 1 && reader_common::EXTRA == 2);
        KernelDescriptor::RTArgList args;
        args.push_back(t.input.buffer());
        args.push_back(per);
        args.push_back(extra);
        reader.emplace_common_runtime_args(args);
    }
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = pair_pool_kernel_path("pair_pool_writer.cpp");
    writer.core_ranges = cores;
    writer.compile_time_args.resize(writer_ct::COUNT);
    writer.compile_time_args[writer_ct::CB_OUT] = cb::OUT;
    writer.compile_time_args[writer_ct::HT] = ht;
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.defines = defines;
    {
        static_assert(writer_common::OUT_ADDR == 0 && writer_common::PER == 1 && writer_common::EXTRA == 2);
        KernelDescriptor::RTArgList args;
        args.push_back(output.buffer());
        args.push_back(per);
        args.push_back(extra);
        writer.emplace_common_runtime_args(args);
    }
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = pair_pool_kernel_path("pair_pool_compute.cpp");
    compute.core_ranges = cores;
    compute.compile_time_args.resize(compute_ct::COUNT);
    compute.compile_time_args[compute_ct::CB_IN] = cb::IN;
    compute.compile_time_args[compute_ct::CB_OUT] = cb::OUT;
    compute.defines = defines;
    compute.common_runtime_args.resize(compute_common::COUNT);
    compute.common_runtime_args[compute_common::PER] = per;
    compute.common_runtime_args[compute_common::EXTRA] = extra;
    std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    modes[cb::IN] = UnpackToDestMode::UnpackToDestFp32;  // exact fp32 copies into DEST
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = std::move(modes),
        .math_approx_mode = false,
    };

    desc.kernels.resize(kernel_index::COUNT);
    desc.kernels[kernel_index::READER] = std::move(reader);
    desc.kernels[kernel_index::WRITER] = std::move(writer);
    desc.kernels[kernel_index::COMPUTE] = std::move(compute);
    return desc;
}

void PairPoolProgramFactory::override_runtime_arguments(
    Program& program,
    const PairPoolParams& /*attrs*/,
    const PairPoolInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins PER / EXTRA, the grid and every compile-time arg; only the two addresses vary.
    GetCommonRuntimeArgs(program, kernel_index::READER)[reader_common::SRC_ADDR] = t.input.buffer()->address();
    GetCommonRuntimeArgs(program, kernel_index::WRITER)[writer_common::OUT_ADDR] = output.buffer()->address();
}

}  // namespace ttnn::operations::pair_pool
