// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_program_factory.hpp"
#include "mhc_post_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t CB_IN = tt::CBIndex::c_0;      // y tile followed by the n residual-stream tiles
constexpr uint32_t CB_PC = tt::CBIndex::c_1;      // raw post tile, raw comb tile
constexpr uint32_t CB_CONSTS = tt::CBIndex::c_2;  // n*n column-broadcast tiles, resident
constexpr uint32_t CB_COEF = tt::CBIndex::c_3;    // broadcast post_j then broadcast comb_(i,j)
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;

void add_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t index,
    uint32_t n_tiles,
    uint32_t tile_size,
    tt::DataFormat df) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = df,
            .page_size = tile_size,
        }}},
    });
}
}  // namespace

ProgramDescriptor MhcPostProgramFactory::create_descriptor(
    const MhcPostParams& operation_attributes,
    const MhcPostTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;

    const tt::DataFormat df = datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    const uint32_t tile_size = tt::tile_size(df);
    const uint32_t n = operation_attributes.n;

    auto* y_buffer = tensor_args.y.buffer();
    auto* residual_buffer = tensor_args.residual.buffer();
    auto* post_buffer = tensor_args.post.buffer();
    auto* comb_buffer = tensor_args.comb.buffer();
    auto* consts_buffer = tensor_args.consts.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    const auto& yps = tensor_args.y.padded_shape();
    const uint32_t token_tiles = yps[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t col_tiles = yps[-1] / tt::constants::TILE_WIDTH;

    // One work unit is one (token-tile, column-tile) pair: it reads y plus the n residual tiles at
    // that position and writes the n output tiles, so every stream of a column is mixed while its
    // inputs are still in L1. Units are numbered row-major (unit = t0*col_tiles + c0) and each core
    // takes a contiguous run, so a core normally stays inside one token-tile and extracts that
    // token-tile's coefficients once.
    const uint32_t num_units = token_tiles * col_tiles;
    auto grid = tensor_args.y.device()->compute_with_storage_grid_size();
    if (operation_attributes.max_cores == 1) {
        grid = CoreCoord{1, 1};
    }
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, num_units);

    add_cb(desc, all_cores, CB_IN, 2 * (1 + n), tile_size, df);
    add_cb(desc, all_cores, CB_PC, 4, tile_size, df);
    add_cb(desc, all_cores, CB_CONSTS, n * n, tile_size, df);
    add_cb(desc, all_cores, CB_COEF, n + n * n, tile_size, df);
    add_cb(desc, all_cores, CB_OUT, 2 * n, tile_size, df);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_post/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_IN, CB_PC, CB_CONSTS, n, col_tiles};
    TensorAccessorArgs(y_buffer).append_to(reader_ct);
    TensorAccessorArgs(residual_buffer).append_to(reader_ct);
    TensorAccessorArgs(post_buffer).append_to(reader_ct);
    TensorAccessorArgs(comb_buffer).append_to(reader_ct);
    TensorAccessorArgs(consts_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_post.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, n, col_tiles};
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_post.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_post_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {n, col_tiles};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
    };

    // Buffer* first so program-cache hits patch addresses.
    for (uint32_t i = 0, start_unit = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t units = core_group_1.contains(core) ? units_per_core_1 : units_per_core_2;

        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(y_buffer);
            rt.push_back(residual_buffer);
            rt.push_back(post_buffer);
            rt.push_back(comb_buffer);
            rt.push_back(consts_buffer);
            rt.push_back(units);
            rt.push_back(start_unit);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(out_buffer);
            rt.push_back(units);
            rt.push_back(start_unit);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(units);
            rt.push_back(start_unit);
            compute.emplace_runtime_args(core, rt);
        }
        start_unit += units;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
