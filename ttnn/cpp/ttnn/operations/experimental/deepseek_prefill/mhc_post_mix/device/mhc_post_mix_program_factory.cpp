// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>

#include "mhc_post_mix_program_factory.hpp"
#include "mhc_post_mix_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
// CB layout (all fp32 tiles). The work unit is one (token-tile r, column-tile c) position: the n
// stream tiles x[r, i, c] and y[r, c] in, the n output tiles out[r, j, c] (or 1 with collapse) out.
constexpr uint32_t CB_X = tt::CBIndex::c_0;      // n tiles per position, double buffered
constexpr uint32_t CB_Y = tt::CBIndex::c_1;      // 1 tile per position, double buffered
constexpr uint32_t CB_RAW = tt::CBIndex::c_2;    // reader scratch: the comb and post tiles of token-tile r
constexpr uint32_t CB_COL = tt::CBIndex::c_3;    // n*n + n coefficient column tiles (coefficient in column 0)
constexpr uint32_t CB_BCAST = tt::CBIndex::c_4;  // the same, broadcast across the 32 columns (compute-made)
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;   // n (or 1) tiles per position, double buffered

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

ProgramDescriptor MhcPostMixProgramFactory::create_descriptor(
    const MhcPostMixParams& args,
    const MhcPostMixTensorArgs& t,
    tensor_return_value_t& out,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;

    const tt::DataFormat df = datatype_to_dataformat_converter(DataType::FLOAT32);
    const uint32_t tile_size = tt::tile_size(df);
    const uint32_t n = args.n;
    const uint32_t n_coef = n * n + n;
    const uint32_t T = t.x.padded_shape()[2];
    const uint32_t C = t.y.padded_shape()[3];
    const uint32_t R = T / tt::constants::TILE_HEIGHT;
    const uint32_t Ct = C / tt::constants::TILE_WIDTH;
    const uint32_t num_positions = R * Ct;
    const uint32_t out_tiles_per_pos = args.collapse ? 1 : n;

    auto* x_buffer = t.x.buffer();
    auto* y_buffer = t.y.buffer();
    auto* post_buffer = t.post.buffer();
    auto* comb_buffer = t.comb.buffer();
    auto* out_buffer = out.buffer();

    const auto grid = t.x.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, pos_per_core_1, pos_per_core_2] =
        split_work_to_cores(grid, num_positions);

    add_cb(desc, all_cores, CB_X, 2 * n, tile_size, df);
    add_cb(desc, all_cores, CB_Y, 2, tile_size, df);
    add_cb(desc, all_cores, CB_RAW, 2, tile_size, df);
    add_cb(desc, all_cores, CB_COL, n_coef, tile_size, df);
    add_cb(desc, all_cores, CB_BCAST, n_coef, tile_size, df);
    add_cb(desc, all_cores, CB_OUT, 2 * out_tiles_per_pos, tile_size, df);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_post_mix/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_X, CB_Y, CB_RAW, CB_COL, n, Ct};
    TensorAccessorArgs(x_buffer).append_to(reader_ct);
    TensorAccessorArgs(y_buffer).append_to(reader_ct);
    TensorAccessorArgs(post_buffer).append_to(reader_ct);
    TensorAccessorArgs(comb_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_post_mix.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, n, Ct, args.collapse ? 1u : 0u};
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_post_mix.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    const float inv_n = 1.0f / static_cast<float>(n);
    std::vector<uint32_t> compute_ct = {
        CB_X, CB_Y, CB_COL, CB_BCAST, CB_OUT, n, Ct, args.collapse ? 1u : 0u, std::bit_cast<uint32_t>(inv_n)};
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_post_mix_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct);
    // The fp32 inputs must be unpacked straight into the fp32 dest register (SrcA/SrcB are 19 bits):
    // this drives both copy_tile's UnpackToDestEn and the 32-bit path of the column-broadcast datacopy.
    std::vector<UnpackToDestMode> unpack_to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : {CB_X, CB_Y, CB_COL, CB_BCAST}) {
        unpack_to_dest[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest),
    };

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t count = core_group_1.contains(core) ? pos_per_core_1 : pos_per_core_2;
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(x_buffer);
            rt.push_back(y_buffer);
            rt.push_back(post_buffer);
            rt.push_back(comb_buffer);
            rt.push_back(start);
            rt.push_back(count);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(out_buffer);
            rt.push_back(start);
            rt.push_back(count);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(start);
            rt.push_back(count);
            compute.emplace_runtime_args(core, rt);
        }
        start += count;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
