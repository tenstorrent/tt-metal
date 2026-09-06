// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cstdlib>

#include "mhc_split_sinkhorn_program_factory.hpp"
#include "mhc_split_sinkhorn_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t CB_MIXES = tt::CBIndex::c_0;
constexpr uint32_t CB_CONSTS = tt::CBIndex::c_1;  // 8 tiles
constexpr uint32_t CB_PRE = tt::CBIndex::c_2;
constexpr uint32_t CB_POST = tt::CBIndex::c_3;
constexpr uint32_t CB_COMB = tt::CBIndex::c_4;
constexpr uint32_t CB_M = tt::CBIndex::c_16;
constexpr uint32_t CB_MM = tt::CBIndex::c_17;
constexpr uint32_t CB_RECIP = tt::CBIndex::c_18;
constexpr uint32_t CB_PRE_W = tt::CBIndex::c_24;
constexpr uint32_t CB_POST_W = tt::CBIndex::c_25;

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

ProgramDescriptor MhcSplitSinkhornProgramFactory::create_descriptor(
    const MhcSplitSinkhornParams& operation_attributes,
    const MhcSplitSinkhornTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;

    const tt::DataFormat df = datatype_to_dataformat_converter(tt::tt_metal::DataType::FLOAT32);
    const uint32_t tile_size = tt::tile_size(df);

    auto* mixes_buffer = tensor_args.mixes.buffer();
    auto* consts_buffer = tensor_args.consts.buffer();
    auto* pre_buffer = tensor_return_value.at(0).buffer();
    auto* post_buffer = tensor_return_value.at(1).buffer();
    auto* comb_buffer = tensor_return_value.at(2).buffer();

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/";

    // ---- Sharded path (#40720): mixes already resident per-core in L1; alias the input and
    // output CBs straight to the shards (zero-copy), so the reader only fetches consts and there
    // is no writer. The compute kernel is identical -- the FIFO advances through the aliased shard.
    if (tensor_args.mixes.memory_config().is_sharded()) {
        const auto& shard = tensor_args.mixes.shard_spec().value();
        const CoreRangeSet& cores = shard.grid;
        const uint32_t tiles = shard.shape[0] / tt::constants::TILE_HEIGHT;  // token-tiles per core

        auto alias_cb = [&](uint32_t idx, Buffer* buf) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = tiles * tile_size,
                .core_ranges = cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = tile_size}}},
                .buffer = buf});
        };
        alias_cb(CB_MIXES, mixes_buffer);
        add_cb(desc, cores, CB_CONSTS, 8, tile_size, df);  // consts stay in DRAM, read per core
        alias_cb(CB_PRE, pre_buffer);
        alias_cb(CB_POST, post_buffer);
        alias_cb(CB_COMB, comb_buffer);
        add_cb(desc, cores, CB_M, 2, tile_size, df);
        add_cb(desc, cores, CB_MM, 2, tile_size, df);
        add_cb(desc, cores, CB_RECIP, 2, tile_size, df);
        add_cb(desc, cores, CB_PRE_W, 2, tile_size, df);
        add_cb(desc, cores, CB_POST_W, 2, tile_size, df);

        std::vector<uint32_t> reader_ct = {CB_MIXES, CB_CONSTS};
        TensorAccessorArgs(consts_buffer).append_to(reader_ct);
        KernelDescriptor reader;
        reader.kernel_source = kdir + "dataflow/reader_mhc_split_sinkhorn_sharded.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = cores;
        reader.compile_time_args = std::move(reader_ct);
        reader.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> compute_ct = {
            operation_attributes.sinkhorn_iters, std::bit_cast<uint32_t>(operation_attributes.eps)};
        KernelDescriptor compute;
        compute.kernel_source = kdir + "compute/mhc_split_sinkhorn_compute.cpp";
        compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute.core_ranges = cores;
        compute.compile_time_args = std::move(compute_ct);
        compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};

        for (const CoreCoord& core : corerange_to_cores(cores)) {
            KernelDescriptor::RTArgList rr;
            rr.push_back(consts_buffer);
            rr.push_back(tiles);
            reader.emplace_runtime_args(core, rr);
            KernelDescriptor::RTArgList rc;
            rc.push_back(tiles);
            compute.emplace_runtime_args(core, rc);
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(compute));
        return desc;
    }

    // ---- Interleaved (DRAM) path.
    // Parametrization is independent per token, so distribute the token-tiles across the grid.
    const uint32_t num_token_tiles = tensor_args.mixes.physical_volume() / tt::constants::TILE_HW;
    auto grid = tensor_args.mixes.device()->compute_with_storage_grid_size();
    // max_cores==1 pins to a single core -- the multi-core A/B baseline. It rides in the hashed
    // attributes (read from MHC_MAX_CORES at op launch), so the program cache stays correct.
    if (operation_attributes.max_cores == 1) {
        grid = CoreCoord{1, 1};
    }
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(grid, num_token_tiles);

    add_cb(desc, all_cores, CB_MIXES, 2, tile_size, df);
    add_cb(desc, all_cores, CB_CONSTS, 8, tile_size, df);
    add_cb(desc, all_cores, CB_PRE, 2, tile_size, df);
    add_cb(desc, all_cores, CB_POST, 2, tile_size, df);
    add_cb(desc, all_cores, CB_COMB, 2, tile_size, df);
    add_cb(desc, all_cores, CB_M, 2, tile_size, df);
    add_cb(desc, all_cores, CB_MM, 2, tile_size, df);
    add_cb(desc, all_cores, CB_RECIP, 2, tile_size, df);
    add_cb(desc, all_cores, CB_PRE_W, 2, tile_size, df);
    add_cb(desc, all_cores, CB_POST_W, 2, tile_size, df);

    std::vector<uint32_t> reader_ct = {CB_MIXES, CB_CONSTS};
    TensorAccessorArgs(mixes_buffer).append_to(reader_ct);
    TensorAccessorArgs(consts_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/dataflow/"
        "reader_mhc_split_sinkhorn.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_PRE, CB_POST, CB_COMB};
    TensorAccessorArgs(pre_buffer).append_to(writer_ct);
    TensorAccessorArgs(post_buffer).append_to(writer_ct);
    TensorAccessorArgs(comb_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/dataflow/"
        "writer_mhc_split_sinkhorn.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_ct = {
        operation_attributes.sinkhorn_iters,
        std::bit_cast<uint32_t>(operation_attributes.eps),
    };
    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/compute/"
        "mhc_split_sinkhorn_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
    };

    // Per-core work split. start_tile is the page offset into mixes/outputs; consts (8
    // tiles) are read whole on every core. Buffer* first so cache hits patch addresses.
    for (uint32_t i = 0, start_tile = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t tiles = core_group_1.contains(core) ? tiles_per_core_1 : tiles_per_core_2;

        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(mixes_buffer);
            rt.push_back(consts_buffer);
            rt.push_back(tiles);
            rt.push_back(start_tile);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(pre_buffer);
            rt.push_back(post_buffer);
            rt.push_back(comb_buffer);
            rt.push_back(tiles);
            rt.push_back(start_tile);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(tiles);
            compute.emplace_runtime_args(core, rt);
        }
        start_tile += tiles;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
