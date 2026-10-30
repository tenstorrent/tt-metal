// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/attn_res_scores_program_factory.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/kernels/";

CBDescriptor make_cb(
    CBIndex index, const CoreRangeSet& cores, uint32_t num_tiles, uint32_t tile_size, DataFormat format) {
    return CBDescriptor{
        .total_size = num_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = tile_size,
        }}},
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResScoresProgramFactory::create_descriptor(
    const AttnResScoresParams& operation_attributes,
    const AttnResScoresInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* device = tensor_args.stats.device();
    const auto& stats = tensor_args.stats;
    auto& [exponentials, shift, mass] = tensor_return_value;

    const auto data_format = datatype_to_dataformat_converter(stats.dtype());
    const auto tile_size = tt::tile_size(data_format);

    const auto& shape = stats.padded_shape();
    const uint32_t num_candidates = shape[1];
    const uint32_t Ht = shape[-2] / TILE_HEIGHT;
    const uint32_t num_partials = shape[-1] / TILE_WIDTH;
    const uint32_t num_sites = operation_attributes.num_sites;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // The arithmetic is a few tiles per token row-tile; what costs is the writes. Each
    // site is a column of the result, and the consumers want it as column 0 of its own
    // tile, so a token row-tile turns into `num_sites * (C + 2)` output tiles. A core per
    // row-tile would leave most of the grid idle behind those writes, so the sites are
    // split into groups as well and every (row-tile, group) is a work item. The cores of
    // one row-tile each redo its few tiles of arithmetic, which is cheaper than any way
    // of sharing it.
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t max_groups = std::clamp(grid_cores / Ht, 1u, num_sites);
    const uint32_t sites_per_group = (num_sites + max_groups - 1) / max_groups;
    // Recounted from the group size so the last group is never empty.
    const uint32_t num_groups = (num_sites + sites_per_group - 1) / sites_per_group;
    const uint32_t num_items = Ht * num_groups;
    const uint32_t num_cores = std::min(num_items, grid_cores);
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // A row-tile's candidates are one unit everywhere: the max and the sum both run
    // over all of them, so nothing can leave before the last arrives.
    desc.cbs.push_back(
        make_cb(CBIndex::c_0, all_cores, 2 * num_candidates * num_partials, tile_size, data_format));  // stats
    desc.cbs.push_back(make_cb(CBIndex::c_4, all_cores, num_candidates, tile_size, data_format));      // summed
    desc.cbs.push_back(make_cb(CBIndex::c_1, all_cores, num_candidates, tile_size, data_format));      // rsqrt
    desc.cbs.push_back(make_cb(CBIndex::c_2, all_cores, num_candidates, tile_size, data_format));      // scores
    desc.cbs.push_back(make_cb(CBIndex::c_3, all_cores, 1, tile_size, data_format));   // shift, for compute
    desc.cbs.push_back(make_cb(CBIndex::c_16, all_cores, 2, tile_size, data_format));  // shift, for the writer
    desc.cbs.push_back(make_cb(CBIndex::c_17, all_cores, 2 * num_candidates, tile_size, data_format));  // exp
    desc.cbs.push_back(make_cb(CBIndex::c_18, all_cores, 2, tile_size, data_format));                   // mass
    // The writer's staging: one column-0 tile per site of a group, zeroed once so the
    // padding columns read as zero, never pushed.
    desc.cbs.push_back(make_cb(CBIndex::c_24, all_cores, sites_per_group, tile_size, data_format));

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    // A core finds its items from its own position, so a launch ships no per-core args.
    const std::vector<uint32_t> split_ct_args = {grid.x, num_cores, num_items, num_groups, sites_per_group};

    std::vector<uint32_t> reader_ct_args = {num_candidates, Ht, num_partials};
    reader_ct_args.insert(reader_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*stats.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {num_candidates, Ht, num_sites};
    writer_ct_args.insert(writer_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*exponentials.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*shift.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*mass.buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {
        num_candidates,
        num_partials,
        std::bit_cast<uint32_t>(operation_attributes.inv_hidden_size),
        std::bit_cast<uint32_t>(operation_attributes.eps)};
    compute_ct_args.insert(compute_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    KernelDescriptor reader;
    reader.kernel_source = std::string(kKernelDir) + "reader_attn_res_scores.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct_args);
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args({stats.buffer()});

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "writer_attn_res_scores.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct_args);
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args({exponentials.buffer(), shift.buffer(), mass.buffer()});

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "attn_res_scores.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct_args);
    // The scores and the shift are only ever copied into dest, never fed to the FPU, so
    // they unpack straight to it: through srcA they would be cut to tf32, and the max,
    // the subtraction and the exp would all see the cut values. The partials stay on
    // srcA: they are the hardware-startup operand, and unpacking them to dest returned
    // NaN. The summed statistics and the reciprocal RMS feed the column-broadcast
    // multiply and have to take srcA anyway.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_2] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[CBIndex::c_3] = UnpackToDestMode::UnpackToDestFp32;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx_mode,
    };

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    return desc;
}

}  // namespace ttnn::experimental::prim
