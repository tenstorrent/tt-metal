// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_fused_swiglu_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "moe_fused_swiglu_geometry.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {
namespace {

using namespace tt::tt_metal;
using tt::DataFormat;
namespace geo = geometry;

constexpr const char* KERNEL_ROOT =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels";

std::pair<uint32_t, uint32_t> virtual_core(IDevice* device, uint32_t x, uint32_t y) {
    const auto core = device->worker_core_from_logical_core(CoreCoord{x, y});
    return {core.x, core.y};
}

// Host-side wire encoder for this operation's rotating multicasts. The device
// kernels consume a NOC-ordered rectangle followed by row-major senders.
std::vector<uint32_t> rotating_mcast_args(
    IDevice* device, NOC noc, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    uint32_t xlo = UINT32_MAX;
    uint32_t ylo = UINT32_MAX;
    uint32_t xhi = 0;
    uint32_t yhi = 0;
    std::vector<std::pair<uint32_t, uint32_t>> senders;
    senders.reserve((x1 - x0 + 1) * (y1 - y0 + 1));
    for (uint32_t y = y0; y <= y1; ++y) {
        for (uint32_t x = x0; x <= x1; ++x) {
            const auto v = virtual_core(device, x, y);
            xlo = std::min(xlo, v.first);
            ylo = std::min(ylo, v.second);
            xhi = std::max(xhi, v.first);
            yhi = std::max(yhi, v.second);
            senders.push_back(v);
        }
    }
    std::vector<uint32_t> args;
    args.reserve(4 + 2 * senders.size());
    if (noc == NOC::NOC_1) {
        args.insert(args.end(), {xhi, yhi, xlo, ylo});
    } else {
        args.insert(args.end(), {xlo, ylo, xhi, yhi});
    }
    for (const auto& sender : senders) {
        args.push_back(sender.first);
        args.push_back(sender.second);
    }
    return args;
}

std::array<uint32_t, 4> mcast_rect_args(IDevice* device, NOC noc, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    const auto args = rotating_mcast_args(device, noc, x0, y0, x1, y1);
    return {args[0], args[1], args[2], args[3]};
}

std::array<uint32_t, 5> mcast_compile_time_args(
    uint32_t data_ready_sem, uint32_t consumer_ready_sem, uint32_t receivers, bool handshake) {
    return {receivers == 0 ? 0u : 1u, data_ready_sem, consumer_ready_sem, receivers, handshake ? 1u : 0u};
}

template <typename Range>
void append(std::vector<uint32_t>& destination, const Range& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

DataFormat format_for(
    geo::FormatKey key, DataFormat weight_format, DataFormat output_format, DataFormat activation_format) {
    switch (key) {
        case geo::FormatKey::Bfp8: return DataFormat::Bfp8_b;
        case geo::FormatKey::Bf16: return DataFormat::Float16_b;
        case geo::FormatKey::Weight: return weight_format;
        case geo::FormatKey::Out: return output_format;
        case geo::FormatKey::U32: return DataFormat::UInt32;
        case geo::FormatKey::XIn: return activation_format;
    }
    TT_THROW("moe_fused_swiglu: unknown CB format key");
}

std::vector<uint32_t> make_reader_ct(
    const geo::Blocking& blocking,
    const OperationArguments& operation_arguments,
    bool activations_are_row_major,
    bool phase_alias,
    bool direct_write,
    uint32_t activation_page,
    uint32_t activation_slice,
    uint32_t counts_page,
    uint32_t idx_page,
    uint32_t start_page,
    uint32_t weight_tile,
    uint32_t bfp8_tile,
    uint32_t wg_shard_w,
    uint32_t wd_shard_w) {
    return {
        activations_are_row_major ? 0u : 1u,
        operation_arguments.m_tiles,
        operation_arguments.local_expert_id,
        blocking.emb_t,
        blocking.hid_t,
        blocking.kr_pad,
        blocking.hn_pad,
        blocking.ec_max,
        blocking.wd_ec_max,
        blocking.ec_group_max,
        geo::M_BLOCK,
        blocking.hgroups,
        blocking.kgroups,
        blocking.num_cores,
        geo::SEM_GO,
        geo::SEM_DATA,
        geo::SEM_HSLICE,
        geo::SEM_XSTAGED,
        geo::SEM_H_RDY_BASE,
        geo::SEM_H_FREE,
        geo::SEM_WDSPLIT,
        geo::SEM_HROW_FREE,
        geo::SEM_PHASE_FREE,
        phase_alias,
        geo::H_ROUND_NOC1_MASK,
        geo::SCATTER_ONE_SIGNAL,
        activation_page,
        activation_slice,
        counts_page,
        idx_page,
        weight_tile,
        bfp8_tile,
        geo::MAILBOX_MAGIC,
        blocking.wd_ahead,
        blocking.m_eff_min,
        geo::W_RESIDENT,
        blocking.wd_resident,
        blocking.wd_mrow_rounds && blocking.wd_resident,
        blocking.wd_mgroups,
        geo::WD_MGROUP_MIN_BLOCKS,
        blocking.gu_chunks,
        geo::XPRIO,
        blocking.hack_ahead,
        blocking.depth_h,
        blocking.depth_x,
        blocking.wd_split,
        wg_shard_w,
        wd_shard_w,
        blocking.gather_pages,
        direct_write || operation_arguments.read_x_at_offset,
        operation_arguments.read_x_at_offset,
        start_page,
        geo::CB_X_IN,
        geo::CB_X_TILES,
        geo::CB_X_STAGE,
        geo::CB_W_GATE,
        geo::CB_W_DOWN,
        geo::CB_H,
        geo::CB_H_LOCAL,
        geo::CB_IDX_SCRATCH,
        geo::CB_COUNTS_SCRATCH,
        geo::CB_GATHER_GATE,
        geo::CB_GATHER_UP,
        geo::CB_UP_ACC,
        geo::CB_MAILBOX_COMPUTE,
        geo::CB_MAILBOX_WRITER,
    };
}

std::vector<uint32_t> make_writer_ct(
    const geo::Blocking& blocking,
    bool phase_alias,
    bool direct_write,
    uint32_t output_m_tiles,
    uint32_t weight_tile,
    uint32_t bfp8_tile,
    uint32_t output_tile,
    uint32_t wg_shard_w,
    uint32_t wd_shard_w) {
    return {
        blocking.emb_t,
        blocking.hid_t,
        blocking.kr_pad,
        blocking.hn_pad,
        blocking.ec_max,
        blocking.wd_ec_max,
        blocking.ec_group_max,
        geo::M_BLOCK,
        blocking.hgroups,
        blocking.kgroups,
        blocking.num_cores,
        geo::SEM_GO,
        geo::SEM_DATA,
        geo::SEM_HSLICE,
        geo::SEM_XSTAGED,
        geo::SEM_H_RDY_BASE,
        geo::SEM_H_FREE,
        geo::SEM_WDSPLIT,
        geo::SEM_PHASE_FREE,
        geo::SEM_HROW_FREE,
        phase_alias,
        weight_tile,
        bfp8_tile,
        output_tile,
        geo::MAILBOX_MAGIC,
        blocking.m_eff_min,
        geo::W_RESIDENT,
        blocking.wd_resident,
        blocking.gu_chunks,
        geo::XPRIO,
        blocking.wd_mrow_rounds && blocking.wd_resident,
        blocking.wd_mgroups,
        geo::WD_MGROUP_MIN_BLOCKS,
        blocking.depth_h,
        geo::H_ROUND_NOC1_MASK,
        geo::SCATTER_ONE_SIGNAL,
        blocking.wd_split,
        wg_shard_w,
        wd_shard_w,
        blocking.gather_pages,
        phase_alias ? blocking.phase_cb_alias_pages(output_tile) : 0,
        direct_write,
        output_m_tiles,
        geo::CB_W_UP,
        geo::CB_W_DOWN,
        geo::CB_OUT_TILES,
        geo::CB_GATE_ACC,
        geo::CB_UP_ACC,
        geo::CB_GATHER_GATE,
        geo::CB_GATHER_UP,
        geo::CB_H_SLICE,
        geo::CB_H_LOCAL,
        geo::CB_H,
        geo::CB_MAILBOX_WRITER,
    };
}

std::vector<uint32_t> make_compute_ct(const geo::Blocking& blocking, bool activations_are_row_major) {
    return {
        geo::M_BLOCK,
        blocking.kr_pad,
        blocking.hn_pad,
        blocking.ec_max,
        blocking.wd_ec_max,
        blocking.ec_group_max,
        blocking.hgroups,
        blocking.kgroups,
        blocking.hid_t,
        activations_are_row_major ? 0u : 1u,
        geo::OUT_SUBBLOCK_H_GU,
        blocking.out_subblock_h_dn,
        geo::OUT_SUBBLOCK_H_DN_MAX,
        geo::MAILBOX_MAGIC,
        blocking.m_eff_min,
        blocking.depth_x,
        blocking.hn_block,
        blocking.wd_resident,
        blocking.wd_mrow_rounds && blocking.wd_resident,
        blocking.wd_mgroups,
        geo::WD_MGROUP_MIN_BLOCKS,
        blocking.gu_chunks,
        geo::ELTWISE_BLK,
        geo::DEST_LIMIT,
        blocking.gather_pages,
        geo::CB_X_IN,
        geo::CB_X_TILES,
        geo::CB_X_STAGE,
        geo::CB_MAILBOX_COMPUTE,
        geo::CB_W_GATE,
        geo::CB_W_UP,
        geo::CB_W_DOWN,
        geo::CB_GATE_ACC,
        geo::CB_UP_ACC,
        geo::CB_GATE_SILU,
        geo::CB_H_LOCAL,
        geo::CB_H,
        geo::CB_OUT_INTERM,
        geo::CB_OUT_TILES,
        geo::CB_GATHER_GATE,
        geo::CB_GATHER_UP,
        geo::CB_SLICE_GATE,
        geo::CB_SLICE_UP,
        geo::CB_H_SLICE,
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor create_moe_fused_swiglu_program_descriptor(
    const OperationArguments& operation_arguments, const TensorArguments& tensor_arguments, Tensor& output) {
    ProgramDescriptor descriptor;
    auto* device = tensor_arguments.activations.device();
    const uint32_t hgroups = operation_arguments.grid_x;
    const uint32_t kgroups = operation_arguments.grid_y;
    const uint32_t num_cores = hgroups * kgroups;
    const CoreRangeSet all_cores{CoreRange({0, 0}, {hgroups - 1, kgroups - 1})};

    const uint32_t emb = tensor_arguments.activations.logical_shape()[-1];
    const uint32_t hidden = tensor_arguments.w_gate.logical_shape()[-1];
    const bool activations_are_row_major = tensor_arguments.activations.layout() == Layout::ROW_MAJOR;
    const DataFormat weight_format = datatype_to_dataformat_converter(tensor_arguments.w_gate.dtype());
    const DataFormat output_format = datatype_to_dataformat_converter(output.dtype());
    const DataFormat activation_format = datatype_to_dataformat_converter(tensor_arguments.activations.dtype());
    const uint32_t weight_tile = tile_size(weight_format);
    const uint32_t bfp8_tile = tile_size(DataFormat::Bfp8_b);
    const uint32_t bf16_tile = tile_size(DataFormat::Float16_b);
    const uint32_t output_tile = tile_size(output_format);
    const uint32_t kr_pad = ((emb / geo::TILE) + kgroups - 1) / kgroups;
    const uint32_t activation_slice =
        activations_are_row_major ? kr_pad * geo::TILE * tensor_arguments.activations.element_size() : bfp8_tile;
    const uint32_t l1_max = hal::get_max_worker_l1_unreserved_size();
    TT_FATAL(l1_max > geo::L1_CB_RESERVE, "moe_fused_swiglu: invalid worker L1 budget");

    geo::Blocking blocking(
        hgroups,
        kgroups,
        emb,
        hidden,
        operation_arguments.m_tiles,
        weight_tile,
        bfp8_tile,
        bf16_tile,
        activation_slice,
        l1_max - geo::L1_CB_RESERVE,
        output_tile,
        /*enable_phase_alias=*/true,
        activations_are_row_major);

    const bool direct_write = tensor_arguments.expert_region_offsets.has_value();
    const Tensor& start_tensor = direct_write ? *tensor_arguments.expert_region_offsets : tensor_arguments.counts;
    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t idx_page =
        std::max<uint32_t>(tensor_arguments.global_expert_idx_table.buffer()->aligned_page_size(), dram_alignment);
    const uint32_t start_page = std::max<uint32_t>(start_tensor.buffer()->aligned_page_size(), dram_alignment);
    const uint32_t counts_page =
        std::max(std::max<uint32_t>(tensor_arguments.counts.buffer()->aligned_page_size(), dram_alignment), start_page);
    const bool phase_alias = blocking.phase_cb_alias(output_tile);
    const uint64_t l1_need = blocking.l1_bytes(activations_are_row_major, output_tile, true);
    TT_FATAL(
        l1_need <= blocking.l1_budget,
        "moe_fused_swiglu: needs {} bytes of CB L1 but budget is {} ({})",
        l1_need,
        blocking.l1_budget,
        blocking.describe());

    for (const auto& allocation : blocking.cb_allocations(
             activations_are_row_major, output_tile, idx_page, counts_page, /*enable_phase_alias=*/true)) {
        CBDescriptor cb_descriptor{
            .total_size = allocation.total_size,
            .core_ranges = all_cores,
        };
        for (const auto& view : allocation.views) {
            cb_descriptor.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(view.index),
                .data_format = format_for(view.format, weight_format, output_format, activation_format),
                .page_size = view.page_size,
            });
        }
        descriptor.cbs.push_back(std::move(cb_descriptor));
    }

    // The device kernels use fixed IDs because the mcast wire carries them as
    // compile-time arguments. Descriptor IDs are explicit, so preserve that order.
    for (uint32_t expected = 0; expected < geo::SEM_COUNT; ++expected) {
        descriptor.semaphores.push_back(SemaphoreDescriptor{
            .id = expected,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    const auto x_mcast_ct =
        mcast_compile_time_args(geo::SEM_X_BASE, geo::SEM_X_BASE + 1, hgroups - 1, /*handshake=*/true);
    const auto h_mcast_ct =
        mcast_compile_time_args(geo::SEM_H_BASE, geo::SEM_H_BASE + 1, num_cores - 1, /*handshake=*/true);
    const auto h_mcast_noc1_args = mcast_rect_args(device, NOC::NOC_1, 0, 0, hgroups - 1, kgroups - 1);

    std::vector<std::array<uint32_t, 4>> h_group_rect_args(
        (kgroups + blocking.mgroup_rows - 1) / blocking.mgroup_rows, {0, 0, 0, 0});
    if (blocking.wd_mgroups) {
        for (uint32_t group = 0; group < h_group_rect_args.size(); ++group) {
            const uint32_t y0 = group * blocking.mgroup_rows;
            h_group_rect_args[group] = mcast_rect_args(
                device, NOC::NOC_0, 0, y0, hgroups - 1, std::min(y0 + blocking.mgroup_rows - 1, kgroups - 1));
        }
    }

    const uint32_t wg = [&]() {
        const uint32_t gate_width = geo::nd_shard_n_tiles(tensor_arguments.w_gate);
        const uint32_t up_width = geo::nd_shard_n_tiles(tensor_arguments.w_up);
        return gate_width == up_width ? gate_width : 0u;
    }();
    const uint32_t wd = geo::nd_shard_n_tiles(tensor_arguments.w_down);

    auto reader_ct = make_reader_ct(
        blocking,
        operation_arguments,
        activations_are_row_major,
        phase_alias,
        direct_write,
        tensor_arguments.activations.buffer()->page_size(),
        activation_slice,
        std::max<uint32_t>(tensor_arguments.counts.buffer()->aligned_page_size(), dram_alignment),
        idx_page,
        start_page,
        weight_tile,
        bfp8_tile,
        wg,
        wd);
    append(reader_ct, x_mcast_ct);
    append(reader_ct, h_mcast_ct);
    for (auto* buffer :
         {tensor_arguments.activations.buffer(),
          tensor_arguments.w_gate.buffer(),
          tensor_arguments.w_down.buffer(),
          tensor_arguments.counts.buffer(),
          tensor_arguments.global_expert_idx_table.buffer(),
          start_tensor.buffer()}) {
        TensorAccessorArgs(buffer).append_to(reader_ct);
    }

    auto writer_ct = make_writer_ct(
        blocking,
        phase_alias,
        direct_write,
        output.padded_shape()[-2] / geo::TILE,
        weight_tile,
        bfp8_tile,
        output_tile,
        wg,
        wd);
    for (auto* buffer : {tensor_arguments.w_up.buffer(), output.buffer(), tensor_arguments.w_down.buffer()}) {
        TensorAccessorArgs(buffer).append_to(writer_ct);
    }
    auto compute_ct = make_compute_ct(blocking, activations_are_row_major);

    KernelDescriptor reader_descriptor{
        .kernel_source = std::string(KERNEL_ROOT) + "/moe_fused_swiglu_reader.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(reader_ct),
        .defines = {{"H_MCAST_POSTED", geo::H_MCAST_POSTED ? "1" : "0"}},
        .config = ReaderConfigDescriptor{},
    };
    KernelDescriptor writer_descriptor{
        .kernel_source = std::string(KERNEL_ROOT) + "/moe_fused_swiglu_writer.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(writer_ct),
        .defines = {{"H_MCAST_POSTED", geo::H_MCAST_POSTED ? "1" : "0"}},
        .config = WriterConfigDescriptor{},
    };

    const auto& compute_config = *operation_arguments.compute_kernel_config;
    KernelDescriptor compute_descriptor{
        .kernel_source = std::string(KERNEL_ROOT) + "/moe_fused_swiglu_compute.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(compute_ct),
        .config =
            ComputeConfigDescriptor{
                .math_fidelity = compute_config.math_fidelity,
                .fp32_dest_acc_en = compute_config.fp32_dest_acc_en,
                .dst_full_sync_en = compute_config.dst_full_sync_en,
                .bfp8_pack_precise = true,
                .math_approx_mode = compute_config.math_approx_mode,
            },
    };
    if (operation_arguments.activation == RoutedExpertActivation::SituGlu) {
        compute_descriptor.defines.emplace_back("SITU_GLU", "1");
    }

    for (uint32_t y = 0; y < kgroups; ++y) {
        for (uint32_t x = 0; x < hgroups; ++x) {
            const CoreCoord core{x, y};
            const uint32_t index = y * hgroups + x;
            const uint32_t group_index = (y % blocking.mgroup_rows) * hgroups + x;
            KernelDescriptor::RTArgList reader_args;
            const auto x_mcast_args = rotating_mcast_args(device, NOC::NOC_0, 0, y, hgroups - 1, y);
            const auto h_mcast_args = rotating_mcast_args(device, NOC::NOC_0, 0, 0, hgroups - 1, kgroups - 1);
            reader_args.reserve(
                17 + 2 * kgroups + x_mcast_args.size() + h_mcast_args.size() +
                h_group_rect_args[y / blocking.mgroup_rows].size());
            reader_args.push_back(0u);  // reserved runtime slot
            reader_args.push_back(tensor_arguments.activations.buffer());
            reader_args.push_back(tensor_arguments.w_gate.buffer());
            reader_args.push_back(tensor_arguments.w_down.buffer());
            reader_args.push_back(tensor_arguments.counts.buffer());
            reader_args.push_back(tensor_arguments.global_expert_idx_table.buffer());
            reader_args.push_back(blocking.kr_sizes[y]);
            reader_args.push_back(blocking.kr_starts[y]);
            reader_args.push_back(blocking.hn_starts[x]);
            reader_args.push_back(blocking.hn_sizes[x]);
            reader_args.push_back(blocking.ec_sizes[index]);
            reader_args.push_back(blocking.ec_starts[index]);
            reader_args.push_back(blocking.ec_group_sizes[group_index]);
            reader_args.push_back(blocking.ec_group_starts[group_index]);
            reader_args.push_back(x);
            reader_args.push_back(y);
            reader_args.push_back(start_tensor.buffer());
            for (uint32_t row = 0; row < kgroups; ++row) {
                const auto [vx, vy] = virtual_core(device, x, row);
                reader_args.push_back(vx);
                reader_args.push_back(vy);
            }
            reader_args.append(x_mcast_args);
            reader_args.append(h_mcast_args);
            for (const uint32_t arg : h_group_rect_args[y / blocking.mgroup_rows]) {
                reader_args.push_back(arg);
            }
            reader_descriptor.emplace_runtime_args(core, reader_args);

            KernelDescriptor::RTArgList writer_args;
            writer_args.reserve(17 + 2 * kgroups + 4);
            writer_args.push_back(0u);  // reserved runtime slot
            writer_args.push_back(tensor_arguments.w_up.buffer());
            writer_args.push_back(output.buffer());
            writer_args.push_back(tensor_arguments.w_down.buffer());
            writer_args.push_back(blocking.kr_sizes[y]);
            writer_args.push_back(blocking.kr_starts[y]);
            writer_args.push_back(blocking.hn_starts[x]);
            writer_args.push_back(blocking.hn_sizes[x]);
            writer_args.push_back(blocking.ec_sizes[index]);
            writer_args.push_back(blocking.ec_starts[index]);
            writer_args.push_back(blocking.ec_group_sizes[group_index]);
            writer_args.push_back(blocking.ec_group_starts[group_index]);
            writer_args.push_back(x);
            writer_args.push_back(y);
            writer_args.push_back(x % kgroups);
            const auto [diag_x, diag_y] = virtual_core(device, y, y);
            writer_args.push_back(diag_x);
            writer_args.push_back(diag_y);
            for (uint32_t row = 0; row < kgroups; ++row) {
                const auto [vx, vy] = virtual_core(device, x, row);
                writer_args.push_back(vx);
                writer_args.push_back(vy);
            }
            for (uint32_t arg = 0; arg < 4; ++arg) {
                writer_args.push_back(h_mcast_noc1_args[arg]);
            }
            writer_descriptor.emplace_runtime_args(core, writer_args);

            compute_descriptor.emplace_runtime_args(
                core,
                {0,
                 blocking.kr_sizes[y],
                 blocking.hn_sizes[x],
                 blocking.ec_sizes[index],
                 blocking.ec_group_sizes[group_index],
                 x,
                 y});
        }
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    return descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu
