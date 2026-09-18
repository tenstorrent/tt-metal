// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The merged op's one program factory.
//
// Both routed-expert implementations live here, each in its own namespace and each still a
// recognizable copy of the op it was carried from -- that correspondence is what lets either be
// diffed against its upstream when that upstream moves. What is unified is the INTERFACE: a
// caller asks for one descriptor and gets one, and how the two halves are built, converted and
// folded is settled here rather than at the call site.
//
// Order matters: each half is defined before the glue that converts into it and folds it.

#include "hybrid_program_factory.hpp"

// Each half's own types, which its body below is written against. The headers that used to carry
// these were the per-half program factory headers; the interface they declared is now one call.

#include <algorithm>
#include <string>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_geometry.hpp"
#include <initializer_list>
#include <map>
#include <tuple>
#include <unordered_map>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operation.hpp"
#include <limits>
#include "hybrid_half_merge.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused {
namespace {
constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

bool is_dram_interleaved(const Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

bool is_dram_nd_sharded(const Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM && memory_config.created_with_nd_shard_spec();
}

void validate_device_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "moe_fused_swiglu: {} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "moe_fused_swiglu: {} must have an allocated device buffer", name);
}

void validate_aux_shape(const Tensor& tensor, const char* name) {
    const auto& shape = tensor.logical_shape();
    TT_FATAL(
        shape.rank() == 1 || (shape.rank() == 2 && shape[0] == 1),
        "moe_fused_swiglu: {} must be 1D or shape (1, N), got {}",
        name,
        shape);
    TT_FATAL(shape[-1] > 0, "moe_fused_swiglu: {} must not be empty", name);
}

using namespace tt::tt_metal;
using tt::DataFormat;
// The fused half's geometry is used as it ships, not copied: this op carries a modified BODY of
// that implementation, but its blocking maths is the same maths, and a second copy is a second
// thing to keep in step with upstream.
namespace geo = ::ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry;

bool stage_profile_enabled() {
    const char* value = std::getenv("MOE_FUSED_SWIGLU_STAGE_PROFILE");
    if (value == nullptr) {
        return false;
    }
    const std::string_view setting(value);
    return setting != "0" && setting != "false" && setting != "False" && setting != "off";
}

constexpr const char* KERNEL_ROOT =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/kernels";

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
    uint32_t num_global_experts,
    uint32_t start_page,
    uint32_t row_capacity,
    uint32_t weight_tile,
    uint32_t bfp8_tile,
    uint32_t wg_shard_w,
    uint32_t wd_shard_w) {
    return {
        activations_are_row_major ? 0u : 1u,
        operation_arguments.m_tiles,
        operation_arguments.experts_per_chip,
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
        num_global_experts,
        weight_tile,
        bfp8_tile,
        geo::MAILBOX_MAGIC,
        blocking.wd_ahead,
        blocking.m_eff_min,
        geo::W_RESIDENT,
        blocking.wd_resident,
        blocking.wd_mrow_rounds && blocking.wd_resident,
        blocking.wd_mgroups,
        blocking.mgroup_rows,
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
        row_capacity,
        operation_arguments.min_active_tokens,
        operation_arguments.max_active_tokens,
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
    uint32_t experts_per_chip,
    bool phase_alias,
    bool direct_write,
    uint32_t output_m_tiles,
    uint32_t weight_tile,
    uint32_t bfp8_tile,
    uint32_t output_tile,
    uint32_t wg_shard_w,
    uint32_t wd_shard_w) {
    return {
        experts_per_chip,
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
        blocking.mgroup_rows,
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

std::vector<uint32_t> make_compute_ct(
    const geo::Blocking& blocking, uint32_t experts_per_chip, bool activations_are_row_major) {
    return {
        experts_per_chip,
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
        blocking.mgroup_rows,
        geo::WD_MGROUP_MIN_BLOCKS,
        blocking.gu_chunks,
        geo::ELTWISE_BLK,
        geo::DEST_LIMIT,
        blocking.gather_pages,
        blocking.depth_h,
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
        geo::CB_GATE_BIAS,
        geo::CB_UP_BIAS,
        geo::CB_DOWN_BIAS,
    };
}

// A gate/up weight run is ONE NoC transaction only while it stays inside a single shard. The reader
// and the writer walk each core's hidden slice in gu_chunk_w-wide chunks (read_weight_chunk in
// kernels/moe_fused_swiglu_dataflow.hpp), so a shard width that does not tile that walk splits every
// crossing chunk into two transactions -- correct, and half the coalescing an ND-sharded placement
// exists to buy. Counted against the real walk rather than a divisibility rule, because hn_starts is
// ragged whenever the hidden split is balanced.
//
// A warning, not a fatal: the width is the CALLER's, it is pinned to whatever grid the weights were
// built for, and this op takes a core_grid per call -- a mismatch here is slow, never wrong.
void warn_if_gate_up_shard_splits_runs(const geo::Blocking& blocking, uint32_t shard_w) {
    if (shard_w == 0) {
        return;
    }
    uint32_t split = 0;
    uint32_t total = 0;
    for (uint32_t x = 0; x < blocking.hgroups; ++x) {
        for (uint32_t chunk = 0; chunk < blocking.gu_chunks; ++chunk) {
            const uint32_t col0 = chunk * blocking.gu_chunk_w;
            if (col0 >= blocking.hn_sizes[x]) {
                continue;
            }
            const uint32_t width = std::min(blocking.gu_chunk_w, blocking.hn_sizes[x] - col0);
            const uint32_t start = blocking.hn_starts[x] + col0;
            ++total;
            split += static_cast<uint32_t>(start / shard_w != (start + width - 1) / shard_w);
        }
    }
    if (split != 0) {
        log_warning(
            tt::LogOp,
            "moe_fused_swiglu: gate/up DRAM ND shard width {} tiles does not tile this grid's chunk "
            "walk (hn_pad {}, chunk {} tiles): {} of {} weight reads per K-row cross a shard edge "
            "and issue as two transactions",
            shard_w,
            blocking.hn_pad,
            blocking.gu_chunk_w,
            split,
            total);
    }
}

}  // namespace

void append_to_descriptor(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const OperationArguments& operation_arguments,
    const TensorArguments& tensor_arguments,
    Tensor& tensor_return_value) {
    auto* device = tensor_arguments.activations.device();
    const uint32_t hgroups = operation_arguments.grid_x;
    const uint32_t kgroups = operation_arguments.grid_y;
    // Rectangle-relative coordinates everywhere below; these shift them onto the grid.
    const uint32_t OX = operation_arguments.origin_x;
    const uint32_t OY = operation_arguments.origin_y;
    const uint32_t num_cores = hgroups * kgroups;
    const CoreRangeSet all_cores{CoreRange({OX, OY}, {OX + hgroups - 1, OY + kgroups - 1})};

    const uint32_t emb = tensor_arguments.activations.logical_shape()[-1];
    const uint32_t hidden = tensor_arguments.w_gates[0].logical_shape()[-1];
    const bool activations_are_row_major = tensor_arguments.activations.layout() == Layout::ROW_MAJOR;
    const DataFormat weight_format = datatype_to_dataformat_converter(tensor_arguments.w_gates[0].dtype());
    const DataFormat output_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const DataFormat activation_format = datatype_to_dataformat_converter(tensor_arguments.activations.dtype());
    const uint32_t weight_tile = tile_size(weight_format);
    const uint32_t bfp8_tile = tile_size(DataFormat::Bfp8_b);
    const uint32_t bf16_tile = tile_size(DataFormat::Float16_b);
    const uint32_t output_tile = tile_size(output_format);
    const uint32_t kr_pad = ((emb / geo::TILE) + kgroups - 1) / kgroups;
    const uint32_t activation_slice =
        activations_are_row_major ? kr_pad * geo::TILE * tensor_arguments.activations.element_size() : bfp8_tile;
    // The standalone op budgets against hal::get_max_worker_l1_unreserved_size() less a fixed
    // L1_CB_RESERVE -- a static ceiling that assumes the device was opened with the largest possible
    // worker L1, and so breaks outright once the allocator base rises. Here the shared arena IS the
    // budget: it is what a core actually owns, it already excludes the kernel-config ring, and the
    // merge lays these buffers into it. Budgeting against anything else lets the blocking pick a
    // size the arena cannot hold.
    const uint32_t l1_budget = hybrid_l1_arena_bytes(device);
    TT_FATAL(l1_budget > 0, "moe_fused_swiglu: the shared L1 arena is empty");

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
        l1_budget,
        output_tile,
        /*enable_phase_alias_=*/true,
        activations_are_row_major,
        operation_arguments.fuse_bias);

    const bool direct_write = tensor_arguments.expert_region_offsets.has_value();
    const Tensor& start_tensor = direct_write ? *tensor_arguments.expert_region_offsets : tensor_arguments.counts;
    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t idx_page =
        std::max<uint32_t>(tensor_arguments.global_expert_idx_table.buffer()->aligned_page_size(), dram_alignment);
    const uint32_t start_page = std::max<uint32_t>(start_tensor.buffer()->aligned_page_size(), dram_alignment);
    // aligned_page_size() is a 64-bit DeviceAddr, and a narrowing conversion inside a braced-init
    // list is ill-formed, so the cast is what lets the three bounds share one std::max.
    const uint32_t counts_page = std::max<uint32_t>(
        {static_cast<uint32_t>(tensor_arguments.counts.buffer()->aligned_page_size()), dram_alignment, start_page});
    const bool phase_alias = blocking.phase_cb_alias(output_tile);
    const uint64_t l1_need = blocking.l1_bytes(activations_are_row_major, output_tile, true);
    TT_FATAL(
        l1_need <= blocking.l1_budget,
        "moe_fused_swiglu: needs {} bytes of CB L1 but budget is {} ({})",
        l1_need,
        blocking.l1_budget,
        blocking.describe());

    for (const auto& allocation : blocking.cb_allocations(
             activations_are_row_major, output_tile, idx_page, counts_page, /*aliases_enabled=*/true)) {
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
    const auto h_mcast_noc1_args = mcast_rect_args(device, NOC::NOC_1, OX, OY, OX + hgroups - 1, OY + kgroups - 1);

    std::vector<std::array<uint32_t, 4>> h_group_rect_args(
        (kgroups + blocking.mgroup_rows - 1) / blocking.mgroup_rows, {0, 0, 0, 0});
    if (blocking.wd_mgroups) {
        for (uint32_t group = 0; group < h_group_rect_args.size(); ++group) {
            const uint32_t y0 = group * blocking.mgroup_rows;
            h_group_rect_args[group] = mcast_rect_args(
                device,
                NOC::NOC_0,
                OX,
                OY + y0,
                OX + hgroups - 1,
                OY + std::min(y0 + blocking.mgroup_rows - 1, kgroups - 1));
        }
    }

    // Every expert shares expert 0's layout (the device operation validates it), so ONE accessor
    // layout descriptor per role serves the whole loop and only the base address varies.
    const uint32_t wg = [&]() {
        const uint32_t gate_width = geo::nd_shard_n_tiles(tensor_arguments.w_gates[0]);
        const uint32_t up_width = geo::nd_shard_n_tiles(tensor_arguments.w_ups[0]);
        // Both tensors are read through ONE compile-time width, so widths that disagree leave no
        // correct value to take: a run sized to either one crosses the other's real shard boundary
        // and would issue a single transaction spanning two banks.
        TT_FATAL(
            gate_width == up_width,
            "moe_fused_swiglu: w_gate and w_up must share a weight placement; DRAM ND shard widths "
            "are {} and {} tiles",
            gate_width,
            up_width);
        return gate_width;
    }();
    // W_down carries no such check: its shard is deliberately WIDER than the ec slice one core
    // reads, which costs nothing (a slice inside a shard is still one transaction) and keeps the
    // shards-per-row count coprime with the bank count, which is what rotates a core's K-rows
    // across all DRAM banks.
    const uint32_t wd = geo::nd_shard_n_tiles(tensor_arguments.w_downs[0]);
    warn_if_gate_up_shard_splits_runs(blocking, wg);
    const uint32_t experts_per_chip = operation_arguments.experts_per_chip;

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
        static_cast<uint32_t>(tensor_arguments.counts.logical_shape()[-1]),
        start_page,
        static_cast<uint32_t>(tensor_arguments.activations.padded_shape()[-2]),
        weight_tile,
        bfp8_tile,
        wg,
        wd);
    append(reader_ct, x_mcast_ct);
    append(reader_ct, h_mcast_ct);
    for (auto* buffer :
         {tensor_arguments.activations.buffer(),
          tensor_arguments.w_gates[0].buffer(),
          tensor_arguments.w_downs[0].buffer(),
          tensor_arguments.counts.buffer(),
          tensor_arguments.global_expert_idx_table.buffer(),
          start_tensor.buffer()}) {
        TensorAccessorArgs(buffer).append_to(reader_ct);
    }
    // Bias CB ids then the three bias accessors, appended last so every offset the reader already
    // derives is unmoved. Only present when fuse_bias, which is in the program cache key.
    if (operation_arguments.fuse_bias) {
        reader_ct.push_back(geo::CB_GATE_BIAS);
        reader_ct.push_back(geo::CB_UP_BIAS);
        reader_ct.push_back(geo::CB_DOWN_BIAS);
        for (auto* buffer :
             {tensor_arguments.gate_biases[0].buffer(),
              tensor_arguments.up_biases[0].buffer(),
              tensor_arguments.down_biases[0].buffer()}) {
            TensorAccessorArgs(buffer).append_to(reader_ct);
        }
    }

    auto writer_ct = make_writer_ct(
        blocking,
        experts_per_chip,
        phase_alias,
        direct_write,
        tensor_return_value.padded_shape()[-2] / geo::TILE,
        weight_tile,
        bfp8_tile,
        output_tile,
        wg,
        wd);
    for (auto* buffer :
         {tensor_arguments.w_ups[0].buffer(), tensor_return_value.buffer(), tensor_arguments.w_downs[0].buffer()}) {
        TensorAccessorArgs(buffer).append_to(writer_ct);
    }
    auto compute_ct = make_compute_ct(blocking, experts_per_chip, activations_are_row_major);

    KernelDescriptor::Defines dataflow_defines{{"H_MCAST_POSTED", geo::H_MCAST_POSTED ? "1" : "0"}};
    if (operation_arguments.fuse_bias) {
        dataflow_defines.emplace_back("FUSE_BIAS", "1");
    }
    KernelDescriptor::Defines compute_defines;
    if (stage_profile_enabled()) {
        dataflow_defines.emplace_back("MOE_FUSED_SWIGLU_STAGE_PROFILE", "1");
        compute_defines.emplace_back("MOE_FUSED_SWIGLU_STAGE_PROFILE", "1");
    }

    KernelDescriptor reader_descriptor{
        .kernel_source = std::string(KERNEL_ROOT) + "/moe_fused_swiglu_reader.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(reader_ct),
        .defines = dataflow_defines,
        .config = ReaderConfigDescriptor{},
    };
    KernelDescriptor writer_descriptor{
        .kernel_source = std::string(KERNEL_ROOT) + "/moe_fused_swiglu_writer.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(writer_ct),
        .defines = std::move(dataflow_defines),
        .config = WriterConfigDescriptor{},
    };

    const auto& compute_config = *operation_arguments.compute_kernel_config;
    KernelDescriptor compute_descriptor{
        .kernel_source = std::string(KERNEL_ROOT) + "/moe_fused_swiglu_compute.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(compute_ct),
        .defines = std::move(compute_defines),
        .config =
            ComputeConfigDescriptor{
                .math_fidelity = compute_config.math_fidelity,
                .fp32_dest_acc_en = compute_config.fp32_dest_acc_en,
                .dst_full_sync_en = compute_config.dst_full_sync_en,
                .bfp8_pack_precise = true,
                .math_approx_mode = compute_config.math_approx_mode,
            },
    };
    // SwiGlu-OAI's SFPU helper is templated on the dst-accumulator mode, which only the host knows;
    // SiTU-GLU reads DST_ACCUM_MODE itself and ignores this.
    compute_descriptor.defines.emplace_back("FP32_DEST_ACC_EN", compute_config.fp32_dest_acc_en ? "1" : "0");
    // Exactly one variant define, so each activation caches as its own program. The kernel #errors
    // if both ever arrive.
    if (operation_arguments.fuse_bias) {
        compute_descriptor.defines.emplace_back("FUSE_BIAS", "1");
    }
    if (operation_arguments.activation == RoutedExpertActivation::SituGlu) {
        compute_descriptor.defines.emplace_back("SITU_GLU", "1");
    } else if (operation_arguments.activation == RoutedExpertActivation::SwiGluOai) {
        compute_descriptor.defines.emplace_back("SWIGLU_OAI", "1");
    }

    for (uint32_t y = 0; y < kgroups; ++y) {
        for (uint32_t x = 0; x < hgroups; ++x) {
            const CoreCoord core{OX + x, OY + y};
            const uint32_t index = y * hgroups + x;
            const uint32_t group_index = (y % blocking.mgroup_rows) * hgroups + x;
            KernelDescriptor::RTArgList reader_args;
            const auto x_mcast_args = rotating_mcast_args(device, NOC::NOC_0, OX, OY + y, OX + hgroups - 1, OY + y);
            const auto h_mcast_args =
                rotating_mcast_args(device, NOC::NOC_0, OX, OY, OX + hgroups - 1, OY + kgroups - 1);
            reader_args.reserve(
                17 + 2 * kgroups + x_mcast_args.size() + h_mcast_args.size() +
                h_group_rect_args[y / blocking.mgroup_rows].size() + 2u * experts_per_chip);
            reader_args.push_back(0u);  // reserved runtime slot
            reader_args.push_back(tensor_arguments.activations.buffer());
            reader_args.push_back(tensor_arguments.w_gates[0].buffer());
            reader_args.push_back(tensor_arguments.w_downs[0].buffer());
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
                const auto [vx, vy] = virtual_core(device, OX + x, OY + row);
                reader_args.push_back(vx);
                reader_args.push_back(vy);
            }
            reader_args.append(x_mcast_args);
            reader_args.append(h_mcast_args);
            for (const uint32_t arg : h_group_rect_args[y / blocking.mgroup_rows]) {
                reader_args.push_back(arg);
            }
            // Per-expert weight bases, role-major, at the END of the list: every earlier offset the
            // kernels derive from HGROUPS/KGROUPS stays where it was, so only one new constexpr
            // offset per kernel tracks this table.
            for (const auto& w_gate : tensor_arguments.w_gates) {
                reader_args.push_back(w_gate.buffer());
            }
            for (const auto& w_down : tensor_arguments.w_downs) {
                reader_args.push_back(w_down.buffer());
            }
            // Per-expert bias bases, role-major, after the weight table for the same reason it sits
            // last: one new constexpr offset in the reader and nothing above it moves.
            if (operation_arguments.fuse_bias) {
                for (const auto* list :
                     {&tensor_arguments.gate_biases, &tensor_arguments.up_biases, &tensor_arguments.down_biases}) {
                    for (const auto& bias : *list) {
                        reader_args.push_back(bias.buffer());
                    }
                }
            }
            reader_descriptor.emplace_runtime_args(core, reader_args);

            KernelDescriptor::RTArgList writer_args;
            writer_args.reserve(17 + 2 * kgroups + 4 + 2u * experts_per_chip);
            writer_args.push_back(0u);  // reserved runtime slot
            writer_args.push_back(tensor_arguments.w_ups[0].buffer());
            writer_args.push_back(tensor_return_value.buffer());
            writer_args.push_back(tensor_arguments.w_downs[0].buffer());
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
            const auto [diag_x, diag_y] = virtual_core(device, OX + y, OY + y);
            writer_args.push_back(diag_x);
            writer_args.push_back(diag_y);
            for (uint32_t row = 0; row < kgroups; ++row) {
                const auto [vx, vy] = virtual_core(device, OX + x, OY + row);
                writer_args.push_back(vx);
                writer_args.push_back(vy);
            }
            for (uint32_t arg = 0; arg < 4; ++arg) {
                writer_args.push_back(h_mcast_noc1_args[arg]);
            }
            for (const auto& w_up : tensor_arguments.w_ups) {
                writer_args.push_back(w_up.buffer());
            }
            for (const auto& w_down : tensor_arguments.w_downs) {
                writer_args.push_back(w_down.buffer());
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
}

void validate(const OperationArguments& operation_arguments, const TensorArguments& tensor_arguments) {
    TT_FATAL(
        tensor_arguments.w_gates.size() == operation_arguments.experts_per_chip &&
            tensor_arguments.w_ups.size() == operation_arguments.experts_per_chip &&
            tensor_arguments.w_downs.size() == operation_arguments.experts_per_chip,
        "moe_fused_swiglu: weight lists must hold experts_per_chip ({}) entries (got {}, {}, {})",
        operation_arguments.experts_per_chip,
        tensor_arguments.w_gates.size(),
        tensor_arguments.w_ups.size(),
        tensor_arguments.w_downs.size());
    std::vector<std::pair<std::string, const Tensor*>> device_tensors{
        {"activations", &tensor_arguments.activations},
        {"counts", &tensor_arguments.counts},
        {"global_expert_idx_table", &tensor_arguments.global_expert_idx_table}};
    for (uint32_t e = 0; e < operation_arguments.experts_per_chip; ++e) {
        device_tensors.emplace_back(fmt::format("w_gate[{}]", e), &tensor_arguments.w_gates[e]);
        device_tensors.emplace_back(fmt::format("w_up[{}]", e), &tensor_arguments.w_ups[e]);
        device_tensors.emplace_back(fmt::format("w_down[{}]", e), &tensor_arguments.w_downs[e]);
    }
    for (const auto& [name_str, tensor_ptr] : device_tensors) {
        const char* name = name_str.c_str();
        const Tensor& tensor = *tensor_ptr;
        validate_device_tensor(tensor, name);
        TT_FATAL(
            tensor.device() == tensor_arguments.activations.device(),
            "moe_fused_swiglu: {} must be on the activations device",
            name);
    }

    const auto& activation_shape = tensor_arguments.activations.logical_shape();
    TT_FATAL(
        activation_shape.rank() == 2 || activation_shape.rank() == 4,
        "moe_fused_swiglu: activations must have rank 2 or 4, got {}",
        activation_shape.rank());
    if (activation_shape.rank() == 4) {
        TT_FATAL(
            activation_shape[0] == 1 && activation_shape[1] == 1,
            "moe_fused_swiglu: rank-4 activations leading dimensions must be (1, 1), got ({}, {})",
            activation_shape[0],
            activation_shape[1]);
    }
    TT_FATAL(
        activation_shape[-2] % TILE == 0,
        "moe_fused_swiglu: activation capacity ({}) must be tile-aligned",
        activation_shape[-2]);
    const bool activations_are_row_major = tensor_arguments.activations.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const bool activations_are_tiled = tensor_arguments.activations.layout() == tt::tt_metal::Layout::TILE;
    TT_FATAL(
        (activations_are_row_major && tensor_arguments.activations.dtype() == tt::tt_metal::DataType::BFLOAT16) ||
            (activations_are_tiled && tensor_arguments.activations.dtype() == tt::tt_metal::DataType::BFLOAT8_B),
        "moe_fused_swiglu: activations must be BFLOAT16 ROW_MAJOR or BFLOAT8_B TILE "
        "(got dtype {}, layout {})",
        tensor_arguments.activations.dtype(),
        tensor_arguments.activations.layout());
    TT_FATAL(
        is_dram_interleaved(tensor_arguments.activations), "moe_fused_swiglu: activations must be DRAM interleaved");

    const auto& gate_shape = tensor_arguments.w_gates[0].logical_shape();
    const auto& up_shape = tensor_arguments.w_ups[0].logical_shape();
    const auto& down_shape = tensor_arguments.w_downs[0].logical_shape();
    std::vector<std::pair<std::string, const Tensor*>> weights;
    for (uint32_t e = 0; e < operation_arguments.experts_per_chip; ++e) {
        weights.emplace_back(fmt::format("w_gate[{}]", e), &tensor_arguments.w_gates[e]);
        weights.emplace_back(fmt::format("w_up[{}]", e), &tensor_arguments.w_ups[e]);
        weights.emplace_back(fmt::format("w_down[{}]", e), &tensor_arguments.w_downs[e]);
    }
    for (const auto& [name_str, weight_ptr] : weights) {
        const char* name = name_str.c_str();
        const Tensor& weight = *weight_ptr;
        TT_FATAL(weight.logical_shape().rank() == 2, "moe_fused_swiglu: {} must have rank 2", name);
        TT_FATAL(weight.layout() == tt::tt_metal::Layout::TILE, "moe_fused_swiglu: {} must be TILE layout", name);
        TT_FATAL(
            weight.dtype() == tt::tt_metal::DataType::BFLOAT4_B ||
                weight.dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
                weight.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "moe_fused_swiglu: {} dtype must be BFLOAT4_B, BFLOAT8_B, or BFLOAT16 (got {})",
            name,
            weight.dtype());
        TT_FATAL(
            is_dram_interleaved(weight) || is_dram_nd_sharded(weight),
            "moe_fused_swiglu: {} must be DRAM interleaved or DRAM ND-sharded",
            name);
    }
    // One program serves every expert, so a single accessor layout descriptor per role must fit
    // them all: shape and dtype are pinned to expert 0 and only the base address varies.
    const auto weights_dtype = tensor_arguments.w_gates[0].dtype();
    for (const auto& [name_str, weight_ptr] : weights) {
        TT_FATAL(
            weight_ptr->dtype() == weights_dtype,
            "moe_fused_swiglu: {} dtype {} must match w_gate[0] ({})",
            name_str,
            weight_ptr->dtype(),
            weights_dtype);
    }
    for (uint32_t e = 1; e < operation_arguments.experts_per_chip; ++e) {
        TT_FATAL(
            tensor_arguments.w_gates[e].logical_shape() == gate_shape &&
                tensor_arguments.w_ups[e].logical_shape() == up_shape &&
                tensor_arguments.w_downs[e].logical_shape() == down_shape,
            "moe_fused_swiglu: expert {} weight shapes must match expert 0",
            e);
        // The bank walk lives in the accessor layout descriptor, not in the base address, so an
        // expert placed differently would be read through expert 0's page->bank map and give
        // silently wrong numbers rather than fail.
        TT_FATAL(
            tensor_arguments.w_gates[e].memory_config() == tensor_arguments.w_gates[0].memory_config() &&
                tensor_arguments.w_ups[e].memory_config() == tensor_arguments.w_ups[0].memory_config() &&
                tensor_arguments.w_downs[e].memory_config() == tensor_arguments.w_downs[0].memory_config(),
            "moe_fused_swiglu: expert {} weight memory configs must match expert 0",
            e);
    }
    TT_FATAL(
        gate_shape == up_shape,
        "moe_fused_swiglu: gate and up shapes must match (got {} and {})",
        gate_shape,
        up_shape);
    TT_FATAL(
        gate_shape[-2] == activation_shape[-1],
        "moe_fused_swiglu: gate K ({}) must equal activation embedding ({})",
        gate_shape[-2],
        activation_shape[-1]);
    TT_FATAL(
        down_shape[-2] == gate_shape[-1],
        "moe_fused_swiglu: down K ({}) must equal gate/up hidden ({})",
        down_shape[-2],
        gate_shape[-1]);
    TT_FATAL(
        down_shape[-1] == activation_shape[-1],
        "moe_fused_swiglu: down N ({}) must equal activation embedding ({})",
        down_shape[-1],
        activation_shape[-1]);
    TT_FATAL(
        activation_shape[-1] % TILE == 0 && gate_shape[-1] % TILE == 0,
        "moe_fused_swiglu: embedding ({}) and hidden ({}) must be tile-aligned",
        activation_shape[-1],
        gate_shape[-1]);

    for (const auto& [name, aux] : std::initializer_list<std::pair<const char*, const Tensor&>>{
             {"counts", tensor_arguments.counts},
             {"global_expert_idx_table", tensor_arguments.global_expert_idx_table}}) {
        TT_FATAL(aux.dtype() == tt::tt_metal::DataType::UINT32, "moe_fused_swiglu: {} must be UINT32", name);
        TT_FATAL(aux.layout() == tt::tt_metal::Layout::ROW_MAJOR, "moe_fused_swiglu: {} must be ROW_MAJOR", name);
        TT_FATAL(is_dram_interleaved(aux), "moe_fused_swiglu: {} must be DRAM interleaved", name);
        validate_aux_shape(aux, name);
    }
    TT_FATAL(
        operation_arguments.experts_per_chip <= tensor_arguments.global_expert_idx_table.logical_shape()[-1],
        "moe_fused_swiglu: experts_per_chip {} exceeds idx table length {}",
        operation_arguments.experts_per_chip,
        tensor_arguments.global_expert_idx_table.logical_shape()[-1]);
    TT_FATAL(operation_arguments.m_tiles > 0, "moe_fused_swiglu: input_m_tiles must be positive");
    TT_FATAL(
        operation_arguments.m_tiles <= tensor_arguments.activations.padded_shape()[-2] / TILE,
        "moe_fused_swiglu: input_m_tiles {} exceeds activation capacity {} tiles",
        operation_arguments.m_tiles,
        tensor_arguments.activations.padded_shape()[-2] / TILE);
    // An inverted band drops every expert down the same count-0 path a genuine skip takes, so the
    // op would run to completion and write nothing rather than fail.
    TT_FATAL(
        operation_arguments.min_active_tokens <= operation_arguments.max_active_tokens,
        "moe_fused_swiglu: active-token band is inverted: min_active_tokens {} > max_active_tokens {}",
        operation_arguments.min_active_tokens,
        operation_arguments.max_active_tokens);
    TT_FATAL(
        operation_arguments.grid_x >= operation_arguments.grid_y && operation_arguments.grid_y >= 2,
        "moe_fused_swiglu: core grid must have columns >= rows >= 2, got {}x{}",
        operation_arguments.grid_x,
        operation_arguments.grid_y);
    const auto available_grid = tensor_arguments.activations.device()->compute_with_storage_grid_size();
    TT_FATAL(
        operation_arguments.grid_x <= available_grid.x && operation_arguments.grid_y <= available_grid.y,
        "moe_fused_swiglu: requested grid {}x{} exceeds device grid {}x{}",
        operation_arguments.grid_x,
        operation_arguments.grid_y,
        available_grid.x,
        available_grid.y);
    TT_FATAL(
        operation_arguments.activation == RoutedExpertActivation::Silu ||
            operation_arguments.activation == RoutedExpertActivation::SituGlu ||
            operation_arguments.activation == RoutedExpertActivation::SwiGluOai,
        "moe_fused_swiglu: activation must be RoutedExpertActivation::Silu, "
        "RoutedExpertActivation::SituGlu or RoutedExpertActivation::SwiGluOai");
    // All three or none: biasing some projections and not others is wrong numbers with no error.
    const auto& gate_biases = tensor_arguments.gate_biases;
    const auto& up_biases = tensor_arguments.up_biases;
    const auto& down_biases = tensor_arguments.down_biases;
    TT_FATAL(
        gate_biases.empty() == up_biases.empty() && gate_biases.empty() == down_biases.empty(),
        "moe_fused_swiglu: gate/up/down biases must all be supplied or all be omitted, got {}/{}/{}",
        gate_biases.size(),
        up_biases.size(),
        down_biases.size());
    if (!gate_biases.empty()) {
        TT_FATAL(
            gate_biases.size() == operation_arguments.experts_per_chip &&
                up_biases.size() == operation_arguments.experts_per_chip &&
                down_biases.size() == operation_arguments.experts_per_chip,
            "moe_fused_swiglu: expected one bias per local expert ({}), got {}/{}/{}",
            operation_arguments.experts_per_chip,
            gate_biases.size(),
            up_biases.size(),
            down_biases.size());
        // SiLU's kernel path has no bias branch: it applies the activation on the packer thread of
        // the gate reduce, where there is no spare pass to add a bias into.
        TT_FATAL(
            operation_arguments.activation != RoutedExpertActivation::Silu,
            "moe_fused_swiglu: expert biases require RoutedExpertActivation::SituGlu or "
            "RoutedExpertActivation::SwiGluOai; the SiLU path has no bias branch");
        // The CB layout reuses the bf16 tile for all three bias CBs rather than threading a bias
        // format through it, so the tensors have to actually be bf16.
        for (const auto& bias : {std::cref(gate_biases), std::cref(up_biases), std::cref(down_biases)}) {
            for (const auto& tensor : bias.get()) {
                TT_FATAL(
                    tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
                    "moe_fused_swiglu: expert biases must be BFLOAT16, got {}",
                    tensor.dtype());
            }
        }
    }

    TT_FATAL(
        operation_arguments.output_dtype == tt::tt_metal::DataType::BFLOAT8_B ||
            operation_arguments.output_dtype == tt::tt_metal::DataType::BFLOAT16,
        "moe_fused_swiglu: output dtype must be BFLOAT8_B or BFLOAT16");
    TT_FATAL(
        operation_arguments.output_memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
            operation_arguments.output_memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "moe_fused_swiglu: output must be DRAM interleaved");
    TT_FATAL(
        operation_arguments.compute_kernel_config.has_value(),
        "moe_fused_swiglu: compute kernel configuration must be resolved before launching the primitive");
    const auto& compute_config = *operation_arguments.compute_kernel_config;
    TT_FATAL(
        !compute_config.fp32_dest_acc_en,
        "moe_fused_swiglu: fp32_dest_acc_en is unsupported because the kernel blocking requires eight DEST tiles");
    TT_FATAL(
        !compute_config.packer_l1_acc,
        "moe_fused_swiglu: packer_l1_acc is managed explicitly inside the fused kernel and must be false");
    TT_FATAL(
        !compute_config.dst_full_sync_en,
        "moe_fused_swiglu: dst_full_sync_en is unsupported by the BF16 row-major tilize path");

    const bool direct_write = tensor_arguments.expert_region_offsets.has_value();
    TT_FATAL(
        !operation_arguments.read_x_at_offset || direct_write,
        "moe_fused_swiglu: read_x_at_offset requires expert_region_offsets");
    if (direct_write) {
        const auto& offsets = *tensor_arguments.expert_region_offsets;
        validate_device_tensor(offsets, "expert_region_offsets");
        TT_FATAL(
            offsets.device() == tensor_arguments.activations.device(),
            "moe_fused_swiglu: expert_region_offsets must use the activations device");
        TT_FATAL(
            offsets.dtype() == tt::tt_metal::DataType::UINT32 && offsets.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                is_dram_interleaved(offsets),
            "moe_fused_swiglu: expert_region_offsets must be UINT32 ROW_MAJOR DRAM interleaved");
        const auto& offsets_shape = offsets.logical_shape();
        validate_aux_shape(offsets, "expert_region_offsets");
        TT_FATAL(
            offsets_shape[-1] == tensor_arguments.counts.logical_shape()[-1],
            "moe_fused_swiglu: expert_region_offsets length {} must equal counts length {}",
            offsets_shape[-1],
            tensor_arguments.counts.logical_shape()[-1]);
        TT_FATAL(tensor_arguments.optional_output.has_value(), "moe_fused_swiglu: direct-write mode requires output");
    }

    if (tensor_arguments.optional_output.has_value()) {
        const auto& output = *tensor_arguments.optional_output;
        validate_device_tensor(output, "output");
        TT_FATAL(
            output.device() == tensor_arguments.activations.device(),
            "moe_fused_swiglu: output must use the activations device");
        TT_FATAL(output.layout() == tt::tt_metal::Layout::TILE, "moe_fused_swiglu: output must be TILE layout");
        TT_FATAL(
            output.dtype() == operation_arguments.output_dtype,
            "moe_fused_swiglu: output dtype contradicts dtype argument");
        TT_FATAL(
            output.memory_config() == operation_arguments.output_memory_config,
            "moe_fused_swiglu: output memory config contradicts memory_config argument");
        TT_FATAL(
            output.logical_shape().rank() == activation_shape.rank(),
            "moe_fused_swiglu: output rank {} must match activation rank {}",
            output.logical_shape().rank(),
            activation_shape.rank());
        if (activation_shape.rank() == 4) {
            TT_FATAL(
                output.logical_shape()[0] == activation_shape[0] && output.logical_shape()[1] == activation_shape[1],
                "moe_fused_swiglu: output leading dimensions ({}, {}) must match activations ({}, {})",
                output.logical_shape()[0],
                output.logical_shape()[1],
                activation_shape[0],
                activation_shape[1]);
        }
        TT_FATAL(
            output.logical_shape()[-1] == activation_shape[-1],
            "moe_fused_swiglu: output embedding {} must equal activation embedding {}",
            output.logical_shape()[-1],
            activation_shape[-1]);
        TT_FATAL(output.logical_shape()[-2] % TILE == 0, "moe_fused_swiglu: output rows must be tile-aligned");
        if (direct_write) {
            TT_FATAL(
                output.logical_shape()[-2] >= activation_shape[-2],
                "moe_fused_swiglu: shared output rows {} must be >= activation rows {}",
                output.logical_shape()[-2],
                activation_shape[-2]);
        } else {
            TT_FATAL(
                output.logical_shape() == activation_shape,
                "moe_fused_swiglu: output shape {} must equal activation shape {} outside direct-write mode",
                output.logical_shape(),
                activation_shape);
        }
        std::vector<std::pair<std::string, const Tensor*>> alias_candidates{
            {"activations", &tensor_arguments.activations},
            {"counts", &tensor_arguments.counts},
            {"global_expert_idx_table", &tensor_arguments.global_expert_idx_table}};
        for (uint32_t e = 0; e < operation_arguments.experts_per_chip; ++e) {
            alias_candidates.emplace_back(fmt::format("w_gate[{}]", e), &tensor_arguments.w_gates[e]);
            alias_candidates.emplace_back(fmt::format("w_up[{}]", e), &tensor_arguments.w_ups[e]);
            alias_candidates.emplace_back(fmt::format("w_down[{}]", e), &tensor_arguments.w_downs[e]);
        }
        for (const auto& [name, tensor] : alias_candidates) {
            TT_FATAL(
                output.buffer()->address() != tensor->buffer()->address(),
                "moe_fused_swiglu: output must not alias {}; device readers can overlap output writeback",
                name);
        }
        if (tensor_arguments.expert_region_offsets.has_value()) {
            TT_FATAL(
                output.buffer()->address() != tensor_arguments.expert_region_offsets->buffer()->address(),
                "moe_fused_swiglu: output must not alias expert_region_offsets");
        }
    }
}
}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified {

namespace {
bool is_dram_interleaved(const ttnn::Tensor& t) {
    const auto& mem = t.memory_config();
    return mem.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

// Weights may instead be DRAM ND-sharded, which lets a core fetch a whole (K-row x per-core-N)
// slice in ONE NoC request rather than one per tile: that slice is exactly one shard, hence
// contiguous in a single bank.
//
// The shard must be a whole number of tile-rows tall. Height is otherwise free — the reader walks
// shard-row runs, whose stride does not depend on it — but it decides how many DRAM banks a
// K-block touches: shards distribute ROUND_ROBIN_1D, so consecutive K-rows land in DIFFERENT
// banks, and a taller shard trades that rotation for fewer requests. Bank rotation is what buys
// the bandwidth (a core pinned to one bank saturates near 30 GB/s regardless of request size), so
// one tile-row is the height to ship; the program factory pins the WIDTH, which is the part
// correctness depends on.
bool is_dram_nd_sharded_by_tile_rows(const ttnn::Tensor& t) {
    const auto& mem = t.memory_config();
    if (mem.buffer_type() != tt::tt_metal::BufferType::DRAM || !mem.created_with_nd_shard_spec()) {
        return false;
    }
    const auto& spec = mem.nd_shard_spec();
    if (!spec.has_value() || spec->shard_shape.rank() < 2) {
        return false;
    }
    const auto& shard_shape = spec->shard_shape;
    return shard_shape[-2] >= tt::constants::TILE_HEIGHT && shard_shape[-2] % tt::constants::TILE_HEIGHT == 0 &&
           shard_shape[-1] % tt::constants::TILE_WIDTH == 0;
}
constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

// CB index allocation (kept stable across kernels via named compile-time args).
//
// Offset past the fused half's block: both halves run on the same cores, and a CB index is
// one slot of that core's cb_interface array -- sharing an index would be one FIFO driven by
// both. Blackhole has 64 slots, so 25 + 20 fit side by side.
//
// Nothing downstream hardcodes these: every kernel takes its CB ids as compile-time args.
// One past the fused half's highest circular-buffer index, so the unified half's block starts
// clear of it: both halves run on the same cores, and a shared cb_interface slot would have one
// half reading the other's buffer. Lives here rather than with the fused geometry because it is
// a property of the UNION, not of that op -- it must track upstream's highest index.
constexpr uint32_t CB_BASE =
    ::ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry::CB_DOWN_BIAS + 1;
constexpr uint32_t CB_IN0_X = CB_BASE + 0;
constexpr uint32_t CB_IN1_GATE = CB_BASE + 16;
constexpr uint32_t CB_IN1_UP = CB_BASE + 2;
constexpr uint32_t CB_IN1_DOWN = CB_BASE + 3;
constexpr uint32_t CB_GATE_INT = CB_BASE + 4;
constexpr uint32_t CB_UP_INT = CB_BASE + 5;
constexpr uint32_t CB_ACTIVATED = CB_BASE + 6;
constexpr uint32_t CB_PARTIALS_GU = CB_BASE + 7;
constexpr uint32_t CB_PARTIALS_D = CB_BASE + 8;
constexpr uint32_t CB_OUT = CB_BASE + 9;
constexpr uint32_t CB_COUNTS_SCRATCH = CB_BASE + 10;
constexpr uint32_t CB_IDX_SCRATCH = CB_BASE + 11;
constexpr uint32_t CB_IN0_DOWN_FULL = CB_BASE + 12;
// Second gate/up matmul partials CB. With the fused gate+up phase, each
// K-block accumulates simultaneously into partials_gu (gate matmul) and
// partials_up (up matmul) using the SAME shared x K-block — one x read
// per K-block feeds both matmuls instead of two.
constexpr uint32_t CB_PARTIALS_UP = CB_BASE + 13;
// Writer-only scratch for the device-side `start` (expert_region_offsets)
// page: the writer adds start[global_id]/TILE tile-rows to every output row.
constexpr uint32_t CB_START_SCRATCH = CB_BASE + 14;
// Reader's own `start` scratch (x is the shared dispatched buffer, so the
// reader offsets its x reads by start[global_id] too). Separate from the
// writer's so the two RISCs don't share one L1 page.
constexpr uint32_t CB_START_SCRATCH_READER = CB_BASE + 15;
// Row-major bf16 staging for x when x_is_row_major: the reader fills it with
// row-major sticks and the compute kernel tilizes it to bf8_b into CB_IN0_X.
// Allocated ONLY in row-major mode (unlike the tiny start scratches, this is a
// full per-K-block bf16 block, so allocating it unconditionally would grow the
// bf8_b path's L1). The CT-arg index is passed either way; the CB just isn't
// created (and never touched by the kernels) when x is already TILE.
// Swapped down out of the +16 slot it would otherwise take: tilize_helpers.inl asserts both
// the tilize input (this) and its output (CB_IN0_X) are below CB index 32, which is stricter
// than the 64 slots the hardware has, and the fused block already occupies 0..24.
// Swapped down out of the +16 slot: tilize_helpers.inl asserts the tilize input (this) and
// its output (CB_IN0_X) are both below CB index 32, and the fused block occupies 0..24.
constexpr uint32_t CB_X_RM = CB_BASE + 1;
// Optional per-expert projection biases (gpt-oss, FUSE_BIAS). Each holds this
// core's N-column slice of the (1, N) bias, read once by the reader and added
// by the compute kernel (gate/up before the activation, down after the down
// matmul). Allocated only when op.fuse_bias.
constexpr uint32_t CB_GATE_BIAS = CB_BASE + 17;
constexpr uint32_t CB_UP_BIAS = CB_BASE + 18;
constexpr uint32_t CB_DOWN_BIAS = CB_BASE + 19;

// Tile columns per DRAM ND shard of a weight tensor, or 0 when it is not ND-sharded (the
// interleaved default). The kernels take this as a compile-time arg and coalesce a shard row into
// one NoC transaction; see kernels/weight_runs.hpp. A shard whose width is not a whole number of
// tiles reports 0, so an unusable spec degrades to the per-tile read instead of misaddressing.
uint32_t nd_shard_n_tiles(const ttnn::Tensor& w) {
    const auto& mem = w.memory_config();
    if (mem.buffer_type() != tt::tt_metal::BufferType::DRAM || !mem.created_with_nd_shard_spec()) {
        return 0;
    }
    const auto& spec = mem.nd_shard_spec();
    if (!spec.has_value() || spec->shard_shape.rank() < 2 || spec->shard_shape[-1] % TILE != 0) {
        return 0;
    }
    return static_cast<uint32_t>(spec->shard_shape[-1]) / TILE;
}
}  // namespace

void append_to_descriptor(
    tt::tt_metal::ProgramDescriptor& descriptor,
    uint32_t& next_semaphore_id,
    const UnifiedRoutedExpertFfnParams& op,
    const UnifiedRoutedExpertFfnInputs& t,
    Tensor& tensor_return_value) {
    // All local experts share one shape/dtype (validated), so the program is
    // built once against expert 0's weights; the kernels loop over experts and
    // vary only the per-expert base address.
    const uint32_t experts_per_chip = op.experts_per_chip;
    const auto& x_shape = t.x.padded_shape();
    const auto& gate_shape = t.gate_projs[0].padded_shape();
    const auto& down_shape = t.down_projs[0].padded_shape();

    // This expert's M (not x's allocated M): x may be a shared buffer wider
    // than one expert's region. K still comes from x's last dim (emb).
    const uint32_t M_tiles_full = op.m_tiles;
    const uint32_t K_gate_tiles = x_shape[-1] / TILE;          // = N_gate K = emb / TILE
    const uint32_t N_gate_tiles_full = gate_shape[-1] / TILE;  // = hidden / TILE
    const uint32_t K_down_tiles = down_shape[-2] / TILE;       // = hidden / TILE
    const uint32_t N_down_tiles_full = down_shape[-1] / TILE;  // = emb / TILE

    // Blackhole compute grid is 13x10 worker cores; we use the bottom-left
    // 11x8 = 88 to leave headroom for dispatch and to give per_core_M /
    // per_core_N clean divisors of common chunk sizes (chunk_M_tiles values
    // {16, 24, ..., 64} all divide by GRID_Y=8). N-axis is rounded UP to a
    // multiple of GRID_X via ceil_div per_core_N. Phantom tiles past the
    // actual tensor dims (col 64-65 of hidden, col 224-230 of emb) are
    // zero-padded in the reader (zero-fill L1 instead of DRAM read).
    // Compute runs uniform per_core_N; writer skips DRAM writes past
    // actual_N. K dim of the down matmul is also padded to N_gate_padded so
    // the activated L1 mcast (one sender per K-block, sender = gx == kb)
    // covers exactly per_core_N_gu cols per step; activated cols past
    // actual_hidden are 0 (gate/up weight OOB zero-fill propagates through
    // silu and multiply).
    // Full 2D grid, always. The short-sequence special case was removed: real
    // dispatch buffers are always in the long-sequence range, and the runtime
    // kernel picker (adaptive_chunk.hpp) already shrinks per_core_M for small
    // token counts, so no host-side small-M grid tuning is needed.
    //
    // chunk_M_tiles here is the CB-sized MAXIMUM chunk (op.chunk_M_tiles, default
    // 32 => per_core_M_max 4). All three kernels pick the ACTUAL chunk_M /
    // per_core_M / num_chunks at runtime from the device token count, never
    // exceeding this max; the CBs below are sized to the max so a smaller pick
    // simply uses fewer of the reserved tiles.
    uint32_t GRID_X = op.grid_x;
    uint32_t GRID_Y = op.grid_y;
    const uint32_t ORIGIN_X = op.origin_x;
    const uint32_t ORIGIN_Y = op.origin_y;
    // chunk_M_tiles is the CB-sized MAXIMUM chunk (per_core_M_max = 4). The host
    // deliberately does NOT pick a chunk from M_tiles_full any more: all three
    // kernels derive the ACTUAL chunk_M_tiles / per_core_M / num_chunks at runtime
    // from the device-read per-expert token count (adaptive_chunk.hpp) and never
    // exceed this max, so the CBs sized below always fit and a smaller runtime
    // pick just uses fewer of the reserved tiles. Sizing per EXPERT at runtime
    // also beats any single host-side seed here, since each local expert carries a
    // different token count. (Owned by the op, not the caller.)
    //
    // The REQUESTED per_core_M_max, not necessarily the final one. 4 rather than the
    // 8 L1 can hold because per_core_M_max and the gate/up K-block width
    // in0_block_w_gu compete for the same L1, and on most shapes the width is worth
    // more: the x-staging CBs are sized M * in0_block_w_gu tiles, so per_core_M_max
    // sets the PRICE of a wider K-block. At M=8 width 16 does not fit and the L1 fit
    // below narrows it, multiplying the K-block count; each block costs a fixed
    // mcast-ready barrier round plus read tail that does NOT shrink with width. On
    // shapes where the halved chunk count outweighs that, the L1 fit doubles this
    // back to 8 -- see kWidenMMinTilesPerBlock.
    //
    // Keep this a POWER OF TWO * kCoreGridY: per_core_M_for_chunk() quantizes tail
    // chunks to divisors of per_core_M_max.
    const uint32_t kMaxChunkMTiles = 4 * GRID_Y;  // per_core_M <= 4 (see above)
    uint32_t chunk_M_tiles = kMaxChunkMTiles;
    uint32_t in0_block_w_gu = 16;
    const auto grid_size = t.x.device()->compute_with_storage_grid_size();
    TT_FATAL(
        grid_size.x >= ORIGIN_X + GRID_X && grid_size.y >= ORIGIN_Y + GRID_Y,
        "unified_routed_expert_ffn: rectangle {}x{} at ({},{}) does not fit the {}x{} compute grid",
        GRID_X,
        GRID_Y,
        ORIGIN_X,
        ORIGIN_Y,
        grid_size.x,
        grid_size.y);
    // per_core_M upper bound (the CB-sized max). The adaptive L1-budget guard
    // below may shrink per_core_M / in0_block_w_gu (and hence chunk_M_tiles) to
    // fit the device's per-core L1 on large models.
    const uint32_t per_core_M_max = chunk_M_tiles / GRID_Y;
    TT_FATAL(
        per_core_M_max * GRID_Y == chunk_M_tiles && per_core_M_max >= 1,
        "chunk_M_tiles ({}) must be a positive multiple of GRID_Y ({})",
        chunk_M_tiles,
        GRID_Y);
    // Effective (CB-sized) per_core_M. The L1 guard below may reduce it to fit L1.
    uint32_t per_core_M = per_core_M_max;
    // M_tiles_full is NOT required to divide chunk_M_tiles. The kernels run
    // ceil(count_tiles / chunk_M_tiles_runtime) chunks; the reader zero-fills L1
    // rows past min(count_tiles, M_tiles_full) in the last chunk; the writer
    // skips OOB writes for output rows >= M_tiles_full. Avoids the host-side
    // pad/slice round-trip in the composite for non-aligned M.

    // Bias tile size (FUSE_BIAS): all three bias CBs share the gate-bias dtype
    // (enforced in validation). Zero when unused so the L1 footprint estimate
    // (cb_footprint_bytes below) is unchanged on the bias-free path.
    const uint32_t bias_ts =
        op.fuse_bias ? tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(t.gate_biases[0].dtype())) : 0;

    // in0_block_w_gu (the gate/up K-loop block width) must divide K_gate_tiles.
    // The short_seq picker above already restricts itself to divisors, but the
    // general 2D path leaves the default 16 unchanged — valid only when emb is a
    // multiple of 512 (=> K_gate_tiles a multiple of 16). Models with a different
    // emb (e.g. GPT-OSS 120B: emb 2880 => K_gate_tiles 90) need the default
    // snapped down to the largest divisor of K_gate_tiles that does not exceed
    // it. No-op for short_seq and for every shipped 512-multiple emb; the L1
    // guard below may narrow it further. Previously divisor-snapping only ran as
    // a side effect of the L1-overflow guard, so non-512 emb slipped through on
    // the TILE x layout — whose smaller footprint fits at 16 and skips the guard.
    while (in0_block_w_gu > 1 && (K_gate_tiles % in0_block_w_gu) != 0) {
        --in0_block_w_gu;
    }

    const uint32_t per_core_N_gu = (N_gate_tiles_full + GRID_X - 1) / GRID_X;
    const uint32_t per_core_N_d = (N_down_tiles_full + GRID_X - 1) / GRID_X;
    const uint32_t N_gate_tiles_padded = per_core_N_gu * GRID_X;
    const uint32_t K_down_tiles_padded = N_gate_tiles_padded;  // down K = gate N

    // DRAM ND-sharded weights (opt-in; 0 = the DRAM-interleaved default). The shard width must be
    // exactly a core's N slice, because that is what makes the slice ONE shard and therefore one
    // contiguous NoC transaction per K-row instead of per_core_N of them. A wider or narrower
    // shard would still read correctly through the run loop but would split or straddle the slice,
    // losing the point, so it fails host-side instead. gate and up are pinned equal: the reader
    // and the writer read them through a single shared compile-time width.
    const uint32_t gate_shard_w = nd_shard_n_tiles(t.gate_projs[0]);
    const uint32_t up_shard_w = nd_shard_n_tiles(t.up_projs[0]);
    const uint32_t d_shard_w = nd_shard_n_tiles(t.down_projs[0]);
    TT_FATAL(
        gate_shard_w == up_shard_w,
        "gate_proj and up_proj must share a weight layout: ND shard widths {} and {} tiles",
        gate_shard_w,
        up_shard_w);
    const uint32_t gu_shard_w = gate_shard_w;
    for (const auto& [name, shard_w, per_core_n] : std::initializer_list<std::tuple<const char*, uint32_t, uint32_t>>{
             {"gate_proj/up_proj", gu_shard_w, per_core_N_gu}, {"down_proj", d_shard_w, per_core_N_d}}) {
        TT_FATAL(
            shard_w == 0 || shard_w == per_core_n,
            "{}: DRAM ND shard width must be per_core_N ({} tiles) so a K-row slice is one shard, got {}",
            name,
            per_core_n,
            shard_w);
    }

    (void)K_down_tiles;  // actual K_down; used by reader for OOB; suppress unused warning here

    // down-matmul K-block width (= gate N per-core slice). Independent of the
    // adaptive levers below.
    const uint32_t in0_block_w_d = per_core_N_gu;
    TT_FATAL(
        K_down_tiles_padded % in0_block_w_d == 0,
        "K_down_tiles_padded ({}) must be divisible by in0_block_w_d ({})",
        K_down_tiles_padded,
        in0_block_w_d);

    // Subblock dims. DST tile-register file is 16 tiles wide; fp32_dest_acc_en
    // halves usable capacity (fp32 accumulator occupies two tile slots). With
    // fp32_dest_acc_en=false (bf16 dst), per-thread DST capacity is 8 tiles.
    // Single source of truth for the dst-accumulator mode: drives DST_CAPACITY,
    // the ComputeConfig below, and (via -DFP32_DEST_ACC_EN) the compute kernel's
    // SwiGLU-OAI dst budget + SFPU fp32-dest template, so they can't drift.
    constexpr bool kFp32DestAccEn = false;
    constexpr uint32_t DST_CAPACITY = kFp32DestAccEn ? 4u : 8u;
    const uint32_t gu_out_subblock_h = 1;
    uint32_t gu_sub_w = 1;
    for (uint32_t cand = DST_CAPACITY; cand >= 1; --cand) {
        if (per_core_N_gu % cand == 0) {
            gu_sub_w = cand;
            break;
        }
    }
    const uint32_t gu_out_subblock_w = gu_sub_w;
    TT_FATAL(
        gu_out_subblock_h * gu_out_subblock_w <= DST_CAPACITY,
        "gu subblock h*w ({}) exceeds dst capacity",
        gu_out_subblock_h * gu_out_subblock_w);
    const uint32_t d_out_subblock_h = 1;
    uint32_t d_sub_w = 1;
    for (uint32_t cand = DST_CAPACITY; cand >= 1; --cand) {
        if (per_core_N_d % cand == 0) {
            d_sub_w = cand;
            break;
        }
    }
    // A subblock narrower than 3 tiles is where the matmul stops being math-bound and the
    // operand unpack starts to dominate: measured on 7168x2048, forcing the down subblock from
    // 1x7 to 1x3 costs 1% but 1x1 costs 28% of total op time. per_core_N_d = 11 (prime, from
    // ceil(112/11) on the 11-wide grid) is the shape that lands there, and no divisor can save
    // it -- so below the cliff, give up the exact tiling and take a RAGGED last subblock
    // instead. Shapes whose divisor rule already clears 3 keep their exact tiling untouched.
    if (d_sub_w < 3) {
        d_sub_w = std::min<uint32_t>(DST_CAPACITY, per_core_N_d);
    }
    const uint32_t d_out_subblock_w = d_sub_w;

    // -------------------------- data formats / tile sizes -----------------
    const tt::DataFormat x_df = tt::tt_metal::datatype_to_dataformat_converter(t.x.dtype());
    const tt::DataFormat gate_df = tt::tt_metal::datatype_to_dataformat_converter(t.gate_projs[0].dtype());
    const tt::DataFormat up_df = tt::tt_metal::datatype_to_dataformat_converter(t.up_projs[0].dtype());
    const tt::DataFormat down_df = tt::tt_metal::datatype_to_dataformat_converter(t.down_projs[0].dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    // Partials vs intermediates deliberately differ in format; the compute
    // kernel pack-reconfigs between them (partials <-> intermed) explicitly.
    //   * partials_gu/partials_d are Float16_b: they hold the K-loop matmul
    //     accumulator (PACKER_L1_ACC adds each K-block's result into the same L1
    //     tiles), so block-float bf8 would lose precision across K-blocks. bf16
    //     keeps a per-element mantissa for the running sum.
    //   * intermediates (gate/up_intermed, activated) are Bfp8_b: post-activation
    //     per-element values feeding the next matmul, not accumulators, so
    //     1KB/tile (half the bf16 cost) is enough and saves L1.
    const tt::DataFormat intermed_df = tt::DataFormat::Bfp8_b;
    const tt::DataFormat partials_gu_df = tt::DataFormat::Float16_b;
    const tt::DataFormat partials_d_df = tt::DataFormat::Float16_b;

    // See in0_x_ts above: cb_in0_x is bf8_b on the row-major path (tilize output),
    // else it matches x's dtype (bf8_b on the TILE path).
    const tt::DataFormat in0_x_df = op.x_is_row_major ? tt::DataFormat::Bfp8_b : x_df;
    const uint32_t in0_x_tile_size = tt::tile_size(in0_x_df);
    const uint32_t gate_tile_size = tt::tile_size(gate_df);
    const uint32_t up_tile_size = tt::tile_size(up_df);
    const uint32_t down_tile_size = tt::tile_size(down_df);
    const uint32_t out_tile_size = tt::tile_size(out_df);
    const uint32_t intermed_tile_size = tt::tile_size(intermed_df);
    const uint32_t partials_gu_tile_size = tt::tile_size(partials_gu_df);
    const uint32_t partials_d_tile_size = tt::tile_size(partials_d_df);

    // ---------------------- adaptive L1-budget sizing ---------------------
    // Per-core CB footprint scales with per_core_M (= chunk_M_tiles / GRID_Y)
    // and in0_block_w_gu (the gate/up K-block width). A fixed chunk_M_tiles=64
    // / in0_block_w_gu=16 fit the DeepSeek-V3 / MiniMax-M2.7 dims with headroom
    // but overflow L1 on larger models (MiniMax-M3: emb 6144 / hidden 3072, 2x
    // both axes). Instead of hard-coding per shape, fit the requested config to
    // the real device L1 budget, shrinking only when it overflows:
    //   1. keep per_core_M as large as possible — fewer M chunks => fewer full
    //      weight re-reads, the dominant DRAM cost;
    //   2. then the largest in0_block_w_gu (divisor of K_gate_tiles) that fits —
    //      wider gate/up K-blocks pipeline DRAM I/O better.
    // The resulting per_core_M is the CB-sized MAX; the runtime picker never
    // exceeds it. This mirrors the CB allocations in the "circular buffers"
    // section below; keep the two in sync.
    const auto cb_footprint_bytes = [&](uint32_t M, uint32_t w_gu) -> uint64_t {
        uint64_t total = 0;
        total += static_cast<uint64_t>(M * w_gu * (op.x_is_row_major ? 1 : 2)) *
                 in0_x_tile_size;  // cb_in0_x (RM: single-buf)
        if (op.x_is_row_major) {
            total += static_cast<uint64_t>(M * w_gu * 2) * partials_gu_tile_size;  // cb_x_rm (bf16 staging)
        }
        total += static_cast<uint64_t>(w_gu * per_core_N_gu * 2) * gate_tile_size;                // cb_in1_gate
        total += static_cast<uint64_t>(w_gu * per_core_N_gu * 2) * up_tile_size;                  // cb_in1_up
        total += static_cast<uint64_t>(in0_block_w_d * per_core_N_d * 2) * down_tile_size;        // cb_in1_down
        total += static_cast<uint64_t>(M * per_core_N_gu) * intermed_tile_size;                   // cb_gate_intermed
        total += static_cast<uint64_t>(M * per_core_N_gu) * intermed_tile_size;                   // cb_activated
        total += static_cast<uint64_t>(M * per_core_N_gu) * partials_gu_tile_size;                // cb_mm_partials_gu
        total += static_cast<uint64_t>(M * per_core_N_gu) * partials_gu_tile_size;                // cb_mm_partials_up
        total += static_cast<uint64_t>(M * per_core_N_d) * partials_d_tile_size;                  // cb_mm_partials_d
        total += static_cast<uint64_t>(d_out_subblock_h * d_out_subblock_w * 2) * out_tile_size;  // cb_out
        total += static_cast<uint64_t>(M * in0_block_w_d * 2) * intermed_tile_size;               // cb_in0_down_full
        // Bias CBs (FUSE_BIAS): single-buffered, per_core_N_gu (gate/up) + per_core_N_d
        // (down) tiles. Keep in sync with the CB allocations in the "Bias CBs" section.
        total += static_cast<uint64_t>(2 * per_core_N_gu + per_core_N_d) * bias_ts;
        return total;
    };

    // Real per-core L1 available for CBs (total minus the firmware/kernel
    // reserved base), with a margin for the small UInt32 scratch CBs
    // (counts/idx/start, allocated below) and per-CB allocation alignment.
    auto* l1_device = t.x.device();
    constexpr uint32_t L1_SCRATCH_MARGIN = 48 * 1024;
    const uint32_t l1_reserved = l1_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    TT_FATAL(
        l1_device->l1_size_per_core() > l1_reserved + L1_SCRATCH_MARGIN,
        "unexpected L1 geometry: l1_size_per_core ({}) <= reserved base ({}) + margin ({})",
        l1_device->l1_size_per_core(),
        l1_reserved,
        L1_SCRATCH_MARGIN);
    const uint64_t l1_budget = static_cast<uint64_t>(l1_device->l1_size_per_core()) - l1_reserved - L1_SCRATCH_MARGIN;

    // Fit (per_core_M, in0_block_w_gu) to the real L1 budget. Candidate widths are
    // divisors of K_gate_tiles no wider than the request, largest first; candidate
    // Ms are DIVISORS of the request, largest first. Walking M down by 1 could stop
    // on a prime: at per_core_M=5 a 16-tile tail needs 2 rows/core but the only
    // divisors are {1,5}, so it would run 5 - 2.5x the M-work. Restricting to
    // divisors of the request keeps per_core_M_for_chunk()'s tail ladder as fine as
    // the request's own.
    const auto fit_config = [&](uint32_t requested_M) -> std::pair<uint32_t, uint32_t> {
        for (uint32_t M = requested_M; M >= 1; --M) {
            if (requested_M % M != 0) {
                continue;
            }
            for (uint32_t w = std::min<uint32_t>(in0_block_w_gu, K_gate_tiles); w >= 1; --w) {
                if (K_gate_tiles % w == 0 && cb_footprint_bytes(M, w) <= l1_budget) {
                    return {M, w};
                }
            }
        }
        return {0, 0};
    };

    auto [fit_M, fit_w] = fit_config(per_core_M);
    TT_FATAL(
        fit_M != 0,
        "unified_routed_expert_ffn: per-core CBs do not fit in L1 even at the smallest config "
        "(per_core_M=1, in0_block_w_gu=1): need {} B but only {} B available "
        "(emb={}, hidden={}, grid {}x{}). Reduce model dims.",
        cb_footprint_bytes(1, 1),
        l1_budget,
        N_down_tiles_full * TILE,
        N_gate_tiles_full * TILE,
        GRID_X,
        GRID_Y);

    // Doubling per_core_M halves the M-chunk count, and every chunk re-reads the
    // full gate/up/down weights (adaptive_chunk.hpp) - the dominant DRAM cost. It is
    // not free: per_core_M and in0_block_w_gu compete for the same L1, so the widest
    // width that still fits narrows, and each gate/up K-block carries a fixed
    // mcast-ready barrier plus read tail that does NOT shrink with width. Take the
    // doubling only when each extra K-block buys enough avoided weight traffic:
    //
    //     weight tiles per extra gate/up K-block
    //         = (2*K_gate*N_gate + K_down*N_down) / (K_gate/w_wide - K_gate/w_narrow)
    //
    // Measured on a BH p150b at ISL 5120, tiles per extra block: kimi_k3 4608,
    // kimi_k26 / dsv4_flash / glm_51 3072, gptoss_120b 2700, minimax_m3 1536,
    // dsv4_pro 658. Only kimi_k3 clears the bar, and it is the only shape the
    // doubling measurably helps: +3-4% at ISL >= 512, -3% at ISL <= 256, against
    // 1.5-1.85x LOSSES on the shapes below it. The bar sits between 3072 and 4608;
    // it is calibrated on one favourable shape, so widen it only with measurements.
    constexpr uint32_t kWidenMMinTilesPerBlock = 4096;
    const auto [wide_M, wide_w] = fit_config(per_core_M * 2);
    if (wide_M == per_core_M * 2 && wide_w < fit_w) {
        const uint32_t extra_blocks = K_gate_tiles / wide_w - K_gate_tiles / fit_w;
        const uint32_t weight_tiles = 2 * K_gate_tiles * N_gate_tiles_full + K_down_tiles * N_down_tiles_full;
        if (extra_blocks > 0 && weight_tiles / extra_blocks >= kWidenMMinTilesPerBlock) {
            fit_M = wide_M;
            fit_w = wide_w;
        }
    }

    per_core_M = fit_M;
    in0_block_w_gu = fit_w;
    chunk_M_tiles = per_core_M * GRID_Y;

    // in0_block_w_gu must divide K_gate_tiles (the gate/up K-loop bound); the
    // divisor-snap after the short_seq picker and the L1 guard above both
    // preserve this, so this is a defensive invariant check.
    TT_FATAL(
        K_gate_tiles % in0_block_w_gu == 0,
        "K_gate_tiles ({}) must be divisible by in0_block_w_gu ({})",
        K_gate_tiles,
        in0_block_w_gu);

    // num_chunks is the compile-time UPPER BOUND on the runtime chunk count used
    // only to clamp the kernels' loop defensively. The runtime picker may choose
    // a chunk as small as min(16, chunk_M_tiles) (per_core_M 2), so the worst
    // case is ceil(M_tiles_full / that min). Matches adaptive_chunk.hpp's
    // kMinChunkMTiles = 16.
    constexpr uint32_t kMinChunkMTiles = 16;
    const uint32_t min_chunk = (chunk_M_tiles < kMinChunkMTiles) ? chunk_M_tiles : kMinChunkMTiles;
    const uint32_t num_chunks = (M_tiles_full + min_chunk - 1) / min_chunk;

    // Phase-level numbers.
    const uint32_t gu_in0_num_subblocks = per_core_M / gu_out_subblock_h;
    const uint32_t gu_in1_num_subblocks = per_core_N_gu / gu_out_subblock_w;
    const uint32_t gu_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    const uint32_t gu_in0_subblock_num_tiles = gu_out_subblock_h * in0_block_w_gu;
    const uint32_t gu_in1_block_num_tiles = in0_block_w_gu * per_core_N_gu;
    const uint32_t gu_in1_block_w = per_core_N_gu;
    const uint32_t gu_num_blocks = K_gate_tiles / in0_block_w_gu;
    const uint32_t gu_out_block_num_tiles = per_core_M * per_core_N_gu;

    const uint32_t d_in0_num_subblocks = per_core_M / d_out_subblock_h;
    // Ceil, not exact: the last subblock may be narrower (d_out_subblock_w_tail). Equal to the
    // exact division whenever the width divides, which is every shape but the ragged one.
    const uint32_t d_in1_num_subblocks = (per_core_N_d + d_out_subblock_w - 1) / d_out_subblock_w;
    const uint32_t d_out_subblock_w_tail = per_core_N_d - (d_in1_num_subblocks - 1) * d_out_subblock_w;
    const uint32_t d_in0_block_num_tiles = per_core_M * in0_block_w_d;
    const uint32_t d_in0_subblock_num_tiles = d_out_subblock_h * in0_block_w_d;
    const uint32_t d_in1_block_num_tiles = in0_block_w_d * per_core_N_d;
    const uint32_t d_in1_block_w = per_core_N_d;
    const uint32_t d_num_blocks = K_down_tiles_padded / in0_block_w_d;
    const uint32_t d_out_block_num_tiles = per_core_M * per_core_N_d;

    // -------------------------- compute grid ------------------------------
    const CoreRange core_range({ORIGIN_X, ORIGIN_Y}, {ORIGIN_X + GRID_X - 1, ORIGIN_Y + GRID_Y - 1});
    const CoreRangeSet core_range_set{core_range};

    // Representative expert-0 buffers: one TensorAccessorArgs layout descriptor
    // per weight role covers every expert (identical shape/layout), and the
    // kernels build a per-expert accessor from that descriptor + the per-expert
    // base address (passed as a runtime-arg).
    auto* x_buffer = t.x.buffer();
    auto* gate_buffer = t.gate_projs[0].buffer();
    auto* up_buffer = t.up_projs[0].buffer();
    auto* down_buffer = t.down_projs[0].buffer();
    auto* counts_buffer = t.counts.buffer();
    auto* idx_buffer = t.global_expert_idx_table.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    // expert_region_offsets is a mandatory input (validated in the device op):
    // the writer always writes an expert's output straight into the shared
    // output buffer at start[global_id]/TILE tile-rows (fusing ttnn::insert),
    // and the reader always reads x from the same region.
    auto* start_buffer = t.expert_region_offsets->buffer();
    // dst_M_tiles bounds destination writes: the shared output buffer's
    // tile-row count.
    const uint32_t dst_M_tiles = tensor_return_value.padded_shape()[-2] / TILE;

    // -------------------------- semaphores --------------------------------
    // Weight-multicast semaphores for in1 (gate/up/down). Pattern: per
    // N-col group (gx fixed, gy=0..GRID_Y-1), one sender at gy=0 reads the
    // weight slice from DRAM and mcasts it to the other GRID_Y-1 cores in
    // the same column. Receivers atomic-inc `ready` on the sender to signal
    // "I'm ready"; sender waits for ready==GRID_Y-1, mcasts the block, then
    // mcast-sets `valid` to 1 on all receivers; receivers wait for valid==1.
    // Same sem pair is reused across gate/up/down phases (phases are
    // sequential, sem values reset between K-blocks).
    // Descriptor semaphores carry explicit ids. They come from the caller's counter rather than
    // from 0 so a merged program can place this half's ids above the other half's.
    auto make_sem = [&]() {
        const uint32_t id = next_semaphore_id++;
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_range_set,
            .initial_value = 0,
        });
        return id;
    };

    const uint32_t in1_ready_sem_id = make_sem();
    const uint32_t in1_valid_sem_id = make_sem();
    // in0 (x) multicast within M-row groups: sender at (gx=0, gy) reads x
    // for that M-row, mcasts to (gx=1..GRID_X-1, gy). Used for phases 1 and
    // 2 (gate and up matmul) where every core in a row needs the same x
    // slice. Phase 4 uses cb_in0_down_full (sourced from DRAM scratch) so
    // doesn't use this pair.
    const uint32_t in0_ready_sem_id = make_sem();
    const uint32_t in0_valid_sem_id = make_sem();
    // Activated multicast sems (phase 4): replace the DRAM scratch round-trip
    // with an L1 NoC mcast. For phase-4 K-block kb, sender = core at
    // (gx=kb, my_mt). Sender's reader mcasts its cb_activated block to all
    // 8 M-row cores' cb_in0_down_full (loopback included). Receivers wait on
    // act_valid_sem; sender waits on act_ready_sem reaching GRID_X-1 incs from
    // the 7 receivers. Sender position rotates per K-block so each core takes
    // a turn as sender exactly once per chunk.
    const uint32_t act_ready_sem_id = make_sem();
    const uint32_t act_valid_sem_id = make_sem();
    // Two-RISC weight read: use the writer (BRISC, idle until the down output)
    // as a second read engine for `up`, read on NoC 1 concurrent with the
    // reader's NoC-0 `gate` read. Two delivery schemes:
    //
    //   * UP_WRITER_MCAST (mode 1): writer also NoC-1 multicasts `up` down its
    //     N-column. Bandwidth-optimal, but the NoC-1 worker multicast + posted
    //     atomics collide with fabric CCL ops on NoC 1 and hang the run.
    //     Short-seq is NOT fabric-disabled (it triggers on small dispatch
    //     buffers in real fabric-enabled runs), so this scheme is retired.
    //   * UP_SPLIT (mode 2): writer only reads `up` on NoC 1 (same kind as its
    //     cb_out NoC-1 writes — fabric-safe) into the gy=0 sender's cb_in1_up
    //     slot; the reader multicasts it on NoC 0 alongside `gate`. A local
    //     same-core L1 handshake orders the two. Used on all layouts.
    //
    // up_mode: 0 = LEGACY (reader reads + mcasts `up` on NoC 0), 2 = UP_SPLIT
    // (writer reads `up` on NoC 1, reader mcasts on NoC 0). The retired
    // UP_WRITER_MCAST scheme (writer NoC-1-multicasts `up`) is no longer
    // selectable. kEnableSplitUp picks UP_SPLIT for all layouts.
    constexpr bool kEnableSplitUp = true;
    uint32_t up_mode = kEnableSplitUp ? 2 : 0;
    const bool reader_reads_up = (up_mode == 0);                   // reader issues up DRAM read
    const bool reader_mcasts_up = (up_mode == 0 || up_mode == 2);  // reader NoC-0 mcasts up
    // Local same-core handshake sems (UP_SPLIT only): up_go (reader -> writer:
    // slot reserved) and up_done (writer -> reader: up in L1). Monotonic.
    // COUNTS_BCAST: one core reads counts/idx and multicasts them; this gates the rest.
    const uint32_t counts_valid_sem_id = make_sem();
    // DOWN_SPLIT: share each down K-block's K-rows between the reader (NoC 0) and the
    // writer (NoC 1). The down read was the only weight read left on the reader's
    // critical path once UP_SPLIT hides gate/up.
    //
    // SPLIT, not a full handoff: these reads are issue-bound, not bandwidth-bound
    // (~43 cycles of RISC command-buffer time per 576 B bfp4 tile against ~9 cycles of
    // NoC port), so the limit is one RISC's issue rate. Two RISCs issuing half each
    // roughly doubles it; handing the whole block to one RISC only moves the serial cost
    // from the reader to the writer, and measured 1.03x against 1.13x for the split.
    //
    // A split needs >= 2 K-rows to divide; in0_block_w_d is per_core_N_gu.
    // IN1_WRITER_MCAST: hand the gate/up weight multicast to the WRITER, which owns NoC 1,
    // so it overlaps the reader's NoC-0 weight reads. The reader cannot multicast on NoC 1
    // itself -- in DM_DEDICATED_NOC both RISCs use the same command-buffer indices and only
    // avoid collision by each owning one NoC -- so the other RISC must be SIGNALLED to do it.
    //
    // !! FABRIC HAZARD, KNOWINGLY ACCEPTED !!
    // This puts a WORKER MULTICAST on NoC 1, which is what retired the earlier
    // UP_WRITER_MCAST scheme: the NoC-1 worker multicast + posted atomics collide with
    // fabric CCL ops on NoC 1 and hang the run. UP_SPLIT's fabric-safety argument is
    // specifically that it keeps NoC 1 READ-ONLY, and enabling this voids that argument.
    // Single-device perf/functional tests CANNOT detect it -- the failure is a collision
    // with CCL traffic that is absent there, so a green sweep here is NOT evidence of
    // fabric safety. Validate against a fabric-enabled run with concurrent CCL before
    // trusting this in production. Set DS_NO_WRITER_MCAST=1 to fall back to the reader's
    // NoC-0 multicast.
    const bool kWriterMcastsIn1 = std::getenv("DS_NO_WRITER_MCAST") == nullptr;
    const uint32_t mcast_go_sem_id = kWriterMcastsIn1 ? make_sem() : 0;
    const uint32_t mcast_done_sem_id = kWriterMcastsIn1 ? make_sem() : 0;

    const bool kEnableSplitDown = in0_block_w_d >= 2;
    // Rows the READER keeps; the writer takes [down_split_k, in0_block_w_d).
    const uint32_t down_split_k = kEnableSplitDown ? (in0_block_w_d / 2) : in0_block_w_d;
    // DEDICATED sems, not up_go/up_done: the UP_SPLIT writer indexes its cb_in1_up slot
    // off (up_seq - 1) % slots, which only holds while up_seq counts gate/up blocks
    // alone. Adding down blocks to that counter shifts the up slot by num_blocks_d per
    // chunk and silently corrupts the GATE/UP path from the second chunk on.
    const uint32_t down_go_sem_id = kEnableSplitDown ? make_sem() : 0;
    const uint32_t down_done_sem_id = kEnableSplitDown ? make_sem() : 0;
    const uint32_t up_go_sem_id = (up_mode == 2) ? make_sem() : 0;
    const uint32_t up_done_sem_id = (up_mode == 2) ? make_sem() : 0;

    // -------------------------- circular buffers --------------------------
    // Double-buffered DRAM-streamed inputs.
    auto make_cb = [&](uint32_t cb_idx, tt::DataFormat fmt, uint32_t num_tiles, uint32_t tile_bytes) {
        descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = num_tiles * tile_bytes,
            .core_ranges = core_range_set,
            .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_idx),
                .data_format = fmt,
                .page_size = tile_bytes,
            }},
        });
    };

    // Single-buffered DRAM-streamed inputs (no double-buffer) to fit L1.
    // Double-buffered input CBs so the reader (NCRISC) can fetch K-block N+1
    // while compute consumes K-block N. PM FPU util = 0 today says we're
    // memory-bound; bigger input CBs let the kernel pipeline DRAM I/O with
    // compute instead of serialising.
    // Row-major path: cb_in0_x is compute-internal (tilize output -> matmul input,
    // both on the compute threads sharing DST -> serial), so a second slot buys no
    // pipelining. Single-buffer it to free L1 for a wider in0_block_w_gu. TILE path
    // keeps double-buffering: the reader fills it and compute consumes it (cross-RISC
    // overlap).
    make_cb(CB_IN0_X, in0_x_df, /*tiles=*/gu_in0_block_num_tiles * (op.x_is_row_major ? 1u : 2u), in0_x_tile_size);
    // Row-major bf16 x staging (x_is_row_major only). Double-buffered like
    // cb_in0_x: it is a MULTICAST SOURCE, so the sender must fill K-block N+1
    // while N's posted mcast still drains — single-buffering reuses the slot
    // mid-mcast and deadlocks. Skipped when x is TILE so the bf8_b path's L1 is
    // unchanged.
    if (op.x_is_row_major) {
        make_cb(
            CB_X_RM,
            tt::DataFormat::Float16_b,
            /*tiles=*/gu_in0_block_num_tiles * 2,
            tt::tile_size(tt::DataFormat::Float16_b));
    }
    make_cb(CB_IN1_GATE, gate_df, /*tiles=*/gu_in1_block_num_tiles * 2, gate_tile_size);
    make_cb(CB_IN1_UP, up_df, /*tiles=*/gu_in1_block_num_tiles * 2, up_tile_size);
    make_cb(CB_IN1_DOWN, down_df, /*tiles=*/d_in1_block_num_tiles * 2, down_tile_size);
    // Intermediate L1 buffers hold one full per-core block each.
    make_cb(CB_GATE_INT, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    // cb_up_intermed removed — multiply reads cb_partials_up directly.
    make_cb(CB_ACTIVATED, intermed_df, /*tiles=*/gu_out_block_num_tiles, intermed_tile_size);
    // Partials CBs: sized to the full per-core output block. Within ONE
    // K-block iteration the kernel pushes (in0_num_subblocks *
    // in1_num_subblocks) subblocks to partials before any pops happen
    // (the pops happen on the NEXT K-block iteration's reload). So the CB
    // must hold all those subblocks = the full block.
    make_cb(
        CB_PARTIALS_GU,
        partials_gu_df,
        /*tiles=*/gu_out_block_num_tiles,
        partials_gu_tile_size);
    // Second gate/up matmul accumulator (cb_partials_up), used by the fused
    // gate+up phase to share x reads across both matmuls.
    make_cb(CB_PARTIALS_UP, partials_gu_df, /*tiles=*/gu_out_block_num_tiles, partials_gu_tile_size);
    make_cb(
        CB_PARTIALS_D,
        partials_d_df,
        /*tiles=*/d_out_block_num_tiles,
        partials_d_tile_size);
    // Output CB: writer drains one subblock at a time. 2-subblock staging
    // pipelines compute/writer one-ahead and is safe under the tightest L1
    // budget (the 256-expert / 32-per-chip case the unfused path is run on).
    constexpr uint32_t cb_out_stage_count = 2u;
    // A CB wraps only when a push lands EXACTLY on fifo_limit (adaptive_chunk.hpp). On a ragged
    // subblock grid the pushes alternate width and tail, so a ring sized in SUBBLOCKS is never hit
    // exactly and the write pointer walks off into neighbouring L1. Size the ragged case in whole
    // output ROWS -- the row period per_core_N_d is what the push sequence repeats on. Exact grids
    // keep the original subblock sizing.
    const bool d_subblocks_exact = (d_out_subblock_w_tail == d_out_subblock_w);
    const uint32_t cb_out_tiles =
        d_out_subblock_h * (d_subblocks_exact ? d_out_subblock_w : per_core_N_d) * cb_out_stage_count;
    make_cb(
        CB_OUT,
        out_df,
        /*tiles=*/cb_out_tiles,
        out_tile_size);
    // cb_in0_down_full: reader pushes per_core_M × in0_block_w_d tiles of activated
    // once per down K-block. Single-buffered to save L1.
    // cb_in0_down_full double-buffered — fits because we eliminated
    // cb_up_intermed (multiply reads cb_partials_up directly).
    make_cb(
        CB_IN0_DOWN_FULL,
        intermed_df,
        /*tiles=*/d_in0_block_num_tiles * 2,
        intermed_tile_size);

    // cb_push_back/cb_pop_front wrap the FIFO pointer only when it lands EXACTLY on
    // fifo_limit; a ring that is not a whole number of the granule its kernels move
    // leaves the pointer mid-ring forever and the next push walks into the neighbouring
    // CB's L1. The granule is the FINEST push/pop any kernel issues on that CB, which is
    // not always the block: the ragged down grid repeats on a whole output row, and the
    // runtime per_core_M splits the x and intermediate blocks into strip-sized pieces.
    // Checked here because the kernels cannot -- ASSERT is a no-op in Release.
    const auto check_ring = [](const char* cb_name, uint32_t ring_tiles, uint32_t granule_tiles) {
        TT_FATAL(
            granule_tiles > 0 && ring_tiles % granule_tiles == 0,
            "unified_routed_expert_ffn: {} ring ({} tiles) is not a whole number of the {}-tile "
            "granule its kernels push; the FIFO pointer would never land on fifo_limit",
            cb_name,
            ring_tiles,
            granule_tiles);
    };
    // x and its row-major staging move one in0_block_w_gu-wide tile-row strip at a time
    // (the tilize helper pushes per strip; the per_core_M remainder is a pointer-only pad).
    check_ring("cb_in0_x", gu_in0_block_num_tiles * (op.x_is_row_major ? 1u : 2u), in0_block_w_gu);
    if (op.x_is_row_major) {
        check_ring("cb_x_rm", gu_in0_block_num_tiles * 2, in0_block_w_gu);
    }
    // gate/up intermediates and their accumulators move one gate/up subblock at a time.
    check_ring("cb_gate_intermed", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w);
    check_ring("cb_partials_gu", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w);
    check_ring("cb_partials_up", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w);
    // The reader drains cb_activated one down K-block at a time; in0_block_w_d ==
    // per_core_N_gu makes that granule the whole block, which this pins.
    check_ring("cb_activated", gu_out_block_num_tiles, per_core_M * in0_block_w_d);
    // Down partials and the output ring repeat on a whole output row when the N-subblock
    // grid is ragged, and on a single subblock when it is exact.
    const uint32_t d_row_granule = d_out_subblock_h * (d_subblocks_exact ? d_out_subblock_w : per_core_N_d);
    check_ring("cb_mm_partials_d", d_out_block_num_tiles, d_row_granule);
    check_ring("cb_out", cb_out_tiles, d_row_granule);
    check_ring("cb_in0_down_full", d_in0_block_num_tiles * 2, d_in0_block_num_tiles);
    check_ring("cb_in1_gate", gu_in1_block_num_tiles * 2, gu_in1_block_num_tiles);
    check_ring("cb_in1_up", gu_in1_block_num_tiles * 2, gu_in1_block_num_tiles);
    check_ring("cb_in1_down", d_in1_block_num_tiles * 2, d_in1_block_num_tiles);

    // Scratch CBs for the device-side count lookup. The reader does a single
    // noc_async_read_page(page=0, ...) of each tensor and then indexes
    // counts[global_expert_id] / idx[local_expert_id]. Both indices stay within
    // the tensor's own length (counts: [0, num_global_experts); idx:
    // [0, idx_len)), and the reader's page-0 read contract requires the entire
    // vector to live in page 0 — i.e. aligned_page_size already covers every
    // index the reader can produce. So aligned_page_size is the exact capacity
    // the scratch needs; we keep a num_entries*4B floor purely as a defensive
    // guard in case a future layout reports a sub-row page.
    //
    // Previously this floored to MAX_GLOBAL_EXPERTS * 4B (a fixed 4 KB), which
    // over-allocated ~3 KB/core for the 256-expert DS-V3 path and ~2.5 KB for
    // Kimi (384). That slack is what tipped the mesh-4x2 perf-256 program over
    // the L1 ceiling once MEM_MAILBOX_SIZE grew 256 B (#46526). Sizing to the
    // real per-call requirement reclaims it (~6 KB/core across both scratches)
    // — far more than the 192 B overlap — and still scales correctly up to the
    // host-side-validated MAX_GLOBAL_EXPERTS limit.
    const uint32_t counts_num_entries = static_cast<uint32_t>(t.counts.logical_shape()[-1]);
    const uint32_t idx_num_entries = static_cast<uint32_t>(t.global_expert_idx_table.logical_shape()[-1]);
    const uint32_t counts_scratch_bytes = std::max<uint32_t>(
        static_cast<uint32_t>(counts_buffer->aligned_page_size()),
        counts_num_entries * static_cast<uint32_t>(sizeof(uint32_t)));
    const uint32_t idx_scratch_bytes = std::max<uint32_t>(
        static_cast<uint32_t>(idx_buffer->aligned_page_size()),
        idx_num_entries * static_cast<uint32_t>(sizeof(uint32_t)));
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = counts_scratch_bytes,
        .core_ranges = core_range_set,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CB_COUNTS_SCRATCH),
            .data_format = tt::DataFormat::UInt32,
            .page_size = counts_scratch_bytes,
        }},
    });
    // CB_IDX_SCRATCH holds the device-side global_expert_idx_table page so
    // reader/compute/writer can resolve `global_expert_id =
    // idx_table[local_expert_id]` without re-reading DRAM. Sized the same way:
    // a single-chip deployment can place all experts locally, so the idx table
    // may itself be up to MAX_GLOBAL_EXPERTS entries.
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = idx_scratch_bytes,
        .core_ranges = core_range_set,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CB_IDX_SCRATCH),
            .data_format = tt::DataFormat::UInt32,
            .page_size = idx_scratch_bytes,
        }},
    });

    // CB_START_SCRATCH holds the device-side `start` (expert_region_offsets)
    // page for the writer in direct-write mode. Same sizing rationale as the
    // counts scratch: expert_region_offsets is validated to have the same
    // length as counts, so size it to the real per-call requirement (lands the
    // tensor's page and holds every region-offset entry) rather than the old
    // fixed MAX_GLOBAL_EXPERTS floor.
    const uint32_t start_scratch_bytes = std::max<uint32_t>(
        static_cast<uint32_t>(start_buffer->aligned_page_size()),
        counts_num_entries * static_cast<uint32_t>(sizeof(uint32_t)));
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = start_scratch_bytes,
        .core_ranges = core_range_set,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CB_START_SCRATCH),
            .data_format = tt::DataFormat::UInt32,
            .page_size = start_scratch_bytes,
        }},
    });
    // Reader's `start` scratch. Same sizing; separate CB so
    // reader (NCRISC) and writer (BRISC) never share one scratch page.
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = start_scratch_bytes,
        .core_ranges = core_range_set,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CB_START_SCRATCH_READER),
            .data_format = tt::DataFormat::UInt32,
            .page_size = start_scratch_bytes,
        }},
    });

    // Bias CBs (FUSE_BIAS): one full per-core N-column slice each; single-buffered
    // (read once, reused across all chunks). gate/up: per_core_N_gu tiles; down:
    // per_core_N_d tiles. Bias broadcast across rows in the compute kernel.
    const bool fuse_bias = op.fuse_bias;
    if (fuse_bias) {
        // Validation enforces gate/up/down biases share one dtype, so all three CBs
        // (and the compute kernel's single unpack reconfig for gate/up) use it safely.
        const tt::DataFormat bias_df = tt::tt_metal::datatype_to_dataformat_converter(t.gate_biases[0].dtype());
        const uint32_t bias_tile_size = tt::tile_size(bias_df);
        make_cb(CB_GATE_BIAS, bias_df, /*tiles=*/per_core_N_gu, bias_tile_size);
        make_cb(CB_UP_BIAS, bias_df, /*tiles=*/per_core_N_gu, bias_tile_size);
        make_cb(CB_DOWN_BIAS, bias_df, /*tiles=*/per_core_N_d, bias_tile_size);
    }

    // -------------------------- kernel build ------------------------------
    // Reader compile-time args. Order must exactly match the layout the reader
    // kernel reads via get_compile_time_arg_val(idx) and the TensorAccessor
    // offsets it computes after the named-arg block.
    std::vector<uint32_t> reader_ct_args = {
        CB_IN0_X,
        CB_IN1_GATE,
        CB_IN1_UP,
        CB_IN1_DOWN,
        CB_IN0_DOWN_FULL,
        CB_COUNTS_SCRATCH,
        CB_IDX_SCRATCH,
        experts_per_chip,
        per_core_M,
        per_core_N_gu,
        per_core_N_d,
        K_gate_tiles,
        K_down_tiles,
        in0_block_w_gu,
        in0_block_w_d,
        N_gate_tiles_full,
        N_down_tiles_full,
        M_tiles_full,
        num_chunks,
        chunk_M_tiles,
        // CB_ACTIVATED — consumed by the reader during phase 4 L1 mcast.
        CB_ACTIVATED,
        // GRID_X — M-row mcast group size, used for both num_dests and the
        // NoC-table endpoint index.
        GRID_X,
        // K_down_tiles_padded — phase-4 K-loop bound. K dim of down is
        // padded to N_gate_padded so per-K-block sender = gx == kb holds.
        K_down_tiles_padded,
        // reader_reads_up — 1 only in LEGACY (reader issues the up DRAM read).
        static_cast<uint32_t>(reader_reads_up),
        // reader_mcasts_up — 1 in LEGACY and UP_SPLIT (reader NoC-0 mcasts up).
        static_cast<uint32_t>(reader_mcasts_up),
        // CB_START_SCRATCH_READER — L1 page holding the fetched `start` vector.
        CB_START_SCRATCH_READER,
        // x_is_row_major — 1 => x is ROW_MAJOR bf16; reader streams sticks into
        // CB_X_RM and compute tilizes. 0 => x is TILE bf8_b, read directly.
        static_cast<uint32_t>(op.x_is_row_major),
        // CB_X_RM — row-major bf16 staging (only allocated/used in row-major mode).
        CB_X_RM,
        // TILE_HEIGHT — rows (token-row sticks) per tile-row; sizes the reader's
        // row-major x reads and its token-count -> tile-row conversion.
        TILE,
        // X_RM_ELEM_BYTES — byte size of one row-major x element (x is bf16 in
        // the row-major path).
        tt::datum_size(tt::DataFormat::Float16_b),
        // DOWN_SPLIT: K-rows of each down block this RISC keeps; the writer reads the
        // rest on NoC 1. Equals in0_block_w_d when the split is off.
        down_split_k,  // 30
        // IN1_WRITER_MCAST: 1 => the writer runs the gate/up multicast on NoC 1.
        static_cast<uint32_t>(kWriterMcastsIn1),  // 31
        // Active-token band. Experts outside it are dropped like a zero count, so a
        // hybrid dispatch can hand this op one load regime and moe_fused_swiglu the other
        // over the SAME counts vector. Wide open by default.
        op.min_active_tokens,  // 32
        op.max_active_tokens,  // 33
        // DRAM ND shard width in tiles per weight stream, 0 when interleaved. Drives the
        // kernels' read coalescing only — both layouts run the same code path.
        gu_shard_w,  // 34
        d_shard_w,   // 35
    };
    tt::tt_metal::TensorAccessorArgs(x_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(gate_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(up_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(down_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(counts_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(idx_buffer).append_to(reader_ct_args);
    // `start` (= expert_region_offsets) accessor — appended last, matching the
    // reader's accessor stream. Always read: x is the shared dispatched buffer
    // and each expert's rows begin at start[global_id].
    tt::tt_metal::TensorAccessorArgs(start_buffer).append_to(reader_ct_args);

    // FUSE_BIAS: after the `start` accessor, append the 3 bias CB ids then the 3
    // bias tensor accessors (gate, up, down). The reader reads them at the
    // offset after start_args.next_compile_time_args_offset(). Only present when
    // fuse_bias — a distinct program (FUSE_BIAS define is in the cache key).
    std::map<std::string, std::string> reader_defines{};
    // adaptive_chunk divides the host-sized max_chunk by this; see kGridY.
    reader_defines["UNIFIED_RE_GRID_Y"] = std::to_string(GRID_Y);
    if (fuse_bias) {
        reader_ct_args.push_back(CB_GATE_BIAS);
        reader_ct_args.push_back(CB_UP_BIAS);
        reader_ct_args.push_back(CB_DOWN_BIAS);
        // Representative expert-0 bias buffers describe the layout; per-expert
        // bias base addresses are passed as runtime-arg arrays below.
        tt::tt_metal::TensorAccessorArgs(t.gate_biases[0].buffer()).append_to(reader_ct_args);
        tt::tt_metal::TensorAccessorArgs(t.up_biases[0].buffer()).append_to(reader_ct_args);
        tt::tt_metal::TensorAccessorArgs(t.down_biases[0].buffer()).append_to(reader_ct_args);
        reader_defines["FUSE_BIAS"] = "1";
    }
    tt::tt_metal::KernelDescriptor reader_descriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/kernels/dataflow/"
            "unified_routed_expert_ffn_reader.cpp",
        .source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = core_range_set,
        .compile_time_args = reader_ct_args,
        .defines = {reader_defines.begin(), reader_defines.end()},
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };

    // Writer compile-time args (must match writer's get_compile_time_arg_val order).
    std::vector<uint32_t> writer_ct_args = {
        CB_OUT,             // 0
        per_core_M,         // 1
        per_core_N_gu,      // 2
        per_core_N_d,       // 3
        d_out_subblock_h,   // 4
        d_out_subblock_w,   // 5
        N_gate_tiles_full,  // 6
        N_down_tiles_full,  // 7
        num_chunks,         // 8
        chunk_M_tiles,      // 9
        // device-side count read: writer also waits on the reader's push
        // and bounds its cb_out drain loop by effective_chunks so it does
        // not wait forever on chunks compute never pushes.
        CB_COUNTS_SCRATCH,  // 10
        CB_IDX_SCRATCH,     // 11
        experts_per_chip,   // 12
        // M_tiles_full: needed for the writer to skip OOB output writes when
        // M_tiles_full doesn't divide chunk_M_tiles. The last chunk runs
        // chunk_M_tiles rows per core, of which only those < M_tiles_full
        // correspond to real output rows in the tensor.
        M_tiles_full,  // 13
        // dst_M_tiles: tile-row count of the shared destination buffer, which
        // bounds the writer's destination rows.
        dst_M_tiles,       // 14
        CB_START_SCRATCH,  // 15
        // UP_SPLIT up-weight read: CB + dims let the writer replicate the gate
        // read on NoC 1, and writer_split_up gates it (1 = UP_SPLIT).
        CB_IN1_UP,                            // 16
        in0_block_w_gu,                       // 17
        K_gate_tiles,                         // 18
        static_cast<uint32_t>(up_mode == 2),  // 19 writer_split_up
        // DOWN_SPLIT down-weight read: the writer reads the UPPER K-rows of each down block
        // on NoC 1 while the reader reads the lower rows on NoC 0.
        CB_IN1_DOWN,            // 20
        in0_block_w_d,          // 21
        K_down_tiles,           // 22
        d_num_blocks,           // 23
        d_in1_block_num_tiles,  // 24
        down_split_k,           // 25 rows the READER keeps
        // IN1_WRITER_MCAST: cb_in1_gate so the writer can multicast it, plus the flag.
        CB_IN1_GATE,                              // 26
        static_cast<uint32_t>(kWriterMcastsIn1),  // 27 writer_mcasts_in1
        // Active-token band, after the DOWN_SPLIT block rather than at 20/21.
        op.min_active_tokens,  // 28
        op.max_active_tokens,  // 29
        // DRAM ND shard widths, matching the reader's args 34/35.
        gu_shard_w,  // 30
        d_shard_w,   // 31
        // Width of the LAST down N-subblock; equals d_out_subblock_w unless the subblock grid
        // is ragged (see the program factory). The drain must not wait on tiles the compute
        // kernel never packs.
        d_out_subblock_w_tail,  // 32
    };
    // Accessor compile-arg stream order MUST match the writer kernel:
    // out, then start, then up (UP_SPLIT).
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(start_buffer).append_to(writer_ct_args);
    // up accessor follows start; used only when the writer handles `up`.
    tt::tt_metal::TensorAccessorArgs(up_buffer).append_to(writer_ct_args);
    // DOWN_SPLIT: down accessor follows up in the writer's compile-arg stream.
    tt::tt_metal::TensorAccessorArgs(down_buffer).append_to(writer_ct_args);

    tt::tt_metal::KernelDescriptor writer_descriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/kernels/dataflow/"
            "unified_routed_expert_ffn_writer.cpp",
        .source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = core_range_set,
        .compile_time_args = writer_ct_args,
        .defines = {{"UNIFIED_RE_GRID_Y", std::to_string(GRID_Y)}},
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };

    // Compute kernel compile-time args: positional + named CB ids.
    std::vector<uint32_t> compute_ct_args = {
        // gate
        in0_block_w_gu,
        gu_in0_num_subblocks,
        gu_in0_block_num_tiles,
        gu_in0_subblock_num_tiles,
        gu_in1_num_subblocks,
        gu_in1_block_num_tiles,
        gu_in1_block_w,
        gu_num_blocks,
        // up
        in0_block_w_gu,
        gu_in0_num_subblocks,
        gu_in0_block_num_tiles,
        gu_in0_subblock_num_tiles,
        gu_in1_num_subblocks,
        gu_in1_block_num_tiles,
        gu_in1_block_w,
        gu_num_blocks,
        // down
        in0_block_w_d,
        d_in0_num_subblocks,
        d_in0_block_num_tiles,
        d_in0_subblock_num_tiles,
        d_in1_num_subblocks,
        d_in1_block_num_tiles,
        d_in1_block_w,
        d_num_blocks,
        // gate/up out subblock
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_block_num_tiles,
        // down out subblock
        d_out_subblock_h,
        d_out_subblock_w,
        d_out_subblock_w_tail,
        d_out_block_num_tiles,
        // chunk loop control
        num_chunks,
        experts_per_chip,
        chunk_M_tiles,
        // x_is_row_major — 1 => compute tilizes CB_X_RM -> CB_IN0_X before the
        // gate/up matmul. 0 => x already TILE in CB_IN0_X (no tilize).
        static_cast<uint32_t>(op.x_is_row_major),
        // Real (unpadded) down-K tile count. Lets the compute skip the down
        // matmul over the last K-block's tail padding tiles (zero-activated)
        // instead of computing dead MACs.
        K_down_tiles,
        op.min_active_tokens,
        op.max_active_tokens,
    };
    std::unordered_map<std::string, uint32_t> compute_named_args = {
        // Row-major bf16 x staging (x_is_row_major only); tilize input CB.
        {"cb_x_rm", CB_X_RM},
        {"cb_in0_x", CB_IN0_X},
        {"cb_in1_gate", CB_IN1_GATE},
        {"cb_in1_up", CB_IN1_UP},
        {"cb_in1_down", CB_IN1_DOWN},
        {"cb_gate_intermed", CB_GATE_INT},
        {"cb_up_intermed", CB_UP_INT},
        {"cb_activated", CB_ACTIVATED},
        {"cb_in0_down_full", CB_IN0_DOWN_FULL},
        {"cb_mm_partials_gu", CB_PARTIALS_GU},
        {"cb_mm_partials_up", CB_PARTIALS_UP},
        {"cb_mm_partials_d", CB_PARTIALS_D},
        {"cb_out", CB_OUT},
        // For device-side count read: compute waits on the reader's push and
        // bounds its chunk loop by effective_chunks = ceil(count/chunk_M_tiles).
        {"cb_counts_scratch", CB_COUNTS_SCRATCH},
        {"cb_idx_scratch", CB_IDX_SCRATCH},
        // Region size in tile-rows: caps the device-provided per-expert count
        // (adaptive_chunk::clamp_count_tiles). Reader and writer take it as a
        // positional CT arg; compute has no free positional slot, so it is named.
        {"m_tiles_full", M_tiles_full},
    };
    if (fuse_bias) {
        compute_named_args["cb_gate_bias"] = CB_GATE_BIAS;
        compute_named_args["cb_up_bias"] = CB_UP_BIAS;
        compute_named_args["cb_down_bias"] = CB_DOWN_BIAS;
    }

    // PACKER_L1_ACC controls cross-K-block accumulation via packer L1 RMW.
    std::map<std::string, std::string> compute_defines{};
    compute_defines["UNIFIED_RE_GRID_Y"] = std::to_string(GRID_Y);
    compute_defines["PACKER_L1_ACC"] = "1";
    // Dst-accumulator mode -> compute kernel: the fused-binary-activation dst budget and
    // the SFPU fp32-dest template derive from this, staying in sync with
    // DST_CAPACITY / ComputeConfig.fp32_dest_acc_en (single source above).
    compute_defines["FP32_DEST_ACC_EN"] = kFp32DestAccEn ? "1" : "0";
    if (op.activation == RoutedExpertActivation::SwiGluOai) {
        // SwiGLU-OAI activation (MiniMax-M3 / gpt-oss): clamp(gate,max=L),
        // clamp(up,±L), (up+1)*gate*sigmoid(alpha*gate). Bakes alpha=1.702,
        // limit=7.0 (SwiGLUConfigGPTOSS) in the kernel.
        compute_defines["SWIGLU_OAI"] = "1";
    } else if (op.activation == RoutedExpertActivation::SituGlu) {
        // SiTU-GLU (Kimi K3), with beta_gate=4.0 / beta_up=25.0 baked into the kernel.
        compute_defines["SITU_GLU"] = "1";
    } else if (op.activation == RoutedExpertActivation::ClampedSiluGlu) {
        // Clamped SiLU-GLU (DeepSeek V4), with limit=10.0 (ClampedSiluGluConfigDsV4) baked
        // into the kernel.
        compute_defines["CLAMPED_SILU_GLU"] = "1";
    }
    if (fuse_bias) {
        // FUSE_BIAS: add gate/up bias (broadcast across rows) before the fused binary
        // activation and down bias after the down matmul. Validation restricts this to
        // the activations that have that branch.
        compute_defines["FUSE_BIAS"] = "1";
    }

    tt::tt_metal::KernelDescriptor compute_descriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/kernels/compute/"
            "fused_swiglu.cpp",
        .source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = core_range_set,
        .compile_time_args = compute_ct_args,
        .named_compile_time_args = {compute_named_args.begin(), compute_named_args.end()},
        .defines = {compute_defines.begin(), compute_defines.end()},
        .config =
            tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = MathFidelity::LoFi,
                .fp32_dest_acc_en = kFp32DestAccEn,
                .math_approx_mode = false,
            },
    };

    // -------------------------- per-core runtime args ---------------------
    // Cross-core synchronization is now done entirely via L1-mcast: weights
    // mcast within N-col groups using in1_{ready,valid}_sem, x mcast within
    // M-row groups using in0_{ready,valid}_sem, and activated mcast within
    // M-row groups (rotating sender per phase-4 K-block) using
    // act_{ready,valid}_sem. No global cross-grid barrier is needed.
    std::vector<CoreCoord> cores;
    cores.reserve(GRID_X * GRID_Y);
    for (uint32_t gy = 0; gy < GRID_Y; ++gy) {
        for (uint32_t gx = 0; gx < GRID_X; ++gx) {
            cores.push_back(CoreCoord{ORIGIN_X + gx, ORIGIN_Y + gy});
        }
    }

    auto* device = t.x.device();

    for (uint32_t idx = 0; idx < cores.size(); ++idx) {
        const auto& core = cores[idx];
        const uint32_t gy = idx / GRID_X;
        const uint32_t gx = idx % GRID_X;
        const uint32_t my_mt = gy;
        const uint32_t my_nt_gu = gx;
        const uint32_t my_nt_d = gx;

        // ---------------- DRAM-read sender placement ----------------
        // Every DRAM read here is issued by a "sender" core that multicasts the block
        // to the cores sharing it: one per N-column for the weights (gate/up/down,
        // shared down the column) and one per M-row for x (shared across the row).
        //
        // Placement matters because these reads are ISSUE-bound, not bandwidth-bound: a
        // 576 B bfp4 tile needs ~9 cycles of a 64 B/cycle NoC port but ~43 cycles of
        // RISC time to push into the command buffer, so a sender delivers only
        // ~13 B/cycle (~20% of its port) and its RISC, not DRAM, is the limit. With the
        // weight senders all on row 0 and the x senders all on column 0, core (0,0)
        // owned BOTH streams and serially issued x + gate + down while cores off both
        // lines sat nearly idle.
        //
        // So stagger the two sender sets so no core is in both:
        //   weight sender for column gx -> row    gx % GRID_Y
        //   x      sender for row    gy -> column (gy + 1) % GRID_X
        // A core is in both only if gx == (gx % GRID_Y + 1) % GRID_X, which has no
        // solution on the 11x8 grid; the TT_FATAL keeps that honest for other grids.
        const uint32_t in1_sender_row = gx % GRID_Y;
        const uint32_t in0_sender_col = (gy + 1) % GRID_X;
        TT_FATAL(
            !(gy == in1_sender_row && gx == in0_sender_col),
            "sender placement collision at ({}, {}): a core must not own both the weight and x DRAM read",
            gx,
            gy);

        const bool is_in1_sender = (gy == in1_sender_row);
        const auto sender_noc =
            device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + gx, ORIGIN_Y + in1_sender_row});
        // The rectangle spans the WHOLE column, sender included: a multicast rectangle
        // must be contiguous and the sender is no longer on an edge row, so "everything
        // but the sender" is not expressible as one rectangle. The non-loopback
        // multicast drops the sender's own copy (it already holds the block), so
        // num_dests stays GRID_Y - 1.
        const auto first_recv_noc = device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + gx, ORIGIN_Y});
        const auto last_recv_noc =
            device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + gx, ORIGIN_Y + GRID_Y - 1});
        const uint32_t in1_num_receivers = GRID_Y - 1;
        const uint32_t in1_mcast_nx_start = first_recv_noc.x;
        const uint32_t in1_mcast_ny_start = first_recv_noc.y;
        const uint32_t in1_mcast_nx_end = last_recv_noc.x;
        const uint32_t in1_mcast_ny_end = last_recv_noc.y;
        const uint32_t in1_sender_nx = sender_noc.x;
        const uint32_t in1_sender_ny = sender_noc.y;

        // x (in0) multicast: per M-row, sender at in0_sender_col; the rectangle spans
        // the whole row for the same reason as the weight column above.
        const bool is_in0_sender = (gx == in0_sender_col);
        const auto in0_sender_noc =
            device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + in0_sender_col, ORIGIN_Y + gy});
        const auto in0_first_recv_noc = device->worker_core_from_logical_core(CoreCoord{ORIGIN_X, ORIGIN_Y + gy});
        const auto in0_last_recv_noc =
            device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + GRID_X - 1, ORIGIN_Y + gy});
        const uint32_t in0_num_receivers = GRID_X - 1;
        const uint32_t in0_mcast_nx_start = in0_first_recv_noc.x;
        const uint32_t in0_mcast_ny_start = in0_first_recv_noc.y;
        const uint32_t in0_mcast_nx_end = in0_last_recv_noc.x;
        const uint32_t in0_mcast_ny_end = in0_last_recv_noc.y;
        const uint32_t in0_sender_nx = in0_sender_noc.x;
        const uint32_t in0_sender_ny = in0_sender_noc.y;

        // Reader runtime arg layout (must match unified_routed_expert_ffn_reader.cpp):
        //   0..2: tensor addrs (x, counts, idx). The per-expert gate/up/down
        //     base addresses are passed as the arrays after start_addr below.
        //   3: my_mt
        //   4: my_nt_gu
        //   5: my_nt_d
        //   6..15: in1 multicast args
        //  16..25: in0 multicast args
        //  26: act_ready_sem_id  27: act_valid_sem_id
        //  28: up_go_sem_id  29: up_done_sem_id
        //  30..36: COUNTS_BCAST (is_counts_reader, rect x0,y0,x1,y1, sem, receivers)
        //  37..38: DOWN_SPLIT go/done sem ids
        //  39..39+2*GRID_X-1: M-row NoC coord table (GRID_X pairs of x, y)
        //  39+2*GRID_X: start_addr (expert_region_offsets)
        tt::tt_metal::KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(x_buffer);
        reader_args.push_back(counts_buffer);
        reader_args.push_back(idx_buffer);
        reader_args.append(std::vector<uint32_t>{
            my_mt,
            my_nt_gu,
            my_nt_d,
            static_cast<uint32_t>(is_in1_sender),
            in1_ready_sem_id,
            in1_valid_sem_id,
            in1_num_receivers,
            in1_mcast_nx_start,
            in1_mcast_ny_start,
            in1_mcast_nx_end,
            in1_mcast_ny_end,
            in1_sender_nx,
            in1_sender_ny,
            static_cast<uint32_t>(is_in0_sender),
            in0_ready_sem_id,
            in0_valid_sem_id,
            in0_num_receivers,
            in0_mcast_nx_start,
            in0_mcast_ny_start,
            in0_mcast_nx_end,
            in0_mcast_ny_end,
            in0_sender_nx,
            in0_sender_ny,
            act_ready_sem_id,
            act_valid_sem_id,
            // UP_SPLIT local same-core handshake sems (0 when unused).
            up_go_sem_id,
            up_done_sem_id,
        });
        // COUNTS_BCAST args (7, at COUNTS_BCAST_RT). Every core needs counts/idx to
        // derive its chunking, but all of them reading the same two DRAM pages at once
        // serialises on one bank. Logical (0,0) reads them and multicasts to a rectangle
        // spanning the whole worker grid; the non-loopback multicast drops the sender's
        // own copy, so num_dests is one less than the grid.
        {
            const auto grid_first = device->worker_core_from_logical_core(CoreCoord{ORIGIN_X, ORIGIN_Y});
            const auto grid_last =
                device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + GRID_X - 1, ORIGIN_Y + GRID_Y - 1});
            reader_args.push_back(static_cast<uint32_t>(gx == 0 && gy == 0));  // is_counts_reader
            reader_args.push_back(static_cast<uint32_t>(grid_first.x));
            reader_args.push_back(static_cast<uint32_t>(grid_first.y));
            reader_args.push_back(static_cast<uint32_t>(grid_last.x));
            reader_args.push_back(static_cast<uint32_t>(grid_last.y));
            reader_args.push_back(counts_valid_sem_id);
            reader_args.push_back(GRID_X * GRID_Y - 1);
        }
        // DOWN_SPLIT go/done sems (dedicated; see down_go_sem_id).
        reader_args.push_back(down_go_sem_id);
        reader_args.push_back(down_done_sem_id);
        // IN1_WRITER_MCAST go/done sems.
        reader_args.push_back(mcast_go_sem_id);
        reader_args.push_back(mcast_done_sem_id);
        // M-row NoC coord table: for our M-row (gy=my_mt), the NoC (x, y) of
        // each of the GRID_X cores (gx=0..GRID_X-1). Reader uses this per
        // phase-4 K-block (kb=0..K_down_tiles_padded-1) to find the sender's
        // NoC addr and to build the M-row mcast rectangle.
        for (uint32_t gxi = 0; gxi < GRID_X; ++gxi) {
            const auto noc = device->worker_core_from_logical_core(CoreCoord{ORIGIN_X + gxi, ORIGIN_Y + gy});
            reader_args.push_back(static_cast<uint32_t>(noc.x));
            reader_args.push_back(static_cast<uint32_t>(noc.y));
        }
        // start_addr — reader arg at M_ROW_NOC_RT_OFFSET + 2*GRID_X (see layout
        // comment). Same buffer the writer gets. Always read (x is the shared
        // buffer; each expert's rows begin at start[global_id]).
        reader_args.push_back(start_buffer);
        // Per-expert weight base addresses, appended after start_addr in three
        // contiguous blocks of experts_per_chip each: gate[0..N), up[0..N),
        // down[0..N).
        for (uint32_t e = 0; e < experts_per_chip; ++e) {
            reader_args.push_back(t.gate_projs[e].buffer());
        }
        for (uint32_t e = 0; e < experts_per_chip; ++e) {
            reader_args.push_back(t.up_projs[e].buffer());
        }
        for (uint32_t e = 0; e < experts_per_chip; ++e) {
            reader_args.push_back(t.down_projs[e].buffer());
        }
        // FUSE_BIAS: per-expert bias base addresses in three further blocks
        // (gate_bias[0..N), up_bias[0..N), down_bias[0..N)) after the weights.
        if (fuse_bias) {
            for (uint32_t e = 0; e < experts_per_chip; ++e) {
                reader_args.push_back(t.gate_biases[e].buffer());
            }
            for (uint32_t e = 0; e < experts_per_chip; ++e) {
                reader_args.push_back(t.up_biases[e].buffer());
            }
            for (uint32_t e = 0; e < experts_per_chip; ++e) {
                reader_args.push_back(t.down_biases[e].buffer());
            }
        }
        reader_descriptor.emplace_runtime_args(core, reader_args);

        // Writer runtime arg layout (must match unified_routed_expert_ffn_writer.cpp):
        //   0: output_addr  1: my_mt  2: my_nt_d
        //   3: start_addr (expert_region_offsets)
        //   4: my_nt_gu  5: is_up_sender (gy==0)
        //   6: up_go_sem_id  7: up_done_sem_id  (UP_SPLIT local same-core handshake)
        //   8..8+N-1: per-expert `up` base addresses
        tt::tt_metal::KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(out_buffer);  // 0
        writer_args.append(std::vector<uint32_t>{
            my_mt,    // 1
            my_nt_d,  // 2
        });
        writer_args.push_back(start_buffer);  // 3
        writer_args.append(std::vector<uint32_t>{
            my_nt_gu,                              // 4
            static_cast<uint32_t>(is_in1_sender),  // 5 is_up_sender
            up_go_sem_id,                          // 6
            up_done_sem_id,                        // 7
        });
        // Per-expert `up` base addresses (UP_SPLIT)
        for (uint32_t e = 0; e < experts_per_chip; ++e) {
            writer_args.push_back(t.up_projs[e].buffer());
        }
        // DOWN_SPLIT: per-expert down base addresses follow the up block (DOWN_RT),
        // then the two dedicated go/done sem ids.
        for (uint32_t e = 0; e < experts_per_chip; ++e) {
            writer_args.push_back(t.down_projs[e].buffer());
        }
        writer_args.push_back(down_go_sem_id);
        writer_args.push_back(down_done_sem_id);
        // IN1_WRITER_MCAST (9): the NoC-0 rectangle (the writer swaps the corners for
        // NoC 1), receiver count, the shared in1 ready/valid sems, and its go/done pair.
        writer_args.push_back(in1_mcast_nx_start);
        writer_args.push_back(in1_mcast_ny_start);
        writer_args.push_back(in1_mcast_nx_end);
        writer_args.push_back(in1_mcast_ny_end);
        writer_args.push_back(in1_num_receivers);
        writer_args.push_back(in1_ready_sem_id);
        writer_args.push_back(in1_valid_sem_id);
        writer_args.push_back(mcast_go_sem_id);
        writer_args.push_back(mcast_done_sem_id);
        writer_descriptor.emplace_runtime_args(core, writer_args);

        // Compute: how many of this core's N subblocks hold REAL output columns.
        // per_core_N is the GRID-ceil'd width, so the highest-gx cores own phantom
        // columns past the true N; their weights are never read from DRAM, so
        // MACing them just burns cycles on stale L1. A subblock straddling the
        // boundary still has real columns, hence the ceil — only wholly-phantom
        // subblocks are skipped. The compute keeps its FULL-width reserve/push
        // (cb_activated feeds the fixed-size activated mcast, cb_out the writer's
        // fixed-size drain); this bounds the MAC only.
        const uint32_t valid_n_gu = std::min(
            per_core_N_gu,
            (N_gate_tiles_full > my_nt_gu * per_core_N_gu) ? N_gate_tiles_full - my_nt_gu * per_core_N_gu : 0u);
        const uint32_t valid_n_d = std::min(
            per_core_N_d,
            (N_down_tiles_full > my_nt_d * per_core_N_d) ? N_down_tiles_full - my_nt_d * per_core_N_d : 0u);
        compute_descriptor.emplace_runtime_args(
            core,
            {(valid_n_gu + gu_out_subblock_w - 1) / gu_out_subblock_w,  // 0 gu valid N subblocks
             (valid_n_d + d_out_subblock_w - 1) / d_out_subblock_w});   // 1 down valid N subblocks
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
}

void validate(const UnifiedRoutedExpertFfnParams& op, const UnifiedRoutedExpertFfnInputs& t) {
    TT_FATAL(t.x.storage_type() == ttnn::StorageType::DEVICE, "x must be on device");
    // Scoped to Blackhole, matching ttnn::softcap / ttnn::situ_glu. The underlying SFPU
    // primitives exist on Wormhole, but that combination is unverified.
    if (op.activation == RoutedExpertActivation::SituGlu || op.activation == RoutedExpertActivation::ClampedSiluGlu) {
        TT_FATAL(
            t.x.device()->arch() == tt::ARCH::BLACKHOLE,
            "unified_routed_expert_ffn: SiTU-GLU and clamped SiLU-GLU are implemented for Blackhole only, got arch {}",
            t.x.device()->arch());
    }
    // x layout/dtype depends on x_is_row_major:
    //   false (default): x is TILE BFLOAT8_B — the reader reads tile pages directly.
    //   true: x is ROW_MAJOR BFLOAT16 (the dispatch output) — the reader streams
    //     sticks and the compute kernel tilizes them to bf8_b before the matmul,
    //     fusing the standalone to_layout. Off preserves the pre-fusion path for
    //     standalone / Wormhole callers.
    if (op.x_is_row_major) {
        TT_FATAL(
            t.x.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "x must be BFLOAT16 when x_is_row_major, got {}",
            t.x.dtype());
        TT_FATAL(
            t.x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "x must be ROW_MAJOR when x_is_row_major, got {}",
            t.x.layout());
    } else {
        TT_FATAL(t.x.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "x must be BFLOAT8_B, got {}", t.x.dtype());
        TT_FATAL(t.x.layout() == tt::tt_metal::Layout::TILE, "x must be TILE layout");
    }
    TT_FATAL(is_dram_interleaved(t.x), "x must be DRAM-interleaved");
    TT_FATAL(t.x.logical_shape().rank() >= 2, "x must have rank >= 2, got rank {}", t.x.logical_shape().rank());
    // For rank > 2, all leading dims must be 1 — we treat x as effectively
    // (M, K) using padded_shape[-2:].
    for (int i = 0; i < static_cast<int>(t.x.logical_shape().rank()) - 2; ++i) {
        TT_FATAL(t.x.logical_shape()[i] == 1, "x leading dim {} must be 1, got {}", i, t.x.logical_shape()[i]);
    }

    // Per-local-expert weight lists: one entry per expert, all identical shape.
    // The kernels loop over experts and index a per-expert address array, so the
    // three lists must be non-empty, equal-length, and match experts_per_chip.
    TT_FATAL(
        !t.gate_projs.empty() && t.gate_projs.size() == t.up_projs.size() && t.gate_projs.size() == t.down_projs.size(),
        "gate/up/down projection lists must be non-empty and equal length (got {}, {}, {})",
        t.gate_projs.size(),
        t.up_projs.size(),
        t.down_projs.size());
    TT_FATAL(
        t.gate_projs.size() == op.experts_per_chip,
        "weight-list length ({}) must equal experts_per_chip ({})",
        t.gate_projs.size(),
        op.experts_per_chip);

    const auto& x_shape = t.x.padded_shape();
    const auto& gate_shape = t.gate_projs[0].padded_shape();
    const auto& up_shape = t.up_projs[0].padded_shape();
    const auto& down_shape = t.down_projs[0].padded_shape();

    TT_FATAL(
        x_shape[-1] == gate_shape[-2] && x_shape[-1] == up_shape[-2],
        "x's last dim {} must match gate/up's K dim ({}, {})",
        x_shape[-1],
        gate_shape[-2],
        up_shape[-2]);
    TT_FATAL(
        gate_shape[-1] == up_shape[-1] && gate_shape[-1] == down_shape[-2],
        "gate/up N ({}) must equal down K ({})",
        gate_shape[-1],
        down_shape[-2]);
    TT_FATAL(down_shape[-1] == x_shape[-1], "down N ({}) must equal x K ({})", down_shape[-1], x_shape[-1]);

    constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;
    TT_FATAL(x_shape[-2] % TILE == 0, "x M ({}) must be tile-aligned", x_shape[-2]);
    // m_tiles is this expert's M (grid/chunk/CB sizing). x may be a shared
    // buffer spanning many experts, so its allocated M only bounds m_tiles from
    // above — the reader/writer index into x at the region offset.
    TT_FATAL(op.m_tiles > 0, "m_tiles must be > 0");
    TT_FATAL(
        op.m_tiles <= x_shape[-2] / TILE, "m_tiles ({}) must be <= x M in tiles ({})", op.m_tiles, x_shape[-2] / TILE);

    // Every expert's gate/up/down tensor shares x's storage / layout / memory
    // contract AND must be identical in shape/dtype to expert 0 (the program is
    // built once for all experts, and the kernels reuse one accessor layout
    // descriptor per role with only the base address varying per expert).
    for (uint32_t e = 0; e < op.experts_per_chip; ++e) {
        for (const auto& [name, w, ref] :
             std::initializer_list<std::tuple<const char*, const ttnn::Tensor&, const ttnn::Tensor&>>{
                 {"gate_proj", t.gate_projs[e], t.gate_projs[0]},
                 {"up_proj", t.up_projs[e], t.up_projs[0]},
                 {"down_proj", t.down_projs[e], t.down_projs[0]}}) {
            TT_FATAL(w.storage_type() == ttnn::StorageType::DEVICE, "{}[{}] must be on device", name, e);
            TT_FATAL(w.layout() == tt::tt_metal::Layout::TILE, "{}[{}] must be TILE layout", name, e);
            TT_FATAL(
                is_dram_interleaved(w) || is_dram_nd_sharded_by_tile_rows(w),
                "{}[{}] must be DRAM-interleaved or DRAM ND-sharded with a tile-aligned shard, got {}",
                name,
                e,
                w.memory_config());
            TT_FATAL(
                w.padded_shape() == ref.padded_shape() && w.dtype() == ref.dtype(),
                "{}[{}] shape/dtype ({}, {}) must match expert 0 ({}, {}) — all experts share one program",
                name,
                e,
                w.padded_shape(),
                w.dtype(),
                ref.padded_shape(),
                ref.dtype());
            // The page->bank map lives in the accessor layout descriptor, which is built once from
            // expert 0; only the base address varies per expert. An expert placed differently would
            // be read through expert 0's map and give silently wrong numbers rather than fail.
            TT_FATAL(
                w.memory_config() == ref.memory_config(),
                "{}[{}] memory config ({}) must match expert 0 ({}) — all experts share one program",
                name,
                e,
                w.memory_config(),
                ref.memory_config());
        }
    }

    // Aux tensors: counts / global_expert_idx_table are small UINT32 vectors
    // the reader fetches via DRAM accessor. The reader does a single
    // noc_async_read_page(page=0, ...) and then indexes anywhere in
    // [0, num_global_experts), so the full vector must fit in one page. The
    // L1 scratch CB is sized to hold MAX_GLOBAL_EXPERTS UINT32 entries (see
    // the program factory), which covers DeepSeek V3 (256), Kimi (384) and any
    // model up to MAX_GLOBAL_EXPERTS routed experts. Validate the length here
    // so larger expert counts produce a clean assertion instead of silent OOB
    // reads at runtime.
    for (const auto& [name, a] : std::initializer_list<std::pair<const char*, const ttnn::Tensor&>>{
             {"counts", t.counts}, {"global_expert_idx_table", t.global_expert_idx_table}}) {
        TT_FATAL(a.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(a.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32", name);
        TT_FATAL(is_dram_interleaved(a), "{} must be DRAM-interleaved", name);
        const uint32_t num_entries = a.logical_shape()[-1];
        TT_FATAL(
            num_entries <= MAX_GLOBAL_EXPERTS,
            "{} length ({}) exceeds the maximum supported number of experts ({}) — "
            "the reader fetches only page 0 of this tensor into a fixed-size L1 scratch",
            name,
            num_entries,
            MAX_GLOBAL_EXPERTS);
    }
    TT_FATAL(
        op.experts_per_chip <= t.global_expert_idx_table.logical_shape()[-1],
        "experts_per_chip ({}) must be <= idx_table size ({})",
        op.experts_per_chip,
        t.global_expert_idx_table.logical_shape()[-1]);

    // An inverted band drops every expert down the same count-0 path a genuine skip takes, so the
    // op would run to completion and write nothing rather than fail.
    TT_FATAL(
        op.min_active_tokens <= op.max_active_tokens,
        "unified_routed_expert_ffn: active-token band is inverted: min_active_tokens {} > "
        "max_active_tokens {}",
        op.min_active_tokens,
        op.max_active_tokens);

    // The kernels always read each expert's x slice at its region offset
    // (fusing ttnn::extract) and write that expert's output into `output` at the
    // same offset (fusing ttnn::insert), so expert_region_offsets is mandatory.
    // This op just writes each expert's region into whatever `output` it was given.
    TT_FATAL(
        t.expert_region_offsets.has_value(),
        "expert_region_offsets is required (the kernels read/write per-expert regions)");
    {
        const auto& start = *t.expert_region_offsets;
        // These mirror ttnn::insert's validate_index_tensor for the `start`
        // tensor: by fusing insert into this op, the FFN now owns the
        // region-offset vector the writer fetches device-side, so it must
        // enforce the same invariants insert did. The writer does a single
        // noc_async_read_page(page 0) and indexes start[global_id], which is
        // only correct for a contiguous ROW_MAJOR single-page UINT32 vector.
        TT_FATAL(start.storage_type() == ttnn::StorageType::DEVICE, "expert_region_offsets must be on device");
        TT_FATAL(start.dtype() == tt::tt_metal::DataType::UINT32, "expert_region_offsets must be UINT32");
        TT_FATAL(
            start.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "expert_region_offsets must be ROW_MAJOR layout, got {}",
            start.layout());
        TT_FATAL(is_dram_interleaved(start), "expert_region_offsets must be DRAM-interleaved");
        const auto& start_shape = start.logical_shape();
        const bool start_valid_1d = start_shape.rank() == 1;
        const bool start_valid_2d = start_shape.rank() == 2 && start_shape[0] == 1;
        TT_FATAL(
            start_valid_1d || start_valid_2d,
            "expert_region_offsets must be 1D or 2D with first dimension == 1, got shape {}",
            start_shape);
        TT_FATAL(
            static_cast<uint32_t>(start_shape[-1]) <= MAX_GLOBAL_EXPERTS,
            "expert_region_offsets length ({}) exceeds the maximum supported number of experts ({})",
            start_shape[-1],
            MAX_GLOBAL_EXPERTS);
        // The writer reads start[global_id] and counts[global_id] from the same
        // global-expert index space, so the two vectors must be the same length
        // (mirrors ttnn::insert's start/counts last-dim check).
        TT_FATAL(
            start_shape[-1] == t.counts.logical_shape()[-1],
            "expert_region_offsets length ({}) must equal counts length ({})",
            start_shape[-1],
            t.counts.logical_shape()[-1]);
    }

    {
        const auto& out = t.output;
        TT_FATAL(out.storage_type() == ttnn::StorageType::DEVICE, "output must be on device");
        TT_FATAL(out.layout() == tt::tt_metal::Layout::TILE, "output must be TILE layout");
        TT_FATAL(is_dram_interleaved(out), "output must be DRAM-interleaved");
        // Output dtype must match x EXCEPT in row-major mode: there x is bf16
        // ROW_MAJOR but the tilized output is bf8_b TILE (for downstream
        // combine), so the two legitimately differ. The tilize/down-matmul packs
        // to the output's dtype regardless.
        TT_FATAL(
            op.x_is_row_major || out.dtype() == t.x.dtype(),
            "output dtype ({}) must match x dtype ({})",
            out.dtype(),
            t.x.dtype());
        const auto& out_shape = out.padded_shape();
        TT_FATAL(
            out_shape.rank() == x_shape.rank(),
            "output rank ({}) must match x rank ({})",
            out_shape.rank(),
            x_shape.rank());
        // Common to both modes: the N (emb) dim and all leading dims must match
        // x — the writer's tile-row stride is out_shape[-1]/TILE, and leading
        // dims index the same logical (1,..,1,M,N) tensor.
        TT_FATAL(
            out_shape[-1] == x_shape[-1],
            "output last dim ({}) must match x last dim ({})",
            out_shape[-1],
            x_shape[-1]);
        for (int i = 0; i < static_cast<int>(out_shape.rank()) - 2; ++i) {
            TT_FATAL(
                out_shape[i] == x_shape[i],
                "output leading dim {} ({}) must match x ({})",
                i,
                out_shape[i],
                x_shape[i]);
        }
        constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
        TT_FATAL(out_shape[-2] % TILE_H == 0, "output M ({}) must be tile-aligned", out_shape[-2]);
        TT_FATAL(out_shape[-2] >= x_shape[-2], "output M ({}) must be >= x M ({})", out_shape[-2], x_shape[-2]);
    }

    // Optional per-local-expert expert biases (gpt-oss). All-or-none: the three
    // lists are all empty or all length == experts_per_chip. gate/up bias last
    // dim == gate/up N (hidden); down bias last dim == down N (emb). Same device
    // / TILE / DRAM-interleaved contract as weights, and (like the weights) all
    // experts' biases share one shape/dtype.
    const int bias_lists = static_cast<int>(!t.gate_biases.empty()) + static_cast<int>(!t.up_biases.empty()) +
                           static_cast<int>(!t.down_biases.empty());
    TT_FATAL(
        bias_lists == 0 || bias_lists == 3,
        "gate/up/down bias lists must all be provided together or all omitted (got {} of 3)",
        bias_lists);
    const bool has_bias = bias_lists == 3;
    TT_FATAL(
        op.fuse_bias == has_bias, "op.fuse_bias ({}) must match presence of bias lists ({})", op.fuse_bias, has_bias);
    if (has_bias) {
        TT_FATAL(
            t.gate_biases.size() == op.experts_per_chip && t.up_biases.size() == op.experts_per_chip &&
                t.down_biases.size() == op.experts_per_chip,
            "each bias list must have experts_per_chip ({}) entries (got {}, {}, {})",
            op.experts_per_chip,
            t.gate_biases.size(),
            t.up_biases.size(),
            t.down_biases.size());
        for (uint32_t e = 0; e < op.experts_per_chip; ++e) {
            for (const auto& [name, b, expected_n] :
                 std::initializer_list<std::tuple<const char*, const ttnn::Tensor&, uint32_t>>{
                     {"gate_bias", t.gate_biases[e], static_cast<uint32_t>(gate_shape[-1])},
                     {"up_bias", t.up_biases[e], static_cast<uint32_t>(up_shape[-1])},
                     {"down_bias", t.down_biases[e], static_cast<uint32_t>(down_shape[-1])}}) {
                TT_FATAL(b.storage_type() == ttnn::StorageType::DEVICE, "{}[{}] must be on device", name, e);
                TT_FATAL(b.layout() == tt::tt_metal::Layout::TILE, "{}[{}] must be TILE layout", name, e);
                TT_FATAL(is_dram_interleaved(b), "{}[{}] must be DRAM-interleaved", name, e);
                // Exact LOGICAL shape: a single row of exactly `expected_n` columns. The
                // padded-width check below is necessary (the kernel/reader address tiles by
                // padded width) but not sufficient: shapes like (2, N) or (1, N-1) tile-pad
                // to the same width and would otherwise be accepted and silently mis-applied
                // (the reader loads only tile-row 0 and the compute kernel row-broadcasts it).
                const auto& lshape = b.logical_shape();
                TT_FATAL(
                    static_cast<uint32_t>(lshape[-1]) == expected_n && lshape.volume() == expected_n,
                    "{}[{}] logical shape {} must be a single row of its projection N ({})",
                    name,
                    e,
                    lshape,
                    expected_n);
                TT_FATAL(
                    static_cast<uint32_t>(b.padded_shape()[-1]) == expected_n,
                    "{}[{}] padded last dim ({}) must match its projection N ({})",
                    name,
                    e,
                    b.padded_shape()[-1],
                    expected_n);
            }
            // All bias CBs are configured from the gate-bias dtype (and the compute
            // kernel reuses one unpack format across gate/up), so every bias must share
            // a single dtype; a mixed-dtype call would read the wrong byte counts/formats.
            TT_FATAL(
                t.up_biases[e].dtype() == t.gate_biases[0].dtype() &&
                    t.down_biases[e].dtype() == t.gate_biases[0].dtype() &&
                    t.gate_biases[e].dtype() == t.gate_biases[0].dtype(),
                "all gate/up/down biases must share one dtype");
        }
        // ClampedSiluGlu is excluded because DeepSeek-V4's experts are bias-free, not
        // because the kernel lacks a bias branch.
        TT_FATAL(
            op.activation == RoutedExpertActivation::SwiGluOai || op.activation == RoutedExpertActivation::SituGlu,
            "unified_routed_expert_moe: expert biases are enabled only for SwiGluOai and SituGlu.");
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

namespace {

constexpr auto kKernelRoot =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/kernels/";

MergedKernelSources merged_kernel_sources() {
    return MergedKernelSources{
        .reader = std::string(kKernelRoot) + "hybrid_reader.cpp",
        .writer = std::string(kKernelRoot) + "hybrid_writer.cpp",
        .compute = std::string(kKernelRoot) + "hybrid_compute.cpp",
    };
}

// The same L1 budget the unified half fits its blocking to: everything above the allocator base,
// less the scratch margin. Both halves are checked against it, so it is by construction at least
// as large as either half's buffers -- which is what the arena must hold, since the passes are
// laid out over it one at a time rather than side by side.
// The allocator base a Blackhole device gets at the default worker_l1_size, which is the
// configuration this op is validated at. A caller may only make the arena SMALLER than the
// default, never larger: the kernel-config ring is everything between the fixed firmware region
// and the allocator base, so it grows only as worker_l1_size shrinks. Checked rather than
// assumed: see validate_on_program_cache_miss.
constexpr uint32_t kDefaultWorkerL1Size = 1'461'248;
constexpr uint32_t kMinAllocatorBase = 111'616;

uint32_t arena_bytes_for(tt::tt_metal::IDevice* device) {
    // Everything above the allocator base, and nothing held back. The fused half sizes its
    // blocking against hal::get_max_worker_l1_unreserved_size() less L1_CB_RESERVE when it runs
    // standalone, and at the default worker_l1_size that is the SAME number as this -- the
    // kernel-config ring L1_CB_RESERVE stands for already sits below the base. So this arena is
    // exactly the budget that half would have had on its own, which is the invariant worth
    // holding: the merge must not make a shape unservable that either op serves alone. Holding
    // anything back here did precisely that, and cost the widest shape (emb 7168, hidden 3072 in
    // ROW_MAJOR) 6 KB it needed.
    const uint32_t reserved = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    TT_FATAL(
        device->l1_size_per_core() > reserved,
        "unexpected L1 geometry: l1_size_per_core ({}) <= reserved base ({})",
        device->l1_size_per_core(),
        reserved);
    const uint32_t usable = static_cast<uint32_t>(device->l1_size_per_core()) - reserved;
    // Whole 64B units, so the arena tensor's shard shape is exact.
    return usable & ~static_cast<uint32_t>(63);
}

PassBarrierPlan barrier_plan(tt::tt_metal::IDevice* device, const HybridRoutedExpertFfnParams& op) {
    const tt::tt_metal::CoreCoord master{0, op.origin_y};
    const auto master_noc = device->worker_core_from_logical_core(master);
    const auto first = device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{0, op.origin_y});
    const auto last =
        device->worker_core_from_logical_core(tt::tt_metal::CoreCoord{op.grid_x - 1, op.origin_y + op.grid_y - 1});
    const uint32_t cores = op.grid_x * op.grid_y;
    return PassBarrierPlan{
        .master_logical = master,
        .master_noc_x = static_cast<uint32_t>(master_noc.x),
        .master_noc_y = static_cast<uint32_t>(master_noc.y),
        .rect_x_start = static_cast<uint32_t>(first.x),
        .rect_y_start = static_cast<uint32_t>(first.y),
        .rect_x_end = static_cast<uint32_t>(last.x),
        .rect_y_end = static_cast<uint32_t>(last.y),
        // Non-loopback multicast drops the sender's own copy.
        .num_receivers = cores - 1,
        // Reader and writer both arrive on every core.
        .total_arrivals = 2 * cores,
    };
}

// The fused half rejects packer L1 accumulation, an fp32 dst accumulator and full sync outright:
// it drives L1 accumulation itself per K-block, needs all eight DEST tiles, and its row-major
// tilize path requires half sync. The shared model config sets packer_l1_acc, which the unified
// half wants, so the two halves are handed the same fidelity and approx mode with those three
// flags cleared for the fused one.
std::optional<ttnn::DeviceComputeKernelConfig> fused_compute_config(
    const std::optional<ttnn::DeviceComputeKernelConfig>& caller) {
    if (!caller.has_value()) {
        return std::nullopt;
    }
    ttnn::DeviceComputeKernelConfig cleared = *caller;
    cleared.fp32_dest_acc_en = false;
    cleared.packer_l1_acc = false;
    cleared.dst_full_sync_en = false;
    return cleared;
}

fused::OperationArguments fused_attributes(const HybridRoutedExpertFfnParams& op) {
    return fused::OperationArguments{
        .experts_per_chip = op.experts_per_chip,
        .m_tiles = op.m_tiles,
        // Pass A owns the low band of token counts, on the whole rectangle.
        .grid_x = op.grid_x,
        .grid_y = op.grid_y,
        .origin_x = 0,
        .origin_y = op.origin_y,
        // x is the shared dispatched buffer, so each expert's rows start at its region offset.
        .read_x_at_offset = true,
        // Pass A owns the low band: every expert at or below the threshold.
        .min_active_tokens = 0,
        .max_active_tokens = op.hybrid_token_threshold,
        .activation = op.activation,
        .fuse_bias = op.fuse_bias,
        .compute_kernel_config = fused_compute_config(op.compute_kernel_config),
    };
}

fused::TensorArguments fused_inputs(const HybridRoutedExpertFfnInputs& t) {
    return fused::TensorArguments{
        .activations = t.x,
        .w_gates = t.gate_projs,
        .w_ups = t.up_projs,
        .w_downs = t.down_projs,
        .gate_biases = t.gate_biases,
        .up_biases = t.up_biases,
        .down_biases = t.down_biases,
        .counts = t.counts,
        .global_expert_idx_table = t.global_expert_idx_table,
        .optional_output = t.output,
        .expert_region_offsets = t.expert_region_offsets,
    };
}

unified::UnifiedRoutedExpertFfnParams unified_attributes(const HybridRoutedExpertFfnParams& op) {
    // Pass B owns everything above the threshold. With no threshold the fused pass does not run,
    // so the band is left wide open rather than starting at 1 -- that keeps the program identical
    // to what the unified op alone would build, which is what the port is graded against.
    const bool fused_pass_runs = op.hybrid_token_threshold > 0;
    return unified::UnifiedRoutedExpertFfnParams{
        .m_tiles = op.m_tiles,
        .experts_per_chip = op.experts_per_chip,
        .x_is_row_major = op.x_is_row_major,
        .activation = op.activation,
        .fuse_bias = op.fuse_bias,
        .compute_kernel_config = op.compute_kernel_config,
        .min_active_tokens = fused_pass_runs ? op.hybrid_token_threshold + 1 : 0,
        .max_active_tokens = std::numeric_limits<uint32_t>::max(),
        // The same rectangle the fused half runs on: the two passes are ordered in time, not
        // split in space, so neither loses cores to the other.
        .grid_x = op.grid_x,
        .grid_y = op.grid_y,
        .origin_x = 0,
        .origin_y = op.origin_y,
    };
}

unified::UnifiedRoutedExpertFfnInputs unified_inputs(const HybridRoutedExpertFfnInputs& t) {
    return unified::UnifiedRoutedExpertFfnInputs{
        // Always the real x, never the output. The standalone op aliases the two on the TILE
        // path, but that is a convention of its entry point rather than something the kernels
        // need, and honouring it here would mean seeding the output with a copy of x -- a second
        // dispatch, which is the one thing this op exists to avoid.
        .x = t.x,
        .gate_projs = t.gate_projs,
        .up_projs = t.up_projs,
        .down_projs = t.down_projs,
        .counts = t.counts,
        .global_expert_idx_table = t.global_expert_idx_table,
        .output = t.output,
        .expert_region_offsets = t.expert_region_offsets,
        .gate_biases = t.gate_biases,
        .up_biases = t.up_biases,
        .down_biases = t.down_biases,
    };
}

}  // namespace

void validate_arguments(const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t) {
    // Each half validates what it will actually be handed, including its own band, so a merged
    // dispatch cannot pass a configuration either op alone would reject.
    unified::validate(unified_attributes(op), unified_inputs(t));
    if (op.hybrid_token_threshold > 0) {
        fused::validate(fused_attributes(op), fused_inputs(t));

        // A union program carries BOTH halves' kernel binaries, so its config is far larger than
        // either op's alone, and the kernel-config ring has to hold it. It fits the ring the
        // device gets by default with a little room to spare, so the only way to break it is to
        // open the device with a LARGER arena than the default, which moves the allocator base
        // down and takes those bytes off the ring. Left to tt_metal that surfaces as "Program
        // size too large for kernel config buffer", which reports two numbers but not the knob
        // that moves them.
        const uint32_t base = t.x.device()->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        TT_FATAL(
            base >= kMinAllocatorBase,
            "hybrid_routed_expert_ffn needs the device opened with worker_l1_size <= {} (the Blackhole default): "
            "the union program's kernel config does not fit the kernel-config ring otherwise. The allocator base "
            "is at {} but this op needs it at {} or above.",
            kDefaultWorkerL1Size,
            base,
            kMinAllocatorBase);
    }
}

tt::tt_metal::ProgramDescriptor create_hybrid_program_descriptor(
    const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t, ttnn::Tensor& output) {
    // Both implementations, ONE program, ONE dispatch -- the two-op forward folded into a single
    // launch so the layer can be overlapped with combine.
    //
    // The halves are NOT placed side by side: a program holds at most one kernel per processor
    // per core and both want all 88, so their bodies are compiled into one binary per RISC-V and
    // run in sequence, pass A then a grid-wide barrier then pass B. Each half therefore sees the
    // whole grid, exactly as it does when the two ops are dispatched back to back.
    const bool run_fused_pass = op.hybrid_token_threshold > 0;

    tt::tt_metal::ProgramDescriptor unified_descriptor;
    uint32_t next_semaphore_id = 0;
    unified::append_to_descriptor(
        unified_descriptor, next_semaphore_id, unified_attributes(op), unified_inputs(t), output);

    // No expert reaches the fused half, so it is not carried at all -- not merged, not compiled
    // into the binaries, not present in the defines. Folding it in anyway would be wrong, not
    // merely wasteful: the merge unions both halves' compile-time defines, and the fused half is
    // built here from an activation it may not implement (its own validate is skipped for the
    // same reason it is not running). That silently reshapes the unified half's binary.
    if (!run_fused_pass) {
        return unified_descriptor;
    }

    // Each half is built into its OWN descriptor and then folded in, rather than both appending
    // to one: the fold has to see them separately to pair their kernels by processor class and to
    // join each pair's argument lists behind the right base.
    tt::tt_metal::ProgramDescriptor fused_descriptor;
    fused::append_to_descriptor(fused_descriptor, fused_attributes(op), fused_inputs(t), output);

    TT_FATAL(
        t.l1_arena.has_value(),
        "pass A runs, so both halves' circular buffers need the caller-owned L1 arena to share");

    MergeReport report;
    auto merged = merge_halves(
        std::move(fused_descriptor),
        std::move(unified_descriptor),
        merged_kernel_sources(),
        /*run_fused_pass=*/true,
        t.l1_arena->buffer(),
        barrier_plan(t.x.device(), op),
        report);

    // The merge's own numbers, on a program-cache miss only. Every one of them is a silent-failure
    // surface: a wrong argument base reads the other half's arguments, and the arena footprint is
    // checked inside the merge but is the thing a future CB change will breach first.
    log_debug(
        tt::LogOp,
        "hybrid routed expert: arena {} B/core, {} semaphores (barrier id {}), bases "
        "reader(ct={},rt={}) writer(ct={},rt={}) compute(ct={},rt={})",
        report.arena_bytes_per_core,
        report.semaphore_count,
        report.barrier_semaphore_id,
        report.reader.ct,
        report.reader.rt,
        report.writer.ct,
        report.writer.rt,
        report.compute.ct,
        report.compute.rt);
    return merged;
}

uint32_t hybrid_l1_arena_bytes(tt::tt_metal::IDevice* device) { return arena_bytes_for(device); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
