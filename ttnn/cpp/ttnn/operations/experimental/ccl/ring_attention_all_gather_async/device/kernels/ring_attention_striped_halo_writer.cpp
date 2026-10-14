// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ring_attention_all_gather_metadata.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/striped_sliding_halo.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_inputs = get_compile_time_arg_val(4);
constexpr uint32_t unicast_route_arg0 = get_compile_time_arg_val(5);
constexpr uint32_t unicast_route_arg1 = get_compile_time_arg_val(6);
constexpr bool send_backward = get_compile_time_arg_val(7);
constexpr uint32_t meta_cb_id = get_compile_time_arg_val(8);
// Compile argument 9 is the legacy multicast flag; striped routing is derived below.

constexpr uint32_t page_size_base_idx = 10;

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

void kernel_main() {
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();
    // See the halo reader: appended after the per-output accessors, with a valid fallback offset so the
    // unconditionally-instantiated TensorAccessorArgs<> never names a non-accessor compile arg. Reader and
    // writer use the same link ranges to keep cb_output page counts equal.
    // The tuple itself has no offset accessor; the last element's next offset is where the
    // accessors end.
    constexpr uint32_t halo_meta_flag_idx = std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset();
    constexpr bool has_halo_metadata = get_compile_time_arg_val(halo_meta_flag_idx) == 1;
    constexpr uint32_t kv_meta_args_offset =
        has_halo_metadata ? halo_meta_flag_idx + 1 : page_size_base_idx + num_inputs;
    constexpr auto kv_meta_args = TensorAccessorArgs<kv_meta_args_offset>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_link = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> output_batch_head_stride_pages;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    // Each destination retains full rank-major cache addresses.
    std::array<uint32_t, num_inputs> output_origin_page;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        output_batch_head_stride_pages[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        arg_idx += 2;  // legacy link page range
        output_origin_page[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        arg_idx += 2;  // legacy second origin and tail size
    }

    ring_joint::StripedHaloGeometry geometry{
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++),
        get_arg_val<uint32_t>(arg_idx++)};
    std::array<uint32_t, num_inputs> input_widths;
    for (uint32_t input = 0; input < num_inputs; ++input) {
        input_widths[input] = get_arg_val<uint32_t>(arg_idx++);
    }
    if constexpr (has_halo_metadata) {
        const uint32_t kv_actual_isl_addr = get_arg_val<uint32_t>(arg_idx++);
        arg_idx += 8 + num_inputs;
        Noc meta_noc;
        CircularBuffer cb_meta(meta_cb_id);
        const uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
            meta_noc, kv_meta_args, kv_actual_isl_addr, cb_meta.get_write_ptr());
        geometry.prefix_tokens = trace_metadata::bounded_kv_actual_isl(
            kv_actual_isl, geometry.q_local_rows * geometry.ring_size, geometry.cache_local_rows * geometry.ring_size);
        geometry.logical_tokens = 32 * trace_metadata::logical_tile_rows_clamped_to_cache(
                                           geometry.prefix_tokens,
                                           geometry.q_local_rows * geometry.ring_size,
                                           geometry.cache_local_rows * geometry.ring_size);
    }

    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);
    // Concrete accessors for 1 or 2 inputs avoid the abstract wrapper's indirect call per page.
    const auto output_noc_addr = [&](uint32_t input_idx, uint32_t page) -> uint64_t {
        if constexpr (num_inputs == 1) {
            return std::get<0>(outputs_tuple).get_noc_addr(page);
        } else if constexpr (num_inputs == 2) {
            return input_idx == 0 ? std::get<0>(outputs_tuple).get_noc_addr(page)
                                  : std::get<num_inputs - 1>(outputs_tuple).get_noc_addr(page);
        } else {
            return output_addrgens[input_idx].get_noc_addr(page);
        }
    };
    size_t fabric_args_idx = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(fabric_args_idx);

    Noc noc_obj;
    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    CircularBuffer cb_output(cb_output_id);

    // packet header cb
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);

    // pre-populate packet headers
    constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info = {
        .dst_mesh_id = static_cast<uint16_t>(unicast_route_arg0),
        .dst_chip_id = static_cast<uint16_t>(unicast_route_arg1)};

    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);

    fabric_connection.open();

    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection =
        send_backward ? fabric_connection.get_backward_connection() : fabric_connection.get_forward_connection();
    const uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    // Send consecutive source pages into the same rank-major interval on every receiver
    // in this multicast run. The final packet also publishes readiness.
    static_assert(packet_size_in_pages <= NOC_SCATTER_WRITE_MAX_CHUNKS);
    uint32_t output_batch_head_base = 0;
    const auto send_pages = [&](uint32_t input_idx, uint32_t first_page, uint32_t count, size_t l1_addr, bool signal) {
        if (count == 0) {
            return;
        }
        std::array<uint64_t, NOC_SCATTER_WRITE_MAX_CHUNKS> noc_addrs{};
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = output_origin_page[input_idx] + first_page + i;
            noc_addrs[i] = output_noc_addr(input_idx, output_batch_head_base + tile);
        }
        // Pages that land back to back go as one plain write: one address to encode instead of a scatter of `count`.
        bool contiguous = true;
        for (uint32_t i = 1; contiguous && i < count; ++i) {
            contiguous = noc_addrs[i] == noc_addrs[0] + i * output_page_size;
        }
        const uint32_t bytes = output_page_size * count;
        const uint16_t page = static_cast<uint16_t>(output_page_size);
        if (contiguous && signal) {
            pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    noc_addrs[0], out_ready_sem_noc_addr_in_pkt, 1, true},
                bytes);
        } else if (contiguous) {
            pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc_addrs[0]}, bytes);
        } else if (signal) {
            ASSERT(count == 2);
            pkt_hdr->to_noc_fused_unicast_scatter_write_atomic_inc(
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                    {noc_addrs[0], noc_addrs[1]}, out_ready_sem_noc_addr_in_pkt, {page}, 1, true},
                bytes);
        } else {
            std::array<uint16_t, NOC_SCATTER_WRITE_MAX_CHUNKS - 1> chunk_sizes;
            chunk_sizes.fill(page);
            pkt_hdr->to_noc_unicast_scatter_write(
                tt::tt_fabric::NocUnicastScatterCommandHeader(
                    noc_addrs.data(), chunk_sizes.data(), static_cast<uint8_t>(count)),
                bytes);
        }
        perform_payload_send(fabric_direction_connection, l1_addr, bytes, pkt_hdr);
        noc_async_writes_flushed();
    };

    for (uint32_t run_start = 0; run_start < geometry.peer_count;) {
        const auto rows = geometry.range(run_start);
        const uint32_t run_end = geometry.run_end(run_start, rows);
        ccl_routing_utils::line_multicast_route_info_t route{};
        route.start_distance_in_hops = static_cast<uint16_t>(run_start + 1);
        route.range_hops = static_cast<uint16_t>(run_end - run_start);
        ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr, route);
        run_start = run_end;
        bool signaled = false;
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            const uint32_t width = input_widths[input_idx];
            const auto pages = ring_attention_all_gather::compute_link_page_range(
                (rows.end - rows.begin) * width, geometry.num_links, worker_link);
            output_origin_page[input_idx] = (geometry.source_rank * geometry.cache_local_rows + rows.begin) * width;
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                uint32_t tiles_read = pages.start;
                output_batch_head_base = bh_idx * output_batch_head_stride_pages[input_idx];
                while (tiles_read < pages.end) {
                    const uint32_t count = std::min(pages.end - tiles_read, packet_size_in_pages);
                    cb_output.wait_front(packet_size_in_pages);
                    const size_t l1_read_addr = cb_output.get_read_ptr();
                    const bool last = input_idx + 1 == num_inputs && bh_idx + 1 == input_batch_head_count[input_idx] &&
                                      tiles_read + count == pages.end;
                    const uint32_t signal_pages = last ? std::min<uint32_t>(count, 2) : 0;
                    const uint32_t plain_pages = count - signal_pages;
                    send_pages(input_idx, tiles_read, plain_pages, l1_read_addr, false);
                    send_pages(
                        input_idx,
                        tiles_read + plain_pages,
                        signal_pages,
                        l1_read_addr + plain_pages * output_page_size,
                        true);
                    signaled |= last;
                    tiles_read += count;
                    cb_output.pop_front(packet_size_in_pages);
                }
            }
        }
        if (!signaled) {
            // Empty windows (or links with no pages) still participate in the barrier.
            pkt_hdr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 1});
            ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr, route);
            fabric_direction_connection.wait_for_empty_write_slot();
            fabric_direction_connection.send_payload_flush_blocking_from_address(
                packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
        }
    }

    noc_obj.async_write_barrier();
    // Drain in-flight writes BEFORE closing the EDM connections.
    noc_obj.async_atomic_barrier();
    noc_obj.async_write_barrier();

    fabric_connection.close();
}
