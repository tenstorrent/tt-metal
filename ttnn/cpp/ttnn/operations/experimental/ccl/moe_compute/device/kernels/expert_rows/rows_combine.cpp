// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, the placement program's combine (data movement only, SingleDevice and FullCcl). Sender `sender` of
// num_senders takes the routed rows [sender R / num_senders, (sender + 1) R / num_senders) of the routing table and
// sends each whole row from the row buffer to page k * tokens_per_device + t % tokens_per_device of the combine output
// on token t's device: a NoC write when that is this device, else (COMBINE_FABRIC) the fabric. Rows go in batches of
// batch_rows through two staging buffers, the next batch read while the current one is sent. With the fabric, a
// sender visits the devices in its own order, so that the senders start on different devices.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "expert_rows_common.h"
#ifdef COMBINE_FABRIC
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/combine_fabric.hpp"
#endif

namespace {

constexpr uint32_t num_senders = get_named_compile_time_arg_val("num_senders");
constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");  // the call's tokens over the replicate group
constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
constexpr uint32_t row_stride = get_named_compile_time_arg_val("row_stride");  // a staged row, DRAM-aligned
constexpr uint32_t batch_rows = get_named_compile_time_arg_val("batch_rows");
constexpr uint32_t sender_rows_max = get_named_compile_time_arg_val("sender_rows_max");
// the routing table block rows_reader.cpp wrote
constexpr expert_rows::TableLayout table_block = expert_rows::table_layout(
    tokens, get_named_compile_time_arg_val("top_k"), get_named_compile_time_arg_val("local_experts"));
// byte offsets in the scratch CB
constexpr uint32_t scratch_head = get_named_compile_time_arg_val("scratch_head");
constexpr uint32_t scratch_tokens = get_named_compile_time_arg_val("scratch_tokens");
constexpr uint32_t scratch_k = get_named_compile_time_arg_val("scratch_k");
constexpr uint32_t scratch_order = get_named_compile_time_arg_val("scratch_order");
constexpr uint32_t scratch_stage = get_named_compile_time_arg_val("scratch_stage");

#ifdef COMBINE_FABRIC
constexpr bool use_init_semaphore = get_named_compile_time_arg_val("use_init_semaphore") == 1;
constexpr uint32_t group_devices = combine_fabric::replicate_group_devices;
constexpr uint32_t tokens_per_device = tokens / group_devices;
constexpr uint32_t self_in_group = combine_fabric::replicate_axis == combine_fabric::ReplicateGroup::ROWS
                                       ? combine_fabric::linearized_mesh_coord % combine_fabric::mesh_cols
                                       : combine_fabric::linearized_mesh_coord / combine_fabric::mesh_cols;
constexpr uint32_t max_group_devices = 32;
static_assert(group_devices <= max_group_devices, "a combine reduces over at most 32 devices");
#else
constexpr uint32_t tokens_per_device = tokens;
#endif

// Reads a u16 table part's entries [first, first + count) into L1 at dst; DRAM reads keep the source's offset modulo
// part_alignment, so the read starts at the aligned entry below `first`. Returns the L1 address of entry `first`.
template <typename TableAccessor>
FORCE_INLINE uint32_t read_table_slice(
    const Noc& noc, const TableAccessor& table, uint32_t part, uint32_t first, uint32_t count, uint32_t dst) {
    const uint32_t begin = part + first * expert_rows::u16_bytes;
    const uint32_t aligned_begin = begin / expert_rows::part_alignment * expert_rows::part_alignment;
    const uint32_t end = expert_rows::align_up(begin + count * expert_rows::u16_bytes, expert_rows::part_alignment);
    noc.async_read(
        table, CoreLocalMem<uint32_t>(dst), end - aligned_begin, {.page_id = 0, .offset_bytes = aligned_begin}, {});
    return dst + (begin - aligned_begin);
}

// Rows [batch batch_rows, ...) of this sender's `rows`, at most batch_rows.
FORCE_INLINE uint32_t batch_size(uint32_t batch, uint32_t rows) {
    const uint32_t begin = batch * batch_rows;
    return rows - begin < batch_rows ? rows - begin : batch_rows;
}

FORCE_INLINE uint32_t stage_of(uint32_t scratch, uint32_t batch) {
    return scratch + scratch_stage + (batch % 2) * batch_rows * row_stride;
}

// The batch's rows from the row buffer into its staging buffer.
template <typename RowsAccessor>
FORCE_INLINE void read_batch(
    const Noc& noc,
    const RowsAccessor& row_buffer,
    uint32_t scratch,
    const tt_l1_ptr uint16_t* order,
    uint32_t first,
    uint32_t rows,
    uint32_t batch) {
    for (uint32_t slot = 0; slot < batch_size(batch, rows); ++slot) {
        noc.async_read(
            row_buffer,
            CoreLocalMem<uint32_t>(stage_of(scratch, batch) + slot * row_stride),
            row_bytes,
            {.page_id = first + order[batch * batch_rows + slot]},
            {});
    }
}

#ifdef COMBINE_FABRIC
// The sender's n-th destination in its group: even senders walk up from self + offset, odd ones down; self comes last.
FORCE_INLINE uint32_t visit(uint32_t sender, uint32_t n) {
    if (group_devices == 1 || n == group_devices - 1) {
        return self_in_group;
    }
    const uint32_t offset = 1 + (sender / 2 + n) % (group_devices - 1);
    return sender % 2 == 0 ? (self_in_group + offset) % group_devices
                           : (self_in_group + group_devices - offset) % group_devices;
}

// The rows' local indices in visit order: a counting sort by destination device.
FORCE_INLINE void order_by_destination(
    uint32_t sender, const tt_l1_ptr uint16_t* row_tokens, uint32_t rows, tt_l1_ptr uint16_t* order) {
    uint32_t next[max_group_devices] = {};
    for (uint32_t row = 0; row < rows; ++row) {
        ++next[row_tokens[row] / tokens_per_device];
    }
    uint32_t first = 0;
    for (uint32_t n = 0; n < group_devices; ++n) {
        const uint32_t device = visit(sender, n);
        const uint32_t count = next[device];
        next[device] = first;
        first += count;
    }
    for (uint32_t row = 0; row < rows; ++row) {
        order[next[row_tokens[row] / tokens_per_device]++] = row;
    }
}
#endif

}  // namespace

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr auto rows_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), row_bytes);
    const auto row_buffer = TensorAccessor(rows_args, get_common_arg_val<uint32_t>(1), row_bytes);
    const auto table = TensorAccessor(table_args, get_common_arg_val<uint32_t>(2), table_block.bytes);

    size_t arg = 0;
    const uint32_t sender = get_arg_val<uint32_t>(arg++);
#ifdef COMBINE_FABRIC
    const uint32_t init_semaphore_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t final_semaphore_addr = get_arg_val<uint32_t>(arg++);  // the cross-device semaphore's slot
    const bool is_init_sync_core = get_arg_val<uint32_t>(arg++);
    combine_fabric::Connections fabric(arg);
    fabric.open();
    if (is_init_sync_core && use_init_semaphore) {
        fabric.send_init(init_semaphore_addr);
    }
#endif

    const Noc noc(1);
    CircularBuffer cb_scratch(get_named_compile_time_arg_val("cb_scratch"));
    cb_scratch.reserve_back(1);
    const uint32_t scratch = cb_scratch.get_write_ptr();

    noc.async_read(
        table, CoreLocalMem<uint32_t>(scratch + scratch_head), expert_rows::part_alignment, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t routed_rows = *reinterpret_cast<const tt_l1_ptr uint32_t*>(scratch + scratch_head);
    const uint32_t first = sender * routed_rows / num_senders;
    const uint32_t rows = (sender + 1) * routed_rows / num_senders - first;
    ASSERT(rows <= sender_rows_max);
    uint32_t tokens_addr = scratch + scratch_tokens, k_addr = scratch + scratch_k;
    if (rows) {  // a NoC read moves at least one byte
        tokens_addr = read_table_slice(noc, table, table_block.row_tokens, first, rows, tokens_addr);
        k_addr = read_table_slice(noc, table, table_block.row_k, first, rows, k_addr);
        noc.async_read_barrier();
    }
    const auto* row_tokens = reinterpret_cast<const tt_l1_ptr uint16_t*>(tokens_addr);
    const auto* row_k = reinterpret_cast<const tt_l1_ptr uint16_t*>(k_addr);

    auto* order = reinterpret_cast<tt_l1_ptr uint16_t*>(scratch + scratch_order);
#ifdef COMBINE_FABRIC
    order_by_destination(sender, row_tokens, rows, order);
    if constexpr (use_init_semaphore) {
        fabric.wait_init(noc, init_semaphore_addr, is_init_sync_core, num_senders);
    }
#else
    for (uint32_t row = 0; row < rows; ++row) {
        order[row] = row;
    }
#endif

    const uint32_t batches = expert_rows::div_up(rows, batch_rows);
    if (batches) {
        read_batch(noc, row_buffer, scratch, order, first, rows, 0);
    }
    for (uint32_t batch = 0; batch < batches; ++batch) {
        noc.async_read_barrier();
        if (batch + 1 < batches) {
            noc.async_writes_flushed();  // batch - 1's NoC writes left the stage buffer batch + 1 refills
            read_batch(noc, row_buffer, scratch, order, first, rows, batch + 1);
        }
        for (uint32_t slot = 0; slot < batch_size(batch, rows); ++slot) {
            const uint32_t row = order[batch * batch_rows + slot];
            const uint32_t token = row_tokens[row];
            const uint32_t page = row_k[row] * tokens_per_device + token % tokens_per_device;
            const uint32_t payload = stage_of(scratch, batch) + slot * row_stride;
#ifdef COMBINE_FABRIC
            const uint32_t device = combine_fabric::device_of_token<tokens_per_device>(token);
            if (device != combine_fabric::linearized_mesh_coord) {
                fabric.send(output, device, payload, page, row_bytes, 0);
                continue;
            }
#endif
            noc.async_write(CoreLocalMem<uint32_t>(payload), output, row_bytes, {}, {.page_id = page});
        }
    }
    noc.async_write_barrier();
#ifdef COMBINE_FABRIC
    fabric.finish(noc, final_semaphore_addr);
#endif
}
