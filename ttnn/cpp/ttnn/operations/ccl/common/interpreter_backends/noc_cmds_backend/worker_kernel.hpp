// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Kernel is designed for more "bursty" noc read/write sequences
// e.g. those types of operations where we are reading many chunks
//      of data from a single source (input tensor) to a common destination
//      (circular buffer) before advancing a different source or destination
//      NOTE: different destination != different address - it would imply
//            different base address or CB ID
//
// Example of a good fit command sequence:
//
// set source base address 0x1000
// set dest CB ID 0
// noc read offset=0x0 size=0x100 noc_xy=...
// noc read offset=0x0 size=0x100 noc_xy=...
// noc read offset=0x0 size=0x100 noc_xy=...
// noc read offset=0x0 size=0x100 noc_xy=...
// ... some noc reads later
// update to new source base address 0x2000
// ... some more noc reads
// and so on
//
// Example of a bad fit command sequence (either additional commands or a more efficient kernel for this pattern should
// be writtern): noc read 0x1000 -> CB 0 noc read 0x2000 -> CB 1 noc read 0x3000 -> CB 2 noc read 0x1000 -> CB 0 noc
// read 0x2000 -> CB 1 noc read 0x3000 -> CB 2

// Noc read/write kernel includes
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/noc_cmds_backend/commands.hpp"

// CCL Kernel common includes
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/io_descriptors.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/command_interpreter_base.hpp"

// Metal includes
#include "dataflow_api.h"

// System includes
#include <cstdint>

using arg_idx_t = uint16_t;

constexpr uint16_t my_chip_id = get_compile_time_arg_val(0);
constexpr size_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr size_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr size_t command_stream_count = get_compile_time_arg_val(3);

constexpr size_t MAX_IMPLEMENTABLE_COMMAND_STREAMS = 8;
constexpr size_t total_num_packet_headers_storable = num_packet_headers_storable * command_stream_count;
static_assert(
    command_stream_count < MAX_IMPLEMENTABLE_COMMAND_STREAMS,
    "Kernel implementation only supports a max of 8 command streams");

struct command_context_t final : public command_context_base_t<command_context_t> {
    FORCE_INLINE command_context_t(
        FabricConnectionManager& fabric_connection,
        uint16_t num_commands,
        arg_idx_t start_arg_idx,
        uint8_t packet_size_in_pages,
        size_t packet_header_buffer_addr,
        uint8_t stream_id) :
        command_context_base_t<command_context_t>(
            fabric_connection, num_commands, start_arg_idx, packet_size_in_pages, packet_header_buffer_addr, stream_id),
        cmd_specific_ctx(),
        base_bank_address(0) {
        ASSERT(num_commands == 0 || arg_idx > 4);
    }
    FabricConnectionManager& fabric_connection;
    uint32_t base_bank_address = 0;

    address_info_t src_addr_info;
    address_info_t dest_addr_info;
    core_descriptor_info_t core_desc_info;
    size_t packet_header_buffer_addr = 0;

    bool populated = false;

    static void populate_command(command_context_t& cmd_ctx) { cmd_ctx.arg_idx, cmd_ctx.current_cmd_header; }

    void fetch_next_command_impl() { populate_command(*this); }
};

void try_advance(command_context_t& cmd_ctx) { switch () }

// returns num runtime args consumed
template <size_t num_command_streams>
constexpr int initialize_command_stream_contexts(
    std::array<command_context_t, command_stream_count>& command_stream_contexts,
    FabricConnectionManager& fabric_connection,
    const arg_idx_t arg_idx) {
    size_t arg_idx_offset = 0;
    auto packet_header_cb_id = get_arg_val<uint32_t>(arg_idx + arg_idx_offset++);
    cb_reserve_back(packet_header_cb_id, num_packet_headers_storable);
    auto packet_header_buffer_addr = get_write_ptr(packet_header_cb_id);
    for (size_t i = 0; i < num_command_streams; i++) {
        size_t stream_command_counts = get_arg_val<uint32_t>(arg_idx + arg_idx_offset++);
        arg_idx_t command_stream_start_offsets = get_arg_val<uint32_t>(arg_idx + arg_idx_offset++);
        auto command_stream_packet_header_buffer_addr =
            packet_header_buffer_addr + i * num_packet_headers_storable * sizeof(tt::fabric::PacketHeader);
        command_stream_contexts[i] = command_context_t(
            fabric_connection,
            stream_command_counts,
            command_stream_start_offsets,
            command_stream_packet_header_buffer_addr,
            i);
    }

    return arg_idx_offset;
}

void kernel_main() {
    std::array<command_context_t, command_stream_count> command_stream_contexts;

    size_t arg_idx = 0;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);
    initialize_command_stream_contexts(fabric_connection, arg_idx);

    uint8_t streams_completed_bitfield = 0;
    constexpr uint8_t all_streams_done_bitfield = (1 << command_stream_count) - 1;

    while (streams_completed_bitfield != all_streams_done_bitfield) {
        uint8_t streams_completed_bitfield_copy = streams_completed_bitfield;
        for (uint8_t i = 0; i < command_stream_count; i++) {
            bool stream_done = (streams_completed_bitfield_copy & 0x1);
            if (!stream_done) {
                auto& cmd_ctx = command_stream_contexts[i];
                if (!cmd_ctx.current_command_active()) {
                    // Read the command header
                    cmd_ctx.fetch_next_command();
                }
                try_advance(cmd_ctx);

                streams_completed_bitfield |= (cmd_ctx.is_complete() << i);
            }

            streams_completed_bitfield_copy = streams_completed_bitfield_copy >> 1;
        }
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
}
