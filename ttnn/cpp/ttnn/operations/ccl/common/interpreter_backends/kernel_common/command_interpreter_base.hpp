// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"

#include <cstddef>
#include <cstdint>

using arg_idx_t = uint16_t;

template <typename derived_t>
struct command_context_base_t {
protected:
    command_context_base_t(
        FabricConnectionManager& fabric_connection,
        uint16_t num_commands,
        arg_idx_t start_arg_idx,
        uint8_t packet_size_in_pages,
        size_t packet_header_buffer_addr,
        uint8_t stream_id) :
        packet_header_buffer_addr(packet_header_buffer_addr),
        num_commands(num_commands),
        arg_idx(start_arg_idx),
        command_idx(0) {}
    FabricConnectionManager& fabric_connection;
    ttnn::ccl::cmd::CclCommandHeader current_cmd_header;
    uint16_t num_commands = 0;
    arg_idx_t arg_idx = 0;
    uint16_t command_idx = 0;
    uint8_t stream_id;
    uint8_t packet_size_in_pages = 0;
    bool populated = false;

    bool command_requires_fabric() const {
        return current_cmd_header.dest_type != ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY;
    }

    bool is_complete() const { return command_idx >= num_commands; }

    void complete_current_command() {
        command_idx++;
        populated = false;
    }

    bool current_command_active() const { return populated; }

    void fetch_next_command() {
        this->populated = true;

        this->current_cmd_header =
            ttnn::ccl::cmd::CclCommandHeader::from_uint32(get_arg_val<uint32_t>(this->arg_idx++));
#ifdef DEBUG_PRINT_ENABLED
        DPRINT << "CMD (code=" << (uint32_t)current_cmd_header.code
               << ", args=" << (uint32_t)current_cmd_header.arg_count << ", idx=" << (uint32_t)(arg_idx - 1) << "\n";
#endif
        static_cast<derived_t*>(this)->fetch_next_command_impl();
    }
};
