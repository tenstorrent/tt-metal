// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <optional>

#include "tt_metal/impl/buffers/buffer.hpp"

namespace ttnn {

namespace reports {

void write_l1_buffers(std::ostream &os) {
    std::map<uint32_t, std::map<uint32_t, Buffer *>> buffer_map_per_device;
    for (const auto &[key, buffer] : tt::tt_metal::detail::BUFFER_MAP) {
        auto [device_id, address] = key;
        buffer_map_per_device[device_id][address] = buffer;
    }

    os << "L1 Buffers:" << std::endl;
    for (const auto &[device_id, buffer_map] : buffer_map_per_device) {
        os << "Device: " << device_id << std::endl;
        auto device = buffer_map.begin()->second->device();

        auto core_to_page_map = std::map<CoreCoord, std::map<tt::tt_metal::detail::PageAddress, std::string>>{};
        int buffer_index = 0;
        for (const auto &[address, buffer] : buffer_map) {
            if (buffer->buffer_type() != BufferType::L1) {
                continue;
            }
            uint32_t page_size = buffer->page_size();
            auto num_pages = buffer->num_pages();
            auto num_banks = device->num_banks(buffer->buffer_type());

            if (buffer->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
                uint32_t bank_index = 0;
                for (int page_index = 0; page_index < num_pages; page_index++) {
                    auto address = buffer->page_address(bank_index, page_index);
                    auto core = buffer->logical_core_from_bank_id(bank_index);
                    bank_index = (bank_index + 1) % num_banks;
                    core_to_page_map[core][address] =
                        fmt::format("\tBuffer {:3}\tPage {:4}\tPage Size {:9}", buffer_index, page_index, page_size);
                }
            } else {
                for (int page_index = 0; page_index < num_pages; page_index++) {
                    auto dev_page_index = buffer->get_host_to_dev_mapped_page_id(page_index);
                    auto core = buffer->get_core_from_dev_page_id(dev_page_index);
                    auto bank_index = device->bank_ids_from_logical_core(core)[0];
                    auto address = buffer->sharded_page_address(bank_index, dev_page_index);
                    core_to_page_map[core][address] =
                        fmt::format("\tBuffer {:3}\tPage {:4}\tPage Size {:9}", buffer_index, page_index, page_size);
                }
            }
            buffer_index++;
        }

        for (const auto &[core, page_map] : core_to_page_map) {
            os << "Core: " << core.str() << std::endl;
            for (const auto &[address, page_name] : page_map) {
                os << "  " << fmt::format("Address {:9}", address) << ":" << page_name << std::endl;
            }
            os << std::endl;
        }
    }
    os << std::endl;
}

void print_l1_buffers(const std::optional<std::string> &file_name = std::nullopt) {
    if (file_name.has_value()) {
        tt::log_info("Writing L1 buffers to file: {}", file_name.value());
        std::ofstream os(file_name.value());
        write_l1_buffers(os);
    } else {
        write_l1_buffers(std::cout);
    }
}
}  // namespace reports

using reports::print_l1_buffers;

}  // namespace ttnn
