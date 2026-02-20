// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/interpreter_backends/kernel_common/algorithms.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"

namespace ttnn::ccl {

// For page-aligned reads - never split pages across packets
template <typename AddressGenerator>
void generate_noc_transfer_burst_for_tensor_slice(
    const ttnn::ccl::v2::TensorSlice& tensor_slice,
    ttnn::ccl::cmd::HostCclCommandNocTransferBurst& noc_transfer_burst_out,
    const AddressGenerator& address_generator,
    size_t page_size,
    size_t packet_size_bytes) {
    TT_FATAL(page_size > 0, "Internal error: page size is 0");

    size_t packet_space_in_bytes_left = packet_size_bytes;
    noc_transfer_burst_out.transfer_burst_groupings.push_back({});
    bool empty_last_group = false;
    for (size_t w = 0; w < tensor_slice.tensor_slice_shape.w; w++) {
        for (size_t z = 0; z < tensor_slice.tensor_slice_shape.z; z++) {
            for (size_t y = 0; y < tensor_slice.tensor_slice_shape.y; y++) {
                size_t pages_read = 0;
                for (size_t x = 0; x < tensor_slice.tensor_slice_shape.x; x += pages_read) {
                    empty_last_group = false;
                    auto offset = ttnn::ccl::Shape4D<uint32_t>{w, z, y, x} + tensor_slice.tensor_slice_offset;
                    auto& transfer_burst_grouping = noc_transfer_burst_out.transfer_burst_groupings.back();
                    const size_t curr_page_idx = get_flat_index_from_shape(tensor_slice.tensor_shape, offset);
                    const auto& [noc_yx, page_index_into_shard, contig_pages_] =
                        address_generator.get_page_location_with_contiguous_pages_in_row_in_bank(curr_page_idx);
                    pages_read = std::min<size_t>(
                        {tensor_slice.tensor_slice_shape.x - x, packet_space_in_bytes_left / page_size, contig_pages_});
                    size_t transfer_size_in_bytes = pages_read * page_size;

                    TT_FATAL(pages_read > 0, "Internal error: hit infinite loop indicating a logical error");
                    noc_transfer_burst_out.num_transfers_total++;
                    transfer_burst_grouping.num_transfers_per_packet++;
                    packet_space_in_bytes_left -= transfer_size_in_bytes;
                    auto byte_offset_in_shard = page_index_into_shard * page_size;
                    uint64_t noc_addr_offset = (static_cast<uint64_t>(noc_yx.noc_y) << 48) |
                                               (static_cast<uint64_t>(noc_yx.noc_x) << 32) |
                                               static_cast<uint64_t>(byte_offset_in_shard);
                    transfer_burst_grouping.transfer_infos.push_back(
                        ttnn::ccl::cmd::noc_transfer_info{noc_addr_offset, transfer_size_in_bytes});

                    if (packet_space_in_bytes_left < page_size) {
                        packet_space_in_bytes_left = packet_size_bytes;
                        bool last_w = w == tensor_slice.tensor_slice_shape.w - 1;
                        bool last_z = z == tensor_slice.tensor_slice_shape.z - 1;
                        bool last_y = y == tensor_slice.tensor_slice_shape.y - 1;
                        bool last_x = x + pages_read == tensor_slice.tensor_slice_shape.x;
                        TT_FATAL(
                            x + pages_read <= tensor_slice.tensor_slice_shape.x,
                            "Internal error: Last x is out of bounds");
                        if (!(last_w && last_z && last_y && last_x)) {
                            empty_last_group = true;
                            noc_transfer_burst_out.transfer_burst_groupings.push_back({});
                        }
                    }
                }
            }
        }
    }
    TT_FATAL(!empty_last_group, "Internal error: Empty last group");
    TT_FATAL(
        noc_transfer_burst_out.transfer_burst_groupings.back().num_transfers_per_packet > 0,
        "Internal error: No transfers per packet");
}

void validate_lowered_noc_commands(const ttnn::ccl::cmd::HostCclCommandNocTransferBurst& noc_transfer_burst) {
    TT_FATAL(!noc_transfer_burst.transfer_burst_groupings.empty(), "Internal error: No transfer burst groupings");
    for (const auto& transfer_burst_grouping : noc_transfer_burst.transfer_burst_groupings) {
        TT_FATAL(transfer_burst_grouping.num_transfers_per_packet > 0, "Internal error: No transfers per packet");
        for (const auto& transfer_info : transfer_burst_grouping.transfer_infos) {
            TT_FATAL(transfer_info.noc_transfer_size_bytes > 0, "Internal error: No transfer size bytes");
        }
    }
}

ttnn::ccl::cmd::CclHostLowLevelWorkerCommand lower_tensor_slice_command_to_noc_commands(
    const ttnn::ccl::cmd::CclHostLowLevelWorkerCommand& command,
    const tt::tt_metal::Tensor& tensor,
    size_t packet_size_bytes) {
    using namespace tt::tt_metal::address_generators;
    using namespace tt::tt_metal;

    TT_FATAL(tensor.is_sharded(), "Only tensor slices for sharded tensors are able to be lowered to noc reads/writes");

    ttnn::ccl::cmd::HostCclCommandNocTransferBurst noc_transfer_burst;
    noc_transfer_burst.bank_base_address = tensor.buffer()->address();

    const auto& tensor_slice = std::get<ttnn::ccl::v2::TensorSlice>(command.command_args);
    auto page_size = tensor.buffer()->page_size();

    auto coord_lookup = tt::tt_metal::address_generators::VirtualCoordWormholeWorkerToNocLookup();

    const auto& [pages_per_shard_y, pages_per_shard_x] = tensor.buffer()->shard_spec().shape_in_pages();
    const auto& [shard_grid_start, shard_grid_end] = ttnn::ccl::shard_grid_from_shard_spec(tensor.shard_spec().value());
    const bool shard_grid_transposed = ttnn::ccl::ShardedAddrGenArgBuilder::shard_grid_is_transposed(tensor);
    // shard_grid_height (cores)
    const size_t shard_grid_height = shard_grid_end.y - shard_grid_start.y + 1;
    TT_FATAL(
        shard_grid_height > 0, "Internal error. Computed shard_grid height == 0 to sharded addrgen, which is invalid");
    // shard_grid_width (cores)
    const size_t shard_grid_width = shard_grid_end.x - shard_grid_start.x + 1;
    TT_FATAL(
        shard_grid_width > 0, "Internal error. Computed shard_grid width == 0 to sharded addrgen, which is invalid");
    // Only page aligned for now since tensor slice is page based at the moment
    // Future work to migrate tensor slice to be element based and then at that
    // point we can
    switch (tensor.buffer()->buffer_layout()) {
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: {
            auto address_generator = build_sharded_addr_gen<TensorMemoryLayout::BLOCK_SHARDED>(
                coord_lookup,
                address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::BLOCK_SHARDED>::type(
                    pages_per_shard_y,
                    pages_per_shard_x,
                    shard_grid_height,
                    shard_grid_width,
                    shard_grid_start.y,
                    shard_grid_start.x,
                    shard_grid_transposed),
                noc_transfer_burst.bank_base_address,
                page_size);
            generate_noc_transfer_burst_for_tensor_slice(
                tensor_slice, noc_transfer_burst, address_generator, page_size, packet_size_bytes);
            break;
        }
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: {
            auto address_generator = build_sharded_addr_gen<TensorMemoryLayout::HEIGHT_SHARDED>(
                coord_lookup,
                address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::HEIGHT_SHARDED>::type(
                    pages_per_shard_y,
                    pages_per_shard_x,
                    shard_grid_height,
                    shard_grid_width,
                    shard_grid_start.y,
                    shard_grid_start.x,
                    shard_grid_transposed),
                noc_transfer_burst.bank_base_address,
                page_size);
            generate_noc_transfer_burst_for_tensor_slice(
                tensor_slice, noc_transfer_burst, address_generator, page_size, packet_size_bytes);
            break;
        }
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: {
            auto address_generator = build_sharded_addr_gen<TensorMemoryLayout::WIDTH_SHARDED>(
                coord_lookup,
                address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::WIDTH_SHARDED>::type(
                    pages_per_shard_y,
                    pages_per_shard_x,
                    shard_grid_height,
                    shard_grid_width,
                    shard_grid_start.y,
                    shard_grid_start.x,
                    shard_grid_transposed),
                noc_transfer_burst.bank_base_address,
                page_size);
            generate_noc_transfer_burst_for_tensor_slice(
                tensor_slice, noc_transfer_burst, address_generator, page_size, packet_size_bytes);
            break;
        }
        default: TT_FATAL(false, "Unsupported buffer layout");
    }

    validate_lowered_noc_commands(noc_transfer_burst);

    std::stringstream ss;
    ss << "Lowering noc commands: \n";
    ss << fmt::format(
              "Base_addr: {}, burst_size: {}",
              noc_transfer_burst.bank_base_address,
              noc_transfer_burst.num_transfers_total)
       << "\n";
    for (auto& transfer : noc_transfer_burst.transfer_burst_groupings) {
        ss << fmt::format("\tGroup_size: {}", transfer.num_transfers_per_packet) << "\n";
        for (auto& transfer_info : transfer.transfer_infos) {
            ss << fmt::format("\t\taddr: {}, size: {}", transfer_info.noc_addr, transfer_info.noc_transfer_size_bytes)
               << "\n";
        }
    }
    log_trace(tt::LogOp, "{}", ss.str());

    // Generate the new (lowered to noc read/write) command
    ttnn::ccl::cmd::CclHostLowLevelWorkerCommand lowered_command = command;
    switch (command.command_code) {
        case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR:
            lowered_command.dest_addr_type = ttnn::ccl::cmd::CclCommandAddrType::NONE;
            lowered_command.dest_addr_args = ttnn::ccl::cmd::CclCommandAddrArgs();
            lowered_command.command_code = ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_BURST;
            lowered_command.command_args = ttnn::ccl::cmd::HostCclCommandNocTransferBurst{noc_transfer_burst};
            break;
        case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB:
            lowered_command.source_addr_type = ttnn::ccl::cmd::CclCommandAddrType::NONE;
            lowered_command.source_addr_args = ttnn::ccl::cmd::CclCommandAddrArgs();
            lowered_command.command_code = ttnn::ccl::cmd::CclCommandCode::NOC_READ_BURST;
            lowered_command.command_args = ttnn::ccl::cmd::HostCclCommandNocTransferBurst{noc_transfer_burst};
            break;
        default: TT_FATAL(false, "Only STREAM_CB_TO_TENSOR and STREAM_TENSOR_TO_CB commands are supported");
    }

    return lowered_command;
}

std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> tensor_slice_commands_to_noc_commands(
    const std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand>& command_stream,
    const tt::tt_metal::Tensor& tensor,
    size_t packet_size_bytes) {
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> lowered_command_stream;
    for (const auto& command : command_stream) {
        switch (command.command_code) {
            case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR: [[fallthrough]];
            case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB:
                lowered_command_stream.push_back(
                    lower_tensor_slice_command_to_noc_commands(command, tensor, packet_size_bytes));
                break;

            default: lowered_command_stream.push_back(command); break;
        }
    }
    return lowered_command_stream;
}

}  // namespace ttnn::ccl
