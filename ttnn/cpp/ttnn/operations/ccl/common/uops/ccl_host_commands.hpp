// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::ccl::cmd {

// This file defines commands that are resolved on a per worker level. This is the lowest level of
// command description (Intermediate Representation if you will) before being lowered directly to
// Ccl Command Process KernelCommands

struct CclHostLowLevelWorkerCommand {
    ttnn::ccl::cmd::CclCommandCode command_code;
    ttnn::ccl::cmd::CclCommandArgs command_args;

    // semaphore ID derived address, absolute address, relative address
    ttnn::ccl::cmd::CclCommandAddrType source_addr_type;
    ttnn::ccl::cmd::CclCommandAddrArgs source_addr_args;

    ttnn::ccl::cmd::CclCommandAddrType dest_addr_type;
    ttnn::ccl::cmd::CclCommandAddrArgs dest_addr_args;

    // resolved core-xy, rectangle (for mcast)
    ttnn::ccl::cmd::CclCommandCoreDescriptorType core_desc_type;
    ttnn::ccl::cmd::CclCommandCoreDescriptorArgs core_desc_args;

    // unicast, mcast, local_only
    ttnn::ccl::cmd::CclCommandDestType fabric_transfer_type;
    ttnn::ccl::cmd::CclCommandDestArgs fabric_transfer_args;
};

using CclHostLowLevelCommandSequence = std::vector<CclHostLowLevelWorkerCommand>;

namespace uops {

using semaphore_id_t = std::variant<uint32_t, tt::tt_metal::GlobalSemaphore const*>;

[[nodiscard]] CclHostLowLevelWorkerCommand read_tensor_slice_to_cb_for_eventual_fabric_write(
    ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id);
[[nodiscard]] CclHostLowLevelWorkerCommand read_tensor_slice_to_cb(
    ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id);
[[nodiscard]] CclHostLowLevelWorkerCommand local_write_cb_to_tensor_slice(
    ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id);
[[nodiscard]] CclHostLowLevelWorkerCommand fabric_write_cb_to_tensor_slice(
    ttnn::ccl::v2::TensorSlice const& slice,
    size_t cb_id,
    std::variant<UnicastCommandDestArgs, MulticastCommandDestArgs> const& dest_args_variant);
[[nodiscard]] CclHostLowLevelWorkerCommand local_semaphore_wait(semaphore_id_t const& semaphore_id, size_t value);
[[nodiscard]] CclHostLowLevelWorkerCommand local_chip_noc_semaphore_inc(
    size_t dest_noc0_x, size_t dest_noc0_y, semaphore_id_t const& semaphore_id, size_t value);
[[nodiscard]] CclHostLowLevelWorkerCommand local_core_semaphore_inc(semaphore_id_t const& semaphore_id, size_t value);
[[nodiscard]] CclHostLowLevelWorkerCommand local_core_semaphore_set(semaphore_id_t const& semaphore_id, size_t value);
[[nodiscard]] [[deprecated]] CclHostLowLevelWorkerCommand local_chip_noc_absolute_address_semaphore_inc(
    size_t dest_noc0_x, size_t dest_noc0_y, size_t bank_address, size_t value);
[[nodiscard]] CclHostLowLevelWorkerCommand fabric_multicast_semaphore_inc(
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    MulticastCommandDestArgs const& multicast_args);
[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_semaphore_inc(
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    UnicastCommandDestArgs const& unicast_args);
[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_semaphore_inc_mcast(
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    CclCommandCoreDescriptorTypeMcast const& dest_mcast_spec,
    UnicastCommandDestArgs const& unicast_args);
[[nodiscard]] CclHostLowLevelWorkerCommand local_chip_semaphore_inc_mcast(
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    CclCommandCoreDescriptorTypeMcast const& dest_mcast_spec);
[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_absolute_address_semaphore_inc(
    CclCommandAddrAbsoluteAddress const& address_dest_args,
    CclCommandAtomicInc const& increment_args,
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    UnicastCommandDestArgs const& unicast_args);

// Noc Read/Write commands
// Densely packs as many transfers as possible into a single packet
[[nodiscard]] CclHostLowLevelWorkerCommand local_noc_read_burst_to_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id
);

[[nodiscard]] CclHostLowLevelWorkerCommand local_noc_write_burst_from_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id
);

[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_noc_write_burst_from_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id,
    UnicastCommandDestArgs const& unicast_args
);

[[nodiscard]] CclHostLowLevelWorkerCommand fabric_multicast_noc_write_burst_from_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id,
    MulticastCommandDestArgs const& multicast_args
);


};  // namespace uops
};  // namespace ttnn::ccl::cmd
