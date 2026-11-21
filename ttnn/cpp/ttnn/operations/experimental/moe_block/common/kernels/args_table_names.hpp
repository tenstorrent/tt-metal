// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The named compile-time args of the args table (common/args_table.hpp on the host, kernels/moe_block_args.hpp on the
// device): one name for each meaning, shared by both sides. Plain constexpr only (no includes): the kernels include
// it as is.
//
// A kernel that includes moe_block_args.hpp reads every name below, so the host gives every such kernel all of them
// (fill_args_table_named_args: off, value 0, for a kernel whose args stay runtime args).
//
//   name                                | meaning
//   ------------------------------------+------------------------------------------------------------------------
//   moe_block_args_enabled              | 1: the kernel's args are in the table, 0: runtime args
//   moe_block_args_common_index         | the table's first common arg (see args_table.hpp)
//   moe_block_args_fixups_at            | the word of the block where the fixups start
//   moe_block_args_block_in_cb          | 1: the block lands in CB moe_block_args_block_cb, 0: at the common arg's L1
//   moe_block_args_block_cb             | the CB the block lands in (0 when moe_block_args_block_in_cb is 0)
//   moe_block_args_block_cb_offset      | the block's byte offset in that CB (0 when not in a CB)
//   moe_block_args_common_in_block      | 1: the kernel's common args sit in its block from word
//                                       |    moe_block_args_common_at on, 0: they stay common args
//   moe_block_args_common_at            | that word (0 when the common args stay common args)
//   moe_block_args_peer_page_sender     | 1: after its own load() the kernel hands the compute kernel on its core
//                                       |    its args: moe_block_args_peer_words words from word
//                                       |    moe_block_args_peer_at of its block, as one page of CB
//                                       |    moe_block_args_peer_cb
//   moe_block_args_peer_at              | (0 when not a sender)
//   moe_block_args_peer_words           | (0 when not a sender)
//   moe_block_args_peer_cb              | the CB of the compute kernel's page (sender and compute kernel; 0 else)
//   moe_block_args_compute_reads_page   | 1: a compute kernel reads its args from that page (its common args from word
//                                       |    moe_block_args_common_at on), 0: they stay runtime args
#pragma once

namespace moe_block_args_names {

inline constexpr const char* kEnabled = "moe_block_args_enabled";
inline constexpr const char* kCommonIndex = "moe_block_args_common_index";
inline constexpr const char* kFixupsAt = "moe_block_args_fixups_at";
inline constexpr const char* kBlockInCb = "moe_block_args_block_in_cb";
inline constexpr const char* kBlockCb = "moe_block_args_block_cb";
inline constexpr const char* kBlockCbOffset = "moe_block_args_block_cb_offset";
inline constexpr const char* kCommonInBlock = "moe_block_args_common_in_block";
inline constexpr const char* kCommonAt = "moe_block_args_common_at";
inline constexpr const char* kPeerPageSender = "moe_block_args_peer_page_sender";
inline constexpr const char* kPeerAt = "moe_block_args_peer_at";
inline constexpr const char* kPeerWords = "moe_block_args_peer_words";
inline constexpr const char* kPeerCb = "moe_block_args_peer_cb";
inline constexpr const char* kComputeReadsPage = "moe_block_args_compute_reads_page";

// Every name, for the host to give a kernel the ones it has no value for.
inline constexpr const char* kAll[] = {
    kEnabled,
    kCommonIndex,
    kFixupsAt,
    kBlockInCb,
    kBlockCb,
    kBlockCbOffset,
    kCommonInBlock,
    kCommonAt,
    kPeerPageSender,
    kPeerAt,
    kPeerWords,
    kPeerCb,
    kComputeReadsPage};

}  // namespace moe_block_args_names
