// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Issue-path primitives of the g6_clone perf experiment "cheaper per-transfer issue".
//
// Raw NoC command-buffer writes, by design: they bypass noc_async_read / noc_async_read_page /
// noc_async_write / noc_async_write_page (api/dataflow/dataflow_api.h; there is no kernel_lib helper for a
// DRAM <-> CB page copy, op_design.md "Helpers considered and rejected"). The mechanism exploited:
//  - a request's static fields (CTRL, coordinate, length) and even its addresses can be written into the
//    command buffer long before the request is sent; the send is then one store (NOC_CMD_CTRL). The writer arms
//    its first write while it waits for the reader, so after the CB handshake only that store is left;
//  - the software issue counters the barriers compare against are bumped once per block, not per request;
//  - a request whose length is the block's constant page size does not rewrite NOC_AT_LEN_BE.
// Dedicated-NoC mode only (the op's default); a kernel built in DM_DYNAMIC_NOC keeps the dataflow API path.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_issue {

constexpr uint32_t WRITE_CTRL_BASE = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_RESP_MARKED;

FORCE_INLINE uint32_t coord_of(uint64_t noc_addr) {
    return (uint32_t)(noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

// ---------------------------------------------------------------------------------------------------------
// Reads (read_cmd_buf; in dedicated mode its CTRL is the firmware's static read config, never rewritten).
// ---------------------------------------------------------------------------------------------------------

// Wait for the read command buffer, then set the length of the reads that follow without a length.
// skip_poll: the buffer is known idle (nothing sent on it since its last ready poll, or since kernel start:
// the previous kernel returned with every read flushed, so every request had left the buffer).
template <bool skip_poll = false>
FORCE_INLINE void read_set_len(uint32_t len) {
    if constexpr (!skip_poll) {
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf));
    }
    NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_AT_LEN_BE, len);
}

// One read packet. with_len = false: the length register already holds this packet's length.
template <bool with_len, bool skip_poll = false>
FORCE_INLINE void read_packet(uint64_t src_noc_addr, uint32_t dst_l1, uint32_t len) {
    if constexpr (!skip_poll) {
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf));
    }
    NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_RET_ADDR_LO, dst_l1);
    NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord_of(src_noc_addr));
    if constexpr (with_len) {
        NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_AT_LEN_BE, len);
    }
    NOC_CMD_BUF_WRITE_REG(noc_index, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

// A span of any length as packets; returns the packet count (the caller bumps the counter once).
FORCE_INLINE uint32_t read_span(uint64_t src_noc_addr, uint32_t dst_l1, uint32_t len) {
    uint32_t packets = 0;
    while (len > NOC_MAX_BURST_SIZE) {
        read_packet<true>(src_noc_addr, dst_l1, NOC_MAX_BURST_SIZE);
        src_noc_addr += NOC_MAX_BURST_SIZE;
        dst_l1 += NOC_MAX_BURST_SIZE;
        len -= NOC_MAX_BURST_SIZE;
        ++packets;
    }
    read_packet<true>(src_noc_addr, dst_l1, len);
    return packets + 1;
}

FORCE_INLINE void count_reads(uint32_t packets) { noc_reads_num_issued[noc_index] += packets; }

template <bool invalidate = true>
FORCE_INLINE void reads_barrier() {
    while (!ncrisc_noc_reads_flushed(noc_index));
    if constexpr (invalidate) {
        invalidate_l1_cache();
    }
}

// ---------------------------------------------------------------------------------------------------------
// Writes (write_cmd_buf).
// ---------------------------------------------------------------------------------------------------------

// Program one write packet, CTRL included, without sending it. On return the command buffer is idle and holds
// the request; nothing may issue on write_cmd_buf before write_fire().
FORCE_INLINE void write_arm(uint32_t vc, uint32_t src_l1, uint64_t dst_noc_addr, uint32_t len) {
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CTRL, WRITE_CTRL_BASE | NOC_CMD_STATIC_VC(vc));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO, src_l1);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord_of(dst_noc_addr));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE, len);
}

FORCE_INLINE void write_fire() { NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ); }

// One write packet after write_arm (CTRL stays). with_len = false: the length register already holds it.
template <bool with_len>
FORCE_INLINE void write_packet(uint32_t src_l1, uint64_t dst_noc_addr, uint32_t len) {
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO, src_l1);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, coord_of(dst_noc_addr));
    if constexpr (with_len) {
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE, len);
    }
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

FORCE_INLINE uint32_t write_span(uint32_t src_l1, uint64_t dst_noc_addr, uint32_t len) {
    uint32_t packets = 0;
    while (len > NOC_MAX_BURST_SIZE) {
        write_packet<true>(src_l1, dst_noc_addr, NOC_MAX_BURST_SIZE);
        src_l1 += NOC_MAX_BURST_SIZE;
        dst_noc_addr += NOC_MAX_BURST_SIZE;
        len -= NOC_MAX_BURST_SIZE;
        ++packets;
    }
    write_packet<true>(src_l1, dst_noc_addr, len);
    return packets + 1;
}

FORCE_INLINE void count_writes(uint32_t packets) {
    noc_nonposted_writes_num_issued[noc_index] += packets;
    noc_nonposted_writes_acked[noc_index] += packets;
}

FORCE_INLINE void writes_flushed() {
    while (!ncrisc_noc_nonposted_writes_sent(noc_index));
    invalidate_l1_cache();
}

FORCE_INLINE void writes_barrier() {
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index));
    invalidate_l1_cache();
}

// ---------------------------------------------------------------------------------------------------------
// CB handshake with the expensive part off the critical path.
// ---------------------------------------------------------------------------------------------------------

// The producer is the only writer of tiles_received: read it before the data lands, store it after.
struct EarlyPush {
    volatile tt_reg_ptr uint32_t* received_ptr;
    uint32_t next;

    FORCE_INLINE EarlyPush(uint32_t cb, uint32_t num_pages) :
        received_ptr(get_cb_tiles_received_ptr(cb)), next(received_ptr[0] + num_pages) {}

    template <uint32_t cb, uint32_t num_pages>
    FORCE_INLINE void push() {
        received_ptr[0] = next;
        LocalCBInterface& cbi = get_local_cb_interface(cb);
        cbi.fifo_wr_ptr += num_pages * cbi.fifo_page_size;
        if (cbi.fifo_wr_ptr == cbi.fifo_limit) {
            cbi.fifo_wr_ptr -= cbi.fifo_size;
        }
    }
};

}  // namespace g6_issue
