# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_gather_pair — a two-chip all-gather from DRAM to DRAM, as one ttnn.generic_op.

Every chip in mesh row 0 is paired with the chip below it (row 1). Each chip holds one shard in DRAM
(TILE, interleaved); afterwards both chips hold [shard of row 0 ; shard of row 1] (concatenated on dim -2).

Per link, one core does all of its link's work:
  reader (RISCV_1): reads its share of the local shard from DRAM into a CB, one chunk at a time.
  sender (RISCV_0): for every chunk, writes it into the local output (the local copy) and sends it over the
                    fabric into the same pages of the peer's output. The last packet also increments a
                    semaphore on the peer's link core; after its own sends, each sender waits for the peer's
                    increment, so the program ends only after every page has landed on both chips.

The shard is split between links by DRAM bank: link l owns the pages of banks l, l+L, l+2L, ...

VARIANTS (how many tiles move per DRAM read and per fabric packet):
  page_per_packet  one tile per read and per packet (the obvious page loop).
  bank_run         up to payload/page tiles per read and per packet. Interleaved tiles p, p+B, p+2B, ... (B = number
                   of DRAM banks) sit at consecutive addresses in one bank, and so do their places in the output,
                   so a run of them is one contiguous DRAM read and one contiguous packet.
"""

import math

import ttnn

VARIANTS = ("page_per_packet", "bank_run")
CB_CHUNKS = 0
NOC0 = ttnn.NOC.RISCV_0_default  # routes +X, then +Y
NOC1 = ttnn.NOC.RISCV_1_default  # routes -Y, then -X

_READER_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);  // tiles per chunk
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);      // chunks per barrier (half the CB)
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pages_per_bank = get_arg_val<uint32_t>(a++);  // shard pages / num_banks
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);     // number of links
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);

    // `group` chunk reads in flight per barrier; batches line up with the sender's flush groups.
    uint32_t batch = 0, wptr = 0;
    uint32_t chunks_left = 0;
    for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
        chunks_left += (pages_per_bank + run_pages - 1) / run_pages;
    }
    for (uint32_t m = 0; m < pages_per_bank; m += run_pages) {  // round-robin over my banks: spread the load
        for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
            if (batch == 0) {
                cb_reserve_back(cb, run_pages * group);
                wptr = get_write_ptr(cb);
            }
            const uint32_t n = (pages_per_bank - m) < run_pages ? (pages_per_bank - m) : run_pages;  // last run may be short
            // pages b + m*B, b + (m+1)*B, ... are consecutive in bank b: one read
#ifndef ABLATE_DRAM_READ
            noc_async_read(in.get_noc_addr(b + m * num_banks), wptr + batch * chunk_bytes, n * page_bytes);
#endif
            if (++batch == group || --chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(cb, run_pages * batch);
                batch = 0;
            }
        }
    }
}
"""

_SENDER_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count (ROUTING_MODE from the build).
inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t run_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);  // chunks per flush; the CB holds 2 x group chunks
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;
#ifdef LOCAL_COPY_NOC
    constexpr uint8_t local_noc = LOCAL_COPY_NOC;  // the local copy on its own NoC (its own outbound port)
#else
    constexpr uint8_t local_noc = noc_index;       // same NoC as the fabric sends
#endif

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pages_per_bank = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t out_page_base = get_arg_val<uint32_t>(a++);  // my shard's first page in the output
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);          // peer's link core = my core (same logical core)
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
    for (uint32_t h = 0; h < num_headers; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
    }
    const uint64_t done_noc = get_noc_addr(peer_x, peer_y, sem_addr);
    uint32_t chunks_left = 0;
    for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
        chunks_left += (pages_per_bank + run_pages - 1) / run_pages;
    }
    conn.open();

    // Up to `group` chunks in flight: a group's CB slots (and headers) are released only after one flush. The read
    // pointer always sits on a group boundary of a 2-group CB, so read_ptr + unflushed * chunk never wraps.
    uint32_t h = 0, unflushed = 0;
    for (uint32_t m = 0; m < pages_per_bank; m += run_pages) {  // same order as the reader
        for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
            cb_wait_front(cb, run_pages * (unflushed + 1));
            const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
            const uint32_t n = (pages_per_bank - m) < run_pages ? (pages_per_bank - m) : run_pages;
            const uint32_t bytes = n * page_bytes;
            // output pages out_page_base + b + m*B, ... are consecutive in the same bank on both chips
            const uint32_t page = out_page_base + b + m * num_banks;
            const uint64_t dst = out.get_noc_addr(page, 0, 0);  // packet destinations are NoC0 coordinates
#ifndef ABLATE_LOCAL_COPY
            noc_async_write(src, out.get_noc_addr(page, 0, local_noc), bytes, local_noc);  // local copy
#endif
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
            h = (h + 1 == num_headers) ? 0 : h + 1;
            if (--chunks_left > 0) {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
            } else {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, done_noc, 1, true}, bytes);
            }
#ifndef ABLATE_FABRIC
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
#endif
            if (++unflushed == group || chunks_left == 0) {
                noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
                if constexpr (local_noc != noc_index) {
                    noc_async_writes_flushed(local_noc);
                }
                cb_pop_front(cb, run_pages * unflushed);
                unflushed = 0;
            }
        }
    }
    noc_async_write_barrier(local_noc);  // local copies landed
    conn.close();

    // Both chips send: the peer's last packet increments my semaphore. Wait for it, then re-arm.
#ifndef ABLATE_FABRIC
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait_min(done, 1);
    noc_semaphore_inc(get_noc_addr(sem_addr), static_cast<uint32_t>(-1));
    noc_async_atomic_barrier();
#endif
}
"""


def default_cores(num_links):
    return [ttnn.CoreCoord(l, 0) for l in range(num_links)]


def create_mesh_program_descriptor(
    mesh_device,
    input_tensor,
    output_tensor,
    sem_addr,
    *,
    variant,
    cores,
    cb_bytes=112 * 1024,
    sender_noc=NOC1,
    local_copy_noc=None,
    ablate=(),
):
    """`ablate` (diagnostic only; output is then incomplete): any of "dram_read", "local_copy", "fabric"."""
    defines = [(f"ABLATE_{x.upper()}", "1") for x in ablate]
    if local_copy_noc is not None:
        defines.append(("LOCAL_COPY_NOC", "0" if local_copy_noc == NOC0 else "1"))
    assert variant in VARIANTS
    rows, cols = tuple(mesh_device.shape)
    assert rows == 2, "pairs chips along mesh axis 0: needs exactly 2 rows"
    num_links = len(cores)
    page_bytes = int(input_tensor.buffer_aligned_page_size())
    num_banks = mesh_device.dram_grid_size().x * mesh_device.dram_grid_size().y
    shape = list(input_tensor.padded_shape)
    shard_pages = (shape[-1] // 32) * (shape[-2] // 32) * math.prod(shape[:-2])
    assert input_tensor.layout == ttnn.TILE_LAYOUT
    assert (
        shard_pages % num_banks == 0
    ), f"shard pages ({shard_pages}) must be a multiple of the DRAM banks ({num_banks})"
    assert num_banks % num_links == 0, "links must split the DRAM banks evenly"
    pages_per_bank = shard_pages // num_banks
    run_pages = 1 if variant == "page_per_packet" else max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    chunk_bytes = run_pages * page_bytes
    group = max(1, min(8, cb_bytes // (2 * chunk_bytes)))  # chunks per flush (= headers in flight)

    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    virt = [mesh_device.worker_core_from_logical_core(c) for c in cores]
    in_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    in_addr, out_addr = int(input_tensor.buffer_address()), int(output_tensor.buffer_address())

    mesh_desc = ttnn.MeshProgramDescriptor()
    for r in range(rows):
        for c in range(cols):
            me = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            peer = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(1 - r, c))
            links = ttnn.get_forwarding_link_indices(me, peer)
            assert len(links) >= num_links, f"only {len(links)} links between {me} and {peer}"
            program = ttnn.ProgramDescriptor()
            program.cbs = [
                ttnn.CBDescriptor(
                    total_size=2 * group * chunk_bytes,
                    core_ranges=core_set,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_CHUNKS, input_tensor.dtype, page_bytes)],
                )
            ]
            reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for l, core in enumerate(cores):
                reader_rt[core.x][core.y] = [in_addr, pages_per_bank, l, num_links]
                args = [
                    out_addr,
                    pages_per_bank,
                    l,
                    num_links,
                    r * shard_pages,
                    virt[l].x,
                    virt[l].y,
                    sem_addr,
                    int(peer.mesh_id),
                    int(peer.chip_id),
                ]
                args += list(ttnn.setup_fabric_connection(me, peer, links[l], program, core))
                sender_rt[core.x][core.y] = args
            ct = [CB_CHUNKS, page_bytes, run_pages, num_banks]
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_READER_SOURCE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=core_set,
                    compile_time_args=ct + [group] + in_ct,
                    defines=defines,
                    runtime_args=reader_rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=NOC0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_SENDER_SOURCE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=core_set,
                    compile_time_args=ct + [group] + out_ct,
                    defines=defines,
                    runtime_args=sender_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=sender_noc
                    ),
                ),
            ]
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


def fabric_gather_pair(mesh_device, input_tensor, output_tensor, sem_addr, **kw):
    """One dispatch: every chip ends with [row-0 shard ; row-1 shard] of its column pair in `output_tensor`."""
    return ttnn.generic_op(
        [input_tensor, output_tensor],
        create_mesh_program_descriptor(mesh_device, input_tensor, output_tensor, sem_addr, **kw),
    )
