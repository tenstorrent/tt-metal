// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include <array>
#include <cstdint>
#include <utility>

// Store-and-forward AllGather is Fabric_1D only: every fabric send is a single unicast, one hop, to the
// immediate neighbor.
namespace fabric_api = tt::tt_fabric::linear::experimental;

// Walks the output-tensor chunks of one stripe. Templated on the geometry so the per-stripe starts are
// computed here and the matched/split fast path (output_chunks_per_page == 1) folds away. Re-pointed to an
// arbitrary stripe each relay iteration via init(); the src stripe address on this device equals the dst
// stripe address on the neighbor, so reader and writer share this iterator unchanged.
template <
    uint32_t output_chunks_per_stripe,
    uint32_t output_chunks_per_page,
    uint32_t output_chunk_size,
    uint32_t num_devices>
class OutputStripeIterator {
    static constexpr uint32_t output_page_size = output_chunks_per_page * output_chunk_size;
    static constexpr uint32_t stripe_distance_chunks = num_devices * output_chunks_per_stripe;
    static constexpr uint32_t output_pages_per_row = stripe_distance_chunks / output_chunks_per_page;

public:
    // Point at `stripe` for the chunk range [start, start + count).
    FORCE_INLINE void init(uint32_t stripe, uint32_t start, uint32_t count) {
        const uint32_t s_start = (start / output_chunks_per_stripe) * stripe_distance_chunks +
                                 (start % output_chunks_per_stripe) + stripe * output_chunks_per_stripe;
        page_id_ = s_start / output_chunks_per_page;
        byte_off_ = (s_start % output_chunks_per_page) * output_chunk_size;
        if constexpr (output_chunks_per_page == 1) {
            phase_ = 0;
            stripe_jump_ = output_pages_per_row - (output_chunks_per_stripe - 1);
        } else {
            // In concat mode the page phase (and hence the stripe jump) depends on the stripe.
            const uint32_t off = (stripe * output_chunks_per_stripe) % output_chunks_per_page;
            phase_ = off * output_chunk_size;
            stripe_jump_ = output_pages_per_row - (off + output_chunks_per_stripe - 1) / output_chunks_per_page;
        }
        chunk_in_stripe_ = start % output_chunks_per_stripe;
        sent_ = 0;
        count_ = count;
    }

    FORCE_INLINE bool valid() const { return sent_ < count_; }

    // {output_page_id, byte_offset} of the current chunk, then advance.
    FORCE_INLINE std::pair<uint32_t, uint32_t> next() {
        std::pair<uint32_t, uint32_t> loc{page_id_, byte_off_};
        sent_++;
        if (++chunk_in_stripe_ == output_chunks_per_stripe) {
            chunk_in_stripe_ = 0;
            page_id_ += stripe_jump_;
            byte_off_ = phase_;
        } else {
            byte_off_ += output_chunk_size;
            if (byte_off_ == output_page_size) {
                byte_off_ = 0;
                page_id_++;
            }
        }
        return loc;
    }

private:
    uint32_t page_id_, byte_off_, chunk_in_stripe_, sent_, count_, phase_, stripe_jump_;
};

// Unicasts pages one hop to the single neighbor device. Handles packetization (packing several pages into a
// scatter-write packet when they fit, else splitting a big page across packets).
//
// Templated on the fabric sender type (SenderT*), so the same writer drives either a direct
// WorkerToFabricEdmSender (one worker per direction) or a WorkerToFabricMuxSender (multiple workers per
// direction sharing a fabric mux). The linear-fabric send calls are all base (sender-pointer) overloads that
// accept either sender type; for 1D-linear all routing reduces to to_chip_unicast(1), which set_state does,
// so no route-manager is needed. FabricWriter owns its two packet headers directly.
template <uint32_t page_size, uint32_t packet_size, typename SenderT>
class FabricWriter {
public:
    FabricWriter(const Noc& noc, SenderT* sender) :
        noc{noc},
        sender{sender},
        scatter_hdr{PacketHeaderPool::allocate_header(1)},
        unicast_hdr{PacketHeaderPool::allocate_header(1)},
        scatter_header({}, {}),
        chunk_count{0} {
        std::array<uint64_t, max_pages_per_packet> dummy_addrs{};
        std::array<uint16_t, max_pages_per_packet - 1> chunk_sizes{};
        chunk_sizes.fill(page_size);
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::ChunkSizes>(
            scatter_hdr,
            num_hops,
            NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet));

        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(unicast_hdr, num_hops);
    }

    ~FabricWriter() { ASSERT(chunk_count == 0); }  // outstanding chunks! flush() not called correctly

    void async_write(uint32_t l1_addr, uint64_t remote_noc_addr) {
        if constexpr (use_scatter_write) {
            // Accumulate contiguous (in L1) pages into a single scatter-write packet.
            if (chunk_count == 0) {
                start_l1_addr = l1_addr;
            }
            scatter_header.noc_address[chunk_count++] = remote_noc_addr;
            if (chunk_count == pages_per_packet) {
                noc.async_writes_flushed();
                scatter_header.chunk_count = chunk_count;
                fabric_api::fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                    sender, scatter_hdr, start_l1_addr, scatter_header, payload_size);
                chunk_count = 0;
            }
        } else {
            // Page larger than a packet: split across packets.
            for (uint32_t packet = 0; packet < packets_per_page; ++packet) {
                noc.async_writes_flushed();
                fabric_api::fabric_unicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    sender,
                    unicast_hdr,
                    l1_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{remote_noc_addr},
                    (packet < packets_per_page - 1) ? payload_size : last_payload_size);
                l1_addr += payload_size;
                remote_noc_addr += payload_size;
            }
        }
    }

    // Call before popping the CB entry the outstanding writes read from.
    void async_writes_flushed() {
        if constexpr (use_scatter_write) {
            static_assert(min_pages_per_packet == 2, "hardcoded to assume scatter_write min_pages_per_packet == 2");
            if (chunk_count > 0) {
                noc.async_writes_flushed();
                if (chunk_count == 1) {
                    // scatter_write needs chunk_count >= 2, so a lone leftover page goes as a unicast_write.
                    fabric_api::fabric_unicast_noc_unicast_write_with_state<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        sender,
                        unicast_hdr,
                        start_l1_addr,
                        tt::tt_fabric::NocUnicastCommandHeader{scatter_header.noc_address[0]},
                        page_size);
                } else {
                    scatter_header.chunk_count = chunk_count;
                    fabric_api::fabric_unicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                        sender, scatter_hdr, start_l1_addr, scatter_header, chunk_count * page_size);
                }
                chunk_count = 0;
            }
        }
        noc.async_writes_flushed();
    }

private:
    static constexpr uint32_t max_pages_per_packet = NOC_SCATTER_WRITE_MAX_CHUNKS;
    static constexpr uint32_t min_pages_per_packet = NOC_SCATTER_WRITE_MIN_CHUNKS;
    static constexpr uint32_t pages_per_packet = std::min(packet_size / page_size, max_pages_per_packet);  // div_down
    static constexpr uint32_t packets_per_page = (page_size + packet_size - 1) / packet_size;              // div_up
    static constexpr bool use_scatter_write = pages_per_packet >= min_pages_per_packet;
    static constexpr uint32_t payload_size = use_scatter_write ? (pages_per_packet * page_size) : packet_size;
    static constexpr uint32_t last_payload_size = page_size - ((packets_per_page - 1) * packet_size);

    const Noc& noc;
    SenderT* sender;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_hdr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* unicast_hdr;
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;
    uint32_t start_l1_addr;
};
