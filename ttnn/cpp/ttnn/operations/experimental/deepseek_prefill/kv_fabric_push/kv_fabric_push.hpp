// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Where the push reads one cache of a compressed layer: rows of one source (a page each), the chips' local rows of
// owner stripes. Entry e is local row (e / period) * stripe_rows + e % stripe_rows of owner (e % period) / stripe_rows,
// period = owners * stripe_rows.
struct KvPushCache {
    uint32_t source = 0;            // index into begin()'s sources
    uint32_t first_page = 0;        // the source page of local row 0
    uint32_t stripe_rows = 0;       // rows an owner stripe
    uint32_t owners = 0;            // owner stripes a period
    bool owners_are_chips = false;  // owner k is chip k (TP-dedup); else mesh row k, any of whose chips holds the rows
};

// A compressed source layer: its KV entries (a page each) and index-key blocks (`block` rows tilized: block / 32 *
// key_tile_cols tiles), from `first_entry` on, into every pool (decode chip) that holds the layer.
struct KvPushLayer {
    uint32_t ratio = 0;
    uint32_t first_entry = 0;
    KvPushCache kv;
    std::optional<KvPushCache> keys;
    uint32_t key_tile_cols = 0;
    std::vector<uint32_t> blocks;                // decode cache block of entry block b
    std::vector<std::array<uint32_t, 3>> pools;  // (destination, KV buffer offset, key buffer offset)
};

// `pages` consecutive pages of the last launch's extra source (the rings) from `first_source_page` to pages
// `first_page` on of buffer `addr` on `destination`.
struct KvPushRing {
    uint32_t destination = 0;
    uint32_t addr = 0;
    uint32_t first_source_page = 0;
    uint32_t pages = 0;
    uint32_t first_page = 0;
};

struct KvPushLaunch {
    uint64_t bytes = 0;
    uint64_t pages = 0;
    uint64_t runs = 0;
    double plan_seconds = 0.0;     // building the runs
    double enqueue_seconds = 0.0;  // the program's dispatch
    double stream_seconds = 0.0;   // writing the runs into the sockets (blocks while a socket's FIFO is full)
};

// Writes a prefill's compressed caches (and, last, the window rings) into another mesh's DRAM over TT-Fabric, a
// launch at a time as the prefill's chunks complete them. Each launch enqueues one program a chip on the mesh's
// command queue (after whatever the caller enqueued before it, e.g. the chunk that completed the entries) and streams
// the chip's runs to it through an H2D socket, so the host plans and streams while the device works.
class KvFabricPush {
public:
    KvFabricPush(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, uint32_t fifo_bytes);
    ~KvFabricPush();
    KvFabricPush(const KvFabricPush&) = delete;
    KvFabricPush& operator=(const KvFabricPush&) = delete;

    // A new request. `dst_banks[j]`: (NoC x, NoC y, offset) of each DRAM bank of destination j, flattened.
    void begin(
        std::vector<Tensor> sources,
        uint32_t dst_mesh_id,
        std::vector<uint32_t> dst_chips,
        std::vector<std::vector<uint32_t>> dst_banks,
        uint32_t block,
        std::vector<KvPushLayer> layers);

    // Pushes every entry complete at position `end` and not pushed yet: entries below end / ratio, key blocks whose
    // rows all are (`last`: every block, the last one partial). `rings` (with `last`) is one more source.
    KvPushLaunch push(
        uint32_t end,
        bool last,
        const std::optional<Tensor>& rings = std::nullopt,
        const std::vector<KvPushRing>& ring_runs = {});

    uint64_t total_bytes() const { return total_bytes_; }

private:
    struct Run {
        uint32_t sender, src, count, dst, base, src_page, page;
    };
    void plan_layer(size_t k, uint32_t hi, bool last, std::vector<Run>& runs);

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    uint32_t rows_ = 0, cols_ = 0;
    std::vector<std::unique_ptr<tt::tt_metal::distributed::H2DSocket>> sockets_;  // a chip each, row-major

    std::vector<Tensor> sources_;
    std::vector<uint32_t> tilize_;
    uint32_t dst_mesh_id_ = 0, banks_ = 0, block_ = 0;
    std::vector<uint32_t> dst_chips_;
    std::vector<uint32_t> dst_table_;  // the destination table, padded to whole socket pages
    std::vector<KvPushLayer> layers_;
    std::vector<uint32_t> kv_done_, keys_done_, spread_;  // per layer: next entry, next key block, runs so far
    uint64_t total_bytes_ = 0;
};

}  // namespace ttnn::experimental
