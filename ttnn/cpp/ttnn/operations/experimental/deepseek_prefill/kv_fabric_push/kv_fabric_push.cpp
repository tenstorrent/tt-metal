// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_fabric_push.hpp"

#include <algorithm>
#include <chrono>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "device/kv_fabric_push_device_operation.hpp"

namespace ttnn::experimental {

namespace {

using tt::tt_metal::distributed::H2DMode;
using tt::tt_metal::distributed::H2DSocket;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoreCoord;

constexpr uint32_t kPage = 512;  // socket page: 32 runs of 4 words
constexpr uint32_t kRunsPerPage = kPage / 16;
constexpr uint32_t kStageBytes = 32 * 2048;  // the kernel's staging buffer: a tilized block's rows and its tiles
const tt::tt_metal::CoreCoord kCore{0, 0};

double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

KvFabricPush::KvFabricPush(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, uint32_t fifo_bytes) :
    mesh_(mesh) {
    const auto shape = mesh->shape();
    TT_FATAL(shape.dims() == 2, "KvFabricPush: a 2D mesh");
    rows_ = shape[0];
    cols_ = shape[1];
    TT_FATAL(fifo_bytes % kPage == 0, "KvFabricPush: a FIFO of whole {} B pages", kPage);
    for (uint32_t r = 0; r < rows_; ++r) {
        for (uint32_t c = 0; c < cols_; ++c) {
            auto socket = std::make_unique<H2DSocket>(
                mesh,
                MeshCoreCoord(MeshCoordinate(r, c), kCore),
                tt::tt_metal::BufferType::L1,
                fifo_bytes,
                H2DMode::DEVICE_PULL);
            socket->set_page_size(kPage);
            sockets_.push_back(std::move(socket));
        }
    }
}

KvFabricPush::~KvFabricPush() = default;

void KvFabricPush::begin(
    std::vector<Tensor> sources,
    uint32_t dst_mesh_id,
    std::vector<uint32_t> dst_chips,
    std::vector<std::vector<uint32_t>> dst_banks,
    uint32_t block,
    std::vector<KvPushLayer> layers) {
    TT_FATAL(!sources.empty() && sources.size() < 255, "KvFabricPush: 1 to 254 sources (the rings are one more)");
    TT_FATAL(!dst_chips.empty() && dst_chips.size() == dst_banks.size(), "KvFabricPush: banks for every destination");
    TT_FATAL(dst_chips.size() <= (1u << 16), "KvFabricPush: at most 2^16 destinations");
    TT_FATAL(
        block > 0 && block <= 255 && block % 32 == 0, "KvFabricPush: blocks of a multiple of 32 entries, at most 255");
    banks_ = dst_banks.front().size() / 3;
    tilize_.assign(sources.size(), 0);
    for (const auto& layer : layers) {
        TT_FATAL(layer.ratio > 0 && layer.kv.source < sources.size(), "KvFabricPush: a layer's ratio and KV source");
        TT_FATAL(layer.kv.stripe_rows > 0 && layer.kv.owners > 0, "KvFabricPush: a layer's KV owner stripes");
        for (const auto& pool : layer.pools) {
            TT_FATAL(pool[0] < dst_chips.size(), "KvFabricPush: pool of destination {}", pool[0]);
        }
        if (layer.keys.has_value()) {
            const auto& k = *layer.keys;
            TT_FATAL(k.source < sources.size(), "KvFabricPush: a layer's key source");
            TT_FATAL(
                k.stripe_rows > 0 && k.stripe_rows % block == 0,
                "KvFabricPush: {} key rows an owner stripe split {}-entry blocks",
                k.stripe_rows,
                block);
            const uint32_t row_bytes = sources[k.source].buffer()->page_size();
            TT_FATAL(
                row_bytes == layer.key_tile_cols * 64 && 2 * block * row_bytes <= kStageBytes,
                "KvFabricPush: key rows of {} B for {} tile columns, a block of rows and its tiles within {} B",
                row_bytes,
                layer.key_tile_cols,
                kStageBytes);
            tilize_[k.source] = 1;
        }
    }
    for (const auto& layer : layers) {
        TT_FATAL(!tilize_[layer.kv.source], "KvFabricPush: a source both copied and tilized");
    }

    // One connection a chip carries every destination: they must leave the chip by one direction.
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    for (const uint32_t chip : dst_chips) {
        dst_nodes.emplace_back(tt::tt_fabric::MeshId{dst_mesh_id}, chip);
    }
    for (uint32_t r = 0; r < rows_; ++r) {
        for (uint32_t c = 0; c < cols_; ++c) {
            const auto src = mesh_->get_fabric_node_id(MeshCoordinate(r, c));
            const auto first = tt::tt_fabric::get_eth_forwarding_direction(*mesh_, src, dst_nodes.front());
            TT_FATAL(first.has_value(), "KvFabricPush: no route from {} to {}", src, dst_nodes.front());
            for (const auto& dst : dst_nodes) {
                TT_FATAL(
                    tt::tt_fabric::get_eth_forwarding_direction(*mesh_, src, dst) == first,
                    "KvFabricPush: {} reaches {} and {} by different directions",
                    src,
                    dst_nodes.front(),
                    dst);
            }
        }
    }

    dst_table_.clear();
    for (size_t j = 0; j < dst_chips.size(); ++j) {
        TT_FATAL(dst_banks[j].size() == 3 * banks_, "KvFabricPush: destinations with different DRAM banks");
        dst_table_.push_back(dst_chips[j]);
        dst_table_.push_back(dst_mesh_id);
        dst_table_.insert(dst_table_.end(), dst_banks[j].begin(), dst_banks[j].end());
    }
    dst_table_.resize((dst_table_.size() * 4 + kPage - 1) / kPage * kPage / 4, 0);

    sources_ = std::move(sources);
    dst_mesh_id_ = dst_mesh_id;
    dst_chips_ = std::move(dst_chips);
    block_ = block;
    layers_ = std::move(layers);
    kv_done_.resize(layers_.size());
    keys_done_.resize(layers_.size());
    spread_.assign(layers_.size(), 0);
    for (size_t k = 0; k < layers_.size(); ++k) {
        kv_done_[k] = layers_[k].first_entry;
        keys_done_[k] = layers_[k].first_entry / block_;
    }
    total_bytes_ = 0;
}

void KvFabricPush::plan_layer(size_t k, uint32_t hi, bool last, std::vector<Run>& runs) {
    const auto& layer = layers_[k];
    if (hi <= layer.first_entry) {
        return;
    }
    const auto sender = [this](const KvPushCache& c, uint32_t owner, uint32_t spread) {
        return c.owners_are_chips ? owner : owner * cols_ + spread % cols_;
    };

    // A run: the entries of one cache block within one owner stripe, consecutive pages on both sides.
    const auto& kv = layer.kv;
    const uint32_t hs = kv.stripe_rows, period = kv.owners * hs;
    for (uint32_t e = kv_done_[k]; e < hi;) {
        const uint32_t next = std::min({hi, (e / block_ + 1) * block_, (e / hs + 1) * hs});
        const uint32_t owner = (e % period) / hs, local = (e / period) * hs + e % hs;
        const uint32_t page = layer.blocks.at(e / block_) * block_ + e % block_;
        for (const auto& [dst, kv_base, _] : layer.pools) {
            runs.push_back(
                {sender(kv, owner, spread_[k] + dst), kv.source, next - e, dst, kv_base, kv.first_page + local, page});
        }
        ++spread_[k];
        e = next;
    }
    kv_done_[k] = std::max(kv_done_[k], hi);

    if (!layer.keys.has_value()) {
        return;
    }
    // A key block is its block / 32 tile rows; pushed once all its rows are written, the last one partial.
    const auto& keys = *layer.keys;
    const uint32_t khs = keys.stripe_rows, kperiod = keys.owners * khs;
    const uint32_t per_block = block_ / 32 * layer.key_tile_cols;
    const uint32_t end = last ? (hi + block_ - 1) / block_ : hi / block_;
    for (uint32_t b = keys_done_[k]; b < end; ++b) {
        const uint32_t eb = b * block_;
        const uint32_t owner = (eb % kperiod) / khs, local = (eb / kperiod) * khs + eb % khs;
        for (const auto& [dst, _, key_base] : layer.pools) {
            runs.push_back(
                {sender(keys, owner, b + dst),
                 keys.source,
                 block_,
                 dst,
                 key_base,
                 keys.first_page + local,
                 layer.blocks.at(b) * per_block});
        }
    }
    keys_done_[k] = std::max(keys_done_[k], end);
}

KvPushLaunch KvFabricPush::push(
    uint32_t end, bool last, const std::optional<Tensor>& rings, const std::vector<KvPushRing>& ring_runs) {
    TT_FATAL(!sources_.empty(), "KvFabricPush: push before begin");
    TT_FATAL(last || !rings.has_value(), "KvFabricPush: the rings go in the last launch");
    KvPushLaunch out;
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t chips = rows_ * cols_;
    std::vector<Run> runs;
    for (size_t k = 0; k < layers_.size(); ++k) {
        plan_layer(k, end / layers_[k].ratio, last, runs);
    }
    std::vector<Tensor> sources = sources_;
    std::vector<uint32_t> tilize = tilize_;
    if (rings.has_value()) {
        const uint32_t src = sources.size();
        sources.push_back(*rings);
        tilize.push_back(0);
        for (const auto& ring : ring_runs) {
            TT_FATAL(
                ring.destination < dst_chips_.size() && ring.pages > 0 && ring.pages <= 255,
                "KvFabricPush: a ring run");
            const uint32_t layer = ring.first_source_page / ring.pages;
            runs.push_back(
                {(layer + ring.destination) % chips,
                 src,
                 ring.pages,
                 ring.destination,
                 ring.addr,
                 ring.first_source_page,
                 ring.first_page});
        }
    }

    std::vector<std::vector<uint32_t>> words(chips);
    for (const auto& r : runs) {
        words.at(r.sender).insert(
            words[r.sender].end(), {r.src | (r.count << 8) | (r.dst << 16), r.src_page, r.base, r.page});
        const uint32_t page = sources[r.src].buffer()->page_size();
        out.bytes += static_cast<uint64_t>(r.count) * page;
        out.pages += tilize[r.src] ? r.count / 32 * (page / 64) : r.count;
    }
    out.runs = runs.size();
    out.plan_seconds = since(t0);

    const auto t1 = std::chrono::steady_clock::now();
    ttnn::experimental::prim::KvFabricPushParams params;
    params.dst_mesh_id = dst_mesh_id_;
    params.dst_chips = dst_chips_;
    params.dst_banks = banks_;
    params.max_dst_bytes = dst_table_.size() * 4;
    params.socket_page_size = kPage;
    params.mesh_num_cols = cols_;
    for (const auto& socket : sockets_) {
        params.socket_config_addrs.push_back(socket->get_config_buffer_address());
    }
    params.tilize = std::move(tilize);
    ttnn::prim::kv_fabric_push(params, sources);
    out.enqueue_seconds = since(t1);

    // After the enqueue: a launch's stream can outgrow the FIFO, and only its program drains it.
    const auto t2 = std::chrono::steady_clock::now();
    constexpr uint32_t page_words = kPage / 4;
    const uint32_t dst_pages = dst_table_.size() / page_words;
    std::vector<uint32_t> stream;
    for (uint32_t i = 0; i < chips; ++i) {
        const uint32_t n = words[i].size() / 4;
        const uint32_t pages = 1 + dst_pages + (n + kRunsPerPage - 1) / kRunsPerPage;
        stream.assign(pages * page_words, 0);
        stream[0] = n;
        stream[1] = dst_table_.size() * 4;
        std::copy(dst_table_.begin(), dst_table_.end(), stream.begin() + page_words);
        std::copy(words[i].begin(), words[i].end(), stream.begin() + (1 + dst_pages) * page_words);
        sockets_[i]->write(stream.data(), pages);
    }
    out.stream_seconds = since(t2);
    total_bytes_ += out.bytes;
    return out;
}

}  // namespace ttnn::experimental
