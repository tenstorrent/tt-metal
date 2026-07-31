// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// One data movement the caller is asking for: copy `tokens_per_movement` tokens starting at
// `in_base_token` of `src`'s input region to `tokens_per_movement` tokens starting at `out_base_token` of
// `dst`'s output region, 1:1 and in order.
//
// This is the op's contract, and it is deliberately the WHOLE contract: the caller says what moves
// where, and nothing about how. How many cores run it, which eth core each one owns, which producer
// serves which movement — all of that is the op's business, invisible here. `dst` must be a chip this
// device has a cable to, which the op checks against cable truth rather than mesh arithmetic.
//
// Bases are in TOKENS (= pages), not bytes. A byte offset would be meaningless: these are interleaved
// DRAM buffers, so consecutive pages are round-robined across banks and page k is not at base + k*size.
struct CombineFabric2dMovement {
    std::vector<uint32_t> src;    // mesh coordinate of the source device
    uint32_t in_base_token = 0;   // first token of this movement's region in src's input buffer
    std::vector<uint32_t> dst;    // mesh coordinate of the destination device
    uint32_t out_base_token = 0;  // first token of this movement's region in dst's output buffer

    // Constructors are declared (rather than relying on aggregate init) deliberately: a struct that is
    // BOTH an aggregate and carries attribute_names matches two of ttsl::reflection's partial
    // specializations at once, which is an ambiguity error. Declaring a constructor makes it non-aggregate
    // so only the attribute_names path applies.
    CombineFabric2dMovement() = default;
    CombineFabric2dMovement(
        std::vector<uint32_t> src_, uint32_t in_base_token_, std::vector<uint32_t> dst_, uint32_t out_base_token_) :
        src(std::move(src_)), in_base_token(in_base_token_), dst(std::move(dst_)), out_base_token(out_base_token_) {}

    static constexpr auto attribute_names = std::forward_as_tuple("src", "in_base_token", "dst", "out_base_token");
    auto attribute_values() const { return std::forward_as_tuple(src, in_base_token, dst, out_base_token); }
};

// Mesh coordinate as text, for error messages on both the validation and the program-factory side.
inline std::string movement_coord_str(const std::vector<uint32_t>& c) {
    std::string s = "(";
    for (size_t i = 0; i < c.size(); i++) {
        s += (i ? "," : "") + std::to_string(c[i]);
    }
    return s + ")";
}

// Isolated fabric-transfer experiment op: chip-local DRAM -> eth -> neighbour chip's DRAM.
//
// The caller supplies an input region, an output region, and a list of movements between them. The op
// places one producer kernel per fabric eth core (num_links toward each `axis` neighbour, both
// directions, so 2 * num_links per device) plus a reader kernel beside it on the same core, and assigns
// each of that device's movements to a producer whose cable reaches the movement's `dst`. The reader
// streams the movement's tokens out of DRAM into a `num_l1_slots`-deep L1 ring over NOC_0 while the
// producer drains that ring to the peer chip's DRAM over NOC_1, one fabric packet per token. The token
// count is the caller's and is NOT bounded by L1.
//
// There is no receiver kernel and no application-level credit loop: the eth channel's own sender-slot
// backpressure is the only throttle, which is what lets this op sit at ~100% of the fabric's
// per-direction bandwidth.
//
// `device` is FIRST so the framework's get_first_object_of_type<MeshDevice*>() over
// attribute_values() (tuple element 0) finds the mesh.
struct CombineFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t num_links = 2;
    uint32_t tokens_per_movement = 100;  // tokens one movement copies; sets the traffic
    uint32_t token_size_bytes = 14336;   // 7168 bf16 elements = one token = one fabric packet
    uint32_t axis = 0;                   // mesh axis along which the neighbours are chosen
    // Depth of the L1 ring between the reader and the producer, in tokens. Purely an implementation
    // tuning knob — it bounds nothing the caller can observe, and accuracy holds for any value >= 2.
    // Slots are claimed and released in batches of num_l1_slots / 2, so this also sets how much of the
    // per-batch bookkeeping is amortised away.
    uint32_t num_l1_slots = 8;
    // Forwarded tokens between semaphore bumps to the downstream reader. A bump ALWAYS follows a chunk's
    // sentinel regardless, so this only sets how finely the downstream reader can pipeline WITHIN a chunk:
    // a large value makes it wait for the whole chunk, a small value costs an extra header-only packet per
    // bump. Purely a tuning knob — accuracy holds for any value >= 1. Swept in P9.3.
    uint32_t fwd_bump_every = 32;
    // Order in which a reader works through its own assignments and the forwarding chunks it relays.
    //   0 = nearest destination first, then all forwarding chunks (the straightforward order).
    //   1 = furthest first, with forwarding batches interleaved between own assignments, so downstream
    //       cores are handed work as early as possible and are less likely to sit starved.
    // Purely a scheduling choice: accuracy is identical either way.
    uint32_t assignment_order = 1;
    // Fine-grained stall attribution in the producer (eth-slot / issue / ring-wait buckets). Off by
    // default: it costs a few wall-clock register reads per token, which is a few percent of the very
    // number being measured. Turn it on to explain a result, off to quote one.
    uint32_t stall_telemetry = 0;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;
    // Every movement across the whole mesh, in any order. The op picks out the ones whose `src` is the
    // device it is currently building for.
    std::vector<CombineFabric2dMovement> movements;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "device",
        "num_links",
        "tokens_per_movement",
        "token_size_bytes",
        "axis",
        "num_l1_slots",
        "fwd_bump_every",
        "assignment_order",
        "stall_telemetry",
        "topology",
        "movements");
    auto attribute_values() const {
        return std::forward_as_tuple(
            device,
            num_links,
            tokens_per_movement,
            token_size_bytes,
            axis,
            num_l1_slots,
            fwd_bump_every,
            assignment_order,
            stall_telemetry,
            topology,
            movements);
    }
};

// Both regions are caller-owned, so a test can lay down known content and zero the destination before
// the run and then read the destination back afterwards. Interleaved uint32 ROW_MAJOR DRAM, one row =
// one page = one token, and the same page size on both.
struct CombineFabric2dInputs {
    ttnn::Tensor input;
    ttnn::Tensor output;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
