// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The x-ready word's byte offset in each relay's words, past the gate/up cores' freed words (4 B each), which the
// relays zero: a caller's persistent relay words need only X_READY_OFF + 4 bytes a relay.
constexpr uint32_t X_READY_OFF = 496;

// Which of a bank's two Metal DRISC tiles send its gate/up weight slots (slot r in bank r % banks; two slots a bank
// at 16 slots). The NOC1-endpoint tile's NOC1 NIU forwards the bank's NOC1 DRAM accesses (NOC2AXI), so a kernel there
// sends on NOC0 only; the free tile (no endpoint) may send on either NoC.
enum class FlatDriscTiles : uint32_t {
    // both tiles every call: slot r < banks from the free tile (NOC1, DMA stream 0), slot r >= banks from the NOC1
    // endpoint (NOC0, DMA stream 1)
    both = 0,
    // the free tile sends both of its bank's slots (job 0 on NOC1 / stream 0, job 1 on NOC0 / stream 1); the NOC1
    // endpoint tile is left idle
    free_only = 1,
    // the NOC1-endpoint tile sends both of its bank's slots, both on NOC0 (job 0 on stream 1, job 1 on stream 0); the
    // free tile is left to another resident kernel (e.g. a KV manager's data movers)
    noc1_endpoint_only = 2,
};
const char* to_string(FlatDriscTiles tiles);

// Who reads the gate/up weights from DRAM into the gate/up cores' landing rings. Same weight layout, landing slots,
// schedule and bit-identical y either way.
enum class FlatGuSource : uint32_t {
    // the DRAM cores' persistent DRISC senders (DMA from their own bank, NoC writes into the receivers), drisc_tiles
    drisc = 0,
    // every gate/up core reads its own blocks from its slot's bank: the NCRISC (NOC0, beside the x forwarder) issues
    // the reads into the landing slots the receiver grants (the DRISC protocol, locally); the DRISC senders idle
    tensix = 1,
    // as tensix, but the bank's first slot's receivers (slot r < banks) read on NOC1 from their BRISC (the receiver
    // loop), the second slot's on NOC0 from their NCRISC: the DRISC "both" NoC split
    tensix_split = 2,
    // as tensix, every gate/up core reading on NOC1 from its BRISC (the receiver loop)
    tensix_noc1 = 3,
    // layout.gu_place says, per gate/up core, which bank holds its share and who delivers it: a Tensix read on NOC0
    // (NCRISC) / NOC1 (BRISC), or a DRISC (the bank's free DRAM_TILE, drisc_tiles free_only: hybrid plans)
    placed = 4,
    // every gate/up core reads its own share (NOC0 or NOC1), the bank and NoC of each core chosen per chip by a NoC
    // link-load search over that chip's own geometry (flat_gu_place.hpp)
    auto_tensix = 5,
    // as auto_tensix, each bank's farthest-placed slot from its free DRAM_TILE's DRISC (drisc_tiles free_only)
    auto_hybrid = 6,
};
const char* to_string(FlatGuSource source);

// FlatLayoutOptions.gu_place words: logical core (x, y), bank, how its share arrives. A bank's first slot (slot b) is
// sent by its free DRAM_TILE on NOC1 (how drisc_noc1), its second (slot b + banks) on NOC0 (drisc_noc0); a slot is all
// DRISC or all Tensix.
struct FlatGuPlace {
    enum How : uint32_t { tensix_noc0 = 1, tensix_noc1 = 2, drisc_noc1 = 3, drisc_noc0 = 4 };
    static uint32_t word(uint32_t x, uint32_t y, uint32_t bank, How how) {
        return (x << 24) | (y << 16) | (bank << 8) | how;
    }
};

// DRISC sender probes (measurement only: the probes' y is garbage). Every probe also counts the sender's loop cycles
// and events into its L1 (FlatDriscSenders::read_prof).
enum class FlatDriscProbe : uint32_t {
    none = 0,
    timing = 1,      // the normal sender, with cycle counters
    dma_only = 2,    // GDDR DMA as usual; no grant checks, no NoC weight writes (the data semaphores still move)
    write_only = 3,  // no DMA (the L1 slots as they are); grants and NoC weight writes as usual
    loop_only = 4,   // neither DMA nor grant checks nor weight writes: the loop and the semaphore increments
    call_time = 5,   // the normal sender, counting only each call's loop cycles (no per-step counters)
};
const char* to_string(FlatDriscProbe probe);

// The layout / program choices a caller makes for the flat routed expert: part of the plan cache key and of every
// program hash. -1: unset (the plan's own choice). The plan itself is fixed: DRISC-only gate/up weight readers (every
// slot from the DRAM cores, tiles in switch mode), even gate/up and down splits, x read once by the helper relays and
// fed along unicast snakes. The DRISC tiles a bank uses are a choice (drisc_tiles).
struct FlatLayoutOptions {
    int32_t x_helpers = -1;  // x helper relays per rectangle: 1 (-1: 1); the x path needs exactly one
    uint32_t pin_ratio = 8;  // pin only an expert this many times the next one's rows (0: always pin)
    // expected mean rows per local expert in a call (-1: unknown; the capacity m is the whole chunk). Plans with
    // 128-row sub-blocks and gate/up row passes (GLM-5.3, GLM-5.3-Flash) take 64-row sub-blocks at <= 320: one row
    // pass a sub-block and half the drain after the last weights land; -1.2 .. -3.3% at M 32 .. 256, but +3 .. +7%
    // at M >= 512
    int32_t rows_hint = -1;
    // x landed signal (a caller that runs the dispatch in the same program): > 0 - every x relay waits until its
    // x-ready word (the relay words at X_READY_OFF) reaches this count before its first x read, then re-arms it by
    // subtraction; the caller's dispatch ports count in there once their rows have landed. -1: x is in place at launch
    int32_t x_ready = -1;
    // which DRISC tiles of each bank send the gate/up weights (FlatDriscTiles): both, or one tile serving both of its
    // bank's slots
    FlatDriscTiles drisc_tiles = FlatDriscTiles::both;
    // DRISC sender tuning (-1: the plan's choice): blocks per unit (se_drisc.hpp SD_UB), DMA reads in flight a job
    // (SD_DMA_CAP, 1..3), the GDDR MPFE arbitration word (SD_MPFE; 0: untouched)
    int32_t drisc_ub = -1;
    int32_t drisc_dma_cap = -1;
    uint32_t drisc_mpfe = 0;
    // DRISC L1 slots refill once a unit's writes have left L1 rather than once they are acked (-1: the plan's choice)
    int32_t drisc_early_free = -1;
    // a measurement probe of the DRISC senders (FlatDriscProbe; none in production)
    FlatDriscProbe drisc_probe = FlatDriscProbe::none;
    // who reads the gate/up weights (FlatGuSource): the DRISC senders (default) or the gate/up cores themselves
    FlatGuSource gu_source = FlatGuSource::drisc;
    // gu_source tensix*: weight reads in flight a gate/up core, 1..6 (-1: 2)
    int32_t gu_read_depth = -1;
    // gu_source tensix* / placed: each weight block read as this many pieces, 1 / 2 / 4 (gu_read_depth then counts
    // pieces in flight; -1: 1)
    int32_t gu_read_pieces = -1;
    // gu_source placed: one word per gate/up core (every cell of both rectangles), FlatGuPlace::word(x, y, bank, how):
    // its share in `bank` (each bank takes 2 x RG cores: its two weight slots), delivered `how`
    std::vector<uint32_t> gu_place;

    // every field, for the plan cache key and the program hashes
    std::vector<uint32_t> key() const;
    // one line for the logs
    std::string str() const;
    bool operator==(const FlatLayoutOptions&) const = default;
};

// What the flat routed expert is built for: the device operation's attributes (every field is in the program hash).
struct FlatRoutedExpertParams {
    uint32_t hidden = 0;              // H (x / y width)
    uint32_t intermediate = 0;        // I (per device: I / TP)
    uint32_t experts_per_chip = 0;    // E
    uint32_t num_global_experts = 0;  // NG: the counts / regions rows' length
    uint32_t max_tokens = 0;          // m: tokens per expert the program is built for (dispatch capacity)
    bool weights_bf8 = false;         // weight tiles bfp8 (else bfp4)
    uint32_t activation = 0;   // se3_compute.cpp SE_ACT: 0 SiLU, 1 SwiGLU-OAI, 2 SiTU, 3 clamped SiLU, 4 GELU-tanh
    uint32_t pin = 1;          // SE_PIN_MIN (0: no pinning)
    bool y_row_major = false;  // y as row-major bf16 [rows, H] (else bfp8 tiles); no effect on the plan
    // down projection accumulated in fp32 DEST in a full-sync DST (else bf16 DEST, whose ties-away rounding over
    // K = I inflates y by ~3-4% with bfp4 weights); no effect on the plan
    bool down_fp32 = false;
    // stochastic rounding in the packer of every compute kernel (x tilize, h, y); no effect on the plan
    bool pack_stochastic_rounding = false;
    // y as bf16 tiles (else bfp8; tiled y only); no effect on the plan
    bool y_bf16 = false;
    // the expert matmuls' fidelity (MathFidelity: 0 LoFi, 2 HiFi2, 3 HiFi3, 4 HiFi4): LoFi is exact for bfp4 weights,
    // bfp8 weights need HiFi2; no effect on the plan
    uint32_t math_fidelity = 0;
    // combine overlap (flat_program.hpp): the y writers report each local expert to the combine's collector once its
    // rows have landed; the 4 report args sit at this runtime-arg index (0: off)
    uint32_t cmb_rt = 0;
    // combine overlap: the y writers' L1 hand-off args after the report args (SE_Y_L1)
    bool y_l1 = false;
    // combine overlap report protocol: cumulative counts, no go handshake; steps in slot order
    bool cmb_no_go = false;
    bool cmb_slot_order = false;
    FlatLayoutOptions layout;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "hidden",
        "intermediate",
        "experts_per_chip",
        "num_global_experts",
        "max_tokens",
        "weights_bf8",
        "activation",
        "pin",
        "y_row_major",
        "down_fp32",
        "pack_stochastic_rounding",
        "y_bf16",
        "math_fidelity",
        "cmb_rt",
        "y_l1",
        "cmb_no_go",
        "cmb_slot_order",
        "layout");
    auto attribute_values() const {
        return std::make_tuple(
            hidden,
            intermediate,
            experts_per_chip,
            num_global_experts,
            max_tokens,
            weights_bf8,
            activation,
            pin,
            y_row_major,
            down_fp32,
            pack_stochastic_rounding,
            y_bf16,
            math_fidelity,
            cmb_rt,
            y_l1,
            cmb_no_go,
            cmb_slot_order,
            layout);
    }
};

// The plan's name for the same struct (what the program is built for).
using FlatRoutedExpertConfig = FlatRoutedExpertParams;

struct FlatRoutedExpertInputs {
    Tensor x;                  // row-major bf16 dispatch buffer [rows * P, H / P], DRAM interleaved
    Tensor counts;             // [1, NG] uint32 row major, DRAM interleaved: tokens per global expert
    Tensor regions;            // [1, NG] uint32: each global expert's first row in x
    Tensor global_expert_ids;  // [E] uint32 row major, DRAM: this device's local experts' global ids
    Tensor gate_up_weights;    // per-weight-slot bank regions (the plan's layout)
    Tensor down_weights;       // per-down-core bank regions
    Tensor done_words;         // persistent L1 words on the down coordinators (zero; the op leaves them zero)
    Tensor arena;              // L1 scratch on every role core (FlatRoutedExpertPlan::arena_tiles x 32 rows a core)
    Tensor words;              // L1 words on the x relays
    Tensor output;             // y [rows, H], DRAM interleaved (only active experts' rows written)
    // indexed mode: [1, rows] uint32 row-major DRAM; flat row r reads x row token_index[r] (x = e.g. all-gathered
    // tokens instead of a dispatch buffer)
    std::optional<Tensor> token_index;
};

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
