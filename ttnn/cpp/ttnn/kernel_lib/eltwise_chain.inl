// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_chain.inl
 * @brief Implementation of the eltwise chain pipeline + chain element types + traits.
 *
 * Included from `eltwise_chain.hpp`. Do NOT include directly.
 */

#include <climits>
#include <type_traits>
#include <utility>

// Impl-only includes (the public eltwise_chain.hpp surface — element decls + enums — needs
// none of these; they live here, with the implementation that uses them).
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"  // DataflowBuffer — chain routes CB sync (wait/pop/reserve/push) through it
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace compute_kernel_lib {

enum class WaitPolicy : uint8_t { None, PerTile, PerChunk, PerOuter, Upfront, Cumulative };
enum class PopPolicy : uint8_t { None, PerTile, PerChunk, PerOuter, AtEnd };
enum class ReservePolicy : uint8_t { None, PerTile, PerChunk, Upfront, PerOuter, OneUpfront };
enum class PushPolicy : uint8_t { None, PerTile, PerChunk, AtEnd, PerOuter, OneAtEnd };

inline constexpr InputLifecycle InputLifecycle::Streaming = {WaitPolicy::PerTile, PopPolicy::PerTile};
inline constexpr InputLifecycle InputLifecycle::Chunked = {WaitPolicy::PerChunk, PopPolicy::PerChunk};
inline constexpr InputLifecycle InputLifecycle::Bulk = {WaitPolicy::Upfront, PopPolicy::AtEnd};
inline constexpr InputLifecycle InputLifecycle::Pipelined = {WaitPolicy::Cumulative, PopPolicy::AtEnd};
inline constexpr InputLifecycle InputLifecycle::CallerManaged = {WaitPolicy::None, PopPolicy::None};
inline constexpr InputLifecycle InputLifecycle::BulkDrain = {WaitPolicy::Upfront, PopPolicy::PerTile};
inline constexpr InputLifecycle InputLifecycle::HeldBulk = {WaitPolicy::Upfront, PopPolicy::None};
inline constexpr InputLifecycle InputLifecycle::HeldCumulative = {WaitPolicy::Cumulative, PopPolicy::None};
inline constexpr InputLifecycle InputLifecycle::HeldStream = {WaitPolicy::PerTile, PopPolicy::None};
inline constexpr InputLifecycle InputLifecycle::DeferredPop = {WaitPolicy::None, PopPolicy::AtEnd};
inline constexpr InputLifecycle InputLifecycle::NoWaitPop = {WaitPolicy::None, PopPolicy::PerTile};
inline constexpr InputLifecycle InputLifecycle::OuterStream = {WaitPolicy::PerOuter, PopPolicy::PerOuter};

inline constexpr OutputLifecycle OutputLifecycle::Streaming = {ReservePolicy::PerTile, PushPolicy::PerTile};
inline constexpr OutputLifecycle OutputLifecycle::Chunked = {ReservePolicy::PerChunk, PushPolicy::PerChunk};
inline constexpr OutputLifecycle OutputLifecycle::Bulk = {ReservePolicy::Upfront, PushPolicy::AtEnd};
inline constexpr OutputLifecycle OutputLifecycle::ReserveAllPushPerTile = {ReservePolicy::Upfront, PushPolicy::PerTile};
inline constexpr OutputLifecycle OutputLifecycle::ReserveAllPushPerChunk = {
    ReservePolicy::Upfront, PushPolicy::PerChunk};
inline constexpr OutputLifecycle OutputLifecycle::CallerManaged = {ReservePolicy::None, PushPolicy::None};
inline constexpr OutputLifecycle OutputLifecycle::ReserveNonePushEnd = {ReservePolicy::None, PushPolicy::AtEnd};
inline constexpr OutputLifecycle OutputLifecycle::L1Accumulation = {ReservePolicy::OneUpfront, PushPolicy::OneAtEnd};
inline constexpr OutputLifecycle OutputLifecycle::DestAccumulation = {ReservePolicy::PerOuter, PushPolicy::PerOuter};

constexpr EltwiseShape::EltwiseShape(uint32_t H, uint32_t W, uint32_t blk) : Ht(H), Wt(W), block_size(blk) {}

constexpr EltwiseShape::EltwiseShape(uint32_t n_tiles) : Ht(1), Wt(n_tiles), block_size(1) {}

constexpr EltwiseShape EltwiseShape::tiles(uint32_t n, uint32_t blk) { return {1, n, blk}; }

constexpr EltwiseShape EltwiseShape::grid(uint32_t H, uint32_t W, uint32_t blk) { return {H, W, blk}; }

constexpr EltwiseShape EltwiseShape::of(uint32_t r, uint32_t c) { return {r, c, 1}; }

constexpr EltwiseShape EltwiseShape::row(uint32_t c) { return {1, c, 1}; }

constexpr EltwiseShape EltwiseShape::col(uint32_t r) { return {r, 1, 1}; }

constexpr EltwiseShape EltwiseShape::single() { return {1, 1, 1}; }

constexpr bool is_legal_input_lifecycle(InputLifecycle lc) noexcept {
    return lc == InputLifecycle::Streaming || lc == InputLifecycle::Chunked || lc == InputLifecycle::Bulk ||
           lc == InputLifecycle::Pipelined || lc == InputLifecycle::CallerManaged || lc == InputLifecycle::BulkDrain ||
           lc == InputLifecycle::HeldBulk || lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::HeldStream ||
           lc == InputLifecycle::DeferredPop || lc == InputLifecycle::NoWaitPop || lc == InputLifecycle::OuterStream;
}

constexpr bool is_legal_output_lifecycle(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::Streaming || lc == OutputLifecycle::Chunked || lc == OutputLifecycle::Bulk ||
           lc == OutputLifecycle::ReserveAllPushPerTile || lc == OutputLifecycle::ReserveAllPushPerChunk ||
           lc == OutputLifecycle::CallerManaged || lc == OutputLifecycle::ReserveNonePushEnd ||
           lc == OutputLifecycle::L1Accumulation || lc == OutputLifecycle::DestAccumulation;
}

constexpr bool is_legal_kind_lifecycle(OperandKind kind, InputLifecycle lc) noexcept {
    if (!is_legal_input_lifecycle(lc)) {
        return false;
    }
    if (kind == OperandKind::Block) {
        return lc == InputLifecycle::Bulk || lc == InputLifecycle::Pipelined || lc == InputLifecycle::HeldBulk ||
               lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::Chunked ||
               lc == InputLifecycle::CallerManaged || lc == InputLifecycle::DeferredPop;
    }
    if (lc == InputLifecycle::Pipelined || lc == InputLifecycle::HeldCumulative || lc == InputLifecycle::Chunked) {
        return false;
    }
    if (kind == OperandKind::Scalar) {
        return true;
    }
    return lc == InputLifecycle::Bulk || lc == InputLifecycle::HeldBulk || lc == InputLifecycle::CallerManaged ||
           lc == InputLifecycle::DeferredPop;
}

constexpr bool is_legal_input_lifecycle_with_base(InputLifecycle lc) noexcept {
    return lc == InputLifecycle::Bulk || lc == InputLifecycle::HeldBulk || lc == InputLifecycle::DeferredPop ||
           lc == InputLifecycle::BulkDrain || lc == InputLifecycle::CallerManaged;
}

constexpr bool is_legal_output_lifecycle_with_base(OutputLifecycle lc) noexcept {
    return lc == OutputLifecycle::Bulk || lc == OutputLifecycle::ReserveNonePushEnd ||
           lc == OutputLifecycle::CallerManaged;
}

constexpr uint32_t to_u32(Dst s) noexcept { return static_cast<uint32_t>(s); }

namespace detail {

constexpr uint32_t bit_width_for_max(uint32_t max_value) noexcept {
    uint32_t width = 1;
    while ((max_value >>= 1) != 0) {
        ++width;
    }
    return width;
}

constexpr uint32_t low_bits_mask(uint32_t width) noexcept { return (uint32_t{1} << width) - uint32_t{1}; }

inline constexpr uint32_t first_config_bit = 0;

template <class Value, uint32_t Shift, Value MaxValue>
struct ConfigField {
    static constexpr uint32_t max_value = static_cast<uint32_t>(MaxValue);
    static constexpr uint32_t width = bit_width_for_max(max_value);
    static_assert(Shift + width <= sizeof(uint32_t) * CHAR_BIT, "ConfigField exceeds uint32_t storage");

    static constexpr uint32_t value_mask = low_bits_mask(width);
    static constexpr uint32_t end = Shift + width;

    static constexpr uint32_t encode(Value value) noexcept {
        return (static_cast<uint32_t>(value) & value_mask) << Shift;
    }

    static constexpr Value decode(uint32_t storage) noexcept {
        return static_cast<Value>((storage >> Shift) & value_mask);
    }
};

struct InputSpecConfig {
    // Buffer identity remains a separate Impl NTTP; this key packs only operand behavior.
    using WaitField = ConfigField<WaitPolicy, first_config_bit, WaitPolicy::Cumulative>;
    using PopField = ConfigField<PopPolicy, WaitField::end, PopPolicy::AtEnd>;
    using IndexField = ConfigField<OperandKind, PopField::end, OperandKind::Scalar>;
    using OffsetField = ConfigField<TileOffset, IndexField::end, TileOffset::Set>;
    using ReconfigField = ConfigField<DataFormatReconfig, OffsetField::end, DataFormatReconfig::Enabled>;

    static constexpr uint32_t used_bits = ReconfigField::end;
    static constexpr uint32_t storage_mask = low_bits_mask(used_bits);
    static_assert(used_bits <= sizeof(uint16_t) * CHAR_BIT, "InputSpec behavior exceeds uint16_t storage");

    static constexpr uint16_t encode(InputSpec spec) noexcept {
        return static_cast<uint16_t>(
            WaitField::encode(spec.lifecycle.wait_policy) | PopField::encode(spec.lifecycle.pop_policy) |
            IndexField::encode(spec.index) | OffsetField::encode(spec.offset) | ReconfigField::encode(spec.reconfig));
    }

    static constexpr InputSpec decode(uint16_t storage, uint32_t cb_id) noexcept;
};

struct OutputSpecConfig {
    // Buffer identity remains a separate Impl NTTP; this key packs only operand behavior.
    using ReserveField = ConfigField<ReservePolicy, first_config_bit, ReservePolicy::OneUpfront>;
    using PushField = ConfigField<PushPolicy, ReserveField::end, PushPolicy::OneAtEnd>;
    using ReconfigField = ConfigField<DataFormatReconfig, PushField::end, DataFormatReconfig::Enabled>;
    using ReluField = ConfigField<PackRelu, ReconfigField::end, PackRelu::Zero>;
    using L1AccumulationField = ConfigField<L1Accumulation, ReluField::end, L1Accumulation::SeedFirst>;
    using DestAccumulationField = ConfigField<DestAccumulation, L1AccumulationField::end, DestAccumulation::Enabled>;
    using OffsetField = ConfigField<TileOffset, DestAccumulationField::end, TileOffset::Set>;

    static constexpr uint32_t used_bits = OffsetField::end;
    static constexpr uint32_t storage_mask = low_bits_mask(used_bits);
    static_assert(used_bits <= sizeof(uint16_t) * CHAR_BIT, "OutputSpec behavior exceeds uint16_t storage");

    static constexpr uint16_t encode(OutputSpec spec) noexcept {
        return static_cast<uint16_t>(
            ReserveField::encode(spec.lifecycle.reserve_policy) | PushField::encode(spec.lifecycle.push_policy) |
            ReconfigField::encode(spec.reconfig) | ReluField::encode(spec.relu) |
            L1AccumulationField::encode(spec.l1_accumulation) | DestAccumulationField::encode(spec.dest_accumulation) |
            OffsetField::encode(spec.offset));
    }

    static constexpr OutputSpec decode(uint16_t storage, uint32_t cb_id) noexcept;
};

}  // namespace detail

constexpr bool InputLifecycle::operator==(InputLifecycle other) const noexcept {
    return wait_policy == other.wait_policy && pop_policy == other.pop_policy;
}

constexpr bool InputLifecycle::operator!=(InputLifecycle other) const noexcept { return !(*this == other); }

constexpr bool OutputLifecycle::operator==(OutputLifecycle other) const noexcept {
    return reserve_policy == other.reserve_policy && push_policy == other.push_policy;
}

constexpr bool OutputLifecycle::operator!=(OutputLifecycle other) const noexcept { return !(*this == other); }

namespace detail {

constexpr InputSpec InputSpecConfig::decode(uint16_t storage, uint32_t cb_id) noexcept {
    return {
        cb_id,
        {WaitField::decode(storage), PopField::decode(storage)},
        IndexField::decode(storage),
        ReconfigField::decode(storage),
        OffsetField::decode(storage)};
}

constexpr OutputSpec OutputSpecConfig::decode(uint16_t storage, uint32_t cb_id) noexcept {
    return {
        cb_id,
        {ReserveField::decode(storage), PushField::decode(storage)},
        ReconfigField::decode(storage),
        ReluField::decode(storage),
        L1AccumulationField::decode(storage),
        DestAccumulationField::decode(storage),
        OffsetField::decode(storage)};
}

}  // namespace detail

constexpr InputSpec input(
    uint32_t cb_id,
    InputLifecycle lifecycle,
    OperandKind index,
    DataFormatReconfig reconfig,
    TileOffset offset) noexcept {
    return {cb_id, lifecycle, index, reconfig, offset};
}

constexpr InputSpec input(uint32_t cb_id, InputLifecycle lifecycle, DataFormatReconfig reconfig) noexcept {
    return input(cb_id, lifecycle, OperandKind::Scalar, reconfig);
}

constexpr OutputSpec output(
    uint32_t cb_id,
    OutputLifecycle lifecycle,
    DataFormatReconfig reconfig,
    PackRelu relu,
    L1Accumulation l1_accumulation,
    DestAccumulation dest_accumulation,
    TileOffset offset) noexcept {
    return {cb_id, lifecycle, reconfig, relu, l1_accumulation, dest_accumulation, offset};
}

namespace detail {

struct CopyTileConfig {
    using DstField = ConfigField<Dst, first_config_bit, Dst::D15>;
    using InputField = ConfigField<uint16_t, DstField::end, static_cast<uint16_t>(InputSpecConfig::storage_mask)>;

    uint32_t bits;

    constexpr CopyTileConfig(Dst dst, InputSpec input_spec) noexcept :
        bits(DstField::encode(dst) | InputField::encode(InputSpecConfig::encode(input_spec))) {}
    constexpr explicit CopyTileConfig(uint32_t encoded) noexcept : bits(encoded) {}

    constexpr Dst dst() const noexcept { return DstField::decode(bits); }
    constexpr InputSpec input_spec(uint32_t cb_id) const noexcept {
        return InputSpecConfig::decode(InputField::decode(bits), cb_id);
    }
};

struct PackTileConfig {
    using OutputField = ConfigField<uint16_t, first_config_bit, static_cast<uint16_t>(OutputSpecConfig::storage_mask)>;
    using DstField = ConfigField<Dst, OutputField::end, Dst::D15>;

    uint32_t bits;

    constexpr PackTileConfig(OutputSpec output_spec, Dst dst) noexcept :
        bits(OutputField::encode(OutputSpecConfig::encode(output_spec)) | DstField::encode(dst)) {}
    constexpr explicit PackTileConfig(uint32_t encoded) noexcept : bits(encoded) {}

    constexpr OutputSpec output_spec(uint32_t cb_id) const noexcept {
        return OutputSpecConfig::decode(OutputField::decode(bits), cb_id);
    }
    constexpr Dst dst() const noexcept { return DstField::decode(bits); }
};

struct BinaryFpuConfig {
    using OpField = ConfigField<BinaryFpuOp, first_config_bit, BinaryFpuOp::Mul>;
    using BroadcastField = ConfigField<BroadcastDim, OpField::end, BroadcastDim::Scalar>;
    using AInputField =
        ConfigField<uint16_t, BroadcastField::end, static_cast<uint16_t>(InputSpecConfig::storage_mask)>;
    using BInputField = ConfigField<uint16_t, AInputField::end, static_cast<uint16_t>(InputSpecConfig::storage_mask)>;
    using DstField = ConfigField<Dst, BInputField::end, Dst::D15>;
    using AccumulationField = ConfigField<DestAccumulation, DstField::end, DestAccumulation::Enabled>;

    uint32_t bits;

    constexpr BinaryFpuConfig(
        BinaryFpuOp op, BroadcastDim bcast, InputSpec a, InputSpec b, Dst dst, DestAccumulation accumulation) noexcept :
        bits(
            OpField::encode(op) | BroadcastField::encode(bcast) | AInputField::encode(InputSpecConfig::encode(a)) |
            BInputField::encode(InputSpecConfig::encode(b)) | DstField::encode(dst) |
            AccumulationField::encode(accumulation)) {}
    constexpr explicit BinaryFpuConfig(uint32_t encoded) noexcept : bits(encoded) {}

    constexpr BinaryFpuOp op() const noexcept { return OpField::decode(bits); }
    constexpr BroadcastDim broadcast() const noexcept { return BroadcastField::decode(bits); }
    constexpr InputSpec a_input_spec(uint32_t cb_id) const noexcept {
        return InputSpecConfig::decode(AInputField::decode(bits), cb_id);
    }
    constexpr InputSpec b_input_spec(uint32_t cb_id) const noexcept {
        return InputSpecConfig::decode(BInputField::decode(bits), cb_id);
    }
    constexpr Dst dst() const noexcept { return DstField::decode(bits); }
    constexpr DestAccumulation accumulation() const noexcept { return AccumulationField::decode(bits); }
};

struct DestReuseBinaryConfig {
    using OpField = ConfigField<BinaryFpuOp, first_config_bit, BinaryFpuOp::Mul>;
    using ReuseField = ConfigField<DestReuseType, OpField::end, DestReuseType::DEST_TO_SRCB>;
    using InputField = ConfigField<uint16_t, ReuseField::end, static_cast<uint16_t>(InputSpecConfig::storage_mask)>;
    using DstInField = ConfigField<Dst, InputField::end, Dst::D15>;
    using DstOutField = ConfigField<Dst, DstInField::end, Dst::D15>;

    uint32_t bits;

    constexpr DestReuseBinaryConfig(
        BinaryFpuOp op, DestReuseType reuse, InputSpec input_spec, Dst dst_in, Dst dst_out) noexcept :
        bits(
            OpField::encode(op) | ReuseField::encode(reuse) | InputField::encode(InputSpecConfig::encode(input_spec)) |
            DstInField::encode(dst_in) | DstOutField::encode(dst_out)) {}
    constexpr explicit DestReuseBinaryConfig(uint32_t encoded) noexcept : bits(encoded) {}

    constexpr BinaryFpuOp op() const noexcept { return OpField::decode(bits); }
    constexpr DestReuseType reuse() const noexcept { return ReuseField::decode(bits); }
    constexpr InputSpec input_spec(uint32_t cb_id) const noexcept {
        return InputSpecConfig::decode(InputField::decode(bits), cb_id);
    }
    constexpr Dst dst_in() const noexcept { return DstInField::decode(bits); }
    constexpr Dst dst_out() const noexcept { return DstOutField::decode(bits); }
};

constexpr uint32_t copy_tile_config_bits(Dst dst, InputSpec input_spec) noexcept {
    return CopyTileConfig{dst, input_spec}.bits;
}

constexpr uint32_t pack_tile_config_bits(OutputSpec output_spec, Dst dst) noexcept {
    return PackTileConfig{output_spec, dst}.bits;
}

constexpr uint32_t binary_fpu_config_bits(
    BinaryFpuOp op, BroadcastDim bcast, InputSpec a, InputSpec b, Dst dst, DestAccumulation accumulation) noexcept {
    return BinaryFpuConfig{op, bcast, a, b, dst, accumulation}.bits;
}

constexpr uint32_t dest_reuse_binary_config_bits(
    BinaryFpuOp op, DestReuseType reuse, InputSpec input_spec, Dst dst_in, Dst dst_out) noexcept {
    return DestReuseBinaryConfig{op, reuse, input_spec, dst_in, dst_out}.bits;
}

}  // namespace detail

// Internal sentinel + type-list wrapper + chain-shape trait declarations. These are
// implementation detail of the chain pipeline — no chain caller references them, so they
// live here rather than on the public eltwise_chain.hpp surface.
inline constexpr uint32_t INVALID_DFB = 0xFFFFFFFFu;      // "no CB on this slot" (== NO_PREV_DFB); a real
                                                          // tt::CBIndex is 0..31, so 0xFFFFFFFF never aliases one.
inline constexpr uint32_t CONDITIONAL_DFB = 0xFFFFFFFEu;  // a runtime branch may leave either the old or a new
                                                          // format programmed; force the following element to
                                                          // re-establish its expected input-side format.

template <class... Es>
struct EltwiseChain;  // typed list of elements (defined below)

template <class Chain>
struct chain_has_duplicate_upfront_dfbs;
template <class Chain>
struct chain_pack_writes_collide;
template <class Chain>
struct chain_per_side_dfbs_consistent;
template <class Chain>
struct chain_math_mop_uniform;
template <class Chain>
struct chain_sfpu_inits_uniform;
template <class Chain>
struct chain_hoist_math_mop;
template <class Chain>
struct chain_hoist_sfpu;
template <class Chain>
struct chain_hoist_pack;

template <class Chain>
inline constexpr bool chain_has_duplicate_upfront_cbs_v = chain_has_duplicate_upfront_dfbs<Chain>::value;
template <class Chain>
inline constexpr bool chain_pack_writes_collide_v = chain_pack_writes_collide<Chain>::value;
template <class Chain>
inline constexpr bool chain_per_side_cbs_consistent_v = chain_per_side_dfbs_consistent<Chain>::value;
template <class Chain>
inline constexpr bool chain_math_mop_uniform_v = chain_math_mop_uniform<Chain>::value;
template <class Chain>
inline constexpr bool chain_sfpu_inits_uniform_v = chain_sfpu_inits_uniform<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_math_mop_v = chain_hoist_math_mop<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_sfpu_v = chain_hoist_sfpu<Chain>::value;
template <class Chain>
inline constexpr bool chain_hoist_pack_v = chain_hoist_pack<Chain>::value;

// =============================================================================
// Marker tag hierarchy (data direction → kind)
// =============================================================================
// Internal classification scaffolding — chain callers never name these. Concrete
// element types (declared on the .hpp surface) inherit the leaf tags; the chain
// pipeline + SFPU/FPU op-helper headers classify elements via the is_*_op_v
// predicates below.

/// Element reads ≥1 CB. Pure marker — concrete elements declare `dfb_a_id()` and
/// (if binary) `dfb_b_id()`. No stub defaults, so a missing accessor is a compile
/// error rather than a silently-wrong default CB id.
struct CbReaderTag {};
/// Element writes to a CB. Pure marker — concrete elements declare `pack_dfb_id()`.
/// No stub defaults.
struct CbWriterTag {};
/// Element neither reads nor writes a CB (DEST-internal). Carries only the
/// `is_upfront` default — no CB-id stubs. The chain pipeline SFINAE-detects
/// `dfb_a_id()` / `dfb_b_id()` / `pack_dfb_id()` on the element directly and never
/// reaches a DestOnlyTag default.
struct DestOnlyTag {
    static constexpr bool is_upfront = false;
};

/// Marker mixed into runtime-conditional wrappers. It is orthogonal to the element-kind
/// hierarchy so the wrapped element keeps its ordinary reader / DEST-only classification.
struct RuntimeConditionalTag {};

/// Marks a runtime-conditional sequence handled as one chain stage. Unlike a
/// single-element runtime wrapper, the sequence performs its own ordered walk
/// so one runtime decision guards every element in the selected branch.
struct RuntimeConditionalSequenceTag {};

/// Pure CB → DEST move (no compute).
struct CopyTileTag : CbReaderTag {};
/// 2 CBs → DEST FPU compute (add/sub/mul + bcast variants).
struct BinaryFpuTag : CbReaderTag {};
/// 1 CB + DEST → DEST FPU compute (binary_dest_reuse_tiles).
struct DestReuseBinaryTag : CbReaderTag {};
/// 1 CB → DEST row/col/scalar broadcast (unary_bcast).
struct UnaryBcastTag : CbReaderTag {};

/// DEST → CB store (pack_tile / pack_tile_block).
struct PackTileTag : CbWriterTag {};

/// Constant → DEST (no CB read).
struct FillTileTag : DestOnlyTag {};
/// RNG → DEST (no CB read).
struct RandTileTag : DestOnlyTag {};

// Trait predicates — which predicate drives each chain decision:
//
//  Sweep / decision                                            | predicate
//  ------------------------------------------------------------|---------------------------
//  Duplicate upfront-CB check across all CB-consumers          | is_cb_reader_op_v
//  Output-CB collision / fan-out across all writers            | is_cb_writer_op_v
//  Hoist-safety "chain shape is CopyTile + 1 SFPU op"          | is_copy_tile_op_v
//  FPU-clash reinit                                            | is_binary_fpu_op_v ‖
//                                                              | is_dest_reuse_binary_op_v ‖
//                                                              | is_unary_bcast_op_v
//  Hoist exclusion: element issues a pack inside the loop      | is_pack_tile_op_v
//  No CB lifecycle to check — pure DEST internal               | is_dest_only_op_v
template <class T>
inline constexpr bool is_cb_reader_op_v = std::is_base_of_v<CbReaderTag, T>;
template <class T>
inline constexpr bool is_cb_writer_op_v = std::is_base_of_v<CbWriterTag, T>;
template <class T>
inline constexpr bool is_dest_only_op_v = std::is_base_of_v<DestOnlyTag, T>;
template <class T>
inline constexpr bool is_copy_tile_op_v = std::is_base_of_v<CopyTileTag, T>;
template <class T>
inline constexpr bool is_binary_fpu_op_v = std::is_base_of_v<BinaryFpuTag, T>;
template <class T>
inline constexpr bool is_dest_reuse_binary_op_v = std::is_base_of_v<DestReuseBinaryTag, T>;
template <class T>
inline constexpr bool is_unary_bcast_op_v = std::is_base_of_v<UnaryBcastTag, T>;
template <class T>
inline constexpr bool is_pack_tile_op_v = std::is_base_of_v<PackTileTag, T>;
template <class T>
inline constexpr bool is_fill_tile_op_v = std::is_base_of_v<FillTileTag, T>;
template <class T>
inline constexpr bool is_rand_tile_op_v = std::is_base_of_v<RandTileTag, T>;
template <class T>
inline constexpr bool is_runtime_conditional_op_v = std::is_base_of_v<RuntimeConditionalTag, T>;
template <class T>
inline constexpr bool is_runtime_conditional_sequence_op_v = std::is_base_of_v<RuntimeConditionalSequenceTag, T>;
/// SFPU (DEST-internal, non-RNG, non-fill) element predicate. SFPU ops inherit
/// from `DestOnlyTag` via `UnaryOp` / `BinaryOp` / `TernaryOp`;
/// Fill / Rand share the `DestOnlyTag` lineage but their init programs PRNG /
/// fill state, not the SFPU MOP / ADDR_MOD_7 lane. The hoist gate counts distinct
/// SFPU init types — `is_sfpu_op_v` is the predicate.
template <class T>
inline constexpr bool is_sfpu_op_v = is_dest_only_op_v<T> && !is_fill_tile_op_v<T> && !is_rand_tile_op_v<T>;

/// FPU-kind (non-CopyTile, FPU-MOP-touching) element predicate. Groups
/// `BinaryFpu`, `DestReuseBinary`, `UnaryBcast` — each programs the FPU MOP /
/// ADDR_MOD_0..3 lane on init via the binary-op init path.
template <class T>
inline constexpr bool is_fpu_kind_op_v =
    is_binary_fpu_op_v<T> || is_dest_reuse_binary_op_v<T> || is_unary_bcast_op_v<T>;

/// MATH-MOP-touching element predicate. Groups every element whose init
/// programs the MATH MOP / ADDR_MOD_0..3 lane: `CopyTile` (via
/// `copy_tile_to_dst_init_short`) and the FPU-kind ops (`BinaryFpu`,
/// `DestReuseBinary`, `UnaryBcast`). The hoist gate requires all such
/// elements in a chain (including the CopyTile-versus-FPU init clash) to be
/// the same instantiated type — otherwise the boot-time fold leaves only the
/// last init's MOP programmed and earlier elements run with the wrong MOP.
template <class T>
inline constexpr bool is_math_mop_op_v = is_copy_tile_op_v<T> || is_fpu_kind_op_v<T>;

// =============================================================================
// tile_base_value — orthogonal tile-index offset extractor (impl helper)
// =============================================================================
/// Extract the offset value stored on an element. Returns 0 (compile-time-folded) when the
/// element's `Offset` is `Unset`, so the `+base` term and the stored field vanish.
template <TileOffset Offset>
ALWI uint32_t tile_base_value([[maybe_unused]] uint32_t stored) noexcept {
    if constexpr (Offset == TileOffset::Unset) {
        return 0u;
    } else {
        return stored;
    }
}

// =============================================================================
// CRTP bases — UnaryOp / BinaryOp / TernaryOp
// =============================================================================
//
// Dispatch contract: the DEST-only op bases expose `void exec(uint32_t i, uint32_t slot_offset) const`
// and forward to a static `exec_impl(uint32_t slot_offset)` supplied by the derived op; runtime-param
// ops (Power, Hardtanh, …) override `exec` directly to capture instance state. (CB elements — CopyTile /
// BinaryFpu — define a wider exec: (i_flat, ht, wt, slot_offset) or the per-side form.) Defining
// neither is a compile error (no silent fallthrough).
//
//   template <Approx A = Approx::Exact, Approx F = Approx::Fast, Dst Slot = Dst::D0>
//   struct Exp : UnaryOp<Exp<A, F, Slot>, Slot> {
//       static void init()                        { exp_tile_init<A == Approx::Fast, F == Approx::Fast>(); }
//       static void exec_impl(uint32_t slot_off)  { exp_tile<A == Approx::Fast, F == Approx::Fast>(to_u32(Slot) +
//       slot_off); }
//   };

template <class Derived, Dst Slot>
struct UnaryOp : DestOnlyTag {
    static_assert(
        to_u32(Slot) < DEST_AUTO_LIMIT, "UnaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");

    static constexpr Dst dst_idx = Slot;
    static constexpr uint32_t max_dst() { return to_u32(Slot); }
    /// Per-lane DEST footprint. Used by chain to pick auto BlockSize.
    /// Default = `to_u32(Slot) + 1` (op writes only Slot). Override per-op when the
    /// op references more slots (e.g. Mask uses DataSlot AND DataSlot+1).
    static constexpr uint32_t lane_width = to_u32(Slot) + 1;

    /// Pipeline dispatch — forwards to `Derived::exec_impl(slot_offset)`. Override
    /// in derived to consume runtime payload (per-instance fields). `slot_offset`
    /// is added by the chain to shift DEST writes into lane `j` when `BlockSize > 1`;
    /// `BlockSize == 1` passes 0 (per-tile shape).
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

template <class Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(Out) < DEST_AUTO_LIMIT,
        "BinaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");
    // NOTE: slot-distinctness is *not* enforced here. SFPU binary ops (AddBinary /
    // SubBinary / MulBinary / DivBinary) routinely operate in-place (Out == In0 or
    // Out == In1) and even `In0 == In1` is legal (e.g. squaring). The FPU binary
    // chain element (`BinaryFpu`) reads its inputs from CBs, not DEST slots, so it
    // also doesn't need In0 != In1. If a future op truly requires distinct DEST
    // slots, it should enforce that locally rather than at the CRTP base.

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst out = Out;
    static constexpr uint32_t max_dst() {
        uint32_t a = to_u32(In0), b = to_u32(In1), c = to_u32(Out);
        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }
    static constexpr uint32_t lane_width = max_dst() + 1;

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

template <class Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp : DestOnlyTag {
    static_assert(
        to_u32(In0) < DEST_AUTO_LIMIT && to_u32(In1) < DEST_AUTO_LIMIT && to_u32(In2) < DEST_AUTO_LIMIT &&
            to_u32(Out) < DEST_AUTO_LIMIT,
        "TernaryOp: DEST slot exceeds compile-time DEST capacity (DEST_AUTO_LIMIT)");
    // NOTE: slot-distinctness is *not* enforced here (mirrors BinaryOp). SFPU ternary
    // ops (where / lerp / addcmul / addcdiv) routinely write Out into one of the input
    // slots in-place — the kernel reads all three inputs before overwriting. If a
    // future op truly requires distinct DEST slots, enforce it locally rather than at
    // the CRTP base.

    static constexpr Dst in0 = In0;
    static constexpr Dst in1 = In1;
    static constexpr Dst in2 = In2;
    static constexpr Dst out = Out;
    static constexpr uint32_t lane_width = []() {
        uint32_t m = to_u32(In0);
        if (to_u32(In1) > m) {
            m = to_u32(In1);
        }
        if (to_u32(In2) > m) {
            m = to_u32(In2);
        }
        if (to_u32(Out) > m) {
            m = to_u32(Out);
        }
        return m + 1;
    }();

    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { Derived::exec_impl(slot_offset); }
};

// =============================================================================
// Compile-time prev-CB tracking — drives the reconfig fold
// =============================================================================
//
// Each element exposes uniform static accessors for the CB it routes to each Side
// (SrcA / SrcB / Pack), letting the pipeline compute the most recent CB on each side
// before a given element. NO_PREV_DFB is the "doesn't touch this side" sentinel.

inline constexpr uint32_t NO_PREV_DFB = 0xFFFFFFFFu;

enum class Side : uint8_t { SrcA, SrcB, Pack };

// Compile-time membership test: is `V` one of `Set...`? Empty set → false. Collapses the
// repeated `X == A || X == B || ...` equality ladders into `is_one_of_v<X, A, B, ...>`.
// `auto V` + `decltype(V)...` lets one definition serve every enum (InputLifecycle,
// OutputLifecycle, OperandKind, reconfig enums, …); the fold is a constant expression, usable in
// `if constexpr`, `static_assert`, and `static constexpr` members. Requires V and Set... to be
// constant expressions — it does NOT apply to runtime values or function parameters.
template <auto V, decltype(V)... Set>
inline constexpr bool is_one_of_v = ((V == Set) || ...);

namespace detail {

// =============================================================================
// A0. 2D index-mode helpers (OperandKind → tile index / upfront window)
//
// Compile-time-elided `idx` / `window` (defined below), inlined by every CB-reader's
// `exec` / `wait_upfront` — `if constexpr` collapses to one arithmetic op at run time.
// `TileBase` layers a runtime offset on top. `is_bcast_mode_v<M>` drives the (Policy ×
// Mode) static_asserts (Row/Col reject streaming policies, as in `binary_op_helpers`).
// =============================================================================

template <OperandKind M>
inline constexpr bool is_bcast_mode_v = is_one_of_v<M, OperandKind::Row, OperandKind::Col>;

template <OperandKind M>
ALWI constexpr uint32_t idx(
    [[maybe_unused]] uint32_t i_flat, [[maybe_unused]] uint32_t ht, [[maybe_unused]] uint32_t wt) noexcept {
    if constexpr (M == OperandKind::Scalar) {
        return 0;
    } else if constexpr (M == OperandKind::Block) {
        return i_flat;
    } else if constexpr (M == OperandKind::Row) {
        return wt;
    } else {
        return ht;  // Col
    }
}

template <OperandKind M>
ALWI constexpr uint32_t window([[maybe_unused]] uint32_t Ht, [[maybe_unused]] uint32_t Wt) noexcept {
    if constexpr (M == OperandKind::Block) {
        return Ht * Wt;
    } else if constexpr (M == OperandKind::Row) {
        return Wt;
    } else if constexpr (M == OperandKind::Col) {
        return Ht;
    } else {
        return 1u;  // Scalar
    }
}

// Allowed (Policy × Mode) combinations. Row/Col cannot stream per-tile —
// the producer must stage the full row/col upfront. Matches the
// `binary_op_helpers` static_assert (ROW/SCALAR require InputLifecycle::Bulk-family or NoWait*).
template <InputLifecycle P, OperandKind M>
inline constexpr bool valid_policy_mode_v =
    !(is_bcast_mode_v<M> && is_one_of_v<P, InputLifecycle::Streaming, InputLifecycle::Chunked>);

// =============================================================================
// B. Static cb-id / dst-slot extraction predicates per element
//
// Every CB-reader element exposes dfb_a_id() (primary CB) + a_policy(). Binary readers also expose
// dfb_b_id() (secondary CB) + b_policy(); unary readers omit them — the defaults below supply
// dfb_b = INVALID_DFB (0xFFFFFFFF, "no CB"; not 0, which is a real CB) and b_policy = CallerManaged.
// (default impls below cover non-CB-reader elements.)
//
// Every CB-writer element must expose:
//   static constexpr uint32_t pack_dfb_id();
//   static constexpr Dst pack_dst_slot;
// (output addressing derives from the walk + TileOffset, not a pack_output_index accessor.)
// =============================================================================

template <class T, class = void>
struct has_dfb_a : std::false_type {};
template <class T>
struct has_dfb_a<T, std::void_t<decltype(T::dfb_a_id())>> : std::true_type {};

template <class T, class = void>
struct has_dfb_b : std::false_type {};
template <class T>
struct has_dfb_b<T, std::void_t<decltype(T::dfb_b_id())>> : std::true_type {};

template <class T, class = void>
struct has_pack_dfb : std::false_type {};
template <class T>
struct has_pack_dfb<T, std::void_t<decltype(T::pack_dfb_id())>> : std::true_type {};

// Forward declarations — defined below (used by reader_pair_collide / writer_pair_collide
// earlier in the file's flow).
template <class E>
constexpr uint32_t dfb_a_of();
template <class E>
constexpr uint32_t dfb_b_of();
template <class E>
constexpr uint32_t pack_dfb_of();

// ChainTraits<Es...> — the one value-reflection aggregate for the whole chain (defined
// once below, after the per-element accessors it reads). Forward-declared here so the
// trait wrappers (chain_lane_width etc.) can name it.
template <class... Es>
struct ChainTraits;

// =============================================================================
// Per-Side prev-CB SFINAE probe
//
// `dfb_for_side<Side, E>` reads `E::reconfig_srca_dfb` / `_srcb_cb` / `_pack_cb`
// when present, returns `NO_PREV_DFB` otherwise. Elements that don't declare the
// accessors still participate in the fold transparently (treated as no-prev).
// =============================================================================

template <class E, class = void>
struct has_reconfig_srca : std::false_type {};
template <class E>
struct has_reconfig_srca<E, std::void_t<decltype(E::reconfig_srca_dfb)>> : std::true_type {};

template <class E, class = void>
struct has_reconfig_srcb : std::false_type {};
template <class E>
struct has_reconfig_srcb<E, std::void_t<decltype(E::reconfig_srcb_dfb)>> : std::true_type {};

template <class E, class = void>
struct has_reconfig_pack : std::false_type {};
template <class E>
struct has_reconfig_pack<E, std::void_t<decltype(E::reconfig_pack_dfb)>> : std::true_type {};

template <Side S, class E>
constexpr uint32_t dfb_for_side() {
    if constexpr (S == Side::SrcA) {
        if constexpr (has_reconfig_srca<E>::value) {
            return E::reconfig_srca_dfb;
        } else {
            return NO_PREV_DFB;
        }
    } else if constexpr (S == Side::SrcB) {
        if constexpr (has_reconfig_srcb<E>::value) {
            return E::reconfig_srcb_dfb;
        } else {
            return NO_PREV_DFB;
        }
    } else {  // Pack
        if constexpr (has_reconfig_pack<E>::value) {
            return E::reconfig_pack_dfb;
        } else {
            return NO_PREV_DFB;
        }
    }
}

// True iff NO element in the pack requests any srca / srcb / pack reconfig — i.e. every element's
// reconfig knob is None. Under SetupOwner::Caller the chain emits zero reconfig, so a non-None knob
// is inert and lies about what the helper does; eltwise_chain uses this to forbid that, forcing the
// caller to declare None — which honestly reflects "the chain does no reconfig, I own the format."
template <class... Es>
constexpr bool chain_requests_no_reconfig() {
    return (
        (dfb_for_side<Side::SrcA, Es>() == NO_PREV_DFB && dfb_for_side<Side::SrcB, Es>() == NO_PREV_DFB &&
         dfb_for_side<Side::Pack, Es>() == NO_PREV_DFB) &&
        ...);
}

// Per-side prev-CB history, last opt-in pack CB, and heterogeneous-pack detection are
// single-sweep fields on `ChainTraits` (prev / last_pack_cb / pack_hetero), computed
// once from the reflected ElemDesc array.

}  // namespace detail

// =============================================================================
// 0. Operand streams — runtime state for driver-managed CB lifecycles
//
// Every CB-reader element's wait/pop hooks are a function of exactly one operand
// tuple (Cb, Policy, IndexMode, Offset, tile_base); every CB-writer's reserve/push
// hooks of one (Cb, Policy, Offset, tile_base). Cb and the spec remain compile-time
// element metadata for planning/reconfig. The existing chain workers emit the lifecycle
// operations directly and pass the CB ids to DataflowBuffer as ordinary values; ALWI
// inlining constant-folds those values and the policy gates without creating one
// stream-method specialization per element configuration.
//
// Single-stream elements (CopyTile / DestReuseBinary / UnaryBcast) inherit the
// non-template `InputStream` state; BinaryFpu holds two and the driver handles its
// second input explicitly; PackTile inherits `OutputStream`.
//
// All policy gates below are a verbatim transcription of the per-element bodies —
// a wrong gate is a hang or a PCC failure, never benign.
// =============================================================================

struct InputStream {
    uint32_t tile_base = 0;

    constexpr InputStream() noexcept = default;
    constexpr explicit InputStream(uint32_t base) noexcept : tile_base(base) {}
};

struct OutputStream {
    uint32_t tile_base = 0;

    constexpr OutputStream() noexcept = default;
    constexpr explicit OutputStream(uint32_t base) noexcept : tile_base(base) {}
};

// =============================================================================
// 1. CopyTile chain element
// =============================================================================

template <uint32_t Cb, uint32_t ConfigBits>
struct detail::CopyTileImpl : InputStream, CopyTileTag {
    static constexpr CopyTileConfig Config{ConfigBits};
    static constexpr InputSpec Input = Config.input_spec(Cb);
    static constexpr Dst DstSlot = Config.dst();
    static constexpr InputLifecycle Policy = Input.lifecycle;
    static constexpr DataFormatReconfig Reconfig = Input.reconfig;
    static constexpr OperandKind IndexMode = Input.index;
    static constexpr TileOffset Offset = Input.offset;
    using Base = InputStream;
    using Base::tile_base;

    // ---- compile-time validation ----
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "CopyTile: DEST slot exceeds DEST_AUTO_LIMIT");
    // Comprehensive (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-index pitfall) and
    // PerTile-wait-of-1 (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(
        is_legal_kind_lifecycle(IndexMode, Policy),
        "CopyTile: (IndexMode, Policy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    // 2D: RowBcast / ColBcast require non-streaming policy (matches binary_op_helpers ROW/SCALAR rule).
    static_assert(
        detail::valid_policy_mode_v<Policy, IndexMode>,
        "CopyTile: RowBcast / ColBcast index require non-streaming policy "
        "(WaitUpfrontPopAtEnd, WaitNoPop, InputLifecycle::NoWaitPop, NoWaitNoPop, CumulativeWaitPopAtEnd)");
    // TileOffset::Set requires InputLifecycle::Bulk-family / InputLifecycle::CallerManaged lifecycle — iter-dependent
    // counts
    // (InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative/Held{Stream,Cumulative}/InputLifecycle::NoWaitPop)
    // can't compose with runtime base offsets. Caller must size CB to base+window.
    static_assert(
        Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
        "CopyTile: TileOffset::Set requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged lifecycle "
        "(InputLifecycle::Bulk / InputLifecycle::HeldBulk / InputLifecycle::DeferredPop / InputLifecycle::BulkDrain / "
        "InputLifecycle::CallerManaged)");

    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t dfb_a_id() { return Cb; }
    // CopyTile reads one CB front (srcA via dfb_a_id); dfb_b / b_policy absent -> defaults apply.
    static constexpr InputLifecycle a_policy() { return Policy; }
    static constexpr bool is_upfront =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

    // Prev-CB fold: CopyTile loads CbA only. srcb/pack sides are absent -> dfb_for_side
    // defaults them to NO_PREV_DFB.
    static constexpr uint32_t reconfig_srca_dfb = (Reconfig == DataFormatReconfig::Enabled) ? Cb : NO_PREV_DFB;

    constexpr CopyTileImpl() noexcept = default;
    constexpr explicit CopyTileImpl(uint32_t base) noexcept : Base(base) {}

    // ---- chain pipeline hooks ----
    static ALWI void init() { copy_tile_init(Cb); }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        const uint32_t in_idx = tile_base_value<Offset>(tile_base) + detail::idx<IndexMode>(i_flat, ht, wt);
        copy_tile(Cb, in_idx, to_u32(DstSlot) + slot_offset);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    // The chain driver emits wait/pop operations from the metadata above and the
    // runtime `tile_base` inherited from InputStream.
};

// =============================================================================
// 2. PackTile chain element
// =============================================================================

template <uint32_t Cb, uint32_t ConfigBits>
struct detail::PackTileImpl : OutputStream, PackTileTag {
    static constexpr PackTileConfig Config{ConfigBits};
    static constexpr OutputSpec Output = Config.output_spec(Cb);
    static constexpr OutputLifecycle Policy = Output.lifecycle;
    static constexpr DataFormatReconfig Reconfig = Output.reconfig;
    static constexpr PackRelu Relu = Output.relu;
    static constexpr L1Accumulation L1AccumulationMode = Output.l1_accumulation;
    static constexpr DestAccumulation DestAccumulationMode = Output.dest_accumulation;
    static constexpr Dst DstSlot = Config.dst();
    static constexpr TileOffset Offset = Output.offset;
    using Base = OutputStream;
    using Base::tile_base;
    // Walk vs pinned output addressing is derived from the lifecycle: upfront-reserve
    // policies write distinct tiles in one reserved window; front-advancing policies stay pinned.
    static constexpr bool walk = Policy.reserve_policy == ReservePolicy::Upfront;

    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "PackTile: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(is_legal_output_lifecycle(Policy), "PackTile: output lifecycle is not a named OutputLifecycle");
    static_assert(
        L1AccumulationMode == L1Accumulation::Disabled || Policy == OutputLifecycle::L1Accumulation ||
            Policy == OutputLifecycle::CallerManaged,
        "PackTile: L1 accumulation requires OutputLifecycle::L1Accumulation or CallerManaged");
    static_assert(
        Policy != OutputLifecycle::L1Accumulation || L1AccumulationMode != L1Accumulation::Disabled,
        "PackTile: OutputLifecycle::L1Accumulation requires L1 accumulation");
    static_assert(
        DestAccumulationMode == DestAccumulation::Disabled || Policy == OutputLifecycle::DestAccumulation ||
            Policy == OutputLifecycle::CallerManaged,
        "PackTile: DEST accumulation requires OutputLifecycle::DestAccumulation or CallerManaged");
    static_assert(
        Policy != OutputLifecycle::DestAccumulation || DestAccumulationMode == DestAccumulation::Enabled,
        "PackTile: OutputLifecycle::DestAccumulation requires DEST accumulation");
    static_assert(
        L1AccumulationMode == L1Accumulation::Disabled || DestAccumulationMode == DestAccumulation::Disabled,
        "PackTile: L1 and DEST accumulation cannot be combined");
    static_assert(
        L1AccumulationMode == L1Accumulation::Disabled || Relu == PackRelu::Disabled,
        "PackTile: pack ReLU cannot be combined with L1 accumulation");
    // TileBase != None on pack side requires caller-managed-style lifecycle on the
    // output CB (caller pre-reserved a window large enough for base + kind window).
    // InputLifecycle::Streaming / InputLifecycle::Chunked reserve+push counts can't be inflated by a runtime base
    // without per-iter bookkeeping the chain doesn't own.
    static_assert(
        Offset == TileOffset::Unset || is_legal_output_lifecycle_with_base(Policy),
        "PackTile: TileOffset::Set requires InputLifecycle::Bulk-family or OutputLifecycle::CallerManaged lifecycle "
        "(OutputLifecycle::Bulk / OutputLifecycle::ReserveNonePushEnd / OutputLifecycle::CallerManaged)");

    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t pack_dfb_id() { return Cb; }
    static constexpr Dst pack_dst_slot = DstSlot;
    static constexpr bool uses_l1_accumulation = L1AccumulationMode != L1Accumulation::Disabled;
    static constexpr bool seeds_l1_accumulation = L1AccumulationMode == L1Accumulation::SeedFirst;
    static constexpr bool manages_l1_accumulation_lifecycle = Policy == OutputLifecycle::L1Accumulation;
    static constexpr bool uses_dest_accumulation_lifecycle = DestAccumulationMode == DestAccumulation::Enabled;
    static constexpr bool manages_dest_accumulation_lifecycle = Policy == OutputLifecycle::DestAccumulation;
    static constexpr bool uses_pack_relu = Relu != PackRelu::Disabled;
    static constexpr bool is_upfront = (Policy == OutputLifecycle::Bulk);
    static constexpr bool uses_per_block_pack = (Policy == OutputLifecycle::Chunked);
    // `walk` (walk vs pinned output addressing) is derived from OutputLifecycle above.

    // Prev-CB fold: PackTile writes pack-side; mark Cb under reconfig only when
    // the user opted into pack reconfig (Output). Otherwise no pack reconfig is
    // emitted — fold keeps prior pack target.
    // srca/srcb absent -> dfb_for_side defaults them to NO_PREV_DFB; PackTile programs pack only.
    static constexpr uint32_t reconfig_pack_dfb = (Reconfig == DataFormatReconfig::Enabled) ? Cb : NO_PREV_DFB;

    constexpr PackTileImpl() noexcept = default;
    constexpr explicit PackTileImpl(uint32_t base) noexcept : Base(base) {}

    static ALWI void configure_relu() {
        if constexpr (Relu == PackRelu::Zero) {
            pack_relu_config(ReluConfig::zero());
        } else {
            pack_relu_config(ReluConfig::none());
        }
    }

    // Pack exec — walk the reserved output window (base + i_flat) for the upfront-reserve outputs
    // (Bulk / ReserveAllPushPerTile / ReserveAllPushPerChunk), or stay pinned at base for the
    // front-advancing policies (Streaming / Chunked) whose CB front already advanced. TileOffset adds base.
    //
    // OOO gating: the LLK's sequential pack path (out_of_order_output=false) derives its write
    // address from an internal running `fifo_wr_tile_ptr` and IGNORES `out_idx` entirely. That is
    // correct only when the intended index coincides with the sequential counter — i.e. when there
    // is no base offset (walk: 0,1,2,…; pinned: 0). The moment `Offset == Set`, `out_idx` carries a
    // non-coincident base that the sequential path would silently drop (data lands at index 0, not
    // base). L1 accumulation also has to stay pinned to one output tile. Both cases use
    // `pack_tile<true>`, which honors `out_idx`
    // (addr = fifo_wr_ptr + page_size*out_idx - 1) without advancing the internal counter — exactly
    // matching the explicit `base + i_flat` we pass each iteration. Unset keeps the proven
    // sequential path with zero behavior change.
    ALWI void exec(uint32_t i_flat, uint32_t /*ht*/, uint32_t /*wt*/, uint32_t slot_offset) const {
        const uint32_t base = tile_base_value<Offset>(tile_base);
        const uint32_t out_idx = walk ? (base + i_flat) : base;
        pack_tile<
            /*out_of_order_output=*/Offset == TileOffset::Set || L1AccumulationMode != L1Accumulation::Disabled>(
            to_u32(DstSlot) + slot_offset, Cb, out_idx);
    }

    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    // The chain driver emits reserve/push operations from the metadata above and the
    // runtime `tile_base` inherited from OutputStream.
};

// =============================================================================
// 3. BinaryFpu chain element
// =============================================================================

template <uint32_t CbA, uint32_t CbB, uint32_t ConfigBits>
struct detail::BinaryFpuImpl : BinaryFpuTag {
    static constexpr BinaryFpuConfig Config{ConfigBits};
    static constexpr InputSpec AInput = Config.a_input_spec(CbA);
    static constexpr InputSpec BInput = Config.b_input_spec(CbB);
    static constexpr BinaryFpuOp Op = Config.op();
    static constexpr BroadcastDim Bcast = Config.broadcast();
    static constexpr DestAccumulation Accumulation = Config.accumulation();
    static constexpr InputLifecycle APolicy = AInput.lifecycle;
    static constexpr InputLifecycle BPolicy = BInput.lifecycle;
    static constexpr Dst DstSlot = Config.dst();
    static constexpr OperandKind AIndex = AInput.index;
    static constexpr OperandKind BIndex = BInput.index;
    static constexpr TileOffset OffsetA = AInput.offset;
    static constexpr TileOffset OffsetB = BInput.offset;
    static_assert(to_u32(DstSlot) < DEST_AUTO_LIMIT, "BinaryFpu: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(
        Accumulation == DestAccumulation::Disabled || DstSlot == Dst::D0,
        "BinaryFpu: DEST accumulation requires Dst::D0");
    // Comprehensive per-side (IndexMode, Policy) legality. Block rejects PerTile-pop
    // (InputLifecycle::Streaming/InputLifecycle::BulkDrain/InputLifecycle::NoWaitPop — absolute-index pitfall) and
    // PerTile-wait-of-1 (InputLifecycle::HeldStream — never tracks per-iter requirement). Scalar/Row/Col accept every
    // legal lifecycle — caller-sized.
    static_assert(
        is_legal_kind_lifecycle(AIndex, APolicy),
        "BinaryFpu: (AIndex, APolicy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(
        is_legal_kind_lifecycle(BIndex, BPolicy),
        "BinaryFpu: (BIndex, BPolicy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    // same_dfb dedup safety: when CbA == CbB the B-side wait/pop is skipped. Matching
    // indices ensure both sides walk the same shared-CB range; matching lifecycles ensure
    // the retained A-side wait/pop schedule also satisfies B.
    static_assert(
        (CbA != CbB) || AIndex == BIndex,
        "BinaryFpu: when CbA == CbB, AIndex and BIndex must match "
        "(B-side wait/pop is deduped — asymmetric indices would under-wait).");
    static_assert(
        (CbA != CbB) || APolicy == BPolicy,
        "BinaryFpu: when CbA == CbB, AInput and BInput must use the same "
        "InputLifecycle (B-side wait/pop is deduped).");
    // 2D: RowBcast / ColBcast on either side require non-streaming policy.
    static_assert(
        detail::valid_policy_mode_v<APolicy, AIndex>,
        "BinaryFpu: A-side RowBcast / ColBcast index require non-streaming APolicy");
    static_assert(
        detail::valid_policy_mode_v<BPolicy, BIndex>,
        "BinaryFpu: B-side RowBcast / ColBcast index require non-streaming BPolicy");
    // Per-operand TileBase lifecycle compatibility — InputLifecycle::Streaming/InputLifecycle::Chunked/Cumulative
    // can't compose with runtime base offsets (iter-dependent wait/pop counts).
    static_assert(
        OffsetA == TileOffset::Unset || is_legal_input_lifecycle_with_base(APolicy),
        "BinaryFpu: OffsetA Set requires APolicy to be InputLifecycle::Bulk-family or InputLifecycle::CallerManaged");
    static_assert(
        OffsetB == TileOffset::Unset || is_legal_input_lifecycle_with_base(BPolicy),
        "BinaryFpu: OffsetB Set requires BPolicy to be InputLifecycle::Bulk-family or InputLifecycle::CallerManaged");
    // Per-block streaming uses chunk-local CB front. When the two sides use
    // DIFFERENT regimes (one per-block → chunk-local index `j`; the other upfront /
    // caller-managed → absolute index `base_tile + j`), the chain dispatcher
    // resolves them separately via the 3-arg exec / exec overloads gated by
    // `needs_per_side_idx`. Same-regime hits the 2-arg fast path.

    static constexpr uint32_t dfb_a_id() { return CbA; }
    static constexpr uint32_t dfb_b_id() { return CbB; }
    static constexpr InputLifecycle a_policy() { return APolicy; }
    static constexpr InputLifecycle b_policy() { return BPolicy; }
    static constexpr bool is_upfront =
        is_one_of_v<APolicy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined> ||
        is_one_of_v<BPolicy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;
    static constexpr bool same_dfb = (CbA == CbB);
    static constexpr bool uses_dest_accumulation = (Accumulation == DestAccumulation::Enabled);
    static constexpr Dst accumulated_dst_slot = DstSlot;

    // Per-side local-vs-absolute index resolution. When the two operands declare
    // DIFFERENT regimes (A=PerBlock + B=Upfront, or vice versa), the chain calls
    // the 3-arg exec / exec overload and passes both indices; each side picks.
    // Same-regime falls through to the 2-arg forwarder.
    static constexpr bool a_uses_local_idx = (APolicy == InputLifecycle::Chunked);
    static constexpr bool b_uses_local_idx = (BPolicy == InputLifecycle::Chunked);
    static constexpr bool needs_per_side_idx = (a_uses_local_idx != b_uses_local_idx);

    // Prev-CB fold: BinaryFpu touches srca (CbA) and srcb (CbB) only. Pack-side
    // reconfig is owned by the downstream PackTile element.
    // — BinaryFpu writes to DEST, not to a CB, so it has no pack-side responsibility.
    //
    // Per-side selection (Input / SrcA / SrcB) lets the caller opt into a single-side
    // fold when the other side is already programmed (by a previous chain element on
    // the same side, or by external init).
    static constexpr uint32_t reconfig_srca_dfb = (AInput.reconfig == DataFormatReconfig::Enabled) ? CbA : NO_PREV_DFB;
    static constexpr uint32_t reconfig_srcb_dfb = (BInput.reconfig == DataFormatReconfig::Enabled) ? CbB : NO_PREV_DFB;
    // pack side absent -> dfb_for_side defaults to NO_PREV_DFB (downstream PackTile owns pack).

    InputStream a;
    InputStream b;

    constexpr BinaryFpuImpl() noexcept = default;
    constexpr BinaryFpuImpl(uint32_t base_a, uint32_t base_b) noexcept : a(base_a), b(base_b) {}
    constexpr explicit BinaryFpuImpl(uint32_t base_a) noexcept : a(base_a) {}

    // Lifecycle fan-out lives in the chain driver. When same_dfb, it emits one
    // physical wait/pop and uses max(base_a, base_b) for the shared window.

    // ---- init / reconfig ----
    // srca / srcb / pack reconfig are fold-driven (compile-time-elided
    // when prev_*_cb == cur_*_cb). init() programs only the per-op LLK shape.
    static ALWI void init() {
        // Op-specific init.
        if constexpr (Bcast == BroadcastDim::None) {
            constexpr bool acc_to_dest = Accumulation == DestAccumulation::Enabled;
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles_init(CbA, CbB, acc_to_dest);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles_init(CbA, CbB, acc_to_dest);
            } else {
                mul_tiles_init(CbA, CbB, static_cast<uint32_t>(acc_to_dest), __builtin_LINE());
            }
        } else {
            // Use the *_init_short form from bcast.h:352-446 (math init + unpack init only,
            // no hw_configure / pack_dest_init / sync_init — the full init is undefined
            // mid-MAIN). The operand form reads the actual tensor shape from CB metadata via
            // get_operand_tensor_shape, matching `add_bcast_rows_init_short` /
            // `sub_bcast_cols_init_short` etc. exactly.
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            constexpr auto et = (Op == BinaryFpuOp::Add)   ? ckernel::EltwiseBinaryType::ELWADD
                                : (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB
                                                           : ckernel::EltwiseBinaryType::ELWMUL;
            if constexpr (Op == BinaryFpuOp::Mul) {
                MATH((llk_math_eltwise_binary_init<et, bt, MATH_FIDELITY>(
                    CbA, CbB, Accumulation == DestAccumulation::Enabled)));
            } else {
                MATH((llk_math_eltwise_binary_init<et, bt, MathFidelity::LoFi>(
                    CbA, CbB, Accumulation == DestAccumulation::Enabled)));
            }
            UNPACK((llk_unpack_AB_init<bt>(CbA, CbB)));
        }
    }

    // Per-side index mode. AIndex drives a_idx, BIndex drives b_idx. The canonical
    // bcast walk is A=Block (walks the tile range) + B=Scalar (pins the
    // scaler/vector operand at tile 0). OffsetA / OffsetB add a runtime or
    // compile-time base offset to the per-iter index. The 3-arg overload accepts a
    // chunk-local index (`i_local`) and an absolute index (`i_abs`); each side
    // picks via `a_uses_local_idx` / `b_uses_local_idx`. The 2-arg overload is
    // the same-regime fast path and forwards with i_local == i_abs.
    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;

    // 2D variants — per-side index + window. 3-arg form takes both chunk-local
    // (`i_local`) and absolute (`i_abs`) flat indices; each side picks via the
    // per-side traits. `ht` is unchanged (always absolute row); `wt_local` /
    // `wt_abs` cover the per-side column index when needed. Same-regime fast
    // path forwards through the 2-arg overload.
    ALWI void exec(
        uint32_t i_flat_local,
        uint32_t i_flat_abs,
        uint32_t ht,
        uint32_t wt_local,
        uint32_t wt_abs,
        uint32_t slot_offset) const {
        const uint32_t a_flat = a_uses_local_idx ? i_flat_local : i_flat_abs;
        const uint32_t b_flat = b_uses_local_idx ? i_flat_local : i_flat_abs;
        const uint32_t a_wt = a_uses_local_idx ? wt_local : wt_abs;
        const uint32_t b_wt = b_uses_local_idx ? wt_local : wt_abs;
        const uint32_t a_idx = tile_base_value<OffsetA>(a.tile_base) + detail::idx<AIndex>(a_flat, ht, a_wt);
        const uint32_t b_idx = tile_base_value<OffsetB>(b.tile_base) + detail::idx<BIndex>(b_flat, ht, b_wt);
        const uint32_t dst =
            Accumulation == DestAccumulation::Enabled ? to_u32(DstSlot) : to_u32(DstSlot) + slot_offset;
        if constexpr (Bcast == BroadcastDim::None) {
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles(CbA, CbB, a_idx, b_idx, dst);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles(CbA, CbB, a_idx, b_idx, dst);
            } else {
                mul_tiles(CbA, CbB, a_idx, b_idx, dst);
            }
        } else {
            constexpr auto bt = static_cast<ckernel::BroadcastType>(static_cast<uint8_t>(Bcast));
            if constexpr (Op == BinaryFpuOp::Add) {
                add_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            } else if constexpr (Op == BinaryFpuOp::Sub) {
                sub_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            } else {
                mul_tiles_bcast<bt>(CbA, CbB, a_idx, b_idx, dst);
            }
        }
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        exec(i_flat, i_flat, ht, wt, wt, slot_offset);
    }
};

// =============================================================================
// 5. DestReuseBinary chain element
// =============================================================================

template <uint32_t Cb, uint32_t ConfigBits>
struct detail::DestReuseBinaryImpl : InputStream, DestReuseBinaryTag {
    static constexpr DestReuseBinaryConfig Config{ConfigBits};
    static constexpr BinaryFpuOp Op = Config.op();
    static constexpr DestReuseType ReuseType = Config.reuse();
    static constexpr InputSpec InputConfig = Config.input_spec(Cb);
    static constexpr Dst DstIn = Config.dst_in();
    static constexpr Dst DstOut = Config.dst_out();
    static constexpr InputLifecycle Policy = InputConfig.lifecycle;
    static constexpr OperandKind IndexMode = InputConfig.index;
    static constexpr TileOffset Offset = InputConfig.offset;
    using Base = InputStream;
    using Base::tile_base;

    static_assert(
        to_u32(DstIn) < DEST_AUTO_LIMIT && to_u32(DstOut) < DEST_AUTO_LIMIT,
        "DestReuseBinary: DEST slot exceeds DEST_AUTO_LIMIT");
    static_assert(
        is_legal_kind_lifecycle(IndexMode, Policy),
        "DestReuseBinary: (IndexMode, Policy) is illegal for Block — exclude "
        "InputLifecycle::Streaming / InputLifecycle::HeldStream / InputLifecycle::BulkDrain / "
        "InputLifecycle::NoWaitPop on Block walkers.");
    static_assert(
        detail::valid_policy_mode_v<Policy, IndexMode>,
        "DestReuseBinary: RowBcast / ColBcast index require non-streaming policy");
    static_assert(
        Offset == TileOffset::Unset || is_legal_input_lifecycle_with_base(Policy),
        "DestReuseBinary: TileOffset::Set requires InputLifecycle::Bulk-family or InputLifecycle::CallerManaged "
        "lifecycle");

    // The one CB feeds the src that DEST is NOT routed to: DEST_TO_SRCB -> CB on srcA (dfb_a),
    // DEST_TO_SRCA -> CB on srcB (dfb_b). The other side is the DEST register, not a CB (INVALID_DFB).
    static constexpr uint32_t dfb = Cb;
    static constexpr uint32_t dfb_a_id() { return (ReuseType == DestReuseType::DEST_TO_SRCB) ? Cb : INVALID_DFB; }
    static constexpr uint32_t dfb_b_id() { return (ReuseType == DestReuseType::DEST_TO_SRCA) ? Cb : INVALID_DFB; }
    static constexpr InputLifecycle a_policy() { return Policy; }
    static constexpr bool is_upfront =
        is_one_of_v<Policy, InputLifecycle::Bulk, InputLifecycle::HeldBulk, InputLifecycle::Pipelined>;

    // Prev-CB fold: DestReuseBinary loads CB into srca (when DEST → srcb) or srcb
    // (when DEST → srca). Reconfig only fires when opted in.
    //
    static constexpr uint32_t reconfig_srca_dfb =
        (InputConfig.reconfig == DataFormatReconfig::Enabled && ReuseType == DestReuseType::DEST_TO_SRCB) ? Cb
                                                                                                          : NO_PREV_DFB;
    static constexpr uint32_t reconfig_srcb_dfb =
        (InputConfig.reconfig == DataFormatReconfig::Enabled && ReuseType == DestReuseType::DEST_TO_SRCA) ? Cb
                                                                                                          : NO_PREV_DFB;
    // pack side absent -> dfb_for_side defaults to NO_PREV_DFB.

    constexpr DestReuseBinaryImpl() noexcept = default;
    constexpr explicit DestReuseBinaryImpl(uint32_t base) noexcept : Base(base) {}

    // srca / srcb reconfig is fold-driven; init() programs only the per-op
    // LLK shape.
    static ALWI void init() {
        constexpr auto et = (Op == BinaryFpuOp::Add)   ? ckernel::EltwiseBinaryType::ELWADD
                            : (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB
                                                       : ckernel::EltwiseBinaryType::ELWMUL;
        constexpr auto reuse = (ReuseType == DestReuseType::DEST_TO_SRCA)
                                   ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                   : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
        binary_dest_reuse_tiles_init<et, reuse>(Cb);
    }

    ALWI void exec(uint32_t i_flat, uint32_t ht, uint32_t wt, uint32_t slot_offset) const {
        constexpr auto et = (Op == BinaryFpuOp::Add)   ? ckernel::EltwiseBinaryType::ELWADD
                            : (Op == BinaryFpuOp::Sub) ? ckernel::EltwiseBinaryType::ELWSUB
                                                       : ckernel::EltwiseBinaryType::ELWMUL;
        constexpr auto reuse = (ReuseType == DestReuseType::DEST_TO_SRCA)
                                   ? ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                   : ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB;
        const uint32_t in_idx = tile_base_value<Offset>(tile_base) + detail::idx<IndexMode>(i_flat, ht, wt);
        binary_dest_reuse_tiles<et, reuse>(Cb, in_idx, to_u32(DstIn) + slot_offset);
    }

    static constexpr uint32_t lane_width =
        (to_u32(DstIn) > to_u32(DstOut)) ? (to_u32(DstIn) + 1) : (to_u32(DstOut) + 1);

    // The chain driver emits wait/pop operations from the metadata above and the
    // runtime `tile_base` inherited from InputStream.
};

// Fill and random elements live in their feature headers so their LLK dependencies
// are absent from the base chain include cone.

// =============================================================================
// 8. EltwiseChain typed list — defined here, declared in .hpp
// =============================================================================

template <class... Es>
struct EltwiseChain {
    static constexpr size_t size = sizeof...(Es);
};

// =============================================================================
// 9. Chain-shape trait predicates
// =============================================================================

// chain_lane_width — N-element fold. Max of per-element `lane_width`. The chain clamps block_size
// at runtime so `block_size * chain_lane_width <= DEST_AUTO_LIMIT` (block_size is a runtime
// EltwiseShape field, so this is a clamp, not a static_assert). Each element writes to
// DEST[dst_slot + j * chain_lane_width] for lane j in [0, block_size).
//
// SFINAE fallback: elements that don't expose a `lane_width` member (caller-defined
// chain elements that inherit directly from `CopyTileTag` / `PackTileTag` / `DestOnlyTag`
// without inheriting from `UnaryOp`/`BinaryOp`/`TernaryOp` bases) default
// to 1. Member ambiguity in caller-defined multiple-inheritance elements is sidestepped
// by reading via this detector instead of `Es::lane_width` directly.
namespace detail {
template <class, class = void>
struct elem_lane_width : std::integral_constant<uint32_t, 1> {};

template <class E>
struct elem_lane_width<E, std::void_t<decltype(E::lane_width)>> : std::integral_constant<uint32_t, E::lane_width> {};

template <class E>
constexpr uint32_t elem_lane_width_v = elem_lane_width<E>::value;
}  // namespace detail

template <class Chain>
struct chain_lane_width;

template <class... Es>
struct chain_lane_width<EltwiseChain<Es...>>
    : std::integral_constant<uint32_t, detail::ChainTraits<Es...>::lane_width> {};

template <class Chain>
inline constexpr uint32_t chain_lane_width_v = chain_lane_width<Chain>::value;

template <class Chain>
struct chain_transient_lane_width;

template <class... Es>
struct chain_transient_lane_width<EltwiseChain<Es...>>
    : std::integral_constant<uint32_t, detail::ChainTraits<Es...>::transient_lane_width> {};

template <class Chain>
inline constexpr uint32_t chain_transient_lane_width_v = chain_transient_lane_width<Chain>::value;

// chain_max_block_v — largest block_size that fits in DEST for this chain, given its
// lane-width fold. Caller-facing compile-time constant: pass any value <= this to
// the runtime `block_size` arg on `eltwise_chain`. Caller can `static_assert` their
// chosen block against this value to recover the build-time DEST overflow signal.
template <class Chain>
struct chain_max_block;

namespace detail {
template <class... Es>
constexpr uint32_t chain_max_block_value() {
    using Traits = ChainTraits<Es...>;
    if constexpr (Traits::any_dest_accumulation) {
        if constexpr (Traits::transient_lane_width == 0) {
            return ~uint32_t{0};
        } else {
            return (DEST_AUTO_LIMIT - 1) / Traits::transient_lane_width;
        }
    } else {
        return DEST_AUTO_LIMIT / Traits::lane_width;
    }
}
}  // namespace detail

template <class... Es>
struct chain_max_block<EltwiseChain<Es...>> : std::integral_constant<uint32_t, detail::chain_max_block_value<Es...>()> {
};

template <class Chain>
inline constexpr uint32_t chain_max_block_v = chain_max_block<Chain>::value;

// chain_supports_block — N-element fold. True when every CB-reader element uses a policy that
// stages a multi-tile window per outer iter (an upfront Bulk-family policy or per-chunk Chunked).
// Per-tile policies (Streaming / HeldStream) consume ONE tile per iter and can't do block_size > 1;
// for such chains the chain clamps block_size to 1 at runtime (not a static_assert).
namespace detail {
template <class E>
constexpr InputLifecycle b_policy_of();  // defined below (defaults to CallerManaged)

constexpr bool policy_supports_block(InputLifecycle p) {
    return p == InputLifecycle::Bulk || p == InputLifecycle::HeldBulk || p == InputLifecycle::Pipelined ||
           p == InputLifecycle::HeldCumulative || p == InputLifecycle::CallerManaged || p == InputLifecycle::Chunked;
}

template <class E>
constexpr bool element_supports_block() {
    if constexpr (is_cb_reader_op_v<E>) {
        return policy_supports_block(E::a_policy()) && policy_supports_block(b_policy_of<E>());
    } else {
        return true;  // non-CB-reader elements don't constrain block_size
    }
}

}  // namespace detail

// =============================================================================
// ChainTraits — reflect each element once into an ElemDesc record, then derive every
// value-based chain property as a field. Type-uniformity (chain_*_uniform) stays
// separate — it reads element types, not values. Emission stays separate too.
// All compile-time: the whole struct folds to constants (zero runtime cost).
// =============================================================================
namespace detail {

// SFINAE accessors for the two members the collision derivations read but not every
// element declares (CB readers carry is_upfront; PackTile carries pack_dst_slot).
template <class E, class = void>
struct has_is_upfront_m : std::false_type {};
template <class E>
struct has_is_upfront_m<E, std::void_t<decltype(E::is_upfront)>> : std::true_type {};
template <class E>
constexpr bool is_upfront_of() {
    if constexpr (has_is_upfront_m<E>::value) {
        return E::is_upfront;
    } else {
        return false;
    }
}
template <class E, class = void>
struct has_pack_dst_slot_m : std::false_type {};
template <class E>
struct has_pack_dst_slot_m<E, std::void_t<decltype(E::pack_dst_slot)>> : std::true_type {};
template <class E>
constexpr Dst pack_dst_slot_of() {
    if constexpr (has_pack_dst_slot_m<E>::value) {
        return E::pack_dst_slot;
    } else {
        return Dst::D0;
    }
}
// b_policy defaults to CallerManaged (the unary-reader / no-srcB-CB case), so only elements
// with a genuine srcB operand (BinaryFpu) declare it.
template <class E, class = void>
struct has_b_policy_m : std::false_type {};
template <class E>
struct has_b_policy_m<E, std::void_t<decltype(E::b_policy())>> : std::true_type {};
template <class E>
constexpr InputLifecycle b_policy_of() {
    if constexpr (has_b_policy_m<E>::value) {
        return E::b_policy();
    } else {
        return InputLifecycle::CallerManaged;
    }
}

// One plain-data descriptor per element — reflected once via the existing accessors.
struct ElemDesc {
    bool is_cb_reader;
    bool is_pack;
    uint32_t srca_cb;        // dfb_for_side<SrcA> (NO_PREV_DFB when not programmed) — per-side consistency + prev input
    uint32_t srcb_cb;        // dfb_for_side<SrcB>
    uint32_t pack_side_dfb;  // dfb_for_side<Pack> (reconfig_pack_dfb) — prev / last_pack / hetero input
    uint32_t dfb_a;          // dfb_a_of (INVALID_DFB when n/a) — reader-collision input
    uint32_t dfb_b;          // dfb_b_of (INVALID_DFB when n/a)
    uint32_t pack_dfb;       // pack_dfb_of (INVALID_DFB when n/a) — writer-collision input
    Dst pack_dst_slot;
    bool uses_l1_accumulation;
    bool seeds_l1_accumulation;
    bool manages_l1_accumulation_lifecycle;
    bool uses_dest_accumulation;
    Dst accumulated_dst_slot;
    bool uses_dest_accumulation_lifecycle;
    bool manages_dest_accumulation_lifecycle;
    bool uses_pack_relu;
    bool is_upfront;
    uint32_t lane_width;
    uint32_t transient_lane_width;
    bool supports_block;
    uint32_t outer_input_a_cb;
    uint32_t outer_input_b_cb;
    bool wait_a_per_outer;
    bool wait_b_per_outer;
    bool pop_a_per_outer;
    bool pop_b_per_outer;
    bool reserve_per_outer;
    bool push_per_outer;
};

template <class E>
constexpr bool uses_l1_accumulation_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::uses_l1_accumulation;
    }
    return false;
}
template <class E>
constexpr bool seeds_l1_accumulation_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::seeds_l1_accumulation;
    }
    return false;
}
template <class E>
constexpr bool manages_l1_accumulation_lifecycle_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::manages_l1_accumulation_lifecycle;
    }
    return false;
}
template <class E>
constexpr bool uses_dest_accumulation_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::uses_dest_accumulation;
    }
    return false;
}
template <class E>
constexpr Dst accumulated_dst_slot_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::accumulated_dst_slot;
    }
    return Dst::D0;
}
template <class E>
constexpr bool uses_dest_accumulation_lifecycle_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::uses_dest_accumulation_lifecycle;
    }
    return false;
}
template <class E>
constexpr bool manages_dest_accumulation_lifecycle_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::manages_dest_accumulation_lifecycle;
    }
    return false;
}
template <class E>
constexpr bool uses_pack_relu_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::uses_pack_relu;
    }
    return false;
}
template <class E>
constexpr uint32_t transient_lane_width_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::uses_dest_accumulation ? 0u : elem_lane_width_v<E>;
    } else if constexpr (is_pack_tile_op_v<E>) {
        return 0u;
    }
    return elem_lane_width_v<E>;
}

template <class E>
constexpr uint32_t outer_input_a_cb_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::dfb_a_id();
    } else if constexpr (is_cb_reader_op_v<E>) {
        return E::dfb;
    } else {
        return INVALID_DFB;
    }
}

template <class E>
constexpr uint32_t outer_input_b_cb_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        if constexpr (!E::same_dfb) {
            return E::dfb_b_id();
        }
    }
    return INVALID_DFB;
}

template <class E>
constexpr bool wait_a_per_outer_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::APolicy.wait_policy == WaitPolicy::PerOuter;
    } else if constexpr (is_cb_reader_op_v<E>) {
        return E::Policy.wait_policy == WaitPolicy::PerOuter;
    } else {
        return false;
    }
}

template <class E>
constexpr bool wait_b_per_outer_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        if constexpr (!E::same_dfb) {
            return E::BPolicy.wait_policy == WaitPolicy::PerOuter;
        }
    }
    return false;
}

template <class E>
constexpr bool pop_a_per_outer_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::APolicy.pop_policy == PopPolicy::PerOuter;
    } else if constexpr (is_cb_reader_op_v<E>) {
        return E::Policy.pop_policy == PopPolicy::PerOuter;
    } else {
        return false;
    }
}

template <class E>
constexpr bool pop_b_per_outer_of() {
    if constexpr (is_binary_fpu_op_v<E>) {
        if constexpr (!E::same_dfb) {
            return E::BPolicy.pop_policy == PopPolicy::PerOuter;
        }
    }
    return false;
}

template <class E>
constexpr bool reserve_per_outer_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::Policy.reserve_policy == ReservePolicy::PerOuter;
    }
    return false;
}

template <class E>
constexpr bool push_per_outer_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        return E::Policy.push_policy == PushPolicy::PerOuter;
    }
    return false;
}

template <class E>
constexpr ElemDesc describe() {
    return ElemDesc{
        is_cb_reader_op_v<E>,
        is_pack_tile_op_v<E>,
        dfb_for_side<Side::SrcA, E>(),
        dfb_for_side<Side::SrcB, E>(),
        dfb_for_side<Side::Pack, E>(),
        dfb_a_of<E>(),
        dfb_b_of<E>(),
        pack_dfb_of<E>(),
        pack_dst_slot_of<E>(),
        uses_l1_accumulation_of<E>(),
        seeds_l1_accumulation_of<E>(),
        manages_l1_accumulation_lifecycle_of<E>(),
        uses_dest_accumulation_of<E>(),
        accumulated_dst_slot_of<E>(),
        uses_dest_accumulation_lifecycle_of<E>(),
        manages_dest_accumulation_lifecycle_of<E>(),
        uses_pack_relu_of<E>(),
        is_upfront_of<E>(),
        elem_lane_width_v<E>,
        transient_lane_width_of<E>(),
        element_supports_block<E>(),
        outer_input_a_cb_of<E>(),
        outer_input_b_cb_of<E>(),
        wait_a_per_outer_of<E>(),
        wait_b_per_outer_of<E>(),
        pop_a_per_outer_of<E>(),
        pop_b_per_outer_of<E>(),
        reserve_per_outer_of<E>(),
        push_per_outer_of<E>(),
    };
}

// Derivations over the descriptor array — flat loops bounded by the real element count
// `n` (the array is sized [N?N:1], so an empty chain must NOT read the lone default slot).
constexpr uint32_t ct_lane_width(const ElemDesc* d, int n) {
    uint32_t w = 1;
    for (int i = 0; i < n; ++i) {
        if (d[i].lane_width > w) {
            w = d[i].lane_width;
        }
    }
    return w;
}
constexpr uint32_t ct_transient_lane_width(const ElemDesc* d, int n) {
    uint32_t w = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].transient_lane_width > w) {
            w = d[i].transient_lane_width;
        }
    }
    return w;
}
constexpr bool ct_supports_block(const ElemDesc* d, int n) {
    bool r = true;
    for (int i = 0; i < n; ++i) {
        r = r && d[i].supports_block;
    }
    return r;
}
constexpr bool ct_side_consistent(const ElemDesc* d, int n, uint32_t ElemDesc::* side) {
    uint32_t seen = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        uint32_t dfb = d[i].*side;
        if (dfb == NO_PREV_DFB) {
            continue;
        }
        if (seen == NO_PREV_DFB) {
            seen = dfb;
        } else if (seen != dfb) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_reader_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (!(d[i].is_cb_reader && d[j].is_cb_reader)) {
                continue;
            }
            if (!(d[i].is_upfront && d[j].is_upfront)) {
                continue;
            }
            uint32_t a0 = d[i].dfb_a, a1 = d[i].dfb_b, b0 = d[j].dfb_a, b1 = d[j].dfb_b;
            if ((a0 != INVALID_DFB && (a0 == b0 || a0 == b1)) || (a1 != INVALID_DFB && (a1 == b0 || a1 == b1))) {
                return true;
            }
        }
    }
    return false;
}
constexpr bool ct_writer_collide(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (d[i].is_pack && d[j].is_pack && d[i].pack_dfb == d[j].pack_dfb &&
                d[i].pack_dst_slot == d[j].pack_dst_slot) {
                return true;
            }
        }
    }
    return false;
}
constexpr bool ct_any_l1_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_l1_accumulation) {
            return true;
        }
    }
    return false;
}
constexpr bool ct_all_writers_l1_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && !d[i].uses_l1_accumulation) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_l1_accumulation_modes_consistent(const ElemDesc* d, int n) {
    bool found = false;
    bool seed_first = false;
    for (int i = 0; i < n; ++i) {
        if (!d[i].is_pack || !d[i].uses_l1_accumulation) {
            continue;
        }
        if (!found) {
            found = true;
            seed_first = d[i].seeds_l1_accumulation;
        } else if (seed_first != d[i].seeds_l1_accumulation) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_any_seed_first_l1_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].seeds_l1_accumulation) {
            return true;
        }
    }
    return false;
}
constexpr bool ct_pack_dfbs_consistent(const ElemDesc* d, int n) {
    uint32_t seen = INVALID_DFB;
    for (int i = 0; i < n; ++i) {
        if (!d[i].is_pack) {
            continue;
        }
        if (seen == INVALID_DFB) {
            seen = d[i].pack_dfb;
        } else if (seen != d[i].pack_dfb) {
            return false;
        }
    }
    return true;
}
constexpr uint32_t ct_managed_l1_accumulation_lifecycles(const ElemDesc* d, int n) {
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].manages_l1_accumulation_lifecycle) {
            ++count;
        }
    }
    return count;
}
constexpr bool ct_any_dest_accumulation(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_dest_accumulation) {
            return true;
        }
    }
    return false;
}
constexpr uint32_t ct_dest_accumulation_slot_count(const ElemDesc* d, int n) {
    bool seen[DEST_AUTO_LIMIT] = {};
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (!d[i].uses_dest_accumulation) {
            continue;
        }
        const uint32_t slot = to_u32(d[i].accumulated_dst_slot);
        if (!seen[slot]) {
            seen[slot] = true;
            ++count;
        }
    }
    return count;
}
constexpr uint32_t ct_pack_writer_count(const ElemDesc* d, int n) {
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack) {
            ++count;
        }
    }
    return count;
}
constexpr bool ct_dest_accumulation_pack_matches(const ElemDesc* d, int n) {
    Dst sticky = Dst::D0;
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_dest_accumulation) {
            sticky = d[i].accumulated_dst_slot;
            break;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && d[i].pack_dst_slot != sticky) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_all_writers_dest_accumulation_lifecycle(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].is_pack && !d[i].uses_dest_accumulation_lifecycle) {
            return false;
        }
    }
    return true;
}
constexpr bool ct_any_dest_accumulation_lifecycle(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_dest_accumulation_lifecycle) {
            return true;
        }
    }
    return false;
}
constexpr uint32_t ct_managed_dest_accumulation_lifecycles(const ElemDesc* d, int n) {
    uint32_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (d[i].manages_dest_accumulation_lifecycle) {
            ++count;
        }
    }
    return count;
}
constexpr bool ct_any_pack_relu(const ElemDesc* d, int n) {
    for (int i = 0; i < n; ++i) {
        if (d[i].uses_pack_relu) {
            return true;
        }
    }
    return false;
}

// Per-side "previous programmed CB at each index" tables, built in ONE forward sweep:
// carry a running prev per side, record it BEFORE each element, update it when the
// element programs that side. prev.srca[I] is the most recent CB programmed on SrcA by
// any element before index I (NO_PREV_DFB if none).
template <int M>
struct PrevTable {
    uint32_t srca[M];
    uint32_t srcb[M];
    uint32_t pack[M];
};
template <int M>
constexpr PrevTable<M> ct_build_prev(const ElemDesc* d, int n) {
    PrevTable<M> t{};
    uint32_t pa = NO_PREV_DFB, pb = NO_PREV_DFB, pp = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        t.srca[i] = pa;
        t.srcb[i] = pb;
        t.pack[i] = pp;
        if (d[i].srca_cb != NO_PREV_DFB) {
            pa = d[i].srca_cb;
        }
        if (d[i].srcb_cb != NO_PREV_DFB) {
            pb = d[i].srcb_cb;
        }
        if (d[i].pack_side_dfb != NO_PREV_DFB) {
            pp = d[i].pack_side_dfb;
        }
    }
    return t;
}
// Last opt-in pack CB in chain order (iter-to-iter wraparound prev for pack site 0).
constexpr uint32_t ct_last_pack_cb(const ElemDesc* d, int n) {
    uint32_t last = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        if (d[i].pack_side_dfb != NO_PREV_DFB) {
            last = d[i].pack_side_dfb;
        }
    }
    return last;
}
// True iff ≥2 opt-in pack sites declare different CBs (boot can't program all).
constexpr bool ct_pack_hetero(const ElemDesc* d, int n) {
    uint32_t first = NO_PREV_DFB;
    for (int i = 0; i < n; ++i) {
        if (d[i].pack_side_dfb == NO_PREV_DFB) {
            continue;
        }
        if (first == NO_PREV_DFB) {
            first = d[i].pack_side_dfb;
        } else if (first != d[i].pack_side_dfb) {
            return true;
        }
    }
    return false;
}

template <class... Es>
struct ChainTraits {
    static constexpr int N = int(sizeof...(Es));
    static constexpr ElemDesc d[N ? N : 1] = {describe<Es>()...};  // the one walk

    static constexpr uint32_t lane_width = ct_lane_width(d, N);
    static constexpr uint32_t transient_lane_width = ct_transient_lane_width(d, N);
    static constexpr bool supports_block = ct_supports_block(d, N);
    static constexpr bool srca_consistent = ct_side_consistent(d, N, &ElemDesc::srca_cb);
    static constexpr bool srcb_consistent = ct_side_consistent(d, N, &ElemDesc::srcb_cb);
    static constexpr bool reader_collide = ct_reader_collide(d, N);
    static constexpr bool writer_collide = ct_writer_collide(d, N);
    static constexpr bool any_l1_accumulation = ct_any_l1_accumulation(d, N);
    static constexpr bool any_seed_first_l1_accumulation = ct_any_seed_first_l1_accumulation(d, N);
    static constexpr bool all_writers_l1_accumulation = ct_all_writers_l1_accumulation(d, N);
    static constexpr bool l1_accumulation_modes_consistent = ct_l1_accumulation_modes_consistent(d, N);
    static constexpr bool pack_dfbs_consistent = ct_pack_dfbs_consistent(d, N);
    static constexpr uint32_t managed_l1_accumulation_lifecycles = ct_managed_l1_accumulation_lifecycles(d, N);
    static constexpr bool any_dest_accumulation = ct_any_dest_accumulation(d, N);
    static constexpr uint32_t dest_accumulation_slot_count = ct_dest_accumulation_slot_count(d, N);
    static constexpr uint32_t pack_writer_count = ct_pack_writer_count(d, N);
    static constexpr bool dest_accumulation_pack_matches = ct_dest_accumulation_pack_matches(d, N);
    static constexpr bool all_writers_dest_accumulation_lifecycle = ct_all_writers_dest_accumulation_lifecycle(d, N);
    static constexpr bool any_dest_accumulation_lifecycle = ct_any_dest_accumulation_lifecycle(d, N);
    static constexpr uint32_t managed_dest_accumulation_lifecycles = ct_managed_dest_accumulation_lifecycles(d, N);
    static constexpr bool any_pack_relu = ct_any_pack_relu(d, N);

    // Per-side prev-CB history (one sweep), + pack-side metadata.
    static constexpr PrevTable<N ? N : 1> prev = ct_build_prev<N ? N : 1>(d, N);
    static constexpr uint32_t last_pack_cb = ct_last_pack_cb(d, N);
    static constexpr bool pack_hetero = ct_pack_hetero(d, N);
};

}  // namespace detail

template <class Chain>
struct chain_supports_block;

template <class... Es>
struct chain_supports_block<EltwiseChain<Es...>> : std::bool_constant<detail::ChainTraits<Es...>::supports_block> {};

template <class Chain>
inline constexpr bool chain_supports_block_v = chain_supports_block<Chain>::value;

// element_uses_per_block_index_v — true when the element's CB front is chunk-
// local (per-block streaming reader or pack), so the pipeline passes `j`
// instead of `base_tile + j` as the tile index seen by `exec`. SFINAE on each
// element kind. False for elements that don't read/write CBs (DEST-only ops).
namespace detail {

template <class E, class = void>
struct elem_per_block_reader : std::false_type {};

template <class E>
struct elem_per_block_reader<E, std::enable_if_t<is_cb_reader_op_v<E>>>
    : std::bool_constant<(E::a_policy() == InputLifecycle::Chunked) || (b_policy_of<E>() == InputLifecycle::Chunked)> {
};

template <class E, class = void>
struct elem_per_block_pack : std::false_type {};

template <class E>
struct elem_per_block_pack<E, std::void_t<decltype(E::uses_per_block_pack)>>
    : std::bool_constant<E::uses_per_block_pack> {};

template <class E>
inline constexpr bool element_uses_per_block_index_v = elem_per_block_reader<E>::value || elem_per_block_pack<E>::value;

// elem_needs_per_side_idx_v — true when the element wants per-side
// local-vs-absolute index resolution (BinaryFpu when A/B policies disagree on
// the per-block regime). SFINAE-tolerant: absence of `needs_per_side_idx`
// static member defaults to false.
template <class E, class = void>
struct elem_needs_per_side_idx : std::false_type {};

template <class E>
struct elem_needs_per_side_idx<E, std::void_t<decltype(E::needs_per_side_idx)>>
    : std::bool_constant<E::needs_per_side_idx> {};

template <class E>
inline constexpr bool elem_needs_per_side_idx_v = elem_needs_per_side_idx<E>::value;

}  // namespace detail

// chain_has_duplicate_upfront_dfbs / chain_pack_writes_collide — pairwise collision
// checks, implemented as flat nested loops in ChainTraits (reader_collide / writer_collide).
template <class... Es>
struct chain_has_duplicate_upfront_dfbs<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::reader_collide> {};

template <class... Es>
struct chain_pack_writes_collide<EltwiseChain<Es...>> : std::bool_constant<detail::ChainTraits<Es...>::writer_collide> {
};

// `chain_hoist_math_mop` / `chain_hoist_sfpu` — per-cohort decisions on whether each
// element's init() can run once at boot instead of per tile. The two cohorts (math-MOP =
// ADDR_MOD_0..3 + MATH MOP; SFPU = ADDR_MOD_7 + SFPU CSR) are hardware-disjoint, so the
// decisions are independent in principle; we constrain hoist_sfpu to imply hoist_math_mop
// to keep the dispatcher simple. Eltwise-only scope (no matmul/reduce hazards).
//
// All three conditions guard the same failure mode — boot programs each lane once, so a
// non-uniform chain leaves only the LAST init's state programmed and earlier elements run
// with the wrong MOP/format:
//   - Per-side CB consistency: every element programming a side (srcA/srcB) uses the same CB.
//   - MATH-MOP uniformity: all `is_math_mop_op_v` elements (CopyTile / BinaryFpu /
//     DestReuseBinary / UnaryBcast) are the same instantiated type.
//   - SFPU-init uniqueness: all `is_sfpu_op_v` elements are the same type. (Caught
//     mish FP32 Exp/Log1p/Tanh → tanh saturation PCC 0.988; logit stage-2
//     Rsub/DivBinary/Log → ATOL 9-12.)
//
// "Identical" is `std::is_same_v` — rejects some safe chains (two `Exp<...>` differing only
// in slot), but false negatives only cost a per-tile init, never correctness.

namespace detail {

// Per-side CB consistency is ChainTraits::srca_consistent / srcb_consistent.

// Trait wrappers (Pred<E>::value form) — `is_sfpu_op_v` / `is_math_mop_op_v`
// are `inline constexpr bool` variable templates; wrap them so they fit the
// `Pred<E>::value` interface used by the uniformity fold.
template <class E>
struct is_sfpu_op_t : std::bool_constant<is_sfpu_op_v<E>> {};
template <class E>
struct is_math_mop_op_t : std::bool_constant<is_math_mop_op_v<E>> {};

// Uniformity helper (used for both the MATH-MOP and SFPU cohorts): across all
// `Es...` satisfying `Pred<E>`, require every instantiated type to be
// `std::is_same_v` with every other (≤ 1 distinct).
//
// Strategy: find the first `E` in `Es...` with `Pred<E>::value == true`;
// call it `Rep`. Then every other `E` with `Pred<E>::value == true` must
// satisfy `std::is_same_v<Rep, E>`. If no element matches, the chain is
// vacuously uniform (returns true).
template <template <class> class Pred, class... Es>
struct first_match {
    using type = void;
};

template <template <class> class Pred, class First, class... Rest>
struct first_match<Pred, First, Rest...> {
    using type = std::conditional_t<Pred<First>::value, First, typename first_match<Pred, Rest...>::type>;
};

template <template <class> class Pred, class Rep, class... Es>
struct all_match_rep : std::bool_constant<(((!Pred<Es>::value) || std::is_same_v<Rep, Es>) && ...)> {};

// Empty `Rep == void` short-circuits to `true` (no element matched Pred at all,
// so the "must equal Rep" condition is vacuously satisfied).
template <template <class> class Pred, class... Es>
constexpr bool chain_all_pred_uniform_v = []() {
    using Rep = typename first_match<Pred, Es...>::type;
    if constexpr (std::is_same_v<Rep, void>) {
        return true;
    } else {
        return all_match_rep<Pred, Rep, Es...>::value;
    }
}();

}  // namespace detail

template <class Chain>
struct chain_per_side_dfbs_consistent : std::true_type {};

template <class... Es>
struct chain_per_side_dfbs_consistent<EltwiseChain<Es...>>
    : std::bool_constant<detail::ChainTraits<Es...>::srca_consistent && detail::ChainTraits<Es...>::srcb_consistent> {};

template <class Chain>
struct chain_math_mop_uniform : std::true_type {};

template <class... Es>
struct chain_math_mop_uniform<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_all_pred_uniform_v<detail::is_math_mop_op_t, Es...>> {};

template <class Chain>
struct chain_sfpu_inits_uniform : std::true_type {};

template <class... Es>
struct chain_sfpu_inits_uniform<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_all_pred_uniform_v<detail::is_sfpu_op_t, Es...>> {};

// Math-MOP cohort hoist: per-side CB consistency + math-MOP uniformity.
// True when CopyTile / BinaryFpu / DestReuseBinary / UnaryBcast inits can be
// emitted once at boot instead of per tile. Independent of SFPU uniformity.
template <class Chain>
struct chain_hoist_math_mop : std::false_type {};

template <class... Es>
struct chain_hoist_math_mop<EltwiseChain<Es...>>
    : std::bool_constant<
          ((!is_runtime_conditional_sequence_op_v<Es>) && ...) &&
          chain_per_side_cbs_consistent_v<EltwiseChain<Es...>> && chain_math_mop_uniform_v<EltwiseChain<Es...>>> {};

// SFPU cohort hoist: requires math-MOP hoist AND SFPU init uniformity.
// True when every element's init can be emitted at boot (the fully-hoisted shape).
template <class Chain>
struct chain_hoist_sfpu : std::false_type {};

template <class... Es>
struct chain_hoist_sfpu<EltwiseChain<Es...>>
    : std::bool_constant<
          chain_hoist_math_mop_v<EltwiseChain<Es...>> && chain_sfpu_inits_uniform_v<EltwiseChain<Es...>>> {};

// Pack cohort hoist: pack init + reconfig are fully boot-emitted, with NO per-stage pack
// reconfig in the loop. True iff the chain isn't pack-hetero (homogeneous opt-in pack CBs).
// The output-reconfig leg of "all one-time setup is boot-hoistable" (with math-MOP + SFPU).
template <class Chain>
struct chain_hoist_pack : std::true_type {};

template <class... Es>
struct chain_hoist_pack<EltwiseChain<Es...>> : std::bool_constant<!detail::ChainTraits<Es...>::pack_hetero> {};

// True iff no element requests any reconfig. SetupOwner::Caller requires this so an enabled but
// inert operand reconfig (which the helper would silently ignore) is a compile error
// instead of a lie about what runs inside the chain.
template <class Chain>
struct chain_no_reconfig_requested : std::true_type {};

template <class... Es>
struct chain_no_reconfig_requested<EltwiseChain<Es...>>
    : std::bool_constant<detail::chain_requests_no_reconfig<Es...>()> {};

template <class Chain>
inline constexpr bool chain_no_reconfig_requested_v = chain_no_reconfig_requested<Chain>::value;

// =============================================================================
// 10. Chain pipeline — per-iteration emit
// =============================================================================

namespace detail {

// CB lifecycles are emitted in the existing compute/pack workers and outer adapters below.
// Their policy gates remain `if constexpr`, so unused operations disappear without creating a
// separate lifecycle-method specialization for every element configuration.

// A direct fold still names every chain element at its call site. Do not make that imply that
// every phase worker is instantiated for every element: select the elements that can contribute
// to a phase before entering its worker. The shared empty sentinel carries neither the rejected
// element type nor the chain pack. Selected references carry one element plus only the small set
// of per-position facts consumed by that phase (never `Es...`).
struct UnselectedElement {};

template <class E, class Facts = void>
struct SelectedElement {
    const E& value;
};

template <bool Select, class Facts = void, class E>
ALWI constexpr auto select_element(const E& elem) {
    if constexpr (Select) {
        return SelectedElement<E, Facts>{elem};
    } else {
        return UnselectedElement{};
    }
}

template <class E>
inline constexpr bool participates_in_compute_v = is_cb_reader_op_v<E> || is_dest_only_op_v<E>;

template <class E>
constexpr bool waits_upfront() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::APolicy == InputLifecycle::BulkDrain || E::APolicy.wait_policy == WaitPolicy::Upfront ||
               (!E::same_dfb &&
                (E::BPolicy == InputLifecycle::BulkDrain || E::BPolicy.wait_policy == WaitPolicy::Upfront));
    } else if constexpr (is_cb_reader_op_v<E>) {
        return E::Policy == InputLifecycle::BulkDrain || E::Policy.wait_policy == WaitPolicy::Upfront;
    }
    return false;
}

template <class E>
constexpr bool reserves_upfront() {
    if constexpr (is_cb_writer_op_v<E>) {
        return E::Policy.reserve_policy == ReservePolicy::Upfront ||
               E::Policy.reserve_policy == ReservePolicy::OneUpfront;
    }
    return false;
}

template <class E>
constexpr bool pops_at_end() {
    if constexpr (is_binary_fpu_op_v<E>) {
        return E::APolicy.pop_policy == PopPolicy::AtEnd || (!E::same_dfb && E::BPolicy.pop_policy == PopPolicy::AtEnd);
    } else if constexpr (is_cb_reader_op_v<E>) {
        return E::Policy.pop_policy == PopPolicy::AtEnd;
    }
    return false;
}

template <class E>
constexpr bool pushes_at_end() {
    if constexpr (is_cb_writer_op_v<E>) {
        return E::Policy.push_policy == PushPolicy::AtEnd || E::Policy.push_policy == PushPolicy::OneAtEnd;
    }
    return false;
}

// All CB lifecycle operations terminate in one of these four emitters. `Enabled` is a
// compile-time policy fact, so disabled actions disappear without a runtime sentinel check;
// the emitted function identity is independent of both the element type and the chain pack.
template <bool Enabled>
ALWI void emit_wait(uint32_t cb, uint32_t count) {
    if constexpr (Enabled) {
        DataflowBuffer(cb).wait_front(count);
    }
}

template <bool Enabled>
ALWI void emit_pop(uint32_t cb, uint32_t count) {
    if constexpr (Enabled) {
        DataflowBuffer(cb).pop_front(count);
    }
}

template <bool Enabled>
ALWI void emit_reserve(uint32_t cb, uint32_t count) {
    if constexpr (Enabled) {
        DataflowBuffer(cb).reserve_back(count);
    }
}

template <bool Enabled>
ALWI void emit_push(uint32_t cb, uint32_t count) {
    if constexpr (Enabled) {
        DataflowBuffer(cb).push_back(count);
    }
}

// init() dispatch convention — the compute-cohort init is emitted on the element
// *instance* (`elem.init()`), exactly like `exec`, so an init can read the struct's
// runtime members when it needs to (e.g. Dropout seeding the SFPU with its runtime
// `seed`). Most inits don't and are declared `static`; a static member is callable
// through an instance, so the call site is uniform either way. PackTile is the one
// exception — its init is dispatched by type in the indexed setup fold (no instance
// needed), which is fine because pack init never needs runtime state.

// =============================================================================
// emit_pre_element_transitions<E, PrevA, PrevB, PrevP, PackHetero>()
//
// Emits srca / srcb / pack reconfig for element I, each compile-time-elided when
// prev_*_cb == curr_*_cb. A chain that shares a CB on a side thus reconfigs it once (at
// element 0, prev == NO_PREV_DFB) and never again. Zero run-time cost — `if constexpr`.
//
// srca+srcb coalesce: both-with-known-prev → 4-arg _with_dt; both-without-known-prev →
// 2-arg combined; mixed → independent per-side. CONDITIONAL_DFB is an unknown previous
// state, so it takes the unconditional path and is never passed to an LLK as a CB id.
// Pack is always independent. The LLK _with_dt overloads
// fast-path-skip on format equality, so an emitted reconfig on matching dtypes is a
// hardware no-op (a few compares).
//
// Pack-side hoist: homogeneous chains (≤1 opt-in pack site, or all share a CB) emit at
// boot via the indexed setup fold; heterogeneous chains (≥2 sites, different CBs) emit only
// the first site at boot and defer the rest to per-stage `emit_per_stage_pack_reconfig`.
// DEST accumulation is build-flag-driven (no per-element fp32 fold here).
// =============================================================================

// Takes only the element and the preceding CB descriptors needed to decide its transitions.
template <class E, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
ALWI void emit_pre_element_transitions() {
    constexpr uint32_t curr_a = dfb_for_side<Side::SrcA, E>();
    constexpr uint32_t curr_b = dfb_for_side<Side::SrcB, E>();
    constexpr uint32_t curr_p = dfb_for_side<Side::Pack, E>();

    constexpr uint32_t prev_a = (curr_a != NO_PREV_DFB) ? PrevA : NO_PREV_DFB;
    constexpr uint32_t prev_b = (curr_b != NO_PREV_DFB) ? PrevB : NO_PREV_DFB;
    constexpr uint32_t prev_p = (curr_p != NO_PREV_DFB) ? PrevP : NO_PREV_DFB;

    constexpr bool reconf_a = (curr_a != NO_PREV_DFB) && (curr_a != prev_a);
    constexpr bool reconf_b = (curr_b != NO_PREV_DFB) && (curr_b != prev_b);
    constexpr bool reconf_p = (curr_p != NO_PREV_DFB) && (curr_p != prev_p);
    constexpr bool known_prev_a = prev_a != NO_PREV_DFB && prev_a != CONDITIONAL_DFB;
    constexpr bool known_prev_b = prev_b != NO_PREV_DFB && prev_b != CONDITIONAL_DFB;

    // Pack-side deferral: in heterogeneous chains, only the first opt-in pack
    // site (prev_p == NO_PREV_DFB) emits at boot. Later sites defer to per-stage
    // via `emit_per_stage_pack_reconfig`, where the 2-arg LLK form's cache check
    // handles intra-stage transitions cheaply and the per-iter wraparound from
    // last-pack-cb to first-pack-cb is correctly programmed.
    constexpr bool defer_pack_to_per_stage = PackHetero && (prev_p != NO_PREV_DFB);

    // ---- srca + srcb: coalesce when both sides share prev-state ----
    if constexpr (reconf_a && reconf_b) {
        if constexpr (known_prev_a && known_prev_b) {
            // both sides have prev → 4-arg _with_dt
            reconfig_data_format(prev_a, curr_a, prev_b, curr_b);
        } else if constexpr (!known_prev_a && !known_prev_b) {
            // first/unknown on both sides → 2-arg combined (unconditional reprogram)
            reconfig_data_format(curr_a, curr_b);
        } else {
            // mixed prev-state → independent per-side
            if constexpr (known_prev_a) {
                reconfig_data_format_srca(prev_a, curr_a);
            } else {
                reconfig_data_format_srca(curr_a);
            }
            if constexpr (known_prev_b) {
                reconfig_data_format_srcb(prev_b, curr_b);
            } else {
                reconfig_data_format_srcb(curr_b);
            }
        }
    } else if constexpr (reconf_a) {
        if constexpr (known_prev_a) {
            reconfig_data_format_srca(prev_a, curr_a);
        } else {
            reconfig_data_format_srca(curr_a);
        }
    } else if constexpr (reconf_b) {
        if constexpr (known_prev_b) {
            reconfig_data_format_srcb(prev_b, curr_b);
        } else {
            reconfig_data_format_srcb(curr_b);
        }
    }

    // ---- pack: always independent; deferred to per-stage in heterogeneous chains ----
    if constexpr (reconf_p && !defer_pack_to_per_stage) {
        if constexpr (prev_p != NO_PREV_DFB) {
            pack_reconfig_data_format(prev_p, curr_p);
        } else {
            pack_reconfig_data_format(curr_p);
        }
    }
}

// emit_per_stage_pack_reconfig(curr, prev, last, heterogeneous)
//
// Per-stage pack reconfig used only when the chain has heterogeneous opt-in
// pack CBs (boot can't program all of them). For every opt-in pack site,
// emit the 2-arg `pack_reconfig_data_format(prev, curr)` form before that
// site's per-iter pack work, using wraparound `prev` for the first pack site
// (so iter k+1's site 0 sees iter k's site N-1 as the previous descriptor
// state). The LLK's compare-and-skip on matching formats keeps the cost to
// a few cycles when adjacent stages happen to share a dtype.
ALWI void emit_per_stage_pack_reconfig(uint32_t curr_p, uint32_t prev_chain, uint32_t last_pack_cb, bool pack_hetero) {
    if (!pack_hetero || curr_p == NO_PREV_DFB) {
        return;
    }
    // Wraparound: first opt-in pack site has no in-chain prev; on iter ≥ 1 the
    // packer ended the previous iter on `last_pack_cb`. The LLK 2-arg form does
    // the right thing on iter 0 too (cache check vs. boot-initialized state).
    const uint32_t prev_p = (prev_chain != NO_PREV_DFB) ? prev_chain : last_pack_cb;
    if (prev_p != NO_PREV_DFB) {
        pack_reconfig_data_format(prev_p, curr_p);
    }
}

// Pack-phase setup (Pack* only). Pack is its own cohort (disjoint from math-MOP / SFPU),
// excluded from the compute-init fold and always boot-hoisted by the pack-init fold.
// Reconfig is fold-driven (see emit_pre_element_transitions): homogeneous chains program
// the packer once at boot; heterogeneous chains defer later sites to per-stage emission so
// the per-iter wraparound stays correct.
template <uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
struct TransitionFacts {};

ALWI void elem_pack_init(UnselectedElement) {}

template <class E, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
ALWI void elem_pack_init(SelectedElement<E, TransitionFacts<PrevA, PrevB, PrevP, PackHetero>>) {
    emit_pre_element_transitions<E, PrevA, PrevB, PrevP, PackHetero>();
}

// =============================================================================
// Two-phase per-element apply: compute / pack
//
// Each element owns its full lifecycle slice of the outer iteration. Per outer iter:
//
//   tile_regs_acquire();
//   elem_apply_compute(...);    // per element: wait + init? + for(j) exec + pop
//   tile_regs_commit();
//   tile_regs_wait();
//   elem_apply_pack(...);       // per pack element: reserve + for(j) pack_exec + push
//   tile_regs_release();
//
// Upfront-policy lifecycle (elem_pop_upfront_end / elem_push_at_end) fires after the loop.
//
// pop_per_tile / push_per_tile live inside the apply body. cb_pop_front right after exec
// is safe — by then the unpack-read is queued and the framework manages in-flight reads
// vs producer L1 reuse; push right after pack_exec wakes the consumer while DEST is still
// held. Both are policy-guarded (no-op for upfront / no-pop / no-push policies).
// =============================================================================

// Boot-time hoist of compute-cohort init (math-MOP and/or DEST-only ops). The chain dispatcher
// computes `HoistMath` from `chain_hoist_math_mop_v` and `HoistSfpu` from `chain_hoist_sfpu_v`,
// then this walk emits the element's transitions + init() only when its cohort is hoisted. The
// HoistSfpu leg gates on `is_dest_only_op_v`, so Fill/Rand init() ride along (chain_hoist_sfpu_v
// itself is decided from SFPU-op uniformity alone).
//
// PackTile is intentionally excluded from this walk — pack-side reconfig is
// emitted unconditionally at boot via the indexed pack-init fold (PACK cohort is
// disjoint from compute cohorts and is always hoisted).
ALWI void hoist_compute_init_one(UnselectedElement) {}

template <class E, uint32_t PrevA, uint32_t PrevB, uint32_t PrevP, bool PackHetero>
ALWI void hoist_compute_init_one(SelectedElement<E, TransitionFacts<PrevA, PrevB, PrevP, PackHetero>> selected) {
    emit_pre_element_transitions<E, PrevA, PrevB, PrevP, PackHetero>();
    selected.value.init();  // instance dispatch: runtime-stateful init may read element members
}

}  // namespace detail

// =============================================================================
// 11. Public eltwise_chain()
//
// Caller-init contract:
//   - Caller writes `compute_kernel_hw_startup(dfb_a, dfb_b, cb_out)` (or its
//     unary/binary variant) as the FIRST statement of `MAIN()`.
//   - Helper does NOT wrap any "BIG init" (`compute_kernel_hw_startup`,
//     `binary_op_init_common`, `mm_init`, `reduce_init`).
//   - Helper owns per-element init only — `add_tiles_init`, `*_tile_init`,
//     `init_bcast`, `copy_tile_to_dst_init_short`, `reconfig_data_format_*`,
//     `tile_regs_*` lifecycle.
//   - Per `compute_kernel_hw_startup.h:26-30`, mid-`MAIN()` boot is undefined.
//     Multi-stage kernels are the only exception (one boot per stage,
//     immediately before that stage's chain call).
// =============================================================================

// =============================================================================
// 11b. eltwise_chain — unified (Ht, Wt) walk with per-element broadcast indexing
//
// Walks an (Ht, Wt) tile grid (Ht=1 expresses the 1D case). Inner loop blocks W
// (block_size tiles per inner iter). Per-element index mode picks the tile index
// for each CB-reader: Block → flat (ht*Wt + wt), Row → wt, Col → ht,
// Scalar → 0.
// =============================================================================

namespace detail {

template <
    bool EmitMathInit,
    bool EmitSfpuInit,
    uint32_t SlotBase,
    uint32_t PrevA,
    uint32_t PrevB,
    uint32_t PrevP,
    bool PackHetero>
struct ComputeFacts {};

ALWI void elem_apply_compute(UnselectedElement, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}

template <
    class ElemT,
    bool EmitMathInit,
    bool EmitSfpuInit,
    uint32_t SlotBase,
    uint32_t PrevA,
    uint32_t PrevB,
    uint32_t PrevP,
    bool PackHetero>
ALWI void elem_apply_compute(
    SelectedElement<ElemT, ComputeFacts<EmitMathInit, EmitSfpuInit, SlotBase, PrevA, PrevB, PrevP, PackHetero>>
        selected,
    [[maybe_unused]] uint32_t i_flat,
    [[maybe_unused]] uint32_t ht,
    [[maybe_unused]] uint32_t wt,
    [[maybe_unused]] uint32_t inner_count,
    [[maybe_unused]] uint32_t chain_lane_width,
    [[maybe_unused]] uint32_t Ht,
    [[maybe_unused]] uint32_t Wt) {
    const ElemT& elem = selected.value;
    constexpr bool skip_compute = (CKL_ELTWISE_CHAIN_SKIP_COMPUTE != 0);
    // Per-block streaming: pass chunk-local index `j` to exec so Block
    // returns the local CB-front offset (the just-waited window).
    [[maybe_unused]] constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
    if constexpr (is_runtime_conditional_sequence_op_v<ElemT>) {
        if constexpr (!skip_compute) {
            elem.template apply<SlotBase, PrevA, PrevB, PrevP, PackHetero>(
                i_flat, ht, wt, inner_count, chain_lane_width, Ht, Wt);
        }
    } else if constexpr (is_cb_reader_op_v<ElemT>) {
        // Per-block-iteration input waits — mutually exclusive by policy (Streaming forces
        // block_size 1, so per_tile and per_block never both fire). The Bulk upfront wait is NOT
        // here: it's hoisted once to the chain boundary (elem_wait_upfront, pre-loop fold), so it's
        // placed exactly once rather than re-issued per block-iter relying on idempotency.
        if constexpr (is_binary_fpu_op_v<ElemT>) {
            if constexpr (ElemT::APolicy.wait_policy == WaitPolicy::PerTile) {
                emit_wait<true>(ElemT::dfb_a_id(), 1);
            } else if constexpr (ElemT::APolicy.wait_policy == WaitPolicy::Cumulative) {
                emit_wait<true>(ElemT::dfb_a_id(), i_flat + inner_count);
            } else if constexpr (ElemT::APolicy.wait_policy == WaitPolicy::PerChunk) {
                emit_wait<true>(ElemT::dfb_a_id(), inner_count);
            }
            if constexpr (!ElemT::same_dfb) {
                if constexpr (ElemT::BPolicy.wait_policy == WaitPolicy::PerTile) {
                    emit_wait<true>(ElemT::dfb_b_id(), 1);
                } else if constexpr (ElemT::BPolicy.wait_policy == WaitPolicy::Cumulative) {
                    emit_wait<true>(ElemT::dfb_b_id(), i_flat + inner_count);
                } else if constexpr (ElemT::BPolicy.wait_policy == WaitPolicy::PerChunk) {
                    emit_wait<true>(ElemT::dfb_b_id(), inner_count);
                }
            }
        } else {
            if constexpr (ElemT::Policy.wait_policy == WaitPolicy::PerTile) {
                emit_wait<true>(ElemT::dfb, 1);
            } else if constexpr (ElemT::Policy.wait_policy == WaitPolicy::Cumulative) {
                emit_wait<true>(ElemT::dfb, i_flat + inner_count);
            } else if constexpr (ElemT::Policy.wait_policy == WaitPolicy::PerChunk) {
                emit_wait<true>(ElemT::dfb, inner_count);
            }
        }
        if constexpr (EmitMathInit && !skip_compute) {
            emit_pre_element_transitions<ElemT, PrevA, PrevB, PrevP, PackHetero>();
            elem.init();  // instance dispatch (see convention note above)
        }
        if constexpr (!skip_compute) {
            constexpr bool per_side = elem_needs_per_side_idx_v<ElemT>;
            for (uint32_t j = 0; j < inner_count; ++j) {
                if constexpr (per_side) {
                    // Per-side path: chain hands both indices; element picks per operand.
                    elem.exec(
                        /*i_flat_local=*/j,
                        /*i_flat_abs=*/(i_flat + j),
                        ht,
                        /*wt_local=*/j,
                        /*wt_abs=*/(wt + j),
                        SlotBase + j * chain_lane_width);
                } else {
                    const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
                    elem.exec(i_arg, ht, wt + j, SlotBase + j * chain_lane_width);
                }
            }
        }
        if constexpr (is_binary_fpu_op_v<ElemT>) {
            if constexpr (ElemT::APolicy.pop_policy == PopPolicy::PerTile) {
                emit_pop<true>(ElemT::dfb_a_id(), 1);
            } else if constexpr (ElemT::APolicy.pop_policy == PopPolicy::PerChunk) {
                emit_pop<true>(ElemT::dfb_a_id(), inner_count);
            }
            if constexpr (!ElemT::same_dfb) {
                if constexpr (ElemT::BPolicy.pop_policy == PopPolicy::PerTile) {
                    emit_pop<true>(ElemT::dfb_b_id(), 1);
                } else if constexpr (ElemT::BPolicy.pop_policy == PopPolicy::PerChunk) {
                    emit_pop<true>(ElemT::dfb_b_id(), inner_count);
                }
            }
        } else {
            if constexpr (ElemT::Policy.pop_policy == PopPolicy::PerTile) {
                emit_pop<true>(ElemT::dfb, 1);
            } else if constexpr (ElemT::Policy.pop_policy == PopPolicy::PerChunk) {
                emit_pop<true>(ElemT::dfb, inner_count);
            }
        }
    } else if constexpr (is_dest_only_op_v<ElemT>) {
        if constexpr (EmitSfpuInit && !skip_compute) {
            emit_pre_element_transitions<ElemT, PrevA, PrevB, PrevP, PackHetero>();
            elem.init();  // instance dispatch (see convention note above)
        }
        if constexpr (!skip_compute) {
            for (uint32_t j = 0; j < inner_count; ++j) {
                elem.exec(i_flat + j, SlotBase + j * chain_lane_width);
            }
        }
    }
}

template <bool AnyPackRelu, uint32_t PrevPack, uint32_t LastPackCb, bool PackHetero>
struct PackFacts {};

ALWI void elem_apply_pack(UnselectedElement, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}

template <class ElemT, bool AnyPackRelu, uint32_t PrevPack, uint32_t LastPackCb, bool PackHetero>
ALWI void elem_apply_pack(
    SelectedElement<ElemT, PackFacts<AnyPackRelu, PrevPack, LastPackCb, PackHetero>> selected,
    [[maybe_unused]] uint32_t i_flat,
    [[maybe_unused]] uint32_t ht,
    [[maybe_unused]] uint32_t wt,
    [[maybe_unused]] uint32_t inner_count,
    [[maybe_unused]] uint32_t chain_lane_width,
    [[maybe_unused]] uint32_t Ht,
    [[maybe_unused]] uint32_t Wt) {
    const ElemT& elem = selected.value;
    constexpr bool skip_compute = (CKL_ELTWISE_CHAIN_SKIP_COMPUTE != 0);
    [[maybe_unused]] constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
    // upfront reserve is emitted once before the loop (see eltwise_chain_impl)
    if constexpr (!skip_compute) {
        emit_per_stage_pack_reconfig(dfb_for_side<Side::Pack, ElemT>(), PrevPack, LastPackCb, PackHetero);
    }
    if constexpr (ElemT::Policy.reserve_policy == ReservePolicy::PerTile) {
        emit_reserve<true>(ElemT::dfb, 1);
    } else if constexpr (ElemT::Policy.reserve_policy == ReservePolicy::PerChunk) {
        emit_reserve<true>(ElemT::dfb, inner_count);
    }
    if constexpr (!skip_compute) {
        if constexpr (AnyPackRelu) {
            elem.configure_relu();
        }
        for (uint32_t j = 0; j < inner_count; ++j) {
            const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
            elem.exec(i_arg, ht, wt + j, j * chain_lane_width);
        }
    }
    if constexpr (ElemT::Policy.push_policy == PushPolicy::PerTile) {
        emit_push<true>(ElemT::dfb, 1);
    } else if constexpr (ElemT::Policy.push_policy == PushPolicy::PerChunk) {
        emit_push<true>(ElemT::dfb, inner_count);
    }
}

ALWI void elem_apply_seed_first_l1_pack(UnselectedElement, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}

template <class ElemT>
ALWI void elem_apply_seed_first_l1_pack(
    SelectedElement<ElemT> selected,
    [[maybe_unused]] uint32_t i_flat,
    [[maybe_unused]] uint32_t ht,
    [[maybe_unused]] uint32_t wt,
    [[maybe_unused]] uint32_t j,
    [[maybe_unused]] uint32_t chain_lane_width) {
    constexpr bool skip_compute = (CKL_ELTWISE_CHAIN_SKIP_COMPUTE != 0);
    if constexpr (!skip_compute) {
        constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
        const uint32_t i_arg = use_local_idx ? j : (i_flat + j);
        selected.value.exec(i_arg, ht, wt + j, j * chain_lane_width);
    }
}

ALWI void emit_wait_upfront(UnselectedElement, uint32_t, uint32_t) {}

template <class E>
ALWI void emit_wait_upfront(SelectedElement<E> selected, uint32_t Ht, uint32_t Wt) {
    const E& e = selected.value;
    if constexpr (is_binary_fpu_op_v<E>) {
        if constexpr (E::same_dfb) {
            const uint32_t base_a = tile_base_value<E::OffsetA>(e.a.tile_base);
            const uint32_t base_b = tile_base_value<E::OffsetB>(e.b.tile_base);
            const uint32_t base = base_a > base_b ? base_a : base_b;
            if constexpr (E::APolicy == InputLifecycle::BulkDrain) {
                emit_wait<true>(E::dfb_a_id(), Ht * Wt + base);
            } else if constexpr (E::APolicy.wait_policy == WaitPolicy::Upfront) {
                emit_wait<true>(E::dfb_a_id(), detail::window<E::AIndex>(Ht, Wt) + base);
            }
        } else {
            if constexpr (E::APolicy == InputLifecycle::BulkDrain) {
                emit_wait<true>(E::dfb_a_id(), Ht * Wt + tile_base_value<E::OffsetA>(e.a.tile_base));
            } else if constexpr (E::APolicy.wait_policy == WaitPolicy::Upfront) {
                emit_wait<true>(
                    E::dfb_a_id(), detail::window<E::AIndex>(Ht, Wt) + tile_base_value<E::OffsetA>(e.a.tile_base));
            }
            if constexpr (E::BPolicy == InputLifecycle::BulkDrain) {
                emit_wait<true>(E::dfb_b_id(), Ht * Wt + tile_base_value<E::OffsetB>(e.b.tile_base));
            } else if constexpr (E::BPolicy.wait_policy == WaitPolicy::Upfront) {
                emit_wait<true>(
                    E::dfb_b_id(), detail::window<E::BIndex>(Ht, Wt) + tile_base_value<E::OffsetB>(e.b.tile_base));
            }
        }
    } else if constexpr (is_cb_reader_op_v<E>) {
        if constexpr (E::Policy == InputLifecycle::BulkDrain) {
            emit_wait<true>(E::dfb, Ht * Wt + tile_base_value<E::Offset>(e.tile_base));
        } else if constexpr (E::Policy.wait_policy == WaitPolicy::Upfront) {
            emit_wait<true>(E::dfb, detail::window<E::IndexMode>(Ht, Wt) + tile_base_value<E::Offset>(e.tile_base));
        }
    }
}

ALWI void emit_reserve_upfront(UnselectedElement, uint32_t, uint32_t) {}

template <class E>
ALWI void emit_reserve_upfront(SelectedElement<E> selected, uint32_t Ht, uint32_t Wt) {
    const E& e = selected.value;
    if constexpr (E::Policy.reserve_policy == ReservePolicy::Upfront) {
        emit_reserve<true>(E::dfb, (Ht * Wt) + tile_base_value<E::Offset>(e.tile_base));
    } else if constexpr (E::Policy.reserve_policy == ReservePolicy::OneUpfront) {
        emit_reserve<true>(E::dfb, 1);
    }
}

ALWI void emit_pop_at_end(UnselectedElement, uint32_t, uint32_t) {}

template <class E>
ALWI void emit_pop_at_end(SelectedElement<E> selected, uint32_t Ht, uint32_t Wt) {
    const E& e = selected.value;
    if constexpr (is_binary_fpu_op_v<E>) {
        if constexpr (E::same_dfb) {
            if constexpr (E::APolicy.pop_policy == PopPolicy::AtEnd) {
                const uint32_t base_a = tile_base_value<E::OffsetA>(e.a.tile_base);
                const uint32_t base_b = tile_base_value<E::OffsetB>(e.b.tile_base);
                const uint32_t base = base_a > base_b ? base_a : base_b;
                emit_pop<true>(E::dfb_a_id(), detail::window<E::AIndex>(Ht, Wt) + base);
            }
        } else {
            if constexpr (E::APolicy.pop_policy == PopPolicy::AtEnd) {
                emit_pop<true>(
                    E::dfb_a_id(), detail::window<E::AIndex>(Ht, Wt) + tile_base_value<E::OffsetA>(e.a.tile_base));
            }
            if constexpr (E::BPolicy.pop_policy == PopPolicy::AtEnd) {
                emit_pop<true>(
                    E::dfb_b_id(), detail::window<E::BIndex>(Ht, Wt) + tile_base_value<E::OffsetB>(e.b.tile_base));
            }
        }
    } else if constexpr (is_cb_reader_op_v<E>) {
        if constexpr (E::Policy.pop_policy == PopPolicy::AtEnd) {
            emit_pop<true>(E::dfb, detail::window<E::IndexMode>(Ht, Wt) + tile_base_value<E::Offset>(e.tile_base));
        }
    }
}

ALWI void emit_push_at_end(UnselectedElement, uint32_t, uint32_t) {}

template <class E>
ALWI void emit_push_at_end(SelectedElement<E> selected, uint32_t Ht, uint32_t Wt) {
    const E& e = selected.value;
    if constexpr (E::Policy.push_policy == PushPolicy::AtEnd) {
        emit_push<true>(E::dfb, (E::walk ? (Ht * Wt) : 1u) + tile_base_value<E::Offset>(e.tile_base));
    } else if constexpr (E::Policy.push_policy == PushPolicy::OneAtEnd) {
        emit_push<true>(E::dfb, 1);
    }
}

template <bool WaitA, bool WaitB>
ALWI void emit_wait_per_row(uint32_t cb_a, uint32_t cb_b) {
    emit_wait<WaitA>(cb_a, 1);
    emit_wait<WaitB>(cb_b, 1);
}

template <bool PopA, bool PopB>
ALWI void emit_pop_per_row(uint32_t cb_a, uint32_t cb_b) {
    emit_pop<PopA>(cb_a, 1);
    emit_pop<PopB>(cb_b, 1);
}

template <bool Reserve>
ALWI void emit_reserve_per_row(uint32_t cb) {
    emit_reserve<Reserve>(cb, 1);
}

template <bool Push>
ALWI void emit_push_per_row(uint32_t cb) {
    emit_push<Push>(cb, 1);
}

}  // namespace detail

// eltwise_chain_impl — the walk. SetupOwner::Chain (default) emits the chain's one-time setup
// (pack boot init + the uniform math-MOP / SFPU init + their srca/srcb reconfig) before walking.
// SetupOwner::Caller skips ALL of it: the caller emitted the chain's whole one-time setup itself,
// once, before its own loop, so this call is pure per-tile compute. SetupOwner is about WHO emits
// the hoistable setup — it never changes which init is hoistable (that's deduced from uniformity).
template <SetupOwner SO = SetupOwner::Chain, std::size_t... Is, class... Es>
ALWI void eltwise_chain_impl([[maybe_unused]] std::index_sequence<Is...> indices, EltwiseShape shape, Es... elts) {
    using Chain = EltwiseChain<Es...>;
    constexpr bool skip_compute = (CKL_ELTWISE_CHAIN_SKIP_COMPUTE != 0);
    static_assert(
        detail::ChainTraits<Es...>::any_dest_accumulation ==
            detail::ChainTraits<Es...>::any_dest_accumulation_lifecycle,
        "eltwise_chain: DEST accumulation must be enabled on both BinaryFpu and output(...)");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::dest_accumulation_slot_count == 1,
        "eltwise_chain: DEST accumulation requires one sticky DEST slot");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation || detail::ChainTraits<Es...>::pack_writer_count == 1,
        "eltwise_chain: DEST accumulation requires exactly one PackTile");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::dest_accumulation_pack_matches,
        "eltwise_chain: DEST accumulation PackTile must pack the sticky DEST slot");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::all_writers_dest_accumulation_lifecycle,
        "eltwise_chain: DEST accumulation cannot mix ordinary and accumulating outputs");
    static_assert(
        detail::ChainTraits<Es...>::managed_dest_accumulation_lifecycles <= 1,
        "eltwise_chain: only one PackTile may manage the DEST-accumulation output lifecycle");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation ||
            detail::ChainTraits<Es...>::transient_lane_width < DEST_AUTO_LIMIT,
        "eltwise_chain: sticky D0 leaves insufficient DEST capacity for a transient lane");
    static_assert(
        !detail::ChainTraits<Es...>::any_dest_accumulation || !detail::ChainTraits<Es...>::any_l1_accumulation,
        "eltwise_chain: DEST and L1 accumulation cannot be combined");
    static_assert(
        !detail::ChainTraits<Es...>::any_l1_accumulation || detail::ChainTraits<Es...>::pack_dfbs_consistent,
        "eltwise_chain: L1 accumulation requires one output CB");
    static_assert(
        !detail::ChainTraits<Es...>::any_l1_accumulation || detail::ChainTraits<Es...>::all_writers_l1_accumulation,
        "eltwise_chain: L1 accumulation cannot mix ordinary and accumulating PackTile elements");
    static_assert(
        detail::ChainTraits<Es...>::l1_accumulation_modes_consistent,
        "eltwise_chain: accumulating PackTile elements must use one L1 accumulation mode");
    static_assert(
        detail::ChainTraits<Es...>::managed_l1_accumulation_lifecycles <= 1,
        "eltwise_chain: only one PackTile may manage the L1-accumulation output lifecycle");
    static_assert(
        !chain_has_duplicate_upfront_cbs_v<Chain>,
        "eltwise_chain: two CB-reader elements share a CB on upfront-wait policy.");
    static_assert(
        !chain_pack_writes_collide_v<Chain>, "eltwise_chain: two PackTile elements collide on (dfb, dst_slot).");
    // SetupOwner::Caller means "the caller did the chain's whole one-time setup itself, once,
    // before the loop." That's only achievable if EVERY part of it is boot-hoistable: uniform
    // math MOP + SFPU init (so input/srca-srcb reconfig is boot-only) AND homogeneous pack CBs
    // (so output/pack reconfig is boot-only — no per-stage pack reconfig in the loop). Otherwise
    // some setup must re-emit per tile, the caller can't pre-do it once, and the knob silently
    // skips a needed init/reconfig — a footgun. Forbid it at compile time.
    static_assert(
        SO == SetupOwner::Chain ||
            (chain_hoist_math_mop_v<Chain> && chain_hoist_sfpu_v<Chain> && chain_hoist_pack_v<Chain>),
        "SetupOwner::Caller requires a chain whose entire one-time setup is boot-hoistable "
        "(uniform math MOP + SFPU init AND homogeneous pack CBs) so that input AND output "
        "reconfig are boot-only and nothing self-emits per tile. This chain has setup that "
        "must re-emit per tile, so the caller cannot own it once — use SetupOwner::Chain.");
    // Honesty: under SetupOwner::Caller the chain emits NO reconfig at all (the caller owns the
    // setup), so enabled operand reconfig on any element is inert and lies about what runs inside
    // the helper. Require each input/output spec to disable it.
    static_assert(
        SO == SetupOwner::Chain || chain_no_reconfig_requested_v<Chain>,
        "SetupOwner::Caller with enabled operand reconfig: under Caller the chain emits no "
        "reconfig (the caller owns the setup), so the setting is inert and misleading. Disable "
        "reconfig in every input/output spec — the caller's manual setup owns the format.");
    static_assert(
        SO == SetupOwner::Chain || ((!is_runtime_conditional_op_v<Es>) && ...),
        "SetupOwner::Caller cannot be used with runtime-conditional elements because their "
        "selected init must execute inside each chain invocation.");
    // Per-cohort hoist decisions: math-MOP init can be hoisted at boot even when SFPU isn't
    // uniform; the SFPU side then re-inits per tile.
    constexpr bool hoist_math = chain_hoist_math_mop_v<Chain>;
    constexpr bool hoist_sfpu = chain_hoist_sfpu_v<Chain>;
    if constexpr (SO == SetupOwner::Chain && !skip_compute) {
        (detail::elem_pack_init(detail::select_element<
                                is_pack_tile_op_v<Es>,
                                detail::TransitionFacts<
                                    detail::ChainTraits<Es...>::prev.srca[Is],
                                    detail::ChainTraits<Es...>::prev.srcb[Is],
                                    detail::ChainTraits<Es...>::prev.pack[Is],
                                    detail::ChainTraits<Es...>::pack_hetero>>(elts)),
         ...);
        (detail::hoist_compute_init_one(
             detail::select_element < (is_math_mop_op_v<Es> && hoist_math) || (is_dest_only_op_v<Es> && hoist_sfpu),
             detail::TransitionFacts < detail::ChainTraits<Es...>::prev.srca[Is],
             detail::ChainTraits<Es...>::prev.srcb[Is],
             detail::ChainTraits<Es...>::prev.pack[Is],
             detail::ChainTraits<Es...>::pack_hetero >> (elts)),
         ...);
    }
    constexpr uint32_t chain_lane_w = detail::ChainTraits<Es...>::any_dest_accumulation
                                          ? chain_transient_lane_width_v<Chain>
                                          : chain_lane_width_v<Chain>;
    uint32_t block_size = shape.block_size;
    if constexpr (!chain_supports_block_v<Chain>) {
        block_size = 1;
    } else {
        constexpr uint32_t max_block = chain_max_block_v<Chain>;
        if (block_size > max_block) {
            block_size = max_block;
        }
    }

    const uint32_t Ht = shape.Ht;
    const uint32_t Wt = shape.Wt;

    (detail::emit_wait_upfront(detail::select_element<detail::waits_upfront<Es>()>(elts), Ht, Wt), ...);
    (detail::emit_reserve_upfront(detail::select_element<detail::reserves_upfront<Es>()>(elts), Ht, Wt), ...);

    constexpr bool dest_accumulation = detail::ChainTraits<Es...>::any_dest_accumulation;
    if constexpr (detail::ChainTraits<Es...>::any_l1_accumulation && !skip_compute) {
        pack_reconfig_l1_acc(detail::ChainTraits<Es...>::any_seed_first_l1_accumulation ? 0 : 1);
    }
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        const uint32_t row_base = ht * Wt;
        (detail::emit_wait_per_row<
             detail::ChainTraits<Es...>::d[Is].wait_a_per_outer,
             detail::ChainTraits<Es...>::d[Is].wait_b_per_outer>(
             detail::ChainTraits<Es...>::d[Is].outer_input_a_cb, detail::ChainTraits<Es...>::d[Is].outer_input_b_cb),
         ...);
        if constexpr (dest_accumulation) {
            (detail::emit_reserve_per_row<detail::ChainTraits<Es...>::d[Is].reserve_per_outer>(
                 detail::ChainTraits<Es...>::d[Is].pack_dfb),
             ...);
            tile_regs_acquire();
        }
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += block_size) {
            const uint32_t inner_count = (wt_base + block_size <= Wt) ? block_size : (Wt - wt_base);
            const uint32_t i_flat = row_base + wt_base;
            if constexpr (!dest_accumulation) {
                tile_regs_acquire();
            }
            // The workers retain CB wait/pop even in skip-compute builds; only init and execution
            // are elided there. Calling the same workers keeps lifecycle counts identical.
            (detail::elem_apply_compute(
                 detail::select_element<
                     detail::participates_in_compute_v<Es>,
                     detail::ComputeFacts<
                         !hoist_math,
                         !hoist_sfpu,
                         dest_accumulation ? 1 : 0,
                         detail::ChainTraits<Es...>::prev.srca[Is],
                         detail::ChainTraits<Es...>::prev.srcb[Is],
                         detail::ChainTraits<Es...>::prev.pack[Is],
                         detail::ChainTraits<Es...>::pack_hetero>>(elts),
                 i_flat,
                 ht,
                 wt_base,
                 inner_count,
                 chain_lane_w,
                 Ht,
                 Wt),
             ...);
            if constexpr (!dest_accumulation) {
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (detail::ChainTraits<Es...>::any_seed_first_l1_accumulation) {
                    for (uint32_t j = 0; j < inner_count; ++j) {
                        (detail::elem_apply_seed_first_l1_pack(
                             detail::select_element<is_pack_tile_op_v<Es>>(elts), i_flat, ht, wt_base, j, chain_lane_w),
                         ...);
                        if constexpr (!skip_compute) {
                            if (i_flat == 0 && j == 0) {
                                pack_reconfig_l1_acc(1);
                            }
                        }
                    }
                } else {
                    (detail::elem_apply_pack(
                         detail::select_element<
                             is_pack_tile_op_v<Es>,
                             detail::PackFacts<
                                 detail::ChainTraits<Es...>::any_pack_relu,
                                 detail::ChainTraits<Es...>::prev.pack[Is],
                                 detail::ChainTraits<Es...>::last_pack_cb,
                                 detail::ChainTraits<Es...>::pack_hetero>>(elts),
                         i_flat,
                         ht,
                         wt_base,
                         inner_count,
                         chain_lane_w,
                         Ht,
                         Wt),
                     ...);
                }
                tile_regs_release();
            }
        }
        if constexpr (dest_accumulation) {
            tile_regs_commit();
            tile_regs_wait();
            (detail::elem_apply_pack(
                 detail::select_element<
                     is_pack_tile_op_v<Es>,
                     detail::PackFacts<
                         detail::ChainTraits<Es...>::any_pack_relu,
                         detail::ChainTraits<Es...>::prev.pack[Is],
                         detail::ChainTraits<Es...>::last_pack_cb,
                         detail::ChainTraits<Es...>::pack_hetero>>(elts),
                 row_base,
                 ht,
                 0,
                 1,
                 0,
                 Ht,
                 Wt),
             ...);
            tile_regs_release();
            (detail::emit_push_per_row<detail::ChainTraits<Es...>::d[Is].push_per_outer>(
                 detail::ChainTraits<Es...>::d[Is].pack_dfb),
             ...);
        }
        (detail::emit_pop_per_row<
             detail::ChainTraits<Es...>::d[Is].pop_a_per_outer,
             detail::ChainTraits<Es...>::d[Is].pop_b_per_outer>(
             detail::ChainTraits<Es...>::d[Is].outer_input_a_cb, detail::ChainTraits<Es...>::d[Is].outer_input_b_cb),
         ...);
    }
    if constexpr (detail::ChainTraits<Es...>::any_l1_accumulation && !skip_compute) {
        pack_reconfig_l1_acc(0);
    }

    if constexpr (detail::ChainTraits<Es...>::any_pack_relu && !skip_compute) {
        pack_relu_config(ReluConfig::none());
    }
    (detail::emit_pop_at_end(detail::select_element<detail::pops_at_end<Es>()>(elts), Ht, Wt), ...);
    (detail::emit_push_at_end(detail::select_element<detail::pushes_at_end<Es>()>(elts), Ht, Wt), ...);
}

// =============================================================================
// 11c. Public eltwise_chain — forwards every element straight to eltwise_chain_impl.
//
// A compile-time-disabled optional resolves to the shared, members-less,
// tag-less `DisabledChainElement`. Every op-kind trait (is_cb_reader / is_pack / is_dest_only / is_math_mop
// = std::is_base_of on a tag it lacks) is false for it, and every ElemDesc accessor is
// SFINAE-guarded (dfb_* gate on the op-kind, lane_width defaults to 1, etc.), so it reflects
// into a NEUTRAL ElemDesc — no CB, no reconfig, no pack, lane_width 1 — and every stage
// (planner, hoist, reconfig fold, per-tile loop) treats it as inert. It is transparent to
// the prev-CB sweep too: its NO_PREV_DFB sides never update the running prev, so a later
// element sees the same previous CB as if the marker were absent.
//
// Passing the marker through the folds leaves it inert and transparent to neighboring elements.
// =============================================================================

// Public entry. `SetupOwner SO` (default Chain) says who emits the chain's one-time setup:
// Chain = this call emits it; Caller = the caller emitted it once, outside its loop (see the
// SetupOwner enum doc). It never changes which init is hoistable. (default lives on the
// declaration in eltwise_chain.hpp.)
template <SetupOwner SO, class... Es>
ALWI void eltwise_chain(EltwiseShape shape, Es... elts) {
    eltwise_chain_impl<SO>(std::index_sequence_for<Es...>{}, shape, elts...);
}

// =============================================================================
// 12. dfb_a_of / dfb_b_of / pack_dfb_of — single-element CB id deducers used by the
// collision-detection static_asserts (chain_has_duplicate_upfront_cbs_v,
// chain_pack_writes_collide_v). Caller-side hw_startup is the caller's
// responsibility — there is no deduced wrapper.
// =============================================================================

namespace detail {

template <class E>
constexpr uint32_t dfb_a_of() {
    if constexpr (is_cb_reader_op_v<E>) {
        static_assert(has_dfb_a<E>::value, "CbReader element must declare 'static constexpr uint32_t dfb_a_id()'");
        return E::dfb_a_id();
    } else {
        return INVALID_DFB;
    }
}

template <class E>
constexpr uint32_t dfb_b_of() {
    if constexpr (is_binary_fpu_op_v<E> || is_dest_reuse_binary_op_v<E>) {
        static_assert(
            has_dfb_b<E>::value, "Binary CbReader element must declare 'static constexpr uint32_t dfb_b_id()'");
        return E::dfb_b_id();
    } else {
        return INVALID_DFB;
    }
}

template <class E>
constexpr uint32_t pack_dfb_of() {
    if constexpr (is_pack_tile_op_v<E>) {
        static_assert(
            has_pack_dfb<E>::value, "CbWriter element must declare 'static constexpr uint32_t pack_dfb_id()'");
        return E::pack_dfb_id();
    } else {
        return INVALID_DFB;
    }
}

}  // namespace detail

}  // namespace compute_kernel_lib
