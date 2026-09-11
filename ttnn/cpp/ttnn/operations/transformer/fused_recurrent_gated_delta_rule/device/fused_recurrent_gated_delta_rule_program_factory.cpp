// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for fused_recurrent_gated_delta_rule.
// Parallelism: one Tensix core per (B*HV) head; each core walks the T token axis sequentially,
// holding the recurrent state S [K,V] on-core. All math is in the compute kernel, derived from
// flash-linear-attention `naive_recurrent_gated_delta_rule`.

#include "fused_recurrent_gated_delta_rule_program_factory.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

// CB index plan (all fp32). Kept in sync with the compute/reader/writer kernels.
namespace cb {
constexpr uint32_t q = tt::CBIndex::c_0;          // [1,K]  one token
constexpr uint32_t k = tt::CBIndex::c_1;          // [1,K]
constexpr uint32_t v = tt::CBIndex::c_2;          // [1,V]
constexpr uint32_t decay = tt::CBIndex::c_3;      // [1,1]  scalar exp(g_t)
constexpr uint32_t beta = tt::CBIndex::c_4;       // [1,1]  scalar
constexpr uint32_t S = tt::CBIndex::c_5;          // [K,V]  reader-produced initial state
constexpr uint32_t out = tt::CBIndex::c_6;        // [1,V]  output o_t
constexpr uint32_t state_out = tt::CBIndex::c_7;  // [K,V]  state output (per-token or final)
constexpr uint32_t s2 = tt::CBIndex::c_8;         // [K,V]  compute state ping
constexpr uint32_t s3 = tt::CBIndex::c_9;         // [K,V]  compute state pong
constexpr uint32_t sd = tt::CBIndex::c_10;        // [K,V]  decayed state (S*decay)
constexpr uint32_t vread = tt::CBIndex::c_11;     // [1,V]  k . S'
constexpr uint32_t u = tt::CBIndex::c_12;         // [1,V]  beta*(v - vread)
constexpr uint32_t kcol = tt::CBIndex::c_13;      // [K,1]  transpose(k)
constexpr uint32_t supd = tt::CBIndex::c_14;      // [K,V]  k^T (x) u
constexpr uint32_t delta = tt::CBIndex::c_15;     // [1,V]  v - vread
}  // namespace cb

tt::tt_metal::ProgramDescriptor FusedRecurrentGatedDeltaRuleProgramFactory::create_descriptor(
    const FusedRecurrentGatedDeltaRuleParams& attrs,
    const FusedRecurrentGatedDeltaRuleInputs& in,
    std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t T = attrs.T;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t has_s0 = in.initial_state.has_value() ? 1u : 0u;
    const uint32_t per_token = attrs.output_per_token_state ? 1u : 0u;

    const uint32_t kv = Kt * Vt;

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    TT_FATAL(BH <= grid.x * grid_y, "num_heads {} exceeds compute cores {}", BH, grid.x * grid_y);

    std::vector<CoreCoord> head_cores(BH);
    std::set<CoreRange> core_set;
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid_y, h % grid_y};
        core_set.insert(CoreRange{head_cores[h], head_cores[h]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    const tt::DataFormat fmt = tt::DataFormat::Float32;
    const uint32_t ts = tt::tile_size(fmt);
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    add_cb(cb::q, Kt);
    add_cb(cb::k, Kt);
    add_cb(cb::v, Vt);
    add_cb(cb::decay, 1, T > 1 ? 2 : 1);
    add_cb(cb::beta, 1, T > 1 ? 2 : 1);
    add_cb(cb::S, kv);
    add_cb(cb::out, Vt, 2);
    add_cb(cb::state_out, kv, 2);
    add_cb(cb::s2, kv);
    add_cb(cb::s3, kv);
    add_cb(cb::sd, kv);
    add_cb(cb::vread, Vt);
    add_cb(cb::u, Vt);
    add_cb(cb::kcol, Kt);
    add_cb(cb::supd, kv);
    add_cb(cb::delta, Vt);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/";
    const std::vector<uint32_t> compute_ct = {Kt, Vt, per_token};

    std::vector<uint32_t> reader_ct = {Kt, Vt, has_s0};
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.decay.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt, per_token};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_fused_recurrent_gated_delta_rule.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(BH);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_fused_recurrent_gated_delta_rule.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(BH);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/fused_recurrent_gated_delta_rule.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute.runtime_args.reserve(BH);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* decay_buf = in.decay.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* st_buf = outputs[1].buffer();

    for (uint32_t h = 0; h < BH; h++) {
        const auto& core = head_cores[h];
        reader.emplace_runtime_args(core, {h, T, q_buf, k_buf, v_buf, decay_buf, beta_buf, s0_buf});
        // BH is needed by the writer to place per-token state token-major (page t*BH + h).
        writer.emplace_runtime_args(core, {h, T, o_buf, st_buf, BH});
        compute.emplace_runtime_args(core, {T});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
