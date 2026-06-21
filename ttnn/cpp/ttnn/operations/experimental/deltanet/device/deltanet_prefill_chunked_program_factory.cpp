// SPDX-License-Identifier: Apache-2.0
#include "deltanet_prefill_chunked_program_factory.hpp"

#include "deltanet_prefill_chunked_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deltanet {

namespace chunked_factory {

constexpr uint32_t kTileSize = tt::constants::TILE_WIDTH;
constexpr auto kReaderPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_deltanet_prefill_chunked.cpp";
constexpr auto kComputePath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_prefill_chunked_compute.cpp";
constexpr auto kWriterPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_deltanet_prefill_chunked.cpp";

// 29 CBs (c_0..c_28)
constexpr auto cbK=tt::CBIndex::c_0; constexpr auto cbQ=tt::CBIndex::c_1; constexpr auto cbV=tt::CBIndex::c_2;
constexpr auto cbZ=tt::CBIndex::c_3; constexpr auto cbKdec=tt::CBIndex::c_4; constexpr auto cbKiT=tt::CBIndex::c_5;
constexpr auto cbQd=tt::CBIndex::c_6; constexpr auto cbDcol=tt::CBIndex::c_7; constexpr auto cbBetacol=tt::CBIndex::c_8;
constexpr auto cbDlast=tt::CBIndex::c_9; constexpr auto cbNormW=tt::CBIndex::c_10; constexpr auto cbScaler=tt::CBIndex::c_11;
constexpr auto cbEps=tt::CBIndex::c_12; constexpr auto cbStateInit=tt::CBIndex::c_13; constexpr auto cbIdent=tt::CBIndex::c_14;
constexpr auto cbTrils=tt::CBIndex::c_15; constexpr auto cbTrili=tt::CBIndex::c_16; constexpr auto cbKS0=tt::CBIndex::c_17;
constexpr auto cbQS0=tt::CBIndex::c_18; constexpr auto cbRhs=tt::CBIndex::c_19; constexpr auto cbU=tt::CBIndex::c_20;
constexpr auto cbP=tt::CBIndex::c_21; constexpr auto cbP2=tt::CBIndex::c_22; constexpr auto cbInv=tt::CBIndex::c_23;
constexpr auto cbIpp=tt::CBIndex::c_24; constexpr auto cbSn=tt::CBIndex::c_25; constexpr auto cbT0=tt::CBIndex::c_26;
constexpr auto cbT1=tt::CBIndex::c_27; constexpr auto cbOutput=tt::CBIndex::c_28;
constexpr auto cbStateA=tt::CBIndex::c_29; constexpr auto cbStateB=tt::CBIndex::c_30;

tt::tt_metal::CBHandle make_cb(tt::tt_metal::Program& program, const tt::tt_metal::CoreRangeSet& cores,
                               uint32_t cb_index, tt::DataFormat fmt, uint32_t num_tiles) {
    uint32_t tile_size = tt::tile_size(fmt);
    auto config = tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{cb_index, fmt}}).set_page_size(cb_index, tile_size);
    return tt::tt_metal::CreateCircularBuffer(program, cores, config);
}

}  // namespace chunked_factory

DeltaNetPrefillChunkedProgramFactory::cached_program_t DeltaNetPrefillChunkedProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& inputs, tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;
    namespace cf = chunked_factory;

    const uint32_t H = attrs.num_heads;
    const uint32_t Dk = attrs.k_head_dim, Dv = attrs.v_head_dim;
    const uint32_t Dk_tiles = Dk / cf::kTileSize, Dv_tiles = Dv / cf::kTileSize;
    const uint32_t state_tiles = Dk_tiles * Dv_tiles;
    const uint32_t nC = attrs.n_chunks;

    auto* device = inputs.recurrent_state.device();
    Program program{};
    tt::DataFormat fmt = datatype_to_dataformat_converter(inputs.k.dtype());
    auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(H <= grid.x * grid.y, "Need {} cores for {} heads", H, H);

    std::vector<CoreRange> core_ranges;
    for (uint32_t h = 0; h < H; h++) core_ranges.emplace_back(CoreCoord{h % grid.x, h / grid.x}, CoreCoord{h % grid.x, h / grid.x});
    CoreRangeSet all_cores(core_ranges);

    auto mk = [&](auto cb, uint32_t n) { cf::make_cb(program, all_cores, cb, fmt, n); };
    mk(cf::cbK, 2 * Dk_tiles); mk(cf::cbQ, 2 * Dk_tiles); mk(cf::cbV, 2 * Dv_tiles); mk(cf::cbZ, 2 * Dv_tiles);
    mk(cf::cbKdec, 2 * Dk_tiles); mk(cf::cbKiT, 2 * Dk_tiles); mk(cf::cbQd, 2 * Dk_tiles);
    mk(cf::cbDcol, 2 * Dv_tiles); mk(cf::cbBetacol, 2 * Dv_tiles); mk(cf::cbDlast, 2);
    mk(cf::cbNormW, Dv_tiles); mk(cf::cbScaler, 1); mk(cf::cbEps, 1);
    mk(cf::cbStateInit, state_tiles); mk(cf::cbStateA, state_tiles); mk(cf::cbStateB, state_tiles);
    mk(cf::cbIdent, 1); mk(cf::cbTrils, 1); mk(cf::cbTrili, 1);
    mk(cf::cbKS0, 2 * Dv_tiles); mk(cf::cbQS0, 2 * Dv_tiles); mk(cf::cbRhs, 2 * Dv_tiles); mk(cf::cbU, 2 * Dv_tiles);
    mk(cf::cbP, 2); mk(cf::cbP2, 2); mk(cf::cbInv, 2); mk(cf::cbIpp, 2); mk(cf::cbSn, state_tiles);
    mk(cf::cbT0, 2 * Dv_tiles); mk(cf::cbT1, 2 * Dv_tiles); mk(cf::cbOutput, 2 * Dv_tiles);

    auto* k_buf = inputs.k.buffer(); auto* q_buf = inputs.q.buffer(); auto* v_buf = inputs.v.buffer();
    auto* z_buf = inputs.z.buffer(); auto* kdec_buf = inputs.Kdec.buffer(); auto* kit_buf = inputs.KiT.buffer();
    auto* qd_buf = inputs.Qd.buffer(); auto* dcol_buf = inputs.dcol.buffer(); auto* beta_buf = inputs.betacol.buffer();
    auto* dlast_buf = inputs.dlast.buffer(); auto* state_buf = inputs.recurrent_state.buffer();
    auto* normw_buf = inputs.norm_weight.buffer();

    std::vector<uint32_t> reader_ct = {
        cf::cbK, cf::cbQ, cf::cbV, cf::cbZ, cf::cbKdec, cf::cbKiT, cf::cbQd, cf::cbDcol, cf::cbBetacol, cf::cbDlast,
        cf::cbNormW, cf::cbScaler, cf::cbEps, cf::cbStateInit, cf::cbIdent, cf::cbTrils, cf::cbTrili,
        Dk_tiles, Dv_tiles, nC, state_tiles};
    TensorAccessorArgs(k_buf).append_to(reader_ct);
    auto reader_kernel = CreateKernel(program, cf::kReaderPath, all_cores, ReaderDataMovementConfig(reader_ct));

    std::vector<uint32_t> compute_ct = {
        cf::cbK, cf::cbQ, cf::cbV, cf::cbZ, cf::cbKdec, cf::cbKiT, cf::cbQd, cf::cbDcol, cf::cbBetacol, cf::cbDlast,
        cf::cbStateInit, cf::cbIdent, cf::cbTrils, cf::cbTrili, cf::cbNormW, cf::cbScaler, cf::cbEps, cf::cbOutput,
        cf::cbKS0, cf::cbQS0, cf::cbRhs, cf::cbU, cf::cbP, cf::cbP2, cf::cbInv, cf::cbIpp, cf::cbSn, cf::cbT0, cf::cbT1,
        cf::cbStateA, cf::cbStateB, Dk_tiles, Dv_tiles, nC};
    auto compute_kernel = CreateKernel(program, cf::kComputePath, all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                      .math_approx_mode = false, .compile_args = compute_ct});

    auto* out_buf = outputs[0].buffer();
    auto* state_out_buf = outputs[1].buffer();
    std::vector<uint32_t> writer_ct = {cf::cbOutput, cf::cbStateA, cf::cbStateB, Dv_tiles, state_tiles, nC};
    TensorAccessorArgs(out_buf).append_to(writer_ct);
    auto writer_kernel = CreateKernel(program, cf::kWriterPath, all_cores, WriterDataMovementConfig(writer_ct));

    for (uint32_t h = 0; h < H; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};
        SetRuntimeArgs(program, reader_kernel, core, {
            k_buf->address(), q_buf->address(), v_buf->address(), z_buf->address(),
            kdec_buf->address(), kit_buf->address(), qd_buf->address(), dcol_buf->address(),
            beta_buf->address(), dlast_buf->address(), state_buf->address(), normw_buf->address(),
            h,                       // head_idx
            h * nC,                  // row-tile base for [Hv*Sp,D] tensors (per chunk: +c)
            h * Dk_tiles * nC,       // KiT row-base: head h occupies Dk_tiles row-tiles * nC col-tiles
            h * state_tiles,         // state base tile
        });
        SetRuntimeArgs(program, writer_kernel, core, {
            out_buf->address(), state_out_buf->address(),
            h * nC,                  // output row-tile base
            h * state_tiles,         // state_out base tile
        });
    }

    return cached_program_t{std::move(program),
        {.reader_kernel_id = reader_kernel, .compute_kernel_id = compute_kernel,
         .writer_kernel_id = writer_kernel, .all_cores = all_cores}};
}

void DeltaNetPrefillChunkedProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const operation_attributes_t& attrs, const tensor_args_t& inputs,
    tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;
    auto& program = cached.program;
    auto& shared = cached.shared_variables;
    auto* device = inputs.recurrent_state.device();
    auto grid = device->compute_with_storage_grid_size();
    auto& rrt = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& wrt = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (uint32_t h = 0; h < attrs.num_heads; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};
        auto& r = rrt[core.x][core.y];
        r[0] = inputs.k.buffer()->address(); r[1] = inputs.q.buffer()->address(); r[2] = inputs.v.buffer()->address();
        r[3] = inputs.z.buffer()->address(); r[4] = inputs.Kdec.buffer()->address(); r[5] = inputs.KiT.buffer()->address();
        r[6] = inputs.Qd.buffer()->address(); r[7] = inputs.dcol.buffer()->address(); r[8] = inputs.betacol.buffer()->address();
        r[9] = inputs.dlast.buffer()->address(); r[10] = inputs.recurrent_state.buffer()->address();
        r[11] = inputs.norm_weight.buffer()->address();
        auto& w = wrt[core.x][core.y];
        w[0] = outputs[0].buffer()->address(); w[1] = outputs[1].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet
