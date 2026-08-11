// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan_program_factory.hpp"

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
namespace {
namespace scan_cb {
constexpr uint32_t state = tt::CBIndex::c_8;
constexpr uint32_t t_inv = tt::CBIndex::c_13;
constexpr uint32_t v_beta = tt::CBIndex::c_17;
constexpr uint32_t kd = tt::CBIndex::c_18;
constexpr uint32_t q_decay = tt::CBIndex::c_19;
constexpr uint32_t intra = tt::CBIndex::c_20;
constexpr uint32_t state_two = tt::CBIndex::c_21;
constexpr uint32_t value_new = tt::CBIndex::c_22;
constexpr uint32_t final_decay = tt::CBIndex::c_11;
constexpr uint32_t output = tt::CBIndex::c_16;
constexpr uint32_t output_intermediate = tt::CBIndex::c_23;
constexpr uint32_t k_decay_transposed = tt::CBIndex::c_24;
constexpr uint32_t state_update = tt::CBIndex::c_25;
constexpr uint32_t state_temporary = tt::CBIndex::c_26;
constexpr uint32_t final_state = tt::CBIndex::c_27;
constexpr uint32_t scratch = tt::CBIndex::c_28;
constexpr uint32_t summary_raw = tt::CBIndex::c_29;
constexpr uint32_t state_three = tt::CBIndex::c_31;
}  // namespace scan_cb

struct ScanWorkDistribution {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> head;
    std::vector<uint32_t> value_block;
    uint32_t value_tiles_per_core = 1;
    CoreRangeSet core_set;
};

ScanWorkDistribution distribute_scan(CoreCoord grid, uint32_t batch_heads, uint32_t value_tiles, bool summary) {
    const uint32_t num_cores = grid.x * grid.y;
    TT_FATAL(batch_heads <= num_cores, "KDA recurrent scan heads {} exceed compute cores {}", batch_heads, num_cores);
    uint32_t value_blocks = 1;
    if (!summary && batch_heads <= 8) {
        for (uint32_t candidate = value_tiles; candidate >= 1; --candidate) {
            if (value_tiles % candidate == 0 && batch_heads * candidate <= num_cores) {
                value_blocks = candidate;
                break;
            }
        }
    }
    ScanWorkDistribution result;
    result.value_tiles_per_core = value_tiles / value_blocks;
    std::set<CoreRange> ranges;
    for (uint32_t index = 0; index < batch_heads * value_blocks; ++index) {
        const CoreCoord core{index % grid.x, index / grid.x};
        result.cores.push_back(core);
        result.head.push_back(index / value_blocks);
        result.value_block.push_back(index % value_blocks);
        ranges.insert(CoreRange{core, core});
    }
    result.core_set = CoreRangeSet{ranges};
    return result;
}
}  // namespace

ProgramDescriptor RecurrentChunkScanProgramFactory::create_descriptor(
    const RecurrentChunkScanParams& attrs, const RecurrentChunkScanInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.batch_heads;
    const uint32_t NC = attrs.num_chunks;
    constexpr uint32_t Ct = 1;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt_full = attrs.value_dim / TILE_WIDTH;
    const bool summary = attrs.mode == RecurrentChunkScanMode::SUMMARY;
    const uint32_t initial_state_mode = summary ? 1U : 0U;
    const auto output_format = summary ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    auto* device = in.v_beta.device();
    const auto distribution = distribute_scan(device->compute_with_storage_grid_size(), BH, Vt_full, summary);
    const auto& cores = distribution.core_set;
    const uint32_t Vt = distribution.value_tiles_per_core;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch = std::max({cc, ck, cv, kv, kc});

    ProgramDescriptor descriptor;
    const auto add_cb =
        [&](uint32_t index, uint32_t tiles, uint32_t buffers = 1, tt::DataFormat format = tt::DataFormat::Float32) {
            const uint32_t tile_size = tt::tile_size(format);
            descriptor.cbs.push_back(CBDescriptor{
                .total_size = tiles * buffers * tile_size,
                .core_ranges = cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = tile_size}}}});
        };
    const auto input_format = [](const Tensor& tensor) {
        return tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    };
    add_cb(scan_cb::v_beta, cv, 1, input_format(in.v_beta));
    add_cb(scan_cb::kd, ck, 1, input_format(in.kd));
    add_cb(scan_cb::k_decay_transposed, kc, 1, input_format(in.k_dec_t));
    add_cb(scan_cb::final_decay, Kt, 1, input_format(in.final_decay));
    add_cb(scan_cb::t_inv, cc, 1, input_format(in.t_inv));
    add_cb(scan_cb::state, kv);
    add_cb(scan_cb::state_two, kv);
    add_cb(scan_cb::state_three, kv);
    add_cb(scan_cb::final_state, kv);
    add_cb(scan_cb::value_new, cv);
    add_cb(scan_cb::state_update, kv);
    add_cb(scan_cb::state_temporary, kv);
    add_cb(scan_cb::scratch, scratch);
    if (summary) {
        add_cb(scan_cb::q_decay, kv);
        add_cb(scan_cb::intra, kv);
        add_cb(scan_cb::output_intermediate, kv);
        add_cb(scan_cb::summary_raw, kv);
        add_cb(scan_cb::output, kv, 1, output_format);
    } else {
        add_cb(scan_cb::q_decay, ck, 1, input_format(in.q_decay));
        add_cb(scan_cb::intra, cc, 1, input_format(in.intra));
        add_cb(scan_cb::output_intermediate, cv);
        add_cb(scan_cb::output, cv, 2, output_format);
    }

    const std::vector<uint32_t> compile_args = {
        Ct, Kt, Vt, initial_state_mode, Vt_full, summary ? 1U : 0U, summary ? 1U : 0U};
    std::vector<uint32_t> reader_compile_args = compile_args;
    TensorAccessorArgs(*in.v_beta.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.kd.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.q_decay.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.intra.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.k_dec_t.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.final_decay.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.t_inv.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr)
        .append_to(reader_compile_args);
    std::vector<uint32_t> writer_compile_args = compile_args;
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_compile_args);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_compile_args);

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
        "reader_recurrent_chunk_scan.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_compile_args;
    reader.config = ReaderConfigDescriptor{};
    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
        "writer_recurrent_chunk_scan.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_compile_args;
    writer.config = WriterConfigDescriptor{};
    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/"
        "recurrent_chunk_scan.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compile_args;
    compute.config = kda_factory_detail::kda_compute_cfg(device->arch(), attrs.compute_kernel_config, true);

    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t head = distribution.head[index];
        const uint32_t value_block = distribution.value_block[index];
        reader.emplace_runtime_args(
            core,
            {head,
             value_block,
             NC,
             in.v_beta.buffer(),
             in.kd.buffer(),
             in.q_decay.buffer(),
             in.intra.buffer(),
             in.k_dec_t.buffer(),
             in.final_decay.buffer(),
             in.t_inv.buffer(),
             in.initial_state.has_value() ? in.initial_state->buffer() : nullptr});
        writer.emplace_runtime_args(core, {head, value_block, NC, outputs[0].buffer(), outputs[1].buffer()});
        compute.emplace_runtime_args(core, {NC});
    }
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::prim
