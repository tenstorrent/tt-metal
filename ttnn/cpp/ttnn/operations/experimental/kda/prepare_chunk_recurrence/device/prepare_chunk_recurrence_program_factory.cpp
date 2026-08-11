// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence_program_factory.hpp"

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
namespace {
namespace prep_cb {
constexpr uint32_t q = tt::CBIndex::c_0;
constexpr uint32_t k = tt::CBIndex::c_1;
constexpr uint32_t v = tt::CBIndex::c_2;
constexpr uint32_t g = tt::CBIndex::c_3;
constexpr uint32_t beta = tt::CBIndex::c_4;
constexpr uint32_t eye = tt::CBIndex::c_5;
constexpr uint32_t tril = tt::CBIndex::c_6;
constexpr uint32_t ones = tt::CBIndex::c_7;
constexpr uint32_t state = tt::CBIndex::c_8;
constexpr uint32_t decay = tt::CBIndex::c_9;
constexpr uint32_t decay_exp = tt::CBIndex::c_10;
constexpr uint32_t decay_factor = tt::CBIndex::c_11;
constexpr uint32_t lower_mask = tt::CBIndex::c_12;
constexpr uint32_t t_inv = tt::CBIndex::c_13;
constexpr uint32_t v_beta = tt::CBIndex::c_14;
constexpr uint32_t k_beta = tt::CBIndex::c_15;
constexpr uint32_t output = tt::CBIndex::c_16;
constexpr uint32_t u = tt::CBIndex::c_17;
constexpr uint32_t w = tt::CBIndex::c_18;
constexpr uint32_t q_decay = tt::CBIndex::c_19;
constexpr uint32_t intra = tt::CBIndex::c_20;
constexpr uint32_t state_two = tt::CBIndex::c_21;
constexpr uint32_t v_new = tt::CBIndex::c_22;
constexpr uint32_t output_intermediate = tt::CBIndex::c_23;
constexpr uint32_t k_decay_transposed = tt::CBIndex::c_24;
constexpr uint32_t state_update = tt::CBIndex::c_25;
constexpr uint32_t state_temporary = tt::CBIndex::c_26;
constexpr uint32_t final_state = tt::CBIndex::c_27;
constexpr uint32_t scratch_one = tt::CBIndex::c_28;
constexpr uint32_t scratch_two = tt::CBIndex::c_29;
constexpr uint32_t scratch_three = tt::CBIndex::c_30;
constexpr uint32_t state_three = tt::CBIndex::c_31;
}  // namespace prep_cb
}  // namespace

uint32_t prepare_chunk_recurrence_cb_size_bytes(
    uint32_t chunk_size, uint32_t key_dim, uint32_t value_dim, DataType gate_dtype, uint32_t output_bf16_mask) {
    const uint32_t Ct = chunk_size / TILE_HEIGHT;
    const uint32_t Kt = key_dim / TILE_WIDTH;
    const uint32_t Vt = value_dim / TILE_WIDTH;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch = std::max({cc, ck, cv, kv, kc});
    const auto format = [&](uint32_t index) {
        return (output_bf16_mask & (1U << index)) ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
    };
    uint32_t bytes = 0;
    const auto add = [&](uint32_t tiles, uint32_t buffers = 1, tt::DataFormat data_format = tt::DataFormat::Float32) {
        bytes += tiles * buffers * tt::tile_size(data_format);
    };
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    add(ck, 1, bf16);
    add(ck, 1, bf16);
    add(cv, 1, bf16);
    add(ck, 1, tt::tt_metal::datatype_to_dataformat_converter(gate_dtype));
    add(Ct);
    add(cc);
    add(cc);
    add(cc);
    add(kv, 2);
    add(ck);
    add(ck);
    add(ck);
    add(cc);
    add(cc, 1, format(6));
    add(cv, 1, format(0));
    add(ck);
    add(cv, 2, bf16);
    add(std::max(cv, 3U));
    add(ck, 1, format(1));
    add(ck, 1, format(2));
    add(cc, 1, format(3));
    add(kv, 2);
    add(std::max(cv, Kt), 1, format(5));
    add(std::max(cv, ck));
    add(kc, 1, format(4));
    add(kv);
    add(kv);
    add(kv);
    add(scratch);
    add(scratch);
    add(scratch);
    add(kv, 2);
    return bytes;
}

namespace {}  // namespace

ProgramDescriptor PrepareChunkRecurrenceProgramFactory::create_descriptor(
    const PrepareChunkRecurrenceParams& attrs, const PrepareChunkRecurrenceInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    constexpr uint32_t chunk_size = TILE_HEIGHT;
    constexpr uint32_t Ct = 1;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch = std::max({cc, ck, cv, kv, kc});

    auto* device = in.q.device();
    const auto distribution = kda_factory_detail::distribute_prep(
        device->compute_with_storage_grid_size(), BH * NC, std::numeric_limits<uint32_t>::max());
    const auto& cores = distribution.core_set;
    uint32_t cb_size_bytes = 0;
    ProgramDescriptor descriptor;
    const auto add_cb =
        [&](uint32_t index, uint32_t tiles, uint32_t buffers = 1, tt::DataFormat format = tt::DataFormat::Float32) {
            const uint32_t tile_size = tt::tile_size(format);
            const uint32_t total_size = tiles * buffers * tile_size;
            cb_size_bytes += total_size;
            descriptor.cbs.push_back(CBDescriptor{
                .total_size = total_size,
                .core_ranges = cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = tile_size}}}});
        };
    const auto output_format = [&](uint32_t index) {
        return tt::tt_metal::datatype_to_dataformat_converter(outputs[index].dtype());
    };
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    add_cb(prep_cb::q, ck, 1, bf16);
    add_cb(prep_cb::k, ck, 1, bf16);
    add_cb(prep_cb::v, cv, 1, bf16);
    add_cb(prep_cb::g, ck, 1, tt::tt_metal::datatype_to_dataformat_converter(in.g.dtype()));
    add_cb(prep_cb::beta, Ct);
    add_cb(prep_cb::eye, cc);
    add_cb(prep_cb::tril, cc);
    add_cb(prep_cb::ones, cc);
    add_cb(prep_cb::state, kv, 2);
    add_cb(prep_cb::decay, ck);
    add_cb(prep_cb::decay_exp, ck);
    add_cb(prep_cb::decay_factor, ck);
    add_cb(prep_cb::lower_mask, cc);
    add_cb(prep_cb::t_inv, cc, 1, output_format(6));
    add_cb(prep_cb::v_beta, cv, 1, output_format(0));
    add_cb(prep_cb::k_beta, ck);
    add_cb(prep_cb::output, cv, 2, bf16);
    add_cb(prep_cb::u, std::max(cv, 3U));
    add_cb(prep_cb::w, ck, 1, output_format(1));
    add_cb(prep_cb::q_decay, ck, 1, output_format(2));
    add_cb(prep_cb::intra, cc, 1, output_format(3));
    add_cb(prep_cb::state_two, kv, 2);
    add_cb(prep_cb::v_new, std::max(cv, Kt), 1, output_format(5));
    add_cb(prep_cb::output_intermediate, std::max(cv, ck));
    add_cb(prep_cb::k_decay_transposed, kc, 1, output_format(4));
    add_cb(prep_cb::state_update, kv);
    add_cb(prep_cb::state_temporary, kv);
    add_cb(prep_cb::final_state, kv);
    add_cb(prep_cb::scratch_one, scratch);
    add_cb(prep_cb::scratch_two, scratch);
    add_cb(prep_cb::scratch_three, scratch);
    add_cb(prep_cb::state_three, kv, 2);
    TT_FATAL(
        cb_size_bytes == prepare_chunk_recurrence_cb_size_bytes(
                             chunk_size, attrs.key_dim, attrs.value_dim, in.g.dtype(), attrs.output_bf16_mask),
        "KDA prep CB size estimator is out of sync with its program factory");

    const std::vector<uint32_t> dimensions = {Ct, Kt, Vt};
    std::vector<uint32_t> reader_compile_args = dimensions;
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.eye.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.tril.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.ones.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(*in.masks.buffer()).append_to(reader_compile_args);
    reader_compile_args.push_back(1U);  // v is production-flat
    reader_compile_args.push_back(1U);  // q/k are production-flat
    reader_compile_args.push_back(1U);  // g is production-flat
    std::vector<uint32_t> writer_compile_args = dimensions;
    for (auto& output : outputs) {
        TensorAccessorArgs(*output.buffer()).append_to(writer_compile_args);
    }

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/"
        "reader_prepare_chunk_recurrence.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_compile_args;
    reader.config = ReaderConfigDescriptor{};
    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/"
        "writer_prepare_chunk_recurrence.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_compile_args;
    writer.config = WriterConfigDescriptor{};
    const auto float_bits = [](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };
    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/"
        "prepare_chunk_recurrence.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = dimensions;
    compute.compile_time_args.push_back(1U);  // Q/K normalization is unconditional
    compute.compile_time_args.push_back(float_bits(1.0F / std::sqrt(static_cast<float>(attrs.key_dim))));
    compute.compile_time_args.push_back(float_bits(1e-6F));
    compute.config = kda_factory_detail::kda_compute_cfg(device->arch(), attrs.compute_kernel_config, true);

    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t work_start = distribution.wi_start[index];
        const uint32_t work_count = distribution.wi_count[index];
        reader.emplace_runtime_args(
            core,
            {work_start,
             work_count,
             in.q.buffer(),
             in.k.buffer(),
             in.v.buffer(),
             in.g.buffer(),
             in.beta.buffer(),
             in.eye.buffer(),
             in.tril.buffer(),
             in.ones.buffer(),
             in.masks.buffer(),
             NC,
             attrs.num_heads,
             attrs.num_heads});
        writer.emplace_runtime_args(
            core,
            {work_start,
             work_count,
             outputs[0].buffer(),
             outputs[1].buffer(),
             outputs[2].buffer(),
             outputs[3].buffer(),
             outputs[4].buffer(),
             outputs[5].buffer(),
             outputs[6].buffer()});
        compute.emplace_runtime_args(core, {work_count});
    }
    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));
    return descriptor;
}

}  // namespace ttnn::experimental::prim
