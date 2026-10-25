// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/sigmoid_gated_rms_norm_program_factory.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts SigmoidGatedRmsNormProgramFactory::create_program_artifacts(
    const SigmoidGatedRmsNormParams& attrs, const SigmoidGatedRmsNormInputs& in, std::vector<Tensor>& outputs) {
    const auto& input = in.input.mesh_tensor();
    const auto& gate = in.gate.mesh_tensor();
    const auto& weight = in.weight.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = input.device();

    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t total = attrs.batch * attrs.num_heads * Mt;
    const uint32_t Ha = attrs.activated_heads;
    const bool has_activated = Ha > 0;
    // Use the fewest workers that preserve the all-core maximum items/worker.
    const auto grid = device.compute_with_storage_grid_size();
    // Core i runs items first[i] + k * stride[i], k < count[i].
    std::vector<CoreCoord> core_list;
    std::vector<uint32_t> first_item;
    std::vector<uint32_t> item_stride;
    std::vector<uint32_t> item_count;
    const auto& distribution = in.input.buffer()->buffer_distribution_spec();
    if (has_activated || distribution.has_value()) {
        // Round-robin: core i of n takes items i, i + n, ... Items of one class (sigmoid or activated) are contiguous
        // in wi for one batch, so each core gets an even share of both. An input sharded one item per shard,
        // round-robin, puts item wi on its shard's core: every core then reads its inputs from its own L1 (and its
        // activated gate, when that is sharded the same way).
        if (distribution.has_value()) {
            core_list = distribution->cores();
        } else {
            const uint32_t per_core = tt::div_up(total, grid.x * grid.y);
            const uint32_t n = tt::div_up(total, per_core);
            for (uint32_t i = 0; i < n; i++) {
                core_list.emplace_back(i % grid.x, i / grid.x);
            }
        }
        const uint32_t n = std::min<uint32_t>(core_list.size(), total);
        core_list.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            first_item.push_back(i);
            item_stride.push_back(n);
            item_count.push_back((total - i + n - 1) / n);
        }
    } else {
        const uint32_t per_core = tt::div_up(total, grid.x * grid.y);
        const uint32_t rms_core_limit = tt::div_up(total, per_core);
        const auto dist = kda_factory_detail::distribute_prep(grid, total, rms_core_limit);
        core_list = dist.cores;
        first_item = dist.wi_start;
        item_stride.assign(core_list.size(), 1);
        item_count = dist.wi_count;
    }
    const auto cores = kda_factory_detail::merged_core_set(core_list);
    const uint32_t num_cores = core_list.size();
    const uint32_t max_items_per_core = *std::max_element(item_count.begin(), item_count.end());
    // A core runs its items in groups of up to max_group_items, each stage over the whole group. A group's inputs and
    // FP32 intermediates stay in L1 until the group ends.
    constexpr uint32_t max_group_items = 4;
    const uint32_t group_items = std::max<uint32_t>(1, std::min(max_group_items, max_items_per_core));

    // A core runs the items whose gate still needs its sigmoid (heads [Ha, H)) first, then those whose gate arrives
    // activated (heads [0, Ha)) (norm_items.hpp). The most of each a core's group holds sizes their buffers.
    std::vector<uint32_t> raw_count(num_cores, 0);
    uint32_t group_raw = 1;
    uint32_t group_act = 1;
    for (uint32_t i = 0; i < num_cores; i++) {
        for (uint32_t k = 0; k < item_count[i]; k++) {
            const uint32_t wi = first_item[i] + k * item_stride[i];
            raw_count[i] += (wi / Mt % attrs.num_heads >= Ha) ? 1u : 0u;
        }
        for (uint32_t first = 0; first < item_count[i]; first += group_items) {
            const uint32_t items = std::min(group_items, item_count[i] - first);
            const uint32_t raw = first >= raw_count[i] ? 0 : std::min(items, raw_count[i] - first);
            group_raw = std::max(group_raw, raw);
            group_act = std::max(group_act, items - raw);
        }
    }

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::DFBSpecName X_DFB{"x"};
    const m2::DFBSpecName GATE_DFB{"gate"};
    const m2::DFBSpecName WEIGHT_DFB{"weight"};
    const m2::DFBSpecName TMP_DFB{"tmp"};
    const m2::DFBSpecName STATS_DFB{"stats"};
    const m2::DFBSpecName INV_DFB{"inv"};
    const m2::DFBSpecName NORM_DFB{"norm"};
    const m2::DFBSpecName SIG_DFB{"sig"};
    const m2::DFBSpecName ACTIVATED_DFB{"activated"};
    const m2::DFBSpecName OUT_DFB{"out"};
    const m2::DFBSpecName SCALER_DFB{"scaler"};
    const m2::DFBSpecName EPS_DFB{"epsilon"};

    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName GATE{"gate"};
    const m2::TensorParamName ACTIVATED{"activated_gate"};
    const m2::TensorParamName WEIGHT{"weight"};
    const m2::TensorParamName OUTPUT{"output"};

    const auto input_format = datatype_to_dataformat_converter(input.dtype());
    const auto output_format = datatype_to_dataformat_converter(attrs.output_dtype);

    auto make_dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };

    m2::Group<m2::DataflowBufferSpec> dfbs = {
        make_dfb(X_DFB, std::max<uint32_t>(group_items, 2) * Vt, input_format),
        make_dfb(GATE_DFB, std::max<uint32_t>(group_items, 2) * Vt, tt::DataFormat::Float16_b),
        make_dfb(WEIGHT_DFB, Vt, tt::DataFormat::Float16_b),
        make_dfb(TMP_DFB, group_items * Vt, tt::DataFormat::Float32),
        make_dfb(STATS_DFB, group_items, tt::DataFormat::Float32),
        make_dfb(INV_DFB, group_items, tt::DataFormat::Float32),
        make_dfb(NORM_DFB, group_items * Vt, tt::DataFormat::Float32),
        make_dfb(SIG_DFB, (has_activated ? group_raw : group_items) * Vt, tt::DataFormat::Float32),
        make_dfb(OUT_DFB, 2 * Vt, output_format),
        make_dfb(SCALER_DFB, 1, tt::DataFormat::Float32),
        make_dfb(EPS_DFB, 1, tt::DataFormat::Float16_b),
    };

    if (has_activated) {
        dfbs.push_back(make_dfb(ACTIVATED_DFB, group_act * Vt, tt::DataFormat::Float32));
    }
    m2::Table<std::string, std::string> defines;
    if (has_activated) {
        defines.emplace("ACTIVATED_GATE", "1");
    }

    uint32_t eps_bits = 0;
    std::memcpy(&eps_bits, &attrs.epsilon, sizeof(float));

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/dataflow/"
            "reader_sigmoid_gated_rms_norm.cpp",
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {
                m2::DFBBinding{X_DFB, "x", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{WEIGHT_DFB, "weight", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCALER_DFB, "scaler", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{EPS_DFB, "epsilon", m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{INPUT, "input"},
                m2::TensorBinding{WEIGHT, "weight"},
            },
        .compile_time_args =
            {{"Vt", Vt},
             {"epsilon_bits", eps_bits},
             {"G", group_items},
             {"H", attrs.num_heads},
             {"Ha", Ha},
             {"Mt", Mt},
             {"max_items", max_items_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_item", "item_stride", "item_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/dataflow/"
            "writer_sigmoid_gated_rms_norm.cpp",
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {
                m2::DFBBinding{GATE_DFB, "gate", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{OUT_DFB, "out", m2::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = {m2::TensorBinding{GATE, "gate"}, m2::TensorBinding{OUTPUT, "output"}},
        .compile_time_args =
            {{"Vt", Vt},
             {"H", attrs.num_heads},
             {"Ha", Ha},
             {"Mt", Mt},
             {"gate_row_tiles", static_cast<uint32_t>(in.gate.padded_shape()[-1]) / TILE_WIDTH},
             {"gate_offset_tiles", attrs.gate_column_offset / TILE_WIDTH},
             {"G", group_items},
             {"max_items", max_items_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_item", "item_stride", "item_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    auto& unpack_modes = compute_hw.unpack_modes;
    unpack_modes[TMP_DFB] = UnpackMode::UnpackToSrc;
    unpack_modes[STATS_DFB] = UnpackMode::UnpackToSrc;
    unpack_modes[INV_DFB] = UnpackMode::UnpackToSrc;
    unpack_modes[NORM_DFB] = UnpackMode::UnpackToSrc;
    unpack_modes[SIG_DFB] = UnpackMode::UnpackToSrc;
    unpack_modes[SCALER_DFB] = UnpackMode::UnpackToSrc;
    if (input_format == tt::DataFormat::Float32) {
        unpack_modes[X_DFB] = UnpackMode::UnpackToSrc;
    }
    if (has_activated) {
        unpack_modes[ACTIVATED_DFB] = UnpackMode::UnpackToSrc;
    }

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/compute/"
            "sigmoid_gated_rms_norm.cpp",
        .compiler_options = {.defines = defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{X_DFB, "x", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{GATE_DFB, "gate", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{WEIGHT_DFB, "weight", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{TMP_DFB, "tmp", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{TMP_DFB, "tmp", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STATS_DFB, "stats", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STATS_DFB, "stats", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{INV_DFB, "inv", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{INV_DFB, "inv", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{NORM_DFB, "norm", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{NORM_DFB, "norm", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SIG_DFB, "sig", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SIG_DFB, "sig", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{OUT_DFB, "out", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCALER_DFB, "scaler", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{EPS_DFB, "epsilon", m2::DFBEndpointType::CONSUMER},
            },
        .compile_time_args = {{"Vt", Vt}, {"G", group_items}},
        .runtime_arg_schema = {.runtime_arg_names = {"raw_count", "act_count"}},
        .hw_config = std::move(compute_hw),
    };

    if (has_activated) {
        // The reader reads the activated gate once the group's inputs have landed: every core's inputs first.
        reader.dfb_bindings.push_back(m2::DFBBinding{ACTIVATED_DFB, "activated", m2::DFBEndpointType::PRODUCER});
        reader.tensor_bindings.push_back(m2::TensorBinding{ACTIVATED, "activated_gate"});
        compute.dfb_bindings.push_back(m2::DFBBinding{ACTIVATED_DFB, "activated", m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < num_cores; i++) {
        const auto& core = core_list[i];
        const uint32_t act_count = item_count[i] - raw_count[i];
        const std::initializer_list<std::pair<std::string, uint32_t>> items = {
            {"first_item", first_item[i]}, {"item_stride", item_stride[i]}, {"item_count", item_count[i]}};
        m2::AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, items);
        m2::AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, items);
        m2::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"raw_count", raw_count[i]}, {"act_count", act_count}});
    }

    m2::ProgramSpec spec{
        .name = "sigmoid_gated_rms_norm",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = GATE, .spec = gate.tensor_spec()},
                m2::TensorParameter{.unique_id = WEIGHT, .spec = weight.tensor_spec()},
                m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = cores,
                },
            },
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {INPUT, input},
        {GATE, gate},
        {WEIGHT, weight},
        {OUTPUT, output},
    };
    if (has_activated) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = ACTIVATED, .spec = in.activated_gate->mesh_tensor().tensor_spec()});
        run_args.tensor_args.emplace(ACTIVATED, in.activated_gate->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
