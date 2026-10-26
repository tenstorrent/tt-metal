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
    // Use the fewest workers that preserve the all-core maximum items/worker.
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t max_items_per_core = tt::div_up(total, grid.x * grid.y);
    const uint32_t rms_core_limit = tt::div_up(total, max_items_per_core);
    auto dist = kda_factory_detail::distribute_prep(grid, total, rms_core_limit);
    const auto& cores = dist.core_set;
    // A core runs its items in groups of up to max_group_items, each stage over the whole group. A group's inputs and
    // FP32 intermediates stay in L1 until the group ends.
    constexpr uint32_t max_group_items = 4;
    const uint32_t group_items = std::max<uint32_t>(1, std::min(max_group_items, max_items_per_core));

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
    const m2::DFBSpecName OUT_DFB{"out"};
    const m2::DFBSpecName SCALER_DFB{"scaler"};
    const m2::DFBSpecName EPS_DFB{"epsilon"};

    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName GATE{"gate"};
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
        make_dfb(SIG_DFB, group_items * Vt, tt::DataFormat::Float32),
        make_dfb(OUT_DFB, 2 * Vt, output_format),
        make_dfb(SCALER_DFB, 1, tt::DataFormat::Float32),
        make_dfb(EPS_DFB, 1, tt::DataFormat::Float16_b),
    };

    m2::Table<std::string, std::string> defines;
    const bool has_prefetch = in.prefetch_jobs.has_value();
    std::vector<uint32_t> prefetch_job(dist.cores.size(), 0);
    const m2::DFBSpecName JOB_ROW_DFB{"job_row"};
    if (has_prefetch) {
        defines.emplace("L1_PREFETCH", "1");
        // The reader's copy of its job row: below the prefetch region, so no job's multicast can land on it.
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = JOB_ROW_DFB, .entry_size = l1_prefetch_job_words * sizeof(uint32_t), .num_entries = 1});
        for (uint32_t j = 0; j < attrs.prefetch_cores.size(); j++) {
            const auto it = std::find(dist.cores.begin(), dist.cores.end(), attrs.prefetch_cores[j]);
            TT_FATAL(
                it != dist.cores.end(),
                "sigmoid_gated_rms_norm: prefetch core {} is not a norm core",
                attrs.prefetch_cores[j].str());
            TT_FATAL(prefetch_job[it - dist.cores.begin()] == 0, "sigmoid_gated_rms_norm: one prefetch job per core");
            prefetch_job[it - dist.cores.begin()] = j + 1;
        }
        // The jobs write above the norm's circular buffers, which depend only on the attributes, so this program
        // checks them once; the bound below every L1 buffer is checked per call (validate_on_program_cache_miss,
        // which also runs on cache hits). The walk mirrors ProgramImpl::allocate_dataflow_buffers: in DFB order from
        // the L1 allocator base, each start aligned to the allocator's DRAM alignment, entry_size * num_entries
        // (DataflowBuffer::total_size()) long.
        const uint32_t cb_alignment = device.allocator()->get_alignment(BufferType::DRAM);
        uint32_t dfb_end = device.allocator()->get_base_allocator_addr(HalMemType::L1);
        for (const auto& dfb : dfbs) {
            dfb_end = tt::round_up(dfb_end, cb_alignment) + (dfb.entry_size * dfb.num_entries);
        }
        dfb_end = tt::round_up(dfb_end, cb_alignment);
        TT_FATAL(
            attrs.prefetch_l1_begin >= dfb_end,
            "sigmoid_gated_rms_norm: prefetch region [{}, {}) overlaps the norm's circular buffers, which end at {}",
            attrs.prefetch_l1_begin,
            attrs.prefetch_l1_end,
            dfb_end);
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
        .compile_time_args = {{"Vt", Vt}, {"epsilon_bits", eps_bits}, {"G", group_items}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count", "prefetch_job"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    const m2::TensorParamName PREFETCH_SOURCE{"prefetch_source"};
    const m2::TensorParamName PREFETCH_JOBS{"prefetch_jobs"};
    if (has_prefetch) {
        reader.tensor_bindings.push_back(m2::TensorBinding{PREFETCH_SOURCE, "prefetch_source"});
        reader.tensor_bindings.push_back(m2::TensorBinding{PREFETCH_JOBS, "prefetch_jobs"});
        reader.dfb_bindings.push_back(m2::DFBBinding{JOB_ROW_DFB, "job_row", m2::DFBEndpointType::PRODUCER});
        reader.dfb_bindings.push_back(m2::DFBBinding{JOB_ROW_DFB, "job_row", m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/dataflow/"
            "writer_sigmoid_gated_rms_norm.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{GATE_DFB, "gate", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{OUT_DFB, "out", m2::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = {m2::TensorBinding{GATE, "gate"}, m2::TensorBinding{OUTPUT, "output"}},
        .compile_time_args =
            {{"Vt", Vt},
             {"H", attrs.num_heads},
             {"Mt", Mt},
             {"gate_row_tiles", static_cast<uint32_t>(in.gate.padded_shape()[-1]) / TILE_WIDTH},
             {"gate_offset_tiles", attrs.gate_column_offset / TILE_WIDTH},
             {"G", group_items}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
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

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/compute/"
            "sigmoid_gated_rms_norm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
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
        .runtime_arg_schema = {.runtime_arg_names = {"wi_count"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); i++) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}, {"prefetch_job", prefetch_job[i]}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"wi_count", dist.wi_count[i]}});
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
    if (has_prefetch) {
        for (const auto& [name, tensor] :
             {std::pair{PREFETCH_SOURCE, &*in.prefetch_source}, std::pair{PREFETCH_JOBS, &*in.prefetch_jobs}}) {
            spec.tensor_parameters.push_back(
                m2::TensorParameter{.unique_id = name, .spec = tensor->mesh_tensor().tensor_spec()});
            run_args.tensor_args.emplace(name, tensor->mesh_tensor());
        }
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
