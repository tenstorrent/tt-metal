// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_device_operation.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <filesystem>

namespace ttnn::prim::qsr {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

namespace {
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/quasar/routed_expert_ffn/device/kernels/";
constexpr uint32_t kStreamEntries = 2;         // per tile counter: double buffering for streamed tiles
constexpr uint32_t kQuasarComputeThreads = 4;  // every Tensix engine of a Neo cluster
}  // namespace

ProgramArtifacts RoutedExpertFfnDeviceOperation::SingleNodeProgramFactory::create_program_artifacts(
    const RoutedExpertFfnParams& /*operation_attributes*/,
    const RoutedExpertFfnInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& x = tensor_args.x.mesh_tensor();
    const auto& w_gate = tensor_args.w_gate.mesh_tensor();
    const auto& w_up = tensor_args.w_up.mesh_tensor();
    const auto& w_down = tensor_args.w_down.mesh_tensor();
    const auto& y = tensor_return_value.mesh_tensor();

    const m2::DFBSpecName X_DFB{"x"};
    const m2::DFBSpecName W_DFB{"w"};
    const m2::DFBSpecName GATE_DFB{"gate"};
    const m2::DFBSpecName UP_DFB{"up"};
    const m2::DFBSpecName ACT_DFB{"act"};
    const m2::DFBSpecName OUT_DFB{"out"};
    const m2::TensorParamName X{"x"};
    const m2::TensorParamName W_GATE{"w_gate"};
    const m2::TensorParamName W_UP{"w_up"};
    const m2::TensorParamName W_DOWN{"w_down"};
    const m2::TensorParamName Y{"y"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    // One Neo cluster: the whole op runs on node (0, 0).
    const m2::NodeCoord node{0, 0};
    const bool is_quasar = tensor_args.x.device()->arch() == tt::ARCH::QUASAR;

    const DataFormat data_format = datatype_to_dataformat_converter(x.dtype());
    const uint32_t tile_bytes = tile_size(data_format);
    const auto& tile = x.tensor_spec().tile();
    const uint32_t Mt = x.padded_shape()[-2] / tile.get_height();
    const uint32_t Kt = x.padded_shape()[-1] / tile.get_width();
    const uint32_t Ht = w_gate.padded_shape()[-1] / tile.get_width();

    // Compute threads split the tile rows of x: thread t owns rows t, t + T, t + 2T, ... Each row needs the whole
    // gate * up row, so rows are the only split with no data exchange between Tensix engines. The reader and writer
    // rotate over the threads one tile at a time, so every thread must own the same number of rows: T is the largest
    // of 4, 2, 1 that divides Mt.
    uint32_t compute_threads = 1;
    if (is_quasar) {
        compute_threads = kQuasarComputeThreads;
        while (Mt % compute_threads != 0) {
            compute_threads /= 2;
        }
    }

    auto make_dfb = [&](const m2::DFBSpecName& name, uint32_t num_entries) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes,
            .num_entries = num_entries,
            .data_format_metadata = data_format,
        };
    };

    // The kernels use explicit reserve_back/push_back sync, so DFB implicit sync is off (Quasar only).
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{std::string(kKernelDir) + "dataflow/reader_routed_expert_ffn.cpp"},
        .dfb_bindings = {m2::ProducerOf(X_DFB, "x"), m2::ProducerOf(W_DFB, "w")},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = X, .accessor_name = "x"},
             m2::TensorBinding{.tensor_parameter_name = W_GATE, .accessor_name = "w_gate"},
             m2::TensorBinding{.tensor_parameter_name = W_UP, .accessor_name = "w_up"},
             m2::TensorBinding{.tensor_parameter_name = W_DOWN, .accessor_name = "w_down"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Kt", "Ht", "compute_threads"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{std::string(kKernelDir) + "dataflow/writer_routed_expert_ffn.cpp"},
        .dfb_bindings = {m2::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = Y, .accessor_name = "y"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Kt", "compute_threads"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // gate, up and act are compute-only scratch, so compute is both their producer and their consumer.
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{std::string(kKernelDir) + "compute/routed_expert_ffn.cpp"},
        .num_threads = compute_threads,
        .dfb_bindings =
            {m2::ConsumerOf(X_DFB, "x"),
             m2::ConsumerOf(W_DFB, "w"),
             m2::ProducerOf(GATE_DFB, "gate"),
             m2::ConsumerOf(GATE_DFB, "gate"),
             m2::ProducerOf(UP_DFB, "up"),
             m2::ConsumerOf(UP_DFB, "up"),
             m2::ProducerOf(ACT_DFB, "act"),
             m2::ConsumerOf(ACT_DFB, "act"),
             m2::ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"Mt", Mt}, {"Kt", Kt}, {"Ht", Ht}},
        .hw_config = m2::ComputeHardwareConfig{},
    };

    // Every DFB is split into one tile counter per compute thread, so num_entries is per-thread entries times
    // compute_threads. x holds one tile row of x (Kt tiles) per thread for all of phase 1; act holds one tile row of
    // gate * up (Ht tiles) per thread for all of phase 2.
    m2::ProgramSpec spec{
        .name = "routed_expert_ffn",
        .kernels = {reader, writer, compute},
        .dataflow_buffers =
            {make_dfb(X_DFB, Kt * compute_threads),
             make_dfb(W_DFB, kStreamEntries * compute_threads),
             make_dfb(GATE_DFB, kStreamEntries * compute_threads),
             make_dfb(UP_DFB, kStreamEntries * compute_threads),
             make_dfb(ACT_DFB, Ht * compute_threads),
             make_dfb(OUT_DFB, kStreamEntries * compute_threads)},
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = X, .spec = x.tensor_spec()},
             m2::TensorParameter{.unique_id = W_GATE, .spec = w_gate.tensor_spec()},
             m2::TensorParameter{.unique_id = W_UP, .spec = w_up.tensor_spec()},
             m2::TensorParameter{.unique_id = W_DOWN, .spec = w_down.tensor_spec()},
             m2::TensorParameter{.unique_id = Y, .spec = y.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        m2::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                node, {{"Mt", Mt}, {"Kt", Kt}, {"Ht", Ht}, {"compute_threads", compute_threads}})},
        m2::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"Mt", Mt}, {"Kt", Kt}, {"compute_threads", compute_threads}})},
        m2::KernelRunArgs{.kernel = COMPUTE},
    };
    run_args.tensor_args = {
        {X, m2::TensorArgument{x}},
        {W_GATE, m2::TensorArgument{w_gate}},
        {W_UP, m2::TensorArgument{w_up}},
        {W_DOWN, m2::TensorArgument{w_down}},
        {Y, m2::TensorArgument{y}},
    };

    return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
