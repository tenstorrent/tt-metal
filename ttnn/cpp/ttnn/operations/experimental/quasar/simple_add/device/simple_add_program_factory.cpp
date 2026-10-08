// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_add_device_operation.hpp"

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
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/quasar/simple_add/device/kernels/";
constexpr uint32_t kEntriesPerThread = 2;      // per Tensix tile counter: double buffering
constexpr uint32_t kQuasarComputeThreads = 4;  // every Tensix engine of a Neo cluster
}  // namespace

ProgramArtifacts SimpleAddDeviceOperation::SingleNodeProgramFactory::create_program_artifacts(
    const SimpleAddParams& /*operation_attributes*/, const SimpleAddInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_a.mesh_tensor();
    const auto& b = tensor_args.input_b.mesh_tensor();
    const auto& c = tensor_return_value.mesh_tensor();

    const m2::DFBSpecName IN0_DFB{"in0"};
    const m2::DFBSpecName IN1_DFB{"in1"};
    const m2::DFBSpecName OUT_DFB{"out"};
    const m2::TensorParamName A{"a"};
    const m2::TensorParamName B{"b"};
    const m2::TensorParamName C{"c"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    // One Neo cluster: the whole op runs on node (0, 0). On Quasar the compute kernel runs on all 4 Tensix
    // engines of that cluster; Wormhole/Blackhole compute kernels are single-threaded.
    const m2::NodeCoord node{0, 0};
    const bool is_quasar = tensor_args.input_a.device()->arch() == tt::ARCH::QUASAR;
    const uint32_t compute_threads = is_quasar ? kQuasarComputeThreads : 1u;

    const DataFormat data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t tile_bytes = tile_size(data_format);
    const uint32_t num_tiles = a.physical_volume() / a.tensor_spec().tile().get_tile_hw();

    // Each DFB has one DM endpoint and compute_threads Tensix endpoints, so it is split into compute_threads
    // tile counters of kEntriesPerThread slots each: num_entries must be divisible by max(producers, consumers).
    auto make_dfb = [&](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes,
            .num_entries = kEntriesPerThread * compute_threads,
            .data_format_metadata = data_format,
        };
    };

    // The kernels use explicit reserve_back/push_back sync, so DFB implicit sync is off (Quasar only).
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{std::string(kKernelDir) + "dataflow/reader_simple_add.cpp"},
        .dfb_bindings = {m2::ProducerOf(IN0_DFB, "in0"), m2::ProducerOf(IN1_DFB, "in1")},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = A, .accessor_name = "a"},
             m2::TensorBinding{.tensor_parameter_name = B, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{std::string(kKernelDir) + "dataflow/writer_simple_add.cpp"},
        .dfb_bindings = {m2::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = C, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{std::string(kKernelDir) + "compute/simple_add.cpp"},
        .num_threads = compute_threads,
        .dfb_bindings =
            {m2::ConsumerOf(IN0_DFB, "in0"), m2::ConsumerOf(IN1_DFB, "in1"), m2::ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"num_tiles", num_tiles}},
        .hw_config = m2::ComputeHardwareConfig{},
    };

    m2::ProgramSpec spec{
        .name = "simple_add",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {make_dfb(IN0_DFB), make_dfb(IN1_DFB), make_dfb(OUT_DFB)},
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = A, .spec = a.tensor_spec()},
             m2::TensorParameter{.unique_id = B, .spec = b.tensor_spec()},
             m2::TensorParameter{.unique_id = C, .spec = c.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        m2::KernelRunArgs{
            .kernel = READER, .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}})},
        m2::KernelRunArgs{
            .kernel = WRITER, .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}})},
        m2::KernelRunArgs{.kernel = COMPUTE},
    };
    run_args.tensor_args = {
        {A, m2::TensorArgument{a}},
        {B, m2::TensorArgument{b}},
        {C, m2::TensorArgument{c}},
    };

    return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
