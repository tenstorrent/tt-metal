// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_add_device_operation.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>
#include <filesystem>

namespace ttnn::prim::qsr {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

namespace {
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/quasar/simple_add/device/kernels/";
constexpr uint32_t kEntriesPerThread = 2;      // per tile counter: double buffering
constexpr uint32_t kQuasarComputeThreads = 4;  // every Tensix engine of a Neo cluster
// Reader and writer DM cores. Each must divide kQuasarComputeThreads so every DM thread round-robins the same
// number of Tensix tile counters: with 4 readers, reader t feeds only Tensix t; with 2 writers, writer t drains
// Tensix t and t + 2. Quasar keeps DM0 (ISR) and DM1 (remapper) for itself, leaving 6 DM cores (DM2..DM7):
// 4 readers + 2 writers use all of them.
constexpr uint32_t kQuasarReaderThreads = 4;
constexpr uint32_t kQuasarWriterThreads = 2;
static_assert(kQuasarComputeThreads % kQuasarReaderThreads == 0);
static_assert(kQuasarComputeThreads % kQuasarWriterThreads == 0);
static_assert(kQuasarReaderThreads + kQuasarWriterThreads <= 6, "only DM2..DM7 can run user kernels");
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

    // One Neo cluster: the whole op runs on node (0, 0). On Quasar the reader runs on 4 DM cores, the writer on 2
    // and the compute kernel on all 4 Tensix engines. Wormhole/Blackhole kernels are single-threaded.
    const m2::NodeCoord node{0, 0};
    const bool is_quasar = tensor_args.input_a.device()->arch() == tt::ARCH::QUASAR;
    const uint32_t compute_threads = is_quasar ? kQuasarComputeThreads : 1u;
    const uint32_t reader_threads = is_quasar ? kQuasarReaderThreads : 1u;

    const DataFormat data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t tile_bytes = tile_size(data_format);
    const uint32_t num_tiles = a.physical_volume() / a.tensor_spec().tile().get_tile_hw();
    // Every writer thread must write at least one tile: with implicit sync, finish() skips the end-of-kernel credit
    // flush on a thread with no transactions while its siblings wait for it there. So no more writers than tiles.
    const uint32_t writer_threads = is_quasar ? std::min(kQuasarWriterThreads, num_tiles) : 1u;

    // A STRIDED DFB is split into max(producers, consumers) tile counters of kEntriesPerThread slots each, so
    // num_entries is kEntriesPerThread * max(producers, consumers): in0/in1 are reader_threads -> compute_threads,
    // out is compute_threads -> writer_threads.
    const uint32_t in_counters = std::max(reader_threads, compute_threads);
    const uint32_t out_counters = std::max(compute_threads, writer_threads);
    auto make_dfb = [&](const m2::DFBSpecName& name, uint32_t num_counters) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes,
            .num_entries = kEntriesPerThread * num_counters,
            .data_format_metadata = data_format,
        };
    };

    // DFB implicit sync (Quasar only): the reader and writer issue transaction-id tagged NoC reads/writes and the
    // DM0 ISR posts/acks the tile-counter credits. On the craq-sim Quasar simulator a partial transaction-id batch
    // on the reader side, whose credits finish() posts by hand, makes a later Tensix push_back on out go missing
    // and hangs the writer. So the readers process num_input_tiles = num_tiles rounded up to a multiple of
    // reader_threads, which keeps every reader batch full (the in0/in1 batch is reader_threads reads here): the
    // extra tiles carry filler data, compute pops them without producing output, and the writers still see exactly
    // num_tiles tiles.
    const bool implicit_sync = is_quasar;
    const uint32_t num_input_tiles =
        implicit_sync ? (num_tiles + reader_threads - 1) / reader_threads * reader_threads : num_tiles;
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{std::string(kKernelDir) + "dataflow/reader_simple_add.cpp"},
        .num_threads = reader_threads,
        .dfb_bindings = {m2::ProducerOf(IN0_DFB, "in0"), m2::ProducerOf(IN1_DFB, "in1")},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = A, .accessor_name = "a"},
             m2::TensorBinding{.tensor_parameter_name = B, .accessor_name = "b"}},
        .compile_time_args = {{"implicit_sync", implicit_sync ? 1u : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "num_input_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/!implicit_sync),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{std::string(kKernelDir) + "dataflow/writer_simple_add.cpp"},
        .num_threads = writer_threads,
        .dfb_bindings = {m2::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = C, .accessor_name = "c"}},
        .compile_time_args = {{"implicit_sync", implicit_sync ? 1u : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/!implicit_sync),
    };

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{std::string(kKernelDir) + "compute/simple_add.cpp"},
        .num_threads = compute_threads,
        .dfb_bindings =
            {m2::ConsumerOf(IN0_DFB, "in0"), m2::ConsumerOf(IN1_DFB, "in1"), m2::ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"num_tiles", num_tiles}, {"num_input_tiles", num_input_tiles}},
        .hw_config = m2::ComputeHardwareConfig{},
    };

    m2::ProgramSpec spec{
        .name = "simple_add",
        .kernels = {reader, writer, compute},
        .dataflow_buffers =
            {make_dfb(IN0_DFB, in_counters), make_dfb(IN1_DFB, in_counters), make_dfb(OUT_DFB, out_counters)},
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = A, .spec = a.tensor_spec()},
             m2::TensorParameter{.unique_id = B, .spec = b.tensor_spec()},
             m2::TensorParameter{.unique_id = C, .spec = c.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        m2::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                node, {{"num_tiles", num_tiles}, {"num_input_tiles", num_input_tiles}})},
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
