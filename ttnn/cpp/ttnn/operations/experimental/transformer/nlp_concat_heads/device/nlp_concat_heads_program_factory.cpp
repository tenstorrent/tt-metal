// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "nlp_concat_heads_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_concat_heads.cpp";
constexpr const char* SHARDED_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp";
// Private (op-owned) copy of the interleaved unary writer, ported to the Metal 2.0 binding namespaces.
// The legacy factory pointed at the shared eltwise/unary writer_unary_interleaved_start_id.cpp, which is
// still consumed positionally by ~10 legacy ops and must not be touched.
constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsProgramFactory::create_program_artifacts(
    const NlpConcatHeadsParams& /*operation_attributes*/, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    bool in_sharded = a.is_sharded();
    bool out_sharded = output.is_sharded();

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH;  // 142

    // Per output tensor args
    // Output shape is: [B, 1, s, 4544]
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH;    // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT;
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores = CoreRangeSet(), core_group_1 = CoreRangeSet(), core_group_2 = CoreRangeSet();
    bool row_major = false;
    if (in_sharded) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = a.shard_spec().value().shape[0] / a.padded_shape()[-2];
        per_tensor_tiles = a.shard_spec().value().shape[0] * a.shard_spec().value().shape[1] / TILE_HW;
        row_major = a.shard_spec().value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "nlp_concat_heads";

    // Tensor parameters: input is always read; output is always written.
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    m2::ProgramRunArgs run_args;

    if (in_sharded) {
        // ---- DFBs: src0 borrowed from the (sharded) input; out0 borrowed from the (sharded) output ----
        std::vector<m2::DataflowBufferSpec> dfbs;
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = m2::TensorParamName{"input"},
        });
        if (out_sharded) {
            dfbs.push_back(m2::DataflowBufferSpec{
                .unique_id = m2::DFBSpecName{"out0"},
                .entry_size = single_tile_size,
                .num_entries = per_tensor_tiles,
                .data_format_metadata = cb_data_format,
                .borrowed_from = m2::TensorParamName{"output"},
            });
        }
        spec.dataflow_buffers = std::move(dfbs);

        // Shared compile-time args for both sharded kernels (formerly positional CTAs 0..5).
        m2::KernelSpec::CompileTimeArgs cta = {
            {"in0_h_tiles", in0_h_tiles},
            {"head_dim_size_bytes", in0_w_tiles * single_tile_size},
            {"out_row_size_bytes", num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size},
            {"block_size", num_blocks_per_core_group_1 * in0_HtWt},
        };

        // Both kernels run the SAME sharded source; reader produces src0 (and consumes out0), writer
        // consumes the buffers. They are split across the two DM processors (reader→NCRISC, writer→BRISC)
        // so the two data-movement kernels don't collide on the same DM processor.
        auto sharded_dfb_bindings = [&]() {
            std::vector<m2::DFBBinding> b = {m2::ProducerOf(m2::DFBSpecName{"src0"}, "cb_id_in0")};
            if (out_sharded) {
                b.push_back(m2::ProducerOf(m2::DFBSpecName{"out0"}, "cb_id_out0"));
            }
            return b;
        };

        m2::KernelSpec reader{
            .unique_id = m2::KernelSpecName{"reader"},
            .source = std::filesystem::path{SHARDED_KERNEL_PATH},
            .dfb_bindings = sharded_dfb_bindings(),
            .compile_time_args = cta,
            .runtime_arg_schema =
                {.runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"}},
            .hw_config =
                m2::DataMovementHardwareConfig{
                    .gen1_config =
                        m2::DataMovementHardwareConfig::Gen1Config{
                            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                            .noc = tt::tt_metal::NOC::RISCV_1_default}},
        };

        m2::KernelSpec writer{
            .unique_id = m2::KernelSpecName{"writer"},
            .source = std::filesystem::path{SHARDED_KERNEL_PATH},
            .dfb_bindings = sharded_dfb_bindings(),
            .compile_time_args = cta,
            .runtime_arg_schema =
                {.runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"}},
            .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        };

        spec.kernels = {std::move(reader), std::move(writer)};
        spec.work_units = {m2::WorkUnitSpec{
            .name = "nlp_concat_heads",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores}};

        // Per-core runtime args. Mirror SetRuntimeArgs(program, kernel, all_cores, args) by emplacing the
        // same per-core args on every logical core in the sharded range set.
        uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;

        m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
        m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
        for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
            reader_args.runtime_arg_values.push_back(
                {core,
                 {{"nheads", nheads_first_risc},
                  {"start_read_offset_bytes", uint32_t{0}},
                  {"start_write_offset_bytes", uint32_t{0}}}});
            writer_args.runtime_arg_values.push_back(
                {core,
                 {{"nheads", nheads_second_risc},
                  {"start_read_offset_bytes", nheads_first_risc * in0_HtWt * single_tile_size},
                  {"start_write_offset_bytes", nheads_first_risc * in0_w_tiles * single_tile_size}}});
        }
        run_args.kernel_run_args.push_back(std::move(reader_args));
        run_args.kernel_run_args.push_back(std::move(writer_args));

    } else {
        // ---- Interleaved path: src0 DFB is L1-allocated, double-buffered. ----
        spec.dataflow_buffers = {m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles * 2,  // double buffer
            .data_format_metadata = cb_data_format,
        }};

        m2::KernelSpec reader{
            .unique_id = m2::KernelSpecName{"reader"},
            .source = std::filesystem::path{READER_KERNEL_PATH},
            .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"src0"}, "cb_id_in0")},
            .tensor_bindings = {m2::TensorBinding{
                .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "in0_args"}},
            .compile_time_args =
                {{"in0_h_tiles", in0_h_tiles}, {"in0_w_tiles", in0_w_tiles}, {"in0_c", in0_c}, {"in0_HtWt", in0_HtWt}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_h_dim", "in0_tensor_tile_id"}},
            // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — so the two data-movement kernels
            // don't collide on the same DM processor.
            .hw_config =
                m2::DataMovementHardwareConfig{
                    .gen1_config =
                        m2::DataMovementHardwareConfig::Gen1Config{
                            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                            .noc = tt::tt_metal::NOC::RISCV_1_default}},
        };

        m2::KernelSpec writer{
            .unique_id = m2::KernelSpecName{"writer"},
            .source = std::filesystem::path{WRITER_KERNEL_PATH},
            .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"src0"}, "cb_id_out")},
            .tensor_bindings = {m2::TensorBinding{
                .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        };

        spec.kernels = {std::move(reader), std::move(writer)};
        spec.work_units = {m2::WorkUnitSpec{
            .name = "nlp_concat_heads",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores}};

        const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
        m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
        m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
        for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_blocks_per_core = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

            uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
            uint32_t in0_tensor_tile_id = (num_blocks_written / in0_h_tiles * in0_CHtWt) + (in0_h_dim * in0_w_tiles);

            reader_args.runtime_arg_values.push_back(
                {core,
                 {{"num_blocks", num_blocks_per_core},
                  {"in0_h_dim", in0_h_dim},
                  {"in0_tensor_tile_id", in0_tensor_tile_id}}});

            writer_args.runtime_arg_values.push_back(
                {core,
                 {{"num_pages", num_blocks_per_core * per_tensor_tiles},
                  {"start_id", num_blocks_written * per_tensor_tiles}}});
            num_blocks_written += num_blocks_per_core;
        }
        run_args.kernel_run_args.push_back(std::move(reader_args));
        run_args.kernel_run_args.push_back(std::move(writer_args));
    }

    // Tensor arguments: the framework fills the kernels' tensor accessor base addresses from these, and
    // refreshes them on a cache hit (UpdateTensorArgs).
    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
