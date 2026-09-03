// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <vector>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_hc_rm_sharded(
    const Tensor& input_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t H = input_shape[2], C = input_shape[1];

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    IDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    std::vector<uint32_t> shard_grid_x_map;
    for (uint32_t i = 0; i < num_cores_x; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(i, 0));
        shard_grid_x_map.push_back(physical_core.x);
    }
    std::vector<uint32_t> shard_grid_y_map;
    for (uint32_t i = 0; i < num_cores_y; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(0, i));
        shard_grid_y_map.push_back(physical_core.y);
    }

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        uint32_t num_sticks_per_core = shard_height;

        std::vector<uint32_t> reader_runtime_args = {num_sticks_per_core, curr_sticks_read, curr_c, curr_h, curr_n};
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_x_map.begin(), shard_grid_x_map.end());
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_y_map.begin(), shard_grid_y_map.end());

        std::vector<uint32_t> writer_runtime_args;

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }

    return ret_val;
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_hc_rm_sharded_special_case(
    const Tensor& input_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    IDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    uint32_t height = 0;
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        log_debug(tt::LogOp, "core: {}", core);

        height += shard_height;

        if (height <= total_height) {
            cores.push_back(core);
        }
    }

    uint32_t num_H_per_core = shard_height / H > 0 ? shard_height / H : 1;  // the number of H blocks in a shard
    uint32_t num_C_blocks_per_core = shard_height > C ? shard_height / C : 1;

    uint32_t curr_c = 0, curr_h = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; i++) {
        std::vector<uint32_t> read_cores_indices;
        std::vector<uint32_t> read_cores_noc_x;
        std::vector<uint32_t> read_cores_noc_y;
        std::vector<uint32_t> read_stick_offset;

        uint32_t num_sticks_per_core = shard_height;

        std::vector<uint32_t> stick_ids_per_core;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            stick_ids_per_core.push_back(curr_sticks_read);
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            uint32_t stick_id = stick_ids_per_core[j];
            uint32_t shard_id = stick_id / num_sticks_per_core;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x : num_cores_y;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x && worker_y_logical < num_cores_y) {
                auto core_physical =
                    device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});

                read_cores_indices.push_back(shard_id);
                read_stick_offset.push_back(stick_id_in_shard * stick_size_bytes);
                read_cores_noc_x.push_back(core_physical.x);
                read_cores_noc_y.push_back(core_physical.y);
            }
        }

        std::vector<uint32_t> non_repeat_stick_offset_values;
        std::vector<uint32_t> non_repeat_noc_x_values;
        std::vector<uint32_t> non_repeat_noc_y_values;

        uint32_t num_sticks_per_shard_core_reader = 0, num_sticks_per_shard_core_writer = 0;
        uint32_t writer_read_stick_offset = 0, writer_write_stick_offset = 0;
        uint32_t num_C_blocks_per_core_reader = num_C_blocks_per_core, num_C_blocks_per_core_writer = 0;

        uint32_t num_non_repeat_cores = read_cores_indices.size();
        uint32_t read_stick_stride = read_stick_offset.size() > 1 ? read_stick_offset[1] - read_stick_offset[0] : 0;

        if (num_H_per_core == 1) {  // each core only has one H block or part of H block
            for (uint32_t k = 1; k < read_cores_indices.size(); ++k) {
                if (read_cores_indices[k] == read_cores_indices[0]) {
                    num_non_repeat_cores = k;
                    read_stick_stride = read_stick_offset[k] - read_stick_offset[0];
                    break;
                }
            }

            uint32_t num_sticks_per_shard_core = shard_height / num_non_repeat_cores;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            bool split_reader = num_sticks_per_shard_core > 2;
            if (split_reader) {
                num_sticks_per_shard_core_reader = num_sticks_per_shard_core / 2;
                num_sticks_per_shard_core_writer = num_sticks_per_shard_core - num_sticks_per_shard_core_reader;
                writer_read_stick_offset = num_sticks_per_shard_core_reader * read_stick_stride;
                writer_write_stick_offset = writer_read_stick_offset * num_non_repeat_cores;
            }

            for (uint32_t k = 0; k < num_non_repeat_cores; ++k) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[k]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[k]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[k]);
            }
        } else {  // contains multiple H blocks
            std::set<uint32_t> unique_values(read_cores_indices.begin(), read_cores_indices.end());
            num_non_repeat_cores = unique_values.size();
            read_stick_stride = read_stick_offset[1] - read_stick_offset[0];

            // TODO: add the second batch args (num_non_repeat_cores, read_stick_offset, non_repeat_noc_x_values,
            // non_repeat_noc_y_values) to support multiple batch in a shard
            for (uint32_t k = 1; k < num_sticks_per_core; ++k) {
                if ((read_cores_indices[k - 1] == read_cores_indices[k]) &&
                    (read_stick_offset[k] == read_stick_offset[k - 1] + stick_size_bytes)) {
                    break;
                }
            }

            uint32_t num_sticks_per_shard_core = shard_height / num_non_repeat_cores / num_C_blocks_per_core;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            num_sticks_per_shard_core_writer = num_sticks_per_shard_core;
            bool split_reader = num_C_blocks_per_core > 2;
            if (split_reader) {
                num_C_blocks_per_core_reader = num_C_blocks_per_core / 2;
                num_C_blocks_per_core_writer = num_C_blocks_per_core - num_C_blocks_per_core_reader;
                writer_read_stick_offset = num_C_blocks_per_core_reader * stick_size_bytes;
                writer_write_stick_offset =
                    num_C_blocks_per_core_reader * num_non_repeat_cores * num_sticks_per_shard_core * stick_size_bytes;
            }

            for (uint32_t k = 0; k < num_non_repeat_cores; ++k) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[k * num_sticks_per_shard_core]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[k * num_sticks_per_shard_core]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[k * num_sticks_per_shard_core]);
            }
        }

        bool read_single_h_block_per_core = num_H_per_core == 1;

        std::vector<uint32_t> reader_runtime_args = {
            static_cast<uint32_t>(read_single_h_block_per_core),
            num_C_blocks_per_core_reader,
            num_sticks_per_shard_core_reader,
            num_non_repeat_cores,
            read_stick_stride,
        };

        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        std::vector<uint32_t> writer_runtime_args = {
            static_cast<uint32_t>(read_single_h_block_per_core),
            num_C_blocks_per_core_writer,
            num_sticks_per_shard_core_writer,
            num_non_repeat_cores,
            read_stick_stride,
            writer_read_stick_offset,
            writer_write_stick_offset,
        };

        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        ret_val[i] = {reader_runtime_args, writer_runtime_args};
    }

    return ret_val;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCShardedProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN{"cb_in"};    // legacy c_0: input shard (borrowed)
    const DFBSpecName CB_OUT{"cb_out"};  // legacy c_16: output shard (borrowed)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_sharded_rm.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_hc_sharded_rm.cpp";

    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    const uint32_t C = input_tensor.logical_shape()[1], N = input_tensor.logical_shape()[0];
    const uint32_t stick_size_bytes = W * input_tensor.element_size();

    const auto shard_spec = input_tensor.shard_spec().value();
    const uint32_t shard_height = shard_spec.shape[0];
    const bool row_major_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    bool is_special_case = false;
    if ((shard_spec.shape[0] % H == 0 || H % shard_spec.shape[0] == 0) &&
        (shard_spec.shape[0] % C == 0 || C % shard_spec.shape[0] == 0) && (C % H == 0 || H % C == 0) &&
        (shard_height <= C * H)) {
        is_special_case = true;
    }

    const auto& all_cores = shard_spec.grid;
    const uint32_t num_cores = shard_spec.num_cores();

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    const auto bbox = shard_spec.grid.bounding_box();
    const CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y + 1};
    const uint32_t num_cores_x = grid_size.x;
    const uint32_t num_cores_y = grid_size.y;

    // ------------------------------------------------------------------------
    // Borrowed-memory DFBs aliasing the input/output tensor shard buffers (legacy CBDescriptor::buffer
    // = input/output_tensor.buffer()). Both shards are shard_height sticks of stick_size_bytes.
    // ------------------------------------------------------------------------
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_IN,
        .entry_size = stick_size_bytes,
        .num_entries = shard_height,
        .data_format_metadata = src0_cb_data_format,
        .borrowed_from = INPUT_TENSOR,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_OUT,
        .entry_size = stick_size_bytes,
        .num_entries = shard_height,
        .data_format_metadata = dst_cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    });

    // Each tensor parameter is used only as a DFB borrowed_from backing store (read/written by L1
    // address; no kernel-side TensorAccessor), which the framework counts as a legitimate use.
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()};

    // Per-core args from the legacy helpers (host logic unchanged). The generic path returns empty
    // writer args (legacy `KernelHandle writer_kernel_id{}`); the special case splits across reader +
    // writer. The variable-length NoC-coordinate / stick-offset lists become positional runtime
    // varargs; the leading scalars become named runtime args.
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> all_runtime_args;
    if (is_special_case) {
        all_runtime_args =
            get_runtime_args_hc_rm_sharded_special_case(input_tensor, num_cores, num_cores_x, num_cores_y);
    } else {
        all_runtime_args = get_runtime_args_hc_rm_sharded(input_tensor, num_cores, num_cores_x, num_cores_y);
    }

    // Map flat core index -> node, mirroring the legacy row-major / col-major shard ordering exactly.
    auto node_for_index = [&](uint32_t i) -> NodeCoord {
        if (row_major_orientation) {
            return CoreCoord{i % num_cores_x, i / num_cores_x};
        }
        return CoreCoord{i / num_cores_y, i % num_cores_y};
    };

    std::vector<KernelSpec> kernels;
    std::vector<KernelSpecName> wu_kernels;
    ProgramRunArgs run_args;

    if (is_special_case) {
        // Special case: reader + writer split the stick copies. Both touch the borrowed shards purely
        // by L1 address; bind reader as the (nominal) producer and writer as the consumer of each DFB
        // so every DFB has exactly one producer + one consumer per node (no FIFO sync is performed).
        // Vararg payload (3 lists of num_cores_read each) varies per core, so we pad every node to the
        // program-wide max; the device reads only num_cores_read entries from each list.
        uint32_t max_reader_varargs = 0;
        uint32_t max_writer_varargs = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            max_reader_varargs =
                std::max<uint32_t>(max_reader_varargs, static_cast<uint32_t>(all_runtime_args[i].first.size()) - 5);
            max_writer_varargs =
                std::max<uint32_t>(max_writer_varargs, static_cast<uint32_t>(all_runtime_args[i].second.size()) - 7);
        }

        KernelSpec reader_spec{
            .unique_id = READER_KERNEL,
            .source = std::filesystem::path{READER_PATH},
            .dfb_bindings = {ProducerOf(CB_IN, "cb_in"), ProducerOf(CB_OUT, "cb_out")},
            .compile_time_args = {{"stick_size_bytes", stick_size_bytes}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"read_single_h_block_per_core",
                      "num_C_blocks_per_core",
                      "num_sticks_per_shard_core",
                      "num_cores_read",
                      "read_stick_stride"}},
            .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
        };
        reader_spec.compiler_options.defines = {{"USE_SPECIAL_CASE", "1"}};
        reader_spec.advanced_options.num_runtime_varargs = max_reader_varargs;

        KernelSpec writer_spec{
            .unique_id = WRITER_KERNEL,
            .source = std::filesystem::path{WRITER_PATH},
            .dfb_bindings = {ConsumerOf(CB_IN, "cb_in"), ConsumerOf(CB_OUT, "cb_out")},
            .compile_time_args = {{"stick_size_bytes", stick_size_bytes}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"read_single_h_block_per_core",
                      "num_C_blocks_per_core",
                      "num_sticks_per_shard_core",
                      "num_cores_read",
                      "read_stick_stride",
                      "src_read_stick_offset",
                      "dst_write_stick_offset"}},
            .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
        };
        writer_spec.advanced_options.num_runtime_varargs = max_writer_varargs;

        KernelRunArgs reader_run{.kernel = READER_KERNEL};
        KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
        for (uint32_t i = 0; i < num_cores; ++i) {
            const NodeCoord node = node_for_index(i);
            const auto& r = all_runtime_args[i].first;
            const auto& w = all_runtime_args[i].second;

            KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
            AddRuntimeArgsForNode(
                reader_rtas,
                node,
                {
                    {"read_single_h_block_per_core", r[0]},
                    {"num_C_blocks_per_core", r[1]},
                    {"num_sticks_per_shard_core", r[2]},
                    {"num_cores_read", r[3]},
                    {"read_stick_stride", r[4]},
                });
            std::vector<uint32_t> r_varargs(r.begin() + 5, r.end());
            r_varargs.resize(max_reader_varargs, 0);
            reader_run.advanced_options.runtime_varargs[node] = std::move(r_varargs);

            KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
            AddRuntimeArgsForNode(
                writer_rtas,
                node,
                {
                    {"read_single_h_block_per_core", w[0]},
                    {"num_C_blocks_per_core", w[1]},
                    {"num_sticks_per_shard_core", w[2]},
                    {"num_cores_read", w[3]},
                    {"read_stick_stride", w[4]},
                    {"src_read_stick_offset", w[5]},
                    {"dst_write_stick_offset", w[6]},
                });
            std::vector<uint32_t> w_varargs(w.begin() + 7, w.end());
            w_varargs.resize(max_writer_varargs, 0);
            writer_run.advanced_options.runtime_varargs[node] = std::move(w_varargs);
        }

        kernels.push_back(std::move(reader_spec));
        kernels.push_back(std::move(writer_spec));
        wu_kernels = {READER_KERNEL, WRITER_KERNEL};
        run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    } else {
        // Generic path: a single reader gathers every output stick via NoC. It is the sole toucher of
        // both borrowed shards, so it self-loops (producer + consumer) each DFB. The shard-grid x/y
        // physical-coordinate maps are uniform-length varargs (num_cores_x + num_cores_y).
        KernelSpec reader_spec{
            .unique_id = READER_KERNEL,
            .source = std::filesystem::path{READER_PATH},
            .dfb_bindings =
                {ProducerOf(CB_IN, "cb_in"),
                 ConsumerOf(CB_IN, "cb_in"),
                 ProducerOf(CB_OUT, "cb_out"),
                 ConsumerOf(CB_OUT, "cb_out")},
            .compile_time_args =
                {{"N", N},
                 {"H", H},
                 {"C", C},
                 {"W_size_bytes", stick_size_bytes},
                 {"row_major", static_cast<uint32_t>(row_major_orientation)},
                 {"num_cores_x", num_cores_x},
                 {"num_cores_y", num_cores_y}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"num_sticks_per_core", "start_id", "curr_c", "curr_h", "curr_n"}},
            .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
        };
        reader_spec.advanced_options.num_runtime_varargs = num_cores_x + num_cores_y;

        KernelRunArgs reader_run{.kernel = READER_KERNEL};
        for (uint32_t i = 0; i < num_cores; ++i) {
            const NodeCoord node = node_for_index(i);
            const auto& r = all_runtime_args[i].first;
            KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
            AddRuntimeArgsForNode(
                reader_rtas,
                node,
                {
                    {"num_sticks_per_core", r[0]},
                    {"start_id", r[1]},
                    {"curr_c", r[2]},
                    {"curr_h", r[3]},
                    {"curr_n", r[4]},
                });
            reader_run.advanced_options.runtime_varargs[node] = std::vector<uint32_t>(r.begin() + 5, r.end());
        }

        kernels.push_back(std::move(reader_spec));
        wu_kernels = {READER_KERNEL};
        run_args.kernel_run_args = {std::move(reader_run)};
    }

    WorkUnitSpec wu{
        .name = "transpose_hc_sharded_rm",
        .kernels = std::move(wu_kernels),
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "transpose_hc_sharded_rm",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
