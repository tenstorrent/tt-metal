// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_sharded_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <utility>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Nested so the names cannot collide with the sibling factories' in the unity build (this op's
// factories share one translation unit).
namespace hc_sharded {
const DFBSpecName IN0{"in0"};
const DFBSpecName OUT{"out"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
}  // namespace hc_sharded

// The per-core argument builders below return positional vectors: a fixed run of scalars (which
// the factory passes on as named runtime args) followed by the per-core arrays the kernels index
// by core / shard id (passed on as runtime varargs). These counts name the split point.
constexpr size_t num_reader_scalar_args = 5;  // both paths
constexpr size_t num_writer_scalar_args = 7;  // special case only

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_hc_rm_sharded(
    const Tensor& input_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t H = input_shape[2], C = input_shape[1];

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    MeshDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    std::vector<uint32_t> shard_grid_x_map;
    shard_grid_x_map.reserve(num_cores_x);
    for (uint32_t i = 0; i < num_cores_x; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(i, 0));
        shard_grid_x_map.push_back(physical_core.x);
    }
    std::vector<uint32_t> shard_grid_y_map;
    shard_grid_y_map.reserve(num_cores_y);
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

        ret_val[i] = {std::move(reader_runtime_args), std::move(writer_runtime_args)};

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
    const uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    MeshDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    uint32_t height = 0;
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
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

    const uint32_t num_sticks_per_core = shard_height;

    // Scratch buffers reused by every core: each stick contributes at most one entry, and entries are dropped only
    // for out-of-grid workers, so num_sticks_per_core covers the worst case for the whole loop.
    std::vector<uint32_t> stick_ids_per_core;
    std::vector<uint32_t> read_cores_indices;
    std::vector<uint32_t> read_cores_noc_x;
    std::vector<uint32_t> read_cores_noc_y;
    std::vector<uint32_t> read_stick_offset;
    std::vector<uint32_t> non_repeat_stick_offset_values;
    std::vector<uint32_t> non_repeat_noc_x_values;
    std::vector<uint32_t> non_repeat_noc_y_values;
    stick_ids_per_core.reserve(num_sticks_per_core);
    read_cores_indices.reserve(num_sticks_per_core);
    read_cores_noc_x.reserve(num_sticks_per_core);
    read_cores_noc_y.reserve(num_sticks_per_core);
    read_stick_offset.reserve(num_sticks_per_core);

    uint32_t curr_c = 0, curr_h = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; ++i) {
        stick_ids_per_core.clear();
        read_cores_indices.clear();
        read_cores_noc_x.clear();
        read_cores_noc_y.clear();
        read_stick_offset.clear();

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

        non_repeat_stick_offset_values.clear();
        non_repeat_noc_x_values.clear();
        non_repeat_noc_y_values.clear();

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

            const uint32_t num_sticks_per_shard_core = shard_height / num_non_repeat_cores;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            if (const bool split_reader = num_sticks_per_shard_core > 2; split_reader) {
                num_sticks_per_shard_core_reader = num_sticks_per_shard_core / 2;
                num_sticks_per_shard_core_writer = num_sticks_per_shard_core - num_sticks_per_shard_core_reader;
                writer_read_stick_offset = num_sticks_per_shard_core_reader * read_stick_stride;
                writer_write_stick_offset = writer_read_stick_offset * num_non_repeat_cores;
            }

            non_repeat_stick_offset_values.reserve(num_non_repeat_cores);
            non_repeat_noc_x_values.reserve(num_non_repeat_cores);
            non_repeat_noc_y_values.reserve(num_non_repeat_cores);
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

            const uint32_t num_sticks_per_shard_core = shard_height / num_non_repeat_cores / num_C_blocks_per_core;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            num_sticks_per_shard_core_writer = num_sticks_per_shard_core;
            if (const bool split_reader = num_C_blocks_per_core > 2; split_reader) {
                num_C_blocks_per_core_reader = num_C_blocks_per_core / 2;
                num_C_blocks_per_core_writer = num_C_blocks_per_core - num_C_blocks_per_core_reader;
                writer_read_stick_offset = num_C_blocks_per_core_reader * stick_size_bytes;
                writer_write_stick_offset = writer_read_stick_offset * num_non_repeat_cores * num_sticks_per_shard_core;
            }

            non_repeat_stick_offset_values.reserve(num_non_repeat_cores);
            non_repeat_noc_x_values.reserve(num_non_repeat_cores);
            non_repeat_noc_y_values.reserve(num_non_repeat_cores);
            for (uint32_t k = 0; k < num_non_repeat_cores; ++k) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[k * num_sticks_per_shard_core]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[k * num_sticks_per_shard_core]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[k * num_sticks_per_shard_core]);
            }
        }

        const bool read_single_h_block_per_core = num_H_per_core == 1;

        const size_t num_non_repeat_args =
            non_repeat_stick_offset_values.size() + non_repeat_noc_x_values.size() + non_repeat_noc_y_values.size();

        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(num_reader_scalar_args + num_non_repeat_args);
        reader_runtime_args.insert(
            reader_runtime_args.end(),
            {
                static_cast<uint32_t>(read_single_h_block_per_core),
                num_C_blocks_per_core_reader,
                num_sticks_per_shard_core_reader,
                num_non_repeat_cores,
                read_stick_stride,
            });
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        std::vector<uint32_t> writer_runtime_args;
        writer_runtime_args.reserve(num_writer_scalar_args + num_non_repeat_args);
        writer_runtime_args.insert(
            writer_runtime_args.end(),
            {
                static_cast<uint32_t>(read_single_h_block_per_core),
                num_C_blocks_per_core_writer,
                num_sticks_per_shard_core_writer,
                num_non_repeat_cores,
                read_stick_stride,
                writer_read_stick_offset,
                writer_write_stick_offset,
            });
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        ret_val[i] = {std::move(reader_runtime_args), std::move(writer_runtime_args)};
    }

    return ret_val;
}

// Selects the special-case per-core builder.
inline bool hc_rm_sharded_is_special_case(uint32_t shard_height, uint32_t H, uint32_t C) {
    return (shard_height % H == 0 || H % shard_height == 0) && (shard_height % C == 0 || C % shard_height == 0) &&
           (C % H == 0 || H % C == 0) && (shard_height <= C * H);
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCShardedProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    using namespace hc_sharded;

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.is_allocated(), "Operand to transpose_hc needs to be allocated in a buffer on device!");

    tt::DataFormat src0_dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat dst_dfb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t C = input_tensor.logical_shape()[1], N = input_tensor.logical_shape()[0];
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    bool is_special_case = hc_rm_sharded_is_special_case(shard_height, H, C);

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = shard_spec.num_cores();

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y + 1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    ProgramSpec spec{.name = "transpose_hc_sharded"};

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    // Both buffers are built on the io tensors' own sharded L1 memory rather than on
    // Program-lifetime storage; override_runtime_arguments re-points them on a cache hit. The
    // kernels gather other cores' input shards through the NoC at this same L1 address, which
    // holds because a borrowed buffer inherits the one sharded allocation's address on every core.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = stick_size_bytes,
        .num_entries = shard_height,
        .data_format_metadata = src0_dfb_data_format,
        .borrowed_from = INPUT,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT,
        .entry_size = stick_size_bytes,
        .num_entries = shard_height,
        .data_format_metadata = dst_dfb_data_format,
        .borrowed_from = OUTPUT,
    });

    // Writer kernel only exists in the special case path; the generic path puts
    // everything through the reader (writer args returned by the generic helper
    // are empty, matching the legacy `KernelHandle writer_kernel_id{}` behavior).
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> all_runtime_args;
    if (is_special_case) {
        all_runtime_args =
            get_runtime_args_hc_rm_sharded_special_case(input_tensor, num_cores, num_cores_x, num_cores_y);
    } else {
        all_runtime_args = get_runtime_args_hc_rm_sharded(input_tensor, num_cores, num_cores_x, num_cores_y);
    }

    // The array tail of each per-core argument vector rides as runtime varargs. The vararg count is
    // one number per kernel, so it is the longest tail over all cores; shorter tails are zero-padded
    // (the kernels index the arrays only up to their own per-core counts, never past them).
    uint32_t reader_num_varargs = 0;
    uint32_t writer_num_varargs = 0;
    for (const auto& [reader_args, writer_args] : all_runtime_args) {
        reader_num_varargs = std::max<uint32_t>(reader_num_varargs, reader_args.size() - num_reader_scalar_args);
        if (is_special_case) {
            writer_num_varargs = std::max<uint32_t>(writer_num_varargs, writer_args.size() - num_writer_scalar_args);
        }
    }

    // ---- reader ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_hc_sharded_rm.cpp",
        .hw_config = create_reader_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = reader_num_varargs},
    };
    if (is_special_case) {
        reader.compiler_options.defines = {{"USE_SPECIAL_CASE", "1"}};
        // Reader and writer split the gather of remote input shards into the resident output shard
        // between them. Neither drives a FIFO on either buffer (both only peek the base pointers and
        // move data through the NoC), so the producer / consumer labels are cosmetic: one of each
        // per buffer satisfies the endpoint rule without a multi-binding.
        reader.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN0,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        reader.compile_time_args = {{"stick_size_bytes", stick_size_bytes}};
        reader.runtime_arg_schema = {
            .runtime_arg_names = {
                "read_single_h_block_per_core",
                "num_C_blocks_per_core",
                "num_sticks_per_shard_core",
                "num_cores_read",
                "read_stick_stride"}};
    } else {
        // The reader is the only kernel on this path: it peeks the input base pointer and fills the
        // resident output shard that nothing drains, so both buffers are self-looped onto it.
        reader.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN0,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = IN0,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        reader.compile_time_args = {
            {"N", N},
            {"H", H},
            {"C", C},
            {"W_size_bytes", stick_size_bytes},
            {"row_major", static_cast<uint32_t>(row_major_orientation)},
            {"num_cores_x", num_cores_x},
            {"num_cores_y", num_cores_y}};
        reader.runtime_arg_schema = {
            .runtime_arg_names = {"num_sticks_per_core", "start_id", "curr_c", "curr_h", "curr_n"}};
    }
    spec.kernels.push_back(std::move(reader));

    // ---- writer (special case only) ----
    if (is_special_case) {
        spec.kernels.push_back(KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                      "writer_unary_transpose_hc_sharded_rm.cpp",
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN0,
                     .accessor_name = "in0",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 }},
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
            .hw_config = create_writer_datamovement_config(),
            .advanced_options = {.num_runtime_varargs = writer_num_varargs},
        });
    }

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = is_special_case ? Group<KernelSpecName>{READER, WRITER} : Group<KernelSpecName>{READER},
        .target_nodes = all_cores,
    });

    // ---- run args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    auto set_varargs = [](KernelRunArgs& kernel_run_args,
                          const CoreCoord& core,
                          const std::vector<uint32_t>& args,
                          size_t num_scalar_args,
                          uint32_t num_varargs) {
        std::vector<uint32_t> varargs(args.begin() + num_scalar_args, args.end());
        varargs.resize(num_varargs, 0);
        kernel_run_args.advanced_options.runtime_varargs[core] = std::move(varargs);
    };

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major_orientation) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        const auto& [reader_args, writer_args] = all_runtime_args[i];

        if (is_special_case) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"read_single_h_block_per_core", reader_args[0]},
                 {"num_C_blocks_per_core", reader_args[1]},
                 {"num_sticks_per_shard_core", reader_args[2]},
                 {"num_cores_read", reader_args[3]},
                 {"read_stick_stride", reader_args[4]}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"read_single_h_block_per_core", writer_args[0]},
                 {"num_C_blocks_per_core", writer_args[1]},
                 {"num_sticks_per_shard_core", writer_args[2]},
                 {"num_cores_read", writer_args[3]},
                 {"read_stick_stride", writer_args[4]},
                 {"src_read_stick_offset", writer_args[5]},
                 {"dst_write_stick_offset", writer_args[6]}});
            set_varargs(writer_run_args, core, writer_args, num_writer_scalar_args, writer_num_varargs);
        } else {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_sticks_per_core", reader_args[0]},
                 {"start_id", reader_args[1]},
                 {"curr_c", reader_args[2]},
                 {"curr_h", reader_args[3]},
                 {"curr_n", reader_args[4]}});
        }
        set_varargs(reader_run_args, core, reader_args, num_reader_scalar_args, reader_num_varargs);
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    if (is_special_case) {
        run_args.kernel_run_args.push_back(std::move(writer_run_args));
    }
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ProgramRunArgs TransposeHCShardedProgramFactory::override_runtime_arguments(
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace hc_sharded;
    // No custom compute_program_hash, so shape and shard spec are keyed and every reader/writer arg is
    // pinned; only the buffer addresses move, and both ride on the borrowed buffers (in0, out), which
    // re-resolve their L1 address from these tensor arguments.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT, tensor_args.input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output_tensor.mesh_tensor());
    return run_args;
}

}  // namespace ttnn::prim
