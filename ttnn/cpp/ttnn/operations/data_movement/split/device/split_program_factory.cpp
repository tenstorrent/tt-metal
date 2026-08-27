// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/split/device/split_program_factory.hpp"

#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts SplitProgramFactory::create_program_artifacts(
    const SplitParams& operation_attributes, const SplitInputs& tensor_args, std::vector<Tensor>& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    const uint32_t num_chunks = static_cast<uint32_t>(operation_attributes.num_splits);

    auto input_shape = input_tensor.padded_shape();
    IDevice* device = input_tensor.device();
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    // Collect output buffers and validate they are all the same type / page size. This invariant is
    // what lets every per-chunk writer share one identical compile-time-arg set and one src0 DFB shape.
    TT_FATAL(
        tensor_return_value.size() == num_chunks,
        "Number of output tensors ({}) must equal number of chunks ({})",
        tensor_return_value.size(),
        num_chunks);
    std::vector<Buffer*> output_buffers;
    output_buffers.reserve(num_chunks);
    for (uint32_t i = 0; i < num_chunks; i++) {
        Buffer* buf = tensor_return_value[i].buffer();
        TT_FATAL(buf != nullptr, "Output {} buffer should be allocated on device!", i);
        if (i > 0) {
            TT_FATAL(
                buf->buffer_type() == output_buffers[0]->buffer_type(),
                "All output buffers must have the same buffer type");
            TT_FATAL(
                buf->aligned_page_size() == output_buffers[0]->aligned_page_size(),
                "All output buffers must have the same aligned page size");
        }
        output_buffers.push_back(buf);
    }

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / tt::constants::TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;

    // Parallelize the Z (dim 1) dimension across separate core rows.
    uint32_t num_cores_z = z;

    // Parallelize dim-2 (height tiles) across X cores.
    auto [num_cores_x, per_core_tiles_x] =
        get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2, num_cores_x_limit / num_cores_z);

    // Parallelize dim-3 (width tiles) across Y cores, grouped by chunk.
    // We need num_cores_y to be a multiple of num_chunks so each chunk gets an equal group.
    uint32_t tiles_per_chunk = num_tiles_dim_3 / num_chunks;
    uint32_t max_cores_per_chunk = num_cores_y_limit / num_chunks;
    auto [num_cores_per_chunk, per_core_tiles_y] =
        get_max_cores_divisible_by_tiles_per_core_tiles(tiles_per_chunk, max_cores_per_chunk);

    uint32_t num_cores_c = num_cores_per_chunk * num_chunks;  // total Y-cores
    uint32_t num_cores_r = num_cores_x * num_cores_z;         // total X-cores (rows)

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_r - 1, (std::size_t)start_core_y + num_cores_c - 1});

    uint32_t num_tiles_per_z = (per_core_tiles_x * num_cores_x) * (per_core_tiles_y * num_cores_c);
    uint32_t z_stride_read = num_tiles_per_z;
    uint32_t y_stride_read = per_core_tiles_y * num_cores_c;

    uint32_t z_stride_write = num_tiles_per_z / num_chunks;
    uint32_t y_stride_write = per_core_tiles_y * num_cores_per_chunk;

    // ---- Metal 2.0 spec resources ----
    // Typed name constants are function-local (avoids anonymous-namespace symbol clashes under
    // unity builds); the per-chunk writer / output names are generated in the chunk loop below.
    const DFBSpecName SRC0{"src0"};
    const TensorParamName IN0{"in0"};
    const KernelSpecName READER{"reader"};

    const std::string reader_source =
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/"
        "reader_tm_tile_layout_split_two_chunks.cpp";
    const std::string writer_source =
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/"
        "writer_split_n_chunks_tile.cpp";

    // One src0 DFB (double-buffered): produced by the reader, consumed by each chunk's writer.
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    };

    // Reader: one KernelSpec over all cores. Producer of src0; reads the single input tensor.
    KernelSpec reader{
        .unique_id = READER,
        .source = reader_source,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0,
            .accessor_name = "src0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = IN0,
            .accessor_name = "in0",
        }},
        .compile_time_args =
            {
                {"z", z / num_cores_z},
                {"out_num_tiles_per_tensor_y", per_core_tiles_x},
                {"out_num_tiles_per_tensor_x", per_core_tiles_y},
                {"z_stride", z_stride_read},
                {"y_stride", y_stride_read},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"in0_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // N writers of ONE source, one per chunk. Each writer is bound to its chunk's output tensor and is
    // placed on that chunk's disjoint column band (Metal 2.0 binds a tensor per KernelSpec, and the
    // writer kernel uses a single tensor::out accessor, so the N distinct outputs need N KernelSpecs).
    // Every writer carries identical compile-time args; the equal split makes them chunk-independent.
    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.push_back(TensorParameter{.unique_id = IN0, .spec = input_tensor.tensor_spec()});

    Group<WorkUnitSpec> work_units;
    std::vector<KernelRunArgs> writer_run_args;
    writer_run_args.reserve(num_chunks);
    std::vector<TensorParamName> out_param_names;
    out_param_names.reserve(num_chunks);

    for (uint32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        const KernelSpecName writer_name{"writer_" + std::to_string(chunk_id)};
        const TensorParamName out_name{"out_" + std::to_string(chunk_id)};
        out_param_names.push_back(out_name);

        kernels.push_back(KernelSpec{
            .unique_id = writer_name,
            .source = writer_source,
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = SRC0,
                .accessor_name = "src0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = out_name,
                .accessor_name = "out",
            }},
            .compile_time_args =
                {
                    {"out_num_tiles_per_tensor_y", per_core_tiles_x},
                    {"out_num_tiles_per_tensor_x", per_core_tiles_y},
                    {"z", z / num_cores_z},
                    {"z_stride", z_stride_write},
                    {"y_stride", y_stride_write},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"out_tensor_tile_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        });

        tensor_parameters.push_back(
            TensorParameter{.unique_id = out_name, .spec = tensor_return_value[chunk_id].tensor_spec()});

        // Chunk chunk_id owns the column band y in [chunk_id * num_cores_per_chunk, +num_cores_per_chunk)
        // across all rows. The reader is listed in every work unit, so its node set is the union
        // (all_cores); each writer runs only on its band.
        CoreRange chunk_band(
            {(std::size_t)start_core_x, (std::size_t)(start_core_y + chunk_id * num_cores_per_chunk)},
            {(std::size_t)(start_core_x + num_cores_r - 1),
             (std::size_t)(start_core_y + (chunk_id + 1) * num_cores_per_chunk - 1)});
        work_units.push_back(WorkUnitSpec{
            .name = "chunk_" + std::to_string(chunk_id),
            .kernels = {READER, writer_name},
            .target_nodes = chunk_band,
        });

        writer_run_args.push_back(KernelRunArgs{.kernel = writer_name});
    }

    // ---- Runtime args (per-core page offsets) ----
    // Assigns the per-core tile offsets for the TILE N-way split. Each core belongs to exactly one
    // chunk group and writes to one output chunk.
    //
    // Core layout (row-major in metal notation):
    //   rows  (x): z_batch * num_cores_x  — parallelizes Z (dim 1) and dim-2 tiles
    //   cols  (y): num_chunks * num_cores_per_chunk — parallelizes dim-3 tiles, grouped by chunk
    //
    // Within a column group k (k=0..num_chunks-1):
    //   cores in that group write tiles from [k * tiles_per_chunk, (k+1) * tiles_per_chunk) of the last
    //   dim to output chunk k.
    KernelRunArgs reader_run_args{.kernel = READER};
    for (uint32_t id_r_outer = 0; id_r_outer < num_cores_z; id_r_outer++) {
        for (uint32_t id_r_inner = 0; id_r_inner < num_cores_x; id_r_inner++) {
            uint32_t id_r = id_r_outer * num_cores_x + id_r_inner;

            // Starting tile ID in the INPUT buffer for this (z, x) row of cores.
            uint32_t id_r_reader =
                (id_r_outer * num_tiles_per_z) + (id_r_inner * per_core_tiles_y * num_cores_c * per_core_tiles_x);

            // Corresponding starting tile in each OUTPUT buffer (output has 1/num_chunks fewer Y tiles).
            uint32_t id_r_writer = id_r_reader / num_chunks;

            for (uint32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
                for (uint32_t id_c_inner = 0; id_c_inner < num_cores_per_chunk; id_c_inner++) {
                    uint32_t id_c = chunk_id * num_cores_per_chunk + id_c_inner;
                    CoreCoord core = {(std::size_t)id_r, (std::size_t)id_c};

                    uint32_t reader_core_id = id_c * per_core_tiles_y + id_r_reader;
                    uint32_t writer_core_id = id_c_inner * per_core_tiles_y + id_r_writer;

                    AddRuntimeArgsForNode(
                        reader_run_args.runtime_arg_values, core, {{"in0_tensor_tile_id", reader_core_id}});
                    AddRuntimeArgsForNode(
                        writer_run_args[chunk_id].runtime_arg_values, core, {{"out_tensor_tile_id", writer_core_id}});
                }
            }
        }
    }

    // ---- Assemble the spec and run-args ----
    ProgramSpec spec{
        .name = "split",
        .kernels = std::move(kernels),
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    for (auto& kra : writer_run_args) {
        run_args.kernel_run_args.push_back(std::move(kra));
    }
    run_args.tensor_args.emplace(IN0, TensorArgument{input_tensor.mesh_tensor()});
    for (uint32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        run_args.tensor_args.emplace(
            out_param_names[chunk_id], TensorArgument{tensor_return_value[chunk_id].mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
