// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <string_view>

#include "moreh_nll_loss_step1_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step1 {

using namespace tt::tt_metal::experimental;

namespace {

const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};

const DFBSpecName DFB_TARGET{"target"};
const DFBSpecName DFB_WEIGHT{"weight"};
const DFBSpecName DFB_WEIGHT_SCRATCH{"weight_scratch"};
const DFBSpecName DFB_OUTPUT{"output"};

const TensorParamName TENSOR_TARGET{"target"};
const TensorParamName TENSOR_WEIGHT{"weight"};
const TensorParamName TENSOR_OUTPUT{"output"};

// Helper: a dataflow buffer holding a whole number of tiles, sized by an explicit entry size.
DataflowBufferSpec make_dfb(
    const DFBSpecName& unique_id, uint32_t entry_size, uint32_t num_entries, tt::DataFormat data_format) {
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
    };
}

// Helper: bind `dfb` to `kernel` as both producer and consumer under one accessor name.
// Used for the three buffers only the reader touches, which therefore hold both endpoints
// themselves. (`weight_scratch` invokes no FIFO machinery at all, so its labels are cosmetic;
// the buffer still needs both endpoints declared to be a legal dataflow buffer.)
void bind_self_loop(Group<DFBBinding>& bindings, const DFBSpecName& dfb, std::string_view accessor_name) {
    bindings.push_back(DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::string{accessor_name},
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    bindings.push_back(DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::string{accessor_name},
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
}

}  // namespace

ttnn::device_operation::ProgramArtifacts MorehNllLossStep1DeviceOperation::Factory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const Tensor& output = tensor_return_value;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const uint32_t channel_size = operation_attributes.channel_size;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto target_shape = target.padded_shape();
    const bool weight_has_value = weight.has_value();
    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    // copy TILE per core
    uint32_t units_to_divide = target.physical_volume() / H / W * (Ht * Wt);

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const auto target_data_format = tt_metal::datatype_to_dataformat_converter(target.dtype());
    const auto data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    const auto target_tile_size = tt::tile_size(target_data_format);
    const auto data_tile_size = tt::tile_size(data_format);
    const auto intermed_tile_size = tt::tile_size(intermed_data_format);

    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);

    uint32_t target_num_tile = 1;
    uint32_t weight_num_tile = weight_has_value ? div_up(channel_size, tt::constants::TILE_WIDTH) : 0;
    uint32_t intermed_num_tile = 1;
    uint32_t output_num_tile = 1;
    // The `intermed_num_tile * intermed_tile_size` term sizes a buffer this op does not allocate: it
    // is the intermediate a compute kernel would stage results in, and this op instantiates no compute
    // kernel (both readers do the work in L1 with a scalar loop). The term is kept because this sum,
    // not the allocation, is what picks the algorithm just below -- dropping it would move the
    // small/large threshold and change which reader kernel some shapes compile, which is a change in
    // behaviour rather than in plumbing. `intermed_data_format` and `intermed_tile_size` above exist
    // only to feed it.
    uint32_t dfb_usage = (target_num_tile * target_tile_size) + (weight_num_tile * data_tile_size) +
                         (intermed_num_tile * intermed_tile_size) + (output_num_tile * data_tile_size);

    const bool use_large_algorithm = dfb_usage >= available_L1;

    ProgramSpec spec;
    spec.name = "moreh_nll_loss_step1";

    // create dataflow buffers

    // target buffer (always Int32, single tile)
    spec.dataflow_buffers.push_back(make_dfb(DFB_TARGET, target_tile_size, 1, tt::DataFormat::Int32));

    // weight buffer:
    //   - large algorithm: always allocate 1 tile (single-tile streaming through the buffer)
    //   - small algorithm: allocate weight_num_tile tiles (skip it when weight is absent and
    //     weight_num_tile == 0)
    const uint32_t weight_dfb_tiles = use_large_algorithm ? 1u : weight_num_tile;
    if (weight_dfb_tiles > 0) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT, data_tile_size, weight_dfb_tiles, data_format));
    }

    // output buffer
    spec.dataflow_buffers.push_back(make_dfb(DFB_OUTPUT, data_tile_size, 1, data_format));

    // Only the small reader needs the scratch buffer: it reaches it through the shared `read_line`
    // helper, whereas the large reader streams the weight a value at a time and never touches it. The
    // condition is therefore narrower than `weight_has_value` alone -- under the large algorithm the
    // buffer would exist with nothing bound to it, which is not a buffer that can be expressed.
    const bool has_weight_scratch = weight_has_value && !use_large_algorithm;
    if (has_weight_scratch) {
        // This buffer is used as scratch storage when reading data from DRAM into L1, since the two
        // have different alignment requirements on some architectures. Need space for only a single
        // tile of scratch, because content is read immediately after writing.
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHT_SCRATCH, data_tile_size, 1, data_format));
    }

    // declare the tensors the kernels operate on
    const auto& target_mesh = target.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();
    const MeshTensor* const weight_mesh = weight_has_value ? &weight.value().mesh_tensor() : nullptr;

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_TARGET, .spec = target_mesh.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TENSOR_WEIGHT, .spec = weight_mesh->tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TENSOR_OUTPUT, .spec = output_mesh.tensor_spec()});

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;

    if (weight_has_value) {
        reader_defines.emplace("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
    }
    const auto* const reader_kernel_file =
        use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/"
                              "kernels/reader_moreh_nll_loss_step1_large.cpp"
                            : "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/"
                              "kernels/reader_moreh_nll_loss_step1.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/kernels/"
        "writer_moreh_nll_loss_step1.cpp";

    // The reader is the only kernel that touches `target`: it fills the entry through `read_tile`,
    // waits on it, reads it through a local L1 pointer and pops it. So it holds both ends.
    Group<DFBBinding> reader_dfb_bindings;
    bind_self_loop(reader_dfb_bindings, DFB_TARGET, "target");
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = DFB_OUTPUT,
        .accessor_name = "output",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (weight_dfb_tiles > 0) {
        // `weight` is likewise reader-only: read in and then read back out of L1 by the same kernel.
        bind_self_loop(reader_dfb_bindings, DFB_WEIGHT, "weight");
    }
    if (has_weight_scratch) {
        bind_self_loop(reader_dfb_bindings, DFB_WEIGHT_SCRATCH, "weight_scratch");
    }

    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = TENSOR_TARGET, .accessor_name = "target"},
    };
    if (weight_has_value) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TENSOR_WEIGHT, .accessor_name = "weight"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = {{"weight_has_value", static_cast<uint32_t>(weight_has_value)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"ignore_index", "num_units_per_core", "start_id", "C", "weight_num_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_OUTPUT,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TENSOR_OUTPUT, .accessor_name = "output"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_units_per_core", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    // Set Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t num_units_per_core;
        if (core_group_1.contains(core)) {
            num_units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"ignore_index", static_cast<uint32_t>(ignore_index)},
                {"num_units_per_core", num_units_per_core},
                {"start_id", tile_offset},
                {"C", channel_size},
                {"weight_num_tile", weight_num_tile},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_units_per_core", num_units_per_core},
                {"start_id", tile_offset},
            });

        tile_offset += num_units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(TENSOR_TARGET, target_mesh);
    if (weight_has_value) {
        run_args.tensor_args.emplace(TENSOR_WEIGHT, *weight_mesh);
    }
    run_args.tensor_args.emplace(TENSOR_OUTPUT, output_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step1
