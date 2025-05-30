// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_sgd_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "ttnn/operations/sharding_utilities.hpp"
#include <tt-metalium/buffer_distribution_spec.hpp>

namespace ttnn::operations::moreh::moreh_sgd {


using tt::tt_metal::sharded_accessor_utils::ArgConfig;
using tt::tt_metal::sharded_accessor_utils::ArgsConfig;

struct InputOutputBufferParams {
    tt::tt_metal::Shape physical_tensor_shape;
    tt::tt_metal::Shape2D page_shape;
    float bytes_per_element;
    tt::DataFormat data_format;  // Used for setting up CBs

    struct DistributionSpecParams {
        tt::tt_metal::Shape physical_shard_shape;
        tt::tt_metal::CoreRangeSet grid;
        tt::tt_metal::ShardOrientation shard_orientation;
        tt::tt_metal::BufferType buffer_type;
    };
    DistributionSpecParams input_shard_spec;
    DistributionSpecParams output_shard_spec;
};

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_replicated_input_mesh_buffer(
    const Tensor& input, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    // These values would be passed from tensor correctly based on PageConfig
    // const auto host_size_in_bytes = inputs.physical_tensor_shape.volume() * inputs.bytes_per_element;
    // const auto page_size = inputs.page_shape.height() * inputs.page_shape.width() * inputs.bytes_per_element;

    auto tensor_sepc = input.get_tensor_spec();

    // auto phyiscal_tensor_shape = tensor_sepc.physical_shape();
    auto phyiscal_tensor_shape = tensor_sepc.logical_shape();

    // auto host_size_in_bytes = phyiscal_tensor_shape.volume() * input.element_size();

    float bytes_per_element = input.element_size();
    if (input.get_dtype() == DataType::BFLOAT8_B) {
        bytes_per_element = 1088.0f / 1024;  // 1.0625
    }
    std::cout << "bytes_per_element : " << bytes_per_element << "\n";
    auto host_size_in_bytes = phyiscal_tensor_shape.volume() * bytes_per_element;
    std::cout << "host_size_in_bytes : " << host_size_in_bytes << "\n";

    auto tensor_layout = tensor_sepc.tensor_layout();

    auto page_config = tensor_sepc.page_config();

    auto physical_shape = tensor_sepc.physical_shape();
    auto page_shape = tensor_layout.compute_page_shape(physical_shape);
    auto page_size = page_config.get_page_size_bytes(page_shape, input.get_dtype());

    // Mirrors allocate_mesh_buffer_on_device in ttnn
    const tt::tt_metal::distributed::ReplicatedBufferConfig mesh_buffer_config{.size = host_size_in_bytes};

    if (input.nd_shard_spec().has_value() == false) {
        TT_THROW("Input tensor must have a valid nd_shard_spec to create a replicated input mesh buffer.");
    }
    auto nd_shard_spec = input.nd_shard_spec().value();

    // Create input mesh buffer
    auto input_buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        phyiscal_tensor_shape, nd_shard_spec.shard_shape, page_shape, nd_shard_spec.grid, nd_shard_spec.orientation);

    const tt::tt_metal::distributed::DeviceLocalBufferConfig input_device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_parameters = input_buffer_distribution_spec,
    };

    const auto input_mesh_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, input_device_local_config, mesh_device);

    return input_mesh_buffer;
}

MorehSgdOperation::ProgramFactory::cached_program_t MorehSgdOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& param_in = tensor_args.param_in;
    auto& grad = tensor_args.grad;
    const std::optional<Tensor>& momentum_buffer_in = tensor_args.momentum_buffer_in;

    auto& output_tensors = output_tensor;
    auto& param_out = output_tensors.at(0).value();
    auto& momentum_buffer_out = output_tensors.at(1);

    auto lr = operation_attributes.lr;
    auto momentum = operation_attributes.momentum;
    auto dampening = operation_attributes.dampening;
    auto weight_decay = operation_attributes.weight_decay;
    auto nesterov = operation_attributes.nesterov;
    auto momentum_initialized = operation_attributes.momentum_initialized;

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    auto shape = param_in.get_logical_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto num = param_in.volume() / H / W;
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    bool has_momentum_buffer_out = momentum_buffer_out.has_value();

    Program program{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // tt::tt_metal::IDevice* device = param_in.device();
    auto* device = param_in.mesh_device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t units_to_divide = num * Ht * Wt;
    uint32_t core_w = grid.x;
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, units_to_divide);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(param_in.get_dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {tt::CBIndex::c_0, 2},   // param_in
            {tt::CBIndex::c_1, 2},   // grad
            {tt::CBIndex::c_2, 2},   // momentum_in
            {tt::CBIndex::c_16, 2},  // param_out
            {tt::CBIndex::c_17, 2},  // momentum_out

            {tt::CBIndex::c_24, 5, intermed_cb_format},  // cb_scalar_args (lr, momentum, dampening, weight_decay, one)
            {tt::CBIndex::c_25, 1, intermed_cb_format},  //
            {tt::CBIndex::c_26, 1, intermed_cb_format},  //
            {tt::CBIndex::c_27, 1, intermed_cb_format},  //
            {tt::CBIndex::c_28, 1, intermed_cb_format},  //
        });

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
    std::map<string, string> compute_defines;

    if (param_in.is_sharded()) {
        reader_defines["PARAM_IN_SHARDED"] = 1;
    }
    
    if (grad.is_sharded()) {
        reader_defines["GRAD_SHARDED"] = 1;
    }


    if (weight_decay != 0) {
        reader_defines["WEIGHT_DECAY"] = 1;
        compute_defines["WEIGHT_DECAY"] = 1;
    }

    if (momentum != 0) {
        reader_defines["MOMENTUM"] = 1;
        compute_defines["MOMENTUM"] = 1;
        writer_defines["MOMENTUM"] = 1;
    }

    if (momentum_initialized) {
        reader_defines["MOMENTUM_INITIALIZED"] = 1;
        compute_defines["MOMENTUM_INITIALIZED"] = 1;
    }

    if (nesterov) {
        reader_defines["NESTEROV"] = 1;
        compute_defines["NESTEROV"] = 1;
    }

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = 1;
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    //RankCRTA | NumBanksCRTA | TensorShapeCRTA | ShardShapeCRTA | BankCoordsCRTA
    ArgsConfig crta_config = {ArgConfig::NumBanksCRTA | ArgConfig::TensorShapeCRTA | ArgConfig::ShardShapeCRTA | ArgConfig::BankCoordsCRTA};


    // param_in
    const auto input_mesh_buffer = create_replicated_input_mesh_buffer(param_in, device);

    const auto& input_buffer_distribution_spec =
        std::get<BufferDistributionSpec>(input_mesh_buffer->device_local_config().shard_parameters.value());

    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto input_shard_view = input_mesh_buffer->get_device_buffer(mesh_coordinate);

    const auto input_sharded_accessor_args = tt::tt_metal::sharded_accessor_utils::get_sharded_accessor_args(
        *device, input_buffer_distribution_spec, input_shard_view->core_type(), crta_config);

    // grad
    const auto grad_mesh_buffer = create_replicated_input_mesh_buffer(grad, device);

    const auto& grad_buffer_distribution_spec =
        std::get<BufferDistributionSpec>(grad_mesh_buffer->device_local_config().shard_parameters.value());

    const auto grad_shard_view = grad_mesh_buffer->get_device_buffer(mesh_coordinate);

    const auto grad_sharded_accessor_args = tt::tt_metal::sharded_accessor_utils::get_sharded_accessor_args(
        *device, grad_buffer_distribution_spec, grad_shard_view->core_type(), crta_config);


    std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(param_in)),
        static_cast<uint32_t>(is_dram(grad)),
        static_cast<uint32_t>(momentum_buffer_in.has_value() ? is_dram(momentum_buffer_in.value()) : 0),
    };

    reader_compile_time_args.insert(
        reader_compile_time_args.end(),
        input_sharded_accessor_args.compile_time_args.cbegin(),
        input_sharded_accessor_args.compile_time_args.cend());
        
    reader_compile_time_args.insert(
        reader_compile_time_args.end(),
        grad_sharded_accessor_args.compile_time_args.cbegin(),
        grad_sharded_accessor_args.compile_time_args.cend());

    std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(param_out))};
    if (has_momentum_buffer_out) {
        writer_compile_time_args.push_back(static_cast<uint32_t>(is_dram(momentum_buffer_out.value())));
    }

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/"
        "reader_moreh_sgd.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/"
        "writer_moreh_sgd.cpp";

    const auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_tiles_per_core_group_1};

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/"
        "moreh_sgd.cpp";

    auto compute_kernel_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto param_in_addr = param_in.buffer()->address();
    const auto grad_addr = grad.buffer()->address();
    const auto momentum_buffer_in_addr =
        momentum_buffer_in.has_value() ? momentum_buffer_in.value().buffer()->address() : 0;

    const auto param_out_addr = param_out.buffer()->address();
    const auto momentum_buffer_out_addr =
        momentum_buffer_out.has_value() ? momentum_buffer_out->buffer()->address() : 0;

    auto core_x_offset = 0;
    auto core_y_offset = 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        union {
            float f;
            uint32_t u;
        } u_lr, u_momentum, u_dampening, u_weight_decay, u_one;
        u_lr.f = lr;
        u_momentum.f = momentum;
        u_dampening.f = dampening;
        u_weight_decay.f = weight_decay;
        u_one.f = 1.0f;

        std::vector<uint32_t> reader_args = {
            param_in.buffer()->address(),
            grad.buffer()->address(),
            momentum_buffer_in.has_value() ? momentum_buffer_in.value().buffer()->address() : 0,
            num_tiles_per_core,
            tile_offset,
            u_lr.u,
            u_momentum.u,
            u_dampening.u,
            u_weight_decay.u,
            u_one.u,
        };

        std::vector<uint32_t> writer_args = {
            param_out.buffer()->address(),
            momentum_buffer_out.has_value() ? momentum_buffer_out.value().buffer()->address() : 0,
            num_tiles_per_core,
            tile_offset,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    std::vector<uint32_t> reader_crtas = {};

    reader_crtas.insert(
        reader_crtas.end(),
        input_sharded_accessor_args.runtime_args.cbegin(),
        input_sharded_accessor_args.runtime_args.cend());
        
    reader_crtas.insert(
        reader_crtas.end(),
        grad_sharded_accessor_args.runtime_args.cbegin(),
        grad_sharded_accessor_args.runtime_args.cend());
        
    SetCommonRuntimeArgs(program, reader_kernel_id, reader_crtas);

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, core_h, has_momentum_buffer_out}};
}

void MorehSgdOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    auto param_in_buffer = tensor_args.param_in.buffer();
    auto grad_buffer = tensor_args.grad.buffer();
    auto momentum_buffer_in_buffer =
        tensor_args.momentum_buffer_in.has_value() ? tensor_args.momentum_buffer_in->buffer() : 0;

    auto param_out_buffer = tensor_return_value.at(0)->buffer();
    auto momentum_buffer_out_buffer = tensor_return_value.at(1)->buffer();

    auto num_cores = cached_program.shared_variables.num_cores;
    auto core_h = cached_program.shared_variables.core_h;
    auto has_momentum_buffer_out = cached_program.shared_variables.has_momentum_buffer_out;

    TT_ASSERT(has_momentum_buffer_out == false || tensor_return_value.size() == 2);

    for (uint32_t core_i = 0; core_i < num_cores; core_i++) {
        CoreCoord core = {core_i / core_h, core_i % core_h};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = param_in_buffer->address();
            runtime_args[1] = grad_buffer->address();
            if (tensor_args.momentum_buffer_in.has_value()) {
                runtime_args[2] = momentum_buffer_in_buffer->address();
                ;
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = param_out_buffer->address();
            if (has_momentum_buffer_out) {
                runtime_args[1] = momentum_buffer_out_buffer->address();
            }
        }
    }
}
}  // namespace ttnn::operations::moreh::moreh_sgd
