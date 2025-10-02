// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_program.hpp"

namespace ttnn {

// Validate the inputs and optional outputs
void AllGather::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, input tensor size should be 1 but has {}", input_tensors.size());
}

// Calculate output tensor specs. Infra will use this to allocate output tensors.
std::vector<ttnn::TensorSpec> AllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // AllGather concatenates along the given dimension, so multiply that dim
    // by the number of devices we're gathering from.
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    // shape[this->dim] *= this->ring_size; TODO
    // TODO if output_memory_config is not set, use input_tensor.memory_config()
    return {ttnn::TensorSpec(
        shape,
        ttnn::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), this->output_memory_config.value()))};
}

// Host program: across all devices
tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGather::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<ttnn::Tensor>& input_tensors,
    std::vector<ttnn::Tensor>& output_tensors) const {
    // We have a global view of all devices we're operating on, perform any common global
    // tasks here.

    // Use infra's implementation to invoke create_program_at() for each coordinate.
    return tt::tt_metal::operation::default_create_mesh_workload(
        *this, tensor_coords, input_tensors, /* optional_input_tensors */ {}, output_tensors);
}

// Host program: on a single device
tt::tt_metal::operation::ProgramWithCallbacks AllGather::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<ttnn::Tensor>& input_tensors,
    std::vector<ttnn::Tensor>& output_tensors) const {
    // TODO below is for vector addition
    // This example executes on a single Tensix core
    tt::tt_metal::Program program{};

    constexpr CoreCoord core{0, 0};  // execute on a single core

    const auto& input_tensor = input_tensors.at(0);
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const auto& output_tensor = output_tensors.at(0);

    constexpr uint32_t cb_depth = 2;
    const uint32_t page_size = input_tensor.buffer()->aligned_page_size();

    // Configure circular buffer to storge input tensor pages
    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CreateCircularBuffer(
        program,
        core,
        tt::tt_metal::CircularBufferConfig(
            /*total_size_bytes=*/cb_depth * page_size,
            /*data_format_spec=*/{{cb_index, df}})
            .set_page_size(cb_index, page_size));

    /************** Reader kernel **************/

    // Kernel args specific to this program (will be part of run-time args, won't be cached in program cache)
    std::vector<uint32_t> reader_prog_args = {
        input_tensor.buffer()->address(),  // input tensor address
    };

    // Kernel args common to all cores (will be part of compile-time args)
    std::vector<uint32_t> reader_common_args = {
        cb_index,   // circular buffer storing input tensor pages
        page_size,  // page size
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer())
        .append_to(reader_common_args);  // collection of args to recreate TensorAccessor in kernel

    // Kernel args specific to this core (will be part of run-time args)
    std::vector<uint32_t> reader_core_args = {
        // input_tensor.num_pages(),  // number of pages
    };

    // Create reader kernel
    auto reader = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl2/all_gather/device/reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(
            /* compile_args */ reader_common_args,
            /* defines */ {},
            /* named_compile_args */ {},
            /* opt_level */ tt::tt_metal::KernelBuildOptLevel::O2));
    /* TODO HACK */ std::vector<uint32_t> reader_rt_args = reader_prog_args;
    /* TODO HACK */ reader_rt_args.insert(reader_rt_args.end(), reader_core_args.begin(), reader_core_args.end());
    tt::tt_metal::SetRuntimeArgs(program, reader, core, reader_rt_args);

    /************** Writer kernel **************/

    // Kernel args specific to this program (will be part of run-time args, won't be cached in program cache)
    std::vector<uint32_t> writer_prog_args = {
        output_tensor.buffer()->address(),  // output tensor address
    };

    // Kernel args common to all cores (will be part of compile-time args)
    std::vector<uint32_t> writer_common_args = {
        cb_index,   // circular buffer storing input tensor pages
        page_size,  // page size
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer())
        .append_to(writer_common_args);  // collection of args to recreate TensorAccessor in kernel

    // Kernel args specific to this core (will be part of run-time args)
    std::vector<uint32_t> writer_core_args = {
        // input_tensor.num_pages(),  // number of pages
    };

    // Create writer kernel
    auto writer = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl2/all_gather/device/writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(
            /* compile_args */ writer_common_args,
            /* defines */ {},
            /* named_compile_args */ {},
            /* opt_level */ tt::tt_metal::KernelBuildOptLevel::O2));
    /* TODO HACK */ std::vector<uint32_t> writer_rt_args = writer_prog_args;
    /* TODO HACK */ writer_rt_args.insert(writer_rt_args.end(), writer_core_args.begin(), writer_core_args.end());
    tt::tt_metal::SetRuntimeArgs(program, writer, core, writer_rt_args);

    /*auto override_runtime_arguments_callback = [reader, writer](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors) {
        auto& runtime_args = GetRuntimeArgs(program, reader, core);
        runtime_args[0] = input_tensors.at(0).buffer()->address();
        runtime_args[1] = output_tensors.at(0).buffer()->address();
    };*/

    // Infra will execute the program
    return {
        .program = std::move(program),
        //.override_runtime_arguments_callback = override_runtime_arguments_callback
    };
}

tt::tt_metal::operation::Hash AllGather::compute_program_hash(const std::vector<ttnn::Tensor>& input_tensors) const {
    // TODO all of below
    const auto& input_shape = input_tensors[0].padded_shape();
    const auto& input_memory_layout = input_tensors[0].layout();
    const auto& input_dtype = input_tensors[0].dtype();
    const auto& input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<AllGather>(
        this->dim,
        this->topology,
        this->output_memory_config,
        this->subdevice_id,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

}  // namespace ttnn
