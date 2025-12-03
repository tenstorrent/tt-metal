
#include "pad_rm_reader_writer_multi_core_v2_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement::pad::program {
PadRmShardedHeightOnlyProgramFactory::cached_program_t PadRmShardedHeightOnlyProgramFactory::create(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    const float pad_value) {
    Program program{};

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    [[maybe_unused]] uint32_t num_unpadded_sticks = H * C * N;
    uint32_t W_padded = output_padded_shape[3], H_padded = output_padded_shape[2], C_padded = output_padded_shape[1],
             N_padded = output_padded_shape[0];

    const auto& front_pad = input_tensor_start;

    log_debug(tt::LogOp, "H_padded: {}", H_padded);
    log_debug(tt::LogOp, "front_pad: {}", front_pad);

    // stick sizes
    auto stick_size_unpadded = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    uint32_t row_major_min_bytes = 16;

    uint32_t zero_pad_stick_size = tt::tt_metal::find_max_divisor(stick_size_padded, 512);
    uint32_t num_zero_pad_sticks_read = stick_size_padded / zero_pad_stick_size;

    log_debug(tt::LogOp, "zero_pad_stick_size: {}", zero_pad_stick_size);
    log_debug(tt::LogOp, "num_zero_pad_sticks_read: {}", num_zero_pad_sticks_read);

    // TODO: add a general case, where we can pad on any dim.
    TT_FATAL(
        stick_size_unpadded == stick_size_padded,
        "sharded pad does not support pad on last dim currently as that will cause perf degradation");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    IDevice* device = a.device();

    // input shard spec
    auto shard_spec_unpadded = a.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    [[maybe_unused]] auto& all_cores_unpadded = shard_spec_unpadded.grid;
    [[maybe_unused]] uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = shard_spec_unpadded.grid.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    // output shard spec
    auto shard_spec_padded = output.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    auto& all_cores_padded = shard_spec_padded.grid;
    uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_unpadded * stick_size_unpadded, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_unpadded)
            .set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_padded * stick_size_padded, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, stick_size_padded)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(stick_size_padded, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, stick_size_padded);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    std::vector<uint32_t> reader_ct_args = {(std::uint32_t)stick_size_padded, (std::uint32_t)shard_height_padded};

    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)N + front_pad[-4],
        (std::uint32_t)H + front_pad[-2],
        (std::uint32_t)C + front_pad[-3],
        (std::uint32_t)stick_size_padded,
        (std::uint32_t)N_padded,
        (std::uint32_t)H_padded,
        (std::uint32_t)C_padded,
        (std::uint32_t)num_zero_pad_sticks_read,
        (std::uint32_t)zero_pad_stick_size,
        (std::uint32_t)not_pad_by_zero,
        (std::uint32_t)packed_pad_value,
        (std::uint32_t)row_major_min_bytes,
        (std::uint32_t)(stick_size_padded / row_major_min_bytes)};

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp",
        all_cores_padded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp",
        all_cores_padded,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_pad_runtime_args_rm_sharded(
        a,
        output,
        input_tensor_start,
        num_cores_padded,
        row_major,
        num_cores_x_padded,
        num_cores_y_padded,
        shard_height_padded,
        shard_height_unpadded,
        num_cores_x_unpadded,
        num_cores_y_unpadded);

    for (uint32_t i = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_padded, i / num_cores_x_padded};
        } else {
            core = {i / num_cores_y_padded, i % num_cores_y_padded};
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
    }

    auto override_runtime_args_callback = [cb_src0, cb_output](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto* src_buffer_a = input_tensors.at(0).buffer();
        auto* dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::pad::program
