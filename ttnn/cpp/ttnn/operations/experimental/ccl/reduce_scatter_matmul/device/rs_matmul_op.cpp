std::vector<Tensor> rs_matmul(
    const ttnn::Tensor& input_tensor,                           // mm0 used
    const ttnn::Tensor& weight_tensor,                          // mm1 used
    const ttnn::Tensor& rs_tensor,                              // rs1
    ttnn::Tensor& intermediate_packet_buffer,                   // rs2
    uint32_t dim,                                               // rs3
    const GlobalSemaphore& cross_device_semaphore,              // rs4
    const uint32_t cluster_axis,                                // rs 5
    const MeshDevice& mesh_device,                              // rs 6
    const uint32_t num_links,                                   // rs 7 default 1
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,  // rs 8 default std::nullopt
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,  // mm4 used but default std::nullopt
    const std::optional<const ttnn::DeviceComputeKernelConfig>
        compute_kernel_config,                                      // mm8 used but default std::nullopt
    const std::optional<const GlobalCircularBuffer>& global_cb,     // mm12 used but default std::nullopt
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,  // rs and mm13 used same but default std::nullopt
    const std::optional<const ttnn::CoreGrid> core_grid,            // mm9 may use but default std::nullopt
    const bool transpose_a,                                         // mm2 set false
    const bool transpose_b,                                         // mm3 set false
    const std::optional<const DataType> dtype,                      // mm5 set false
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,  // mm6 std::nullopt
    const std::optional<const std::string>& activation,                                  // mm7 set false
    const std::optional<const tt::tt_metal::Tile>& output_tile,                          // mm10 std::nullopt
    const std::optional<Tensor>& optional_output_tensor                                  // mm11 std::nullopt
) {
    return <input_tensor>;
}
