// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"


#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

namespace ttnn {

using ccl::EriscDatamoverBuilder;

struct AllGatherV2 {
    Program* program;
    const ttnn::ccl::EdmLineFabricOpInterface line_fabric;
    // const std::vector<Device*>& devices;
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;

    AllGatherV2(
        Program* prog,
        ttnn::ccl::EdmLineFabricOpInterface fabric,
        uint32_t d,
        uint32_t nl,
        uint32_t rs,
        uint32_t ri,
        MemoryConfig mc,
        ccl::Topology topo)
        : program(prog)
        , line_fabric(std::move(fabric))
        , dim(d)
        , num_links(nl)
        , ring_size(rs)
        , ring_index(ri)
        , output_mem_config(mc)
        , topology(topo) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);

        return attrs;
    }

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

namespace ccl{
namespace all_gather_v2_detail{
AllGatherV2 create_all_gather_struct(
    const std::vector<Program*>& programs,
    const ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<Device*>& devices,
    const ccl::Topology topology
);
} // namespace all_gather_detail
} // namespace ccl

// All Gather Variants
operation::ProgramWithCallbacks all_gather_multi_core_with_workers_new(
    Program& program,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology);



namespace operations {
namespace ccl {

Tensor all_gather_v2(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
    const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);

Tensor all_gather_v2(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
    const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
