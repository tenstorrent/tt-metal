// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// One request to the persistent fused_experts op (see FusedPersistentParams), from every chip that
// holds a sender core of `request_socket`: the [64-byte header with `layer` in word 0 | input row]
// page, multicast to that chip's receivers, then -- unless `layer` is the stop id 0xFFFFFFFF -- the sum
// of the op's chips' partial rows, which they send into `partials` on every chip of this mesh: one L1
// shard per part of the row, each on its own sum core, num_chips slots of that part then a flag word per
// chip. The sum, in chip order, is the output.
//
// The persistent op does not ack the request page; it takes a requester's next page only after this
// one's partials, so this op credits the request socket itself.
struct PersistentMoeRequestParams {
    uint32_t layer;
    tt::tt_metal::distributed::MeshSocket request_socket;
    Tensor partials;

    // Not an aggregate: reflection would try to default-construct the socket.
    PersistentMoeRequestParams(
        uint32_t layer, const tt::tt_metal::distributed::MeshSocket& request_socket, const Tensor& partials) :
        layer(layer), request_socket(request_socket), partials(partials) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("layer", layer);
        attrs.emplace_back("request_socket", request_socket);
        attrs.emplace_back("partials", partials);
        return attrs;
    }
};

struct PersistentMoeRequestDeviceOperation {
    using operation_attributes_t = PersistentMoeRequestParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        // Per chip: its request receivers fix the fabric routes. Chips with no sender core of the request
        // socket get an empty descriptor.
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

        // The layer and the input and output addresses change per request; the socket config and partials
        // addresses are compile-time args, in the program hash.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };
    using program_factory_t = std::variant<MultiCore>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {

Tensor persistent_moe_request(
    const Tensor& input_tensor,
    uint32_t layer,
    const tt::tt_metal::distributed::MeshSocket& request_socket,
    const Tensor& partials);

}  // namespace ttnn::prim
