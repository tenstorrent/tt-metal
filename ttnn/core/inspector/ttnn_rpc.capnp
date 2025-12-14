# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Cap'N Proto file schema: https://capnproto.org/language.html

@0xba5d498ab9873a11;

using Cxx = import "/capnp/c++.capnp";
using Rpc = import "rpc.capnp";
$Cxx.namespace("ttnn::inspector::rpc");

# Inspector RPC interface for querying TTNN runtime state

interface TtnnInspector extends(Rpc.InspectorChannel) {
}
