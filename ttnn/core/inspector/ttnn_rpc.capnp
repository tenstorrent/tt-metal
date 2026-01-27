# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Cap'N Proto file schema: https://capnproto.org/language.html

@0xba5d498ab9873a11;

using Cxx = import "/capnp/c++.capnp";
using Rpc = import "/tt-metalium/experimental/inspector_rpc.capnp";
$Cxx.namespace("ttnn::inspector::rpc");

# Inspector RPC interface for querying TTNN runtime state

struct MeshWorkloadData {
    meshWorkloadId @0 :UInt64;
    # High-level operation metadata
    # Empty if workload was not created by a tracked operation
    name @1 :Text;        # Operation name
    parameters @2 :Text;  # Operation parameters
}

struct MeshWorkloadRuntimeIdEntry {
    workloadId @0 :UInt64;
    runtimeId @1 :UInt64;
}

interface TtnnInspector extends(Rpc.InspectorChannel) {
    # Get mesh workloads we have additional info for
    getMeshWorkloads @0 () -> (meshWorkloads :List(MeshWorkloadData));

    # Get runtime IDs for mesh workloads
    getMeshWorkloadsRuntimeIds @1 () -> (runtimeIds :List(MeshWorkloadRuntimeIdEntry));
}
