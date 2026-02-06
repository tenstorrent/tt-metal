// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::graph {
// Vertex struct
constexpr auto kNodeType = "node_type";
constexpr auto kCounter = "counter";
constexpr auto kConnections = "connections";
constexpr auto kParams = "params";
constexpr auto kArguments = "arguments";
constexpr auto kInputTensors = "input_tensors";
// params keys
constexpr auto kName = "name";
constexpr auto kInputs = "inputs";
constexpr auto kTensorId = "tensor_id";
constexpr auto kType = "type";
constexpr auto kAddress = "address";
constexpr auto kSize = "size";
constexpr auto kLayout = "layout";
constexpr auto kShape = "shape";
constexpr auto kDtype = "dtype";
constexpr auto kMemoryConfig = "memory_config";
constexpr auto kBufferType = "buffer_type";
constexpr auto kDeviceTensors = "device_tensors";
constexpr auto kNumCores = "num_cores";
constexpr auto kPageSize = "page_size";
constexpr auto kCoreRangeSet = "core_range_set";
constexpr auto kGloballyAllocated = "globally_allocated";
constexpr auto kDeviceId = "device_id";
constexpr auto kDurationNs = "duration_ns";

// node names
constexpr auto kNodeBuffer = "buffer";
constexpr auto kNodeBufferAllocate = "buffer_allocate";
constexpr auto kNodeBufferDeallocate = "buffer_deallocate";
constexpr auto kNodeTensor = "tensor";
constexpr auto kNodeCBAllocate = "circular_buffer_allocate";
constexpr auto kNodeCBDeallocateAll = "circular_buffer_deallocate_all";
constexpr auto kNodeFunctionStart = "function_start";
constexpr auto kNodeFunctionEnd = "function_end";
constexpr auto kNodeCaptureStart = "capture_start";
constexpr auto kNodeCaptureEnd = "capture_end";
constexpr auto kNodeError = "error";

// levelized graph keys
constexpr auto kInEdges = "in_edges";
constexpr auto kOutEdges = "out_edges";
constexpr auto kInternals = "internals";
constexpr auto kOutputInfo = "output_info";
constexpr auto kOutputShape = "output_shape";
constexpr auto kStackingLevel = "stacking_level";

// report file keys
constexpr auto kReportVersion = "version";
constexpr auto kReportGraph = "graph";
constexpr auto kReportDevices = "devices";
constexpr auto kReportMetadata = "metadata";
constexpr auto kReportTimestampNs = "capture_timestamp_ns";
constexpr auto kReportTotalDurationNs = "total_duration_ns";

// device info keys
constexpr auto kDeviceNumYCores = "num_y_cores";
constexpr auto kDeviceNumXCores = "num_x_cores";
constexpr auto kDeviceNumYComputeCores = "num_y_compute_cores";
constexpr auto kDeviceNumXComputeCores = "num_x_compute_cores";
constexpr auto kDeviceWorkerL1Size = "worker_l1_size";
constexpr auto kDeviceL1NumBanks = "l1_num_banks";
constexpr auto kDeviceL1BankSize = "l1_bank_size";
constexpr auto kDeviceAddressAtFirstL1Bank = "address_at_first_l1_bank";
constexpr auto kDeviceAddressAtFirstL1CbBuffer = "address_at_first_l1_cb_buffer";
constexpr auto kDeviceNumBanksPerStorageCore = "num_banks_per_storage_core";
constexpr auto kDeviceNumComputeCores = "num_compute_cores";
constexpr auto kDeviceNumStorageCores = "num_storage_cores";
constexpr auto kDeviceTotalL1Memory = "total_l1_memory";
constexpr auto kDeviceTotalL1ForTensors = "total_l1_for_tensors";
constexpr auto kDeviceTotalL1ForInterleavedBuffers = "total_l1_for_interleaved_buffers";
constexpr auto kDeviceTotalL1ForShardedBuffers = "total_l1_for_sharded_buffers";
constexpr auto kDeviceCbLimit = "cb_limit";

// error info keys
constexpr auto kErrorType = "error_type";
constexpr auto kErrorMessage = "error_message";
constexpr auto kErrorOperation = "error_operation";

// stack trace key
constexpr auto kStackTrace = "stack_trace";

// current report format version
constexpr int kCurrentReportVersion = 1;
}  // namespace ttnn::graph
