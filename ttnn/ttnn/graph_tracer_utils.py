# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import re
import json


class GraphTracerUtils:
    @staticmethod
    def replace_control_chars(value):
        """Replaces all control characters (\x00-\x1F) with their Unicode escape \\u00XX representation."""

        def replacer(match):
            char = match.group(0)
            return f"\\u{ord(char):04X}"  # Converts to Unicode format \u0000, \u0001, etc.

        return re.sub(r"[\x00-\x1F]", replacer, value)

    @staticmethod
    def _convert_to_json(input_str, index):
        """Converts TTNN graph output into a properly formatted JSON object."""
        result = input_str
        # Remove '::' notation (for example, `TensorMemoryLayout::INTERLEAVED` → `TensorMemoryLayout~~~INTERLEAVED`)
        result = re.sub(r"::", "~~~", result)

        # Enclose all property names in double quotes and handle key-value pairs
        # for instance  Tensor(storage= → Tensor("storage":
        result = re.sub(r"(\b[a-zA-Z_][a-zA-Z0-9_]*(?:~~~\w*)*)=", r'"\1":', result)

        # Quote bare identifiers as values (identifiers following a colon)
        # MemoryConfig("memory_layout":TensorMemoryLayout~~~INTERLEAVED →
        # MemoryConfig("memory_layout":"TensorMemoryLayou~~~INTERLEAVED"
        result = re.sub(r"(:\s*)(\b[a-zA-Z_][a-zA-Z0-9_~]*(?:__\w*)*)(?=[,}\)\s])", r'\1"\2"', result)

        # Convert parentheses ( ) to curly braces { }
        # Tensor("storage":DeviceStorage( → Tensor{"storage":DeviceStorage{
        result = result.replace("(", "{").replace(")", "}")

        # Convert only the FIRST word before '{' into a quoted property
        # Tensor{"storage": → '"Tensor": {"storage":
        result = re.sub(r"^(\b[a-zA-Z_][a-zA-Z0-9_]*(?:__\w*)*)\s*{", r'"\1":{', result)

        # Remove property names directly between a property and '{' (e.g., 'Tensor{', 'DeviceStorage{', etc.)
        # Tensor{"storage":DeviceStorage{ → '{"storage":{"memory_config"
        result = re.sub(r"(\w+){", r"{", result)

        # Convert standalone words before '{' into properties
        # "memory_config":MemoryConfig{ → "memory_config":{
        result = re.sub(r"(\b[a-zA-Z_][a-zA-Z0-9_]*(?:__\w*)*)\s*{", r'"\1":{', result)

        # Replace back "~~~" to "::"
        # TensorMemoryLayout~~~INTERLEAVED → TensorMemoryLayout::INTERLEAVED
        result = re.sub(r"~~~", "::", result)

        # Replace {[]} by just [] for array data
        # "logical_shape":{[1, 2048, 4, 128]} → "logical_shape":[1, 2048, 4, 128]
        result = re.sub(r"{\s*(\[[^]]*\])\s*}", r"\1", result)

        # Handle the case {n, n, n... n}
        # "tile_shape":{32, 32} → "tile_shape":"{32, 32}"
        result = re.sub(r"(:\s*)\{(\s*\d+(?:\s*,\s*\d+)*\s*)\}", r'\1"{\2}"', result)

        # Handle null bytes and other control characters
        # some strings are '\x00' which breaks the json conversion
        # '\x00' → '\\u0000'
        result = GraphTracerUtils.replace_control_chars(result)

        # '\\u0000' → '{"argument": "\\u0000"}'
        result = (
            '{"arg'
            + str(index)
            + '": '
            + (("{" + result.strip() + "}") if re.match(r'^"\w+"\s*:', result.strip()) else ('"' + result + '"'))
            + "}"
        )

        # Fix quoted arrays: "argX": "[{...}]" -> "argX": [{...}]
        result = re.sub(r'("arg\d+"\s*:\s*)"(\[.*?\])"', r"\1\2", result)
        preresult = result
        # Convert nullopt arrays to simple values: [<nullopt>] -> "<nullopt>"
        result = re.sub(r"(\[\s*)<nullopt>(\s*\])", r'"<nullopt>"', result)
        # Convert unsupported type arrays to objects: [ unsupported type , std::reference... ] -> {"unsupported type": "std::reference..."}
        result = re.sub(r"\[\s*unsupported\s+type\s*,\s*([^]]+)\s*\]", r'{"unsupported type": "\1"}', result)
        # Convert unsupported type arrays to objects: "argX": [ unsupported type , std::reference... ] -> "argX": {"unsupported type": "std::reference..."}
        result = re.sub(
            r'("arg\d+"\s*:\s*)\[\s*unsupported type\s*,\s*([^]]+?)\s*\]', r'\1{"unsupported type": "\2"}', result
        )

        try:
            json_obj = json.loads(result)
            return json_obj
        except json.JSONDecodeError as e:
            return {"UnparsedElement": {"error": e, "element_info": result}}

    @staticmethod
    def serialize_arguments_to_json(operation_name, arguments):
        """Serialize the arguments of an operation into json"""
        serialized_list = []
        if operation_name == "":
            return

        i = 0
        for argument in arguments:
            json_obj = GraphTracerUtils._convert_to_json(argument, i)
            serialized_list.append(json_obj)
            i = i + 1
        return {"operation": operation_name, "arguments": serialized_list}

    @staticmethod
    def _infer_mesh_shape_from_device_ids(device_ids):
        """
        Infer mesh shape from device IDs.

        NOTE: This is a heuristic and cannot accurately distinguish between
        certain configurations (e.g., 1x8 vs 2x4 both use 8 devices).
        For accurate mesh shape, it should be captured from the actual MeshDevice
        at runtime and passed through the graph trace.

        For T3000/T3001 systems, device IDs follow patterns where mesh coordinates
        map to device IDs, but the mapping alone cannot determine the mesh topology.
        """
        if not device_ids or len(device_ids) == 0:
            return None

        num_devices = len(device_ids)

        # Parse device IDs (they come as strings)
        try:
            device_ints = sorted([int(d) for d in device_ids])
        except (ValueError, TypeError):
            return None

        # Single device
        if num_devices == 1:
            return [1, 1]

        # Common patterns for T3000 (1x4 mesh)
        # Devices are typically: [0,1,4,5] or [2,3,6,7]
        if num_devices == 4:
            # Check if it's likely a 1x4 configuration
            # T3000 maps mesh coordinates to device IDs in a specific pattern
            if set(device_ints) in [{0, 1, 4, 5}, {2, 3, 6, 7}]:
                return [1, 4]
            # Could also be 2x2
            elif device_ints == [0, 1, 2, 3] or device_ints == [4, 5, 6, 7]:
                # Consecutive devices might indicate 2x2
                return [2, 2]
            else:
                # Default to 1xN for unknown patterns
                return [1, 4]

        # For 8 devices: Cannot distinguish 1x8 from 2x4 from device IDs alone
        # Default to 1xN (most common for linear mesh)
        # WARNING: This may be incorrect for 2D mesh configurations
        return [1, num_devices]

    @staticmethod
    def serialize_graph(captured_graph):
        """Serialize the graph into a json document with device and placement information"""
        json_result = {"content": []}

        # First pass: extract device IDs and placement info from nodes
        operation_devices = {}  # Maps operation node counter to set of device IDs
        tensor_placements = {}  # Maps tensor_id to placement information

        for node in captured_graph:
            node_type = node.get("node_type", "")

            # Extract placement information from tensor nodes
            if node_type == "tensor":
                params = node.get("params", {})
                tensor_id = params.get("tensor_id")
                if tensor_id and "placement" in params:
                    placement_info = {
                        "placement": params.get("placement"),
                        "distribution_shape": params.get("distribution_shape"),
                    }
                    # Add actual mesh device shape if available (2D shape like [2, 4])
                    if "mesh_device_shape" in params:
                        placement_info["mesh_device_shape"] = params.get("mesh_device_shape")
                    tensor_placements[tensor_id] = placement_info

            # Track buffer allocations/deallocations with device IDs
            if node_type in ["buffer", "buffer_allocate", "buffer_deallocate"]:
                device_id = node.get("params", {}).get("device_id")
                if device_id:
                    # Find which operation this buffer belongs to by traversing connections
                    connections = node.get("connections", [])
                    for conn_id in connections:
                        if conn_id < len(captured_graph):
                            conn_node = captured_graph[conn_id]
                            # Walk up to find the parent function_start node
                            if conn_node.get("node_type") == "tensor":
                                # Check tensor's connections to find operations
                                tensor_conns = conn_node.get("connections", [])
                                for tensor_conn_id in tensor_conns:
                                    if tensor_conn_id < len(captured_graph):
                                        op_node = captured_graph[tensor_conn_id]
                                        if op_node.get("node_type") == "function_start":
                                            if tensor_conn_id not in operation_devices:
                                                operation_devices[tensor_conn_id] = set()
                                            operation_devices[tensor_conn_id].add(device_id)

        # Second pass: serialize operations with device and placement metadata
        for idx, node in enumerate(captured_graph):
            arguments = node.get("arguments", [])
            if not arguments:
                continue

            operation_name = node.get("params", {}).get("name", "")
            operation = GraphTracerUtils.serialize_arguments_to_json(operation_name, arguments)

            # Add device information if available
            if idx in operation_devices and operation:
                device_ids = sorted(list(operation_devices[idx]))
                operation["device_ids"] = device_ids
                operation["device_count"] = len(device_ids)

            # Add placement information from input tensors
            # Tensor nodes connect TO function nodes, so we need to find tensors that connect to this operation
            if operation and node.get("node_type") == "function_start":
                placements_info = []
                # Search all tensor nodes to find which ones connect to this function
                for other_node in captured_graph:
                    if other_node.get("node_type") == "tensor":
                        # Check if this tensor connects to our function
                        if idx in other_node.get("connections", []):
                            tensor_id = other_node.get("params", {}).get("tensor_id")
                            if tensor_id in tensor_placements:
                                placements_info.append(tensor_placements[tensor_id])

                if placements_info:
                    operation["tensor_placements"] = placements_info

            if operation:
                json_result["content"].append(operation)

        return json_result
