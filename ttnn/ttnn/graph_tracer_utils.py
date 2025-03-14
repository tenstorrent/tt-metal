# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
    def serialize_graph(captured_graph):
        """Serialize the graph into a json document"""
        json_result = {"content": []}
        for node in captured_graph:
            arguments = node["arguments"]
            if not arguments:
                continue

            operation_name = node["params"].get("name", "")
            operation = GraphTracerUtils.serialize_arguments_to_json(operation_name, arguments)
            json_result["content"].append(operation)

        return json_result
