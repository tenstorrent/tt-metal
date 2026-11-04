# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


def gen_tensor(dim, per_device_output_shape, mesh_axes, mesh_shape, cluster_axis, scheme="random"):
    factor = 0

    if cluster_axis is None:
        # Linearized grid case - treat all devices as a single linear sequence
        total_devices = mesh_shape[0] * mesh_shape[1]

        # Create output tensors for each device in the linearized grid
        single_device_output_tensors = []
        for device_idx in range(total_devices):
            if scheme == "random":
                single_device_output_tensors.append(torch.rand(per_device_output_shape))
            elif scheme == "sequential":
                single_device_output_tensors.append(torch.ones(per_device_output_shape) * factor)
                factor += 1
            else:
                raise ValueError(f"Invalid scheme: {scheme}")

        # Create input tensor by concatenating all slices along dim
        torch_input_tensor = torch.cat(single_device_output_tensors, dim=dim)

        # For output tensor, we need to arrange slices according to mesh layout
        # First, arrange them in rows and columns according to mesh_shape
        output_tensors_2d = []
        device_idx = 0
        for row in range(mesh_shape[0]):
            row_tensors = []
            for col in range(mesh_shape[1]):
                row_tensors.append(single_device_output_tensors[device_idx])
                device_idx += 1
            # Concatenate along mesh_axes[1] (column dimension)
            row_tensor = torch.cat(row_tensors, dim=mesh_axes[1])
            output_tensors_2d.append(row_tensor)

        # Concatenate rows along mesh_axes[0] (row dimension)
        torch_output_tensor = torch.cat(output_tensors_2d, dim=mesh_axes[0])

        # Create input tensor with same structure as output for mesh mapping
        input_tensors_2d = []
        for row in range(mesh_shape[0]):
            row_tensors = []
            for col in range(mesh_shape[1]):
                row_tensors.append(torch_input_tensor)
            row_tensor = torch.cat(row_tensors, dim=mesh_axes[1])
            input_tensors_2d.append(row_tensor)
        torch_input_tensor = torch.cat(input_tensors_2d, dim=mesh_axes[0])

        return torch_input_tensor, torch_output_tensor

    # Original logic for when cluster_axis is specified
    non_cluster_axes = [i for i in range(len(mesh_axes)) if i != cluster_axis]

    torch_input_tensor = None
    for axis in non_cluster_axes:
        per_axis_tensor = None
        temp_per_axis_tensors = []
        temp_per_axis_output_tensors = []
        for _ in range(mesh_shape[axis]):
            single_device_output_tensors = []
            for _ in range(mesh_shape[cluster_axis]):
                if scheme == "random":
                    single_device_output_tensors.append(torch.rand(per_device_output_shape))
                elif scheme == "sequential":
                    single_device_output_tensors.append(torch.ones(per_device_output_shape) * factor)
                    factor += 1
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
            single_device_input_tensor = torch.cat(single_device_output_tensors, dim=dim)
            cluster_axis_tensors = []
            for _ in range(mesh_shape[cluster_axis]):
                cluster_axis_tensors.append(single_device_input_tensor)

            cluster_axis_tensor = torch.cat(cluster_axis_tensors, dim=mesh_axes[cluster_axis])
            cluster_axis_output_tensor = torch.cat(single_device_output_tensors, dim=mesh_axes[cluster_axis])

            temp_per_axis_tensors.append(cluster_axis_tensor)
            temp_per_axis_output_tensors.append(cluster_axis_output_tensor)
        per_axis_tensor = torch.cat(temp_per_axis_tensors, dim=axis)
        per_axis_output_tensor = torch.cat(temp_per_axis_output_tensors, dim=axis)

        if torch_input_tensor is None:
            torch_input_tensor = per_axis_tensor
            torch_output_tensor = per_axis_output_tensor
        else:
            torch_input_tensor = torch.cat([torch_input_tensor, per_axis_tensor], dim=axis)
            torch_output_tensor = torch.cat([torch_output_tensor, per_axis_output_tensor], dim=axis)

    return torch_input_tensor, torch_output_tensor
