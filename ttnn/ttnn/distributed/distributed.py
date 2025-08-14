# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools

from typing import List, Dict, Optional, Callable, Tuple, Optional, Callable, Union, List

import ttnn


def get_mesh_device_core_grid(mesh_device):
    compute_with_storage_grid_size = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


MeshDevice = ttnn._ttnn.multi_device.MeshDevice
MeshDevice.core_grid = property(get_mesh_device_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType


def _compute_axis_part_representatives(mesh_device, mesh_rank):
    mapping = {axis: {} for axis in range(mesh_rank)}
    rows, cols = mesh_device.shape
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            device_id = mesh_device.get_device_id(coord)
            for axis in range(mesh_rank):
                part_index = int(coord[axis])
                if part_index not in mapping[axis]:
                    mapping[axis][part_index] = device_id
    return mapping


def _compute_axis_offsets_and_sizes(shards, placements, mesh_shape, axis_part_to_rep_device):
    axis_to_offsets = {}
    axis_to_sizes = {}
    axis_to_parts = {}
    for axis in range(len(mesh_shape)):
        placement = placements[axis]
        if isinstance(placement, ttnn.PlacementShard):
            tensor_dim = placement.dim
            parts = int(mesh_shape[axis])
            sizes = []
            for p in range(parts):
                rep_device_id = axis_part_to_rep_device[axis].get(p)
                if rep_device_id is None:
                    sizes.append(0)
                else:
                    sizes.append(int(list(shards[rep_device_id].shape)[tensor_dim]))
            offsets = []
            cumulative = 0
            for size in sizes:
                offsets.append(cumulative)
                cumulative += size
            axis_to_offsets[axis] = offsets
            axis_to_sizes[axis] = sizes
            axis_to_parts[axis] = parts
    return axis_to_offsets, axis_to_sizes, axis_to_parts


def _map_tensor_dim_to_mesh_axis(placements):
    dim_to_axis = {}
    for axis, placement in enumerate(placements):
        if isinstance(placement, ttnn.PlacementShard):
            tensor_dim = placement.dim
            if tensor_dim not in dim_to_axis:
                dim_to_axis[tensor_dim] = axis
    return dim_to_axis


def _format_inclusive_slices_for_device(
    device_id,
    device_coord,
    tensor,
    shards,
    mesh_shape,
    dim_to_axis,
    axis_to_offsets,
    axis_to_sizes,
):
    if device_id >= len(shards):
        return ""
    shard = shards[device_id]
    shape_str = str(list(shard.shape))
    dtype_str = str(shard.dtype).split(".")[-1]
    layout_str = str(shard.layout).split(".")[-1]

    dist_coord = device_coord
    tensor_global_shape = list(tensor.shape)
    slice_tokens = []
    for tensor_dim, dim_size in enumerate(tensor_global_shape):
        if tensor_dim in dim_to_axis:
            axis = dim_to_axis[tensor_dim]
            part_index = int(dist_coord[axis])
            start = axis_to_offsets[axis][part_index]
            size = axis_to_sizes[axis][part_index]
            end_inclusive = start + size - 1
            slice_tokens.append(f"{start}:{end_inclusive}")
        else:
            end_inclusive = int(dim_size) - 1
            slice_tokens.append(f"0:{end_inclusive}")
    header = ", ".join(slice_tokens)
    return f"[{header}]\nShape: {shape_str}\nDtype: {dtype_str}\nLayout: {layout_str}"


def _compute_device_slice_key(
    device_coord,
    tensor,
    mesh_shape,
    dim_to_axis,
    axis_to_offsets,
    axis_to_sizes,
):
    dist_coord = device_coord
    tensor_global_shape = list(tensor.shape)
    key = []
    for tensor_dim, dim_size in enumerate(tensor_global_shape):
        if tensor_dim in dim_to_axis:
            axis = dim_to_axis[tensor_dim]
            part_index = int(dist_coord[axis])
            start = axis_to_offsets[axis][part_index]
            size = axis_to_sizes[axis][part_index]
            end_inclusive = start + size - 1
            key.append((int(start), int(end_inclusive)))
        else:
            end_inclusive = int(dim_size) - 1
            key.append((0, int(end_inclusive)))
    return tuple(key)


def _build_replication_color_map(
    mesh_device,
    tensor,
    mesh_shape,
    dim_to_axis,
    axis_to_offsets,
    axis_to_sizes,
):
    # Distinguish groups by identical slice keys across all tensor dims
    key_to_group = {}
    device_to_group = {}
    group_index = 0
    rows, cols = mesh_device.shape
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            device_id = mesh_device.get_device_id(coord)
            key = _compute_device_slice_key(coord, tensor, mesh_shape, dim_to_axis, axis_to_offsets, axis_to_sizes)
            if key not in key_to_group:
                key_to_group[key] = group_index
                group_index += 1
            device_to_group[device_id] = key_to_group[key]

    # Generate a distinct color palette sized to number of unique groups
    num_groups = len(key_to_group)

    def _hsv_to_hex(h: float, s: float, v: float) -> str:
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    def _generate_distinct_colors(n: int) -> list:
        if n <= 0:
            return []
        # Muted palette (lower saturation/value to dim colors)
        s = 0.48
        v = 0.80
        return [_hsv_to_hex((i / float(n)) % 1.0, s, v) for i in range(n)]

    palette = _generate_distinct_colors(num_groups)
    # Fallback to a neutral color if something goes wrong
    if not palette:
        palette = ["#9e9e9e"]

    device_to_style = {did: f"on {palette[grp % len(palette)]}" for did, grp in device_to_group.items()}
    return device_to_style


def _get_rich_table(
    mesh_device: "ttnn.MeshDevice",
    tensor: "ttnn.Tensor" = None,
    storage_type: ttnn.StorageType = ttnn.StorageType.DEVICE,
    style_cell: Optional[Callable] = None,
    annotate_cell: Optional[Callable] = None,
):
    from rich import box, padding
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    from loguru import logger

    CELL_SIZE = 30

    # Setup rich table
    if storage_type == ttnn.StorageType.DEVICE:
        try:
            rows, cols = mesh_device.shape
            view = mesh_device.get_view()
            fully_local = all(view.is_local(coord) for coord in ttnn.MeshCoordinateRange(view.shape()))
        except AttributeError as e:
            logger.error("Error getting mesh device info: {}.", e)
            rows, cols = 0, 0
    else:
        try:
            host_buffer = tensor.host_buffer()
            rows, cols = host_buffer.shape()
            fully_local = all(host_buffer.is_local(coord) for coord in ttnn.MeshCoordinateRange(host_buffer.shape()))
        except AttributeError as e:
            logger.error("Error getting host buffer info: {}.", e)
            rows, cols = 0, 0

    if tensor:
        table_title = f"Tensor(storage: {storage_type})"
    else:
        table_title = f"MeshDevice(rows={rows}, cols={cols})"

    table_view = Table(
        title=table_title,
        show_header=False,
        show_footer=False,
        box=box.SQUARE,
        expand=False,
        show_lines=True,
        padding=(0, 0),
    )

    for _ in range(cols):
        table_view.add_column(justify="center", vertical="middle", width=CELL_SIZE)

    # Populate table
    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            try:
                coord = ttnn.MeshCoordinate(row_idx, col_idx)
                if storage_type == ttnn.StorageType.DEVICE:
                    locality = "Local\n" if view.is_local(coord) else "Remote\n"
                    device_id = mesh_device.get_device_id(ttnn.MeshCoordinate(row_idx, col_idx))
                    device_id_str = f"Dev. ID: {device_id}\n" if view.is_local(coord) else "Unknown\n"
                else:
                    locality = "Local\n" if host_buffer.is_local(coord) else "Remote\n"
                    device_id = row_idx * cols + col_idx
                    device_id_str = ""

                locality = "" if fully_local else locality
                coords = f"({row_idx}, {col_idx})\n"
                annotation = annotate_cell(device_id, coord) if annotate_cell and device_id is not None else ""

                cell_content = Text(f"{locality}{device_id_str}{coords}{annotation}", justify="center")
                cell_content.truncate(CELL_SIZE * 4, overflow="ellipsis")  # 4 lines max
            except AttributeError as e:
                logger.error("Error formatting cell content at row {}, col {}: {}.", row_idx, col_idx, e)
                cell_content = Text("Error", justify="center")

            cell_style = style_cell(device_id) if style_cell and device_id is not None else None
            cell = Align(cell_content, "center", vertical="middle")
            if cell_style:
                cell.style = cell_style
            row_cells.append(cell)
        table_view.add_row(*row_cells)
    return table_view


def visualize_mesh_device(mesh_device: "ttnn.MeshDevice"):
    """
    Visualize the device mesh.
    """
    from rich.console import Console

    mesh_table = _get_rich_table(mesh_device)
    Console().print(mesh_table)


def visualize_tensor(tensor: "ttnn.Tensor"):
    """
    Visualize tensor distribution across the mesh.
    """
    from rich.console import Console
    from rich.panel import Panel
    from loguru import logger

    if tensor.storage_type() == ttnn.StorageType.HOST:
        visualize_tensor_host(tensor)
        return

    console = Console()

    try:
        shards = ttnn.get_device_tensors(tensor)
        topology = tensor.tensor_topology()
        placements = topology.placements()
        mesh_shape = list(topology.mesh_shape())
        mesh_coords = topology.mesh_coords()

        mesh_rank = len(mesh_shape)
        axis_part_to_rep_device = _compute_axis_part_representatives(tensor.device(), mesh_rank)
        axis_to_offsets, axis_to_sizes, axis_to_parts = _compute_axis_offsets_and_sizes(
            shards, placements, mesh_shape, axis_part_to_rep_device
        )
        dim_to_axis = _map_tensor_dim_to_mesh_axis(placements)

        placement_panel = Panel(str(placements), title="Placement Configuration", border_style="blue")

        # Build color map so devices owning identical global slices share the same background color
        device_to_style = _build_replication_color_map(
            mesh_device=tensor.device(),
            tensor=tensor,
            mesh_shape=mesh_shape,
            dim_to_axis=dim_to_axis,
            axis_to_offsets=axis_to_offsets,
            axis_to_sizes=axis_to_sizes,
        )

        def annotate_with_shard_info(device_id, device_coord):
            return _format_inclusive_slices_for_device(
                device_id,
                device_coord,
                tensor,
                shards,
                mesh_shape,
                dim_to_axis,
                axis_to_offsets,
                axis_to_sizes,
            )

        # Generate the mesh table with shard annotations
        mesh_table = _get_rich_table(
            tensor.device(),
            tensor,
            annotate_cell=annotate_with_shard_info,
            style_cell=lambda device_id: device_to_style.get(device_id),
            storage_type=tensor.storage_type(),
        )

        console.print(placement_panel)
        console.print(mesh_table)

    except Exception as e:
        logger.error(f"Error visualizing tensor: {e}")


def visualize_tensor_host(tensor: "ttnn.Tensor"):
    """
    Visualize tensor distribution when tensor is on HOST storage.
    This function is fully decoupled from device visualization and does not dereference mesh_device.
    """
    from rich.console import Console
    from rich.panel import Panel
    from loguru import logger

    console = Console()
    try:
        # Topology and shards
        shards = ttnn.get_device_tensors(tensor)
        topology = tensor.tensor_topology()
        placements = topology.placements()
        mesh_shape = list(topology.mesh_shape())
        mesh_coords = topology.mesh_coords()

        # Host grid shape
        try:
            host_buffer = tensor.host_buffer()
            rows, cols = host_buffer.shape()
            rows, cols = int(rows), int(cols)
        except Exception:
            rows, cols = int(mesh_shape[0]), int(mesh_shape[1])

        # Representatives derived from logical coords (no device deref)
        mesh_rank = len(mesh_shape)
        axis_part_to_rep_device = {}
        for axis in range(mesh_rank):
            part_map = {}
            for did, mc in enumerate(mesh_coords):
                part_index = int(mc[axis])
                if part_index not in part_map:
                    part_map[part_index] = did
            axis_part_to_rep_device[axis] = part_map

        axis_to_offsets, axis_to_sizes, axis_to_parts = _compute_axis_offsets_and_sizes(
            shards, placements, mesh_shape, axis_part_to_rep_device
        )
        dim_to_axis = _map_tensor_dim_to_mesh_axis(placements)

        placement_panel = Panel(str(placements), title="Placement Configuration", border_style="blue")

        # Map logical coord -> device_id
        coord_to_device_id = {(int(mc[0]), int(mc[1])): did for did, mc in enumerate(mesh_coords)}

        # Group devices by identical global slice keys using logical coords
        key_to_group = {}
        device_to_group = {}
        group_index = 0
        for did, logical_coord in enumerate(mesh_coords):
            key = _compute_device_slice_key(
                logical_coord, tensor, mesh_shape, dim_to_axis, axis_to_offsets, axis_to_sizes
            )
            if key not in key_to_group:
                key_to_group[key] = group_index
                group_index += 1
            device_to_group[did] = key_to_group[key]

        # Muted HSV palette sized to number of groups
        num_groups = len(key_to_group)

        def _hsv_to_hex(h: float, s: float, v: float) -> str:
            i = int(h * 6.0)
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - f * s)
            t = v * (1.0 - (1.0 - f) * s)
            i = i % 6
            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

        def _generate_distinct_colors(n: int) -> list:
            if n <= 0:
                return []
            s = 0.48
            v = 0.80
            return [_hsv_to_hex((i / float(n)) % 1.0, s, v) for i in range(n)]

        palette = _generate_distinct_colors(num_groups) or ["#9e9e9e"]
        device_to_style = {did: f"on {palette[grp % len(palette)]}" for did, grp in device_to_group.items()}

        # Style mapping for host cells (row-major linear ids)
        def style_cell(host_linear_id, _cols=cols, _map=coord_to_device_id, _styles=device_to_style):
            try:
                r = int(host_linear_id // _cols)
                c = int(host_linear_id % _cols)
                did = _map.get((r, c))
                return _styles.get(did) if did is not None else None
            except Exception:
                return None

        # Annotation per host cell using logical coords
        def annotate_cell(host_linear_id, host_coord):
            try:
                r = int(host_coord[0])
                c = int(host_coord[1])
                did = coord_to_device_id.get((r, c))
                if did is None:
                    return ""
                logical_coord = ttnn.MeshCoordinate(r, c)
                return _format_inclusive_slices_for_device(
                    did,
                    logical_coord,
                    tensor,
                    shards,
                    mesh_shape,
                    dim_to_axis,
                    axis_to_offsets,
                    axis_to_sizes,
                )
            except Exception:
                return ""

        # Render using host storage path in _get_rich_table (no device IDs printed)
        mesh_table = _get_rich_table(
            tensor.device(),
            tensor,
            annotate_cell=annotate_cell,
            style_cell=style_cell,
            storage_type=tensor.storage_type(),
        )

        console.print(placement_panel)
        console.print(mesh_table)

    except Exception as e:
        logger.error(f"Error visualizing host tensor: {e}")


def visualize_system_mesh():
    """
    Print SystemMesh global and local shapes.
    """
    from rich.console import Console
    from loguru import logger

    try:
        system_mesh_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()
        global_shape = system_mesh_desc.shape()
        local_shape = system_mesh_desc.local_shape()
    except Exception as e:
        logger.error(f"Error accessing SystemMesh: {e}")
        return

    console = Console()
    console.print(f"\n[bold green]SystemMesh Global Shape: {global_shape}[/bold green]")
    console.print(f"\n[bold blue]SystemMesh Local Shape: {local_shape}[/bold blue]\n")
    console.print(create_system_mesh_table())


def create_system_mesh_table():
    """
    Create a visual table representation of the system mesh layout.
    """
    from rich import box
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    from rich.style import Style
    from loguru import logger

    CELL_SIZE = 30

    try:
        system_mesh_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()

        # TODO: Remove shape indexing workaround after exposing subscripts in pybind11
        global_shape = tuple(system_mesh_desc.shape())
        local_shape = tuple(system_mesh_desc.local_shape())
        rows, cols = global_shape[0], global_shape[1]
        local_rows, local_cols = local_shape[0], local_shape[1]
    except Exception as e:
        logger.error("Error getting system mesh shapes: {}.", e)
        return None

    all_local = system_mesh_desc.all_local()

    mesh_table = Table(
        title=f"SystemMesh Global Shape: ({rows}, {cols}) | Local Shape: ({local_rows}, {local_cols})",
        show_header=False,
        show_footer=False,
        box=box.SQUARE,
        expand=False,
        show_lines=True,
        padding=(0, 0),
    )

    for _ in range(cols):
        mesh_table.add_column(justify="center", vertical="middle", width=CELL_SIZE)

    # Populate table
    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            try:
                coords = f"({row_idx}, {col_idx})"
                coord = ttnn.MeshCoordinate(row_idx, col_idx)

                # Create cell content
                if all_local:
                    device_id = f"Dev. ID: {system_mesh_desc.get_device_id(coord)}"
                    cell_content = Text(f"{device_id}\n{coords}", justify="center")
                    cell_style = Style(bgcolor="dark_green")
                else:
                    is_local = system_mesh_desc.is_local(coord)
                    locality = "Local\n" if is_local else "Remote\n"
                    device_id = f"Dev. ID: {system_mesh_desc.get_device_id(coord)}\n" if is_local else "Unknown\n"
                    cell_content = Text(f"{locality}{device_id}{coords}", justify="center")
                    cell_style = None

                cell_content.truncate(CELL_SIZE * 3, overflow="ellipsis")
            except Exception as e:
                logger.error("Error formatting cell content at row {}, col {}: {}.", row_idx, col_idx, e)
                cell_content = Text("Error", justify="center")
                cell_style = None

            cell = Align(cell_content, "center", vertical="middle")
            if cell_style:
                cell.style = cell_style
            row_cells.append(cell)
        mesh_table.add_row(*row_cells)

    return mesh_table


def get_num_devices() -> List[int]:
    return ttnn._ttnn.device.GetNumAvailableDevices()


def get_num_pcie_devices() -> int:
    return ttnn._ttnn.device.GetNumPCIeDevices()


def get_pcie_device_ids() -> List[int]:
    num_pcie_devices = get_num_pcie_devices()
    return list(range(num_pcie_devices))


def get_device_ids() -> List[int]:
    num_devices = get_num_devices()
    return list(range(num_devices))


def open_mesh_device(
    mesh_shape: ttnn.MeshShape = None,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    num_command_queues: int = 1,
    dispatch_core_config: ttnn.DispatchCoreConfig = None,
    offset: Optional[ttnn.MeshCoordinate] = None,
    physical_device_ids: List[int] = [],
    worker_l1_size: int = ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
):
    """
    Open a mesh device with the specified configuration.

    Args:
        mesh_shape (ttnn.MeshShape, optional): The shape of the mesh device. Defaults to the global shape of the system mesh.
        l1_small_size (int, optional): Size of the L1 small memory. Defaults to ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE.
        trace_region_size (int, optional): Size of the trace region. Defaults to ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE.
        num_command_queues (int, optional): Number of command queues. Defaults to 1.
        dispatch_core_type (int, optional): Type of dispatch core. Defaults to DispatchCoreType.WORKER.
        offset (ttnn.MeshCoordinate, optional): Offset in logical mesh coordinates for the mesh device. Defaults to None.
        physical_device_ids (List[int], optional): List of physical device IDs to use. Defaults to [].
        worker_l1_size (int, optional): Size of the usable worker L1 memory. Defaults to ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE.

    Returns:
        ttnn._ttnn.multi_device.MeshDevice: The opened mesh device.

    """
    return ttnn._ttnn.multi_device.open_mesh_device(
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
        dispatch_core_config=dispatch_core_config or ttnn.DispatchCoreConfig(),
        mesh_shape=mesh_shape,
        offset=offset,
        physical_device_ids=physical_device_ids,
        worker_l1_size=worker_l1_size,
    )


def close_mesh_device(mesh_device):
    """
    close_mesh_device(multi_device: ttnn.Multi) -> None:

    Close the device and remove it from the device cache.
    """
    return ttnn._ttnn.multi_device.close_mesh_device(mesh_device)


@contextlib.contextmanager
def create_mesh_device(*args, **kwargs):
    """
    create_mesh_device(*args, **kwargs) -> ttnn.MeshDevice

    Context manager for opening and closing a device.
    """
    mesh_device = open_mesh_device(*args, **kwargs)
    try:
        yield mesh_device
    finally:
        close_mesh_device(mesh_device)


# Temporary stubs to accomodate migration of Python-based sharding / concatenation to C++.
# TODO: #24114 - When migration of concatenation is complete, remove these stubs.
TensorToMesh = ttnn.CppTensorToMesh
MeshToTensor = ttnn.CppMeshToTensor


# Workaround needed to differentiate mappers created by `ReplicateTensorToMesh`, which use a different file name used for caching.
class ReplicateTensorToMeshWrapper:
    def __init__(self, mapper: ttnn.CppTensorToMesh):
        self._mapper = mapper

    def unwrap(self):
        return self._mapper


# Deprecated. Use `ttnn.replicate_tensor_to_mesh_mapper` directly.
def ReplicateTensorToMesh(mesh_device: MeshDevice):
    mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
    return ReplicateTensorToMeshWrapper(mapper)


# Deprecated. Use `ttnn.shard_tensor_to_mesh_mapper` directly.
def ShardTensorToMesh(mesh_device: MeshDevice, dim: int):
    return ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim)


# Deprecated. Use `ttnn.concat_mesh_to_tensor_composer` directly.
def ConcatMeshToTensor(mesh_device: MeshDevice, dim: int):
    return ttnn.concat_mesh_to_tensor_composer(mesh_device, dim)


# Deprecated. Use `ttnn.create_mesh_mapper` directly.
def ShardTensor2dMesh(mesh_device: MeshDevice, mesh_shape: Tuple[int, int], dims: Tuple[Optional[int], Optional[int]]):
    return ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [
                ttnn.PlacementReplicate() if dims[0] is None else ttnn.PlacementShard(dims[0]),
                ttnn.PlacementReplicate() if dims[1] is None else ttnn.PlacementShard(dims[1]),
            ],
            ttnn.MeshShape(mesh_shape[0], mesh_shape[1]),
        ),
    )


# Deprecated. Use `ttnn.create_mesh_composer` directly.
def ConcatMesh2dToTensor(mesh_device: MeshDevice, mesh_shape: Tuple[int, int], dims: Tuple[int, int]):
    return ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            [dims[0], dims[1]],
            ttnn.MeshShape(mesh_shape[0], mesh_shape[1]),
        ),
    )


@contextlib.contextmanager
def distribute(default: Union[ttnn.CppTensorToMesh, ReplicateTensorToMeshWrapper, ttnn.CppMeshToTensor]):
    """
    Context manager to temporarily modify the behavior of ttnn.from_torch and ttnn.to_torch to use the specified
    mesh_mapper or mesh_composer for tensor distribution and composition to/from MeshDevice.
    Invocations of ttnn.from_torch(..) will use the mesh_mapper as defined by the default in ttnn.distribute.
    Invocations of ttnn.to_torch(..) will use the mesh_composer as defined by the default in ttnn.distribute.

    Args:
        mesh_mapper_or_composer (Union[ttnn.CppTensorToMesh, ReplicateTensorToMeshWrapper, MeshToTensor]): An instance of either TensorToMesh or MeshToTensor
            used to map tensors to a mesh or compose tensors from a mesh.

    Example:
        with distribute(ShardTensorToMesh(mesh_device, dim=3)):
            # Code here will use the default mapper
            result = ttnn.from_torch(torch_tensor)

        is equivalent to:
        result = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMesh(mesh_device, dim=3))
    """
    _original_to_torch = ttnn.to_torch
    _original_from_torch = ttnn.from_torch

    try:
        if isinstance(default, ttnn.CppTensorToMesh) or isinstance(default, ReplicateTensorToMeshWrapper):
            ttnn.from_torch = functools.partial(_original_from_torch, mesh_mapper=default)
        elif isinstance(default, ttnn.CppMeshToTensor):
            ttnn.to_torch = functools.partial(_original_to_torch, mesh_composer=default)
        else:
            raise ValueError("Argument must be an instance of either TensorToMesh or MeshToTensor.")
        yield

    finally:
        # Restore the original functions
        ttnn.from_torch = _original_from_torch
        ttnn.to_torch = _original_to_torch


__all__ = []
