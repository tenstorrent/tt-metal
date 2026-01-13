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


# ====================================================================
# TENSOR VISUALIZATION UTILITIES
# ====================================================================


class TensorShardingInfo:
    """Class to hold all tensor sharding information for visualization."""

    def __init__(self, tensor, shards):
        self.tensor = tensor
        self.shards = shards
        self.topology = tensor.tensor_topology()
        self.placements = self.topology.placements()
        self.distribution_shape = list(self.topology.distribution_shape())
        self.mesh_shape = list(tensor.device().shape) if tensor.device() else list(tensor.host_buffer().shape())
        self.mesh_coords = self.topology.mesh_coords()

        assert len(self.mesh_shape) <= 2, "Tensor visualization only supports up to 2D meshes"

        self.reverse_coord_mapper = self._create_coordinate_mapper()
        self.dim_to_axis = self._compute_dim_to_axis_mapping()
        self.global_tensor_shape = self._compute_global_tensor_shape()
        self.axis_representatives = self._compute_axis_representatives()
        self.device_to_shard_map = self._compute_device_to_shard_mapping()
        self.axis_offsets, self.axis_sizes = self._compute_axis_offsets_and_sizes()

    def _compute_dim_to_axis_mapping(self):
        """Map tensor dimensions to mesh axes based on placements.
        Ex. placements = [PlacementReplicate(), PlacementShard(3)] --> returns {3: 1}"""
        mapping = {}
        for axis, placement in enumerate(self.placements):
            if isinstance(placement, ttnn.PlacementShard):
                tensor_dim = placement.dim
                if tensor_dim not in mapping:
                    mapping[tensor_dim] = axis
        return mapping

    def _create_coordinate_mapper(self):
        """Create mapping from distribution coordinates to mesh coordinates."""
        self.mesh_coords_ordered = ttnn.compute_distribution_to_mesh_mapping(
            ttnn.MeshShape(self.distribution_shape), ttnn.MeshShape(self.mesh_shape)
        )

        mesh_to_distribution_map = {}
        coord_idx = 0
        for distribution_coord in ttnn.MeshCoordinateRange(ttnn.MeshShape(self.distribution_shape)):
            if coord_idx < len(self.mesh_coords_ordered):
                distribution_key = ttnn.MeshCoordinate(
                    [distribution_coord[i] for i in range(distribution_coord.dims())]
                )
                mesh_to_distribution_map[self.mesh_coords_ordered[coord_idx]] = distribution_key
                coord_idx += 1

        def mapper(mesh_coord):
            return mesh_to_distribution_map.get(mesh_coord, mesh_coord)

        return mapper

    def _iter_distribution_to_mesh_coords(self):
        """Generator that yields (distribution_coord, mesh_coord) pairs using C++ results directly."""
        coord_idx = 0
        for distribution_coord in ttnn.MeshCoordinateRange(ttnn.MeshShape(self.distribution_shape)):
            if coord_idx < len(self.mesh_coords_ordered):
                yield distribution_coord, self.mesh_coords_ordered[coord_idx]
                coord_idx += 1

    def _compute_global_tensor_shape(self):
        """Compute the global tensor shape using distribution_shape for scaling."""
        shape = list(self.tensor.shape)
        for tensor_dim, axis in self.dim_to_axis.items():
            shape[tensor_dim] *= self.distribution_shape[axis]
        return ttnn.Shape(shape)

    def _compute_axis_representatives(self):
        """Find representative device for each axis partition in distribution coordinate space."""
        distribution_shape_rank = len(self.distribution_shape)
        mapping = {axis: {} for axis in range(distribution_shape_rank)}

        if self.tensor.storage_type() == ttnn.StorageType.HOST:
            for host_idx, mesh_coord in enumerate(self.mesh_coords):
                distribution_coord = self.reverse_coord_mapper(mesh_coord)

                for axis in range(min(distribution_shape_rank, distribution_coord.dims())):
                    part_index = int(distribution_coord[axis])
                    if part_index not in mapping[axis]:
                        mapping[axis][part_index] = host_idx
        else:
            mesh_device = self.tensor.device()

            # Generate distribution->mesh coordinate pairs using the coord_mapper
            for distribution_coord, mesh_coord in self._iter_distribution_to_mesh_coords():
                try:
                    device_id = mesh_device.get_device_id(mesh_coord)

                    # Record representative for each distribution axis
                    if distribution_shape_rank == 1:
                        mapping[0][distribution_coord[0]] = device_id
                    elif distribution_shape_rank == 2:
                        mapping[0][distribution_coord[0]] = device_id
                        mapping[1][distribution_coord[1]] = device_id
                except:
                    continue

        return mapping

    def _compute_device_to_shard_mapping(self):
        """Create mapping from device IDs to shard indices."""
        from loguru import logger

        if self.tensor.storage_type() == ttnn.StorageType.HOST:
            return {host_idx: host_idx for host_idx, _ in enumerate(self.mesh_coords)}

        mapping = {}
        try:
            mesh_device = self.tensor.device()
            view = mesh_device.get_view()
            for i, mesh_coord in enumerate(self.mesh_coords):
                if not view.is_local(mesh_coord):
                    continue
                dev_id = mesh_device.get_device_id(mesh_coord)
                mapping[dev_id] = i

        except Exception as e:
            logger.warning(
                f"Fallback in TensorShardingInfo._compute_device_to_shard_mapping: {e}. "
                "Assuming device ID equals shard index. This may not be valid in complex distributed scenarios."
            )
            mapping = {i: i for i in range(len(self.shards))}

        return mapping

    def _compute_axis_offsets_and_sizes(self):
        """Compute offset and size information for each axis using distribution_shape."""
        axis_offsets = {}
        axis_sizes = {}

        max_axis = min(len(self.distribution_shape), len(self.placements))

        for axis in range(max_axis):
            placement = self.placements[axis]
            if isinstance(placement, ttnn.PlacementShard):
                tensor_dim = placement.dim
                parts = int(self.distribution_shape[axis])
                sizes = []

                for p in range(parts):
                    rep_device_id = self.axis_representatives[axis].get(p)
                    if rep_device_id is None:
                        sizes.append(0)
                    else:
                        shard_index = self.device_to_shard_map.get(rep_device_id, None)
                        if shard_index is not None and shard_index < len(self.shards):
                            shard_shape = list(self.shards[shard_index].shape)
                            if tensor_dim < len(shard_shape):
                                sizes.append(int(shard_shape[tensor_dim]))
                            else:
                                sizes.append(0)
                        else:
                            sizes.append(0)

                offsets = []
                cumulative = 0
                for size in sizes:
                    offsets.append(cumulative)
                    cumulative += size

                axis_offsets[axis] = offsets
                axis_sizes[axis] = sizes

        return axis_offsets, axis_sizes


class ColorPalette:
    """Utility for generating distinct color palettes."""

    DEFAULT_COLOR = "#9e9e9e"

    @staticmethod
    def hsv_to_hex(h: float, s: float, v: float) -> str:
        """Convert HSV color to hex string."""
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

    @staticmethod
    def generate_muted_colors(n: int) -> list:
        """Generate a list of n distinct muted colors."""
        saturation = 0.48
        value = 0.80
        colors = []

        for i in range(n):
            hue = (i / float(n)) % 1.0
            color = ColorPalette.hsv_to_hex(hue, saturation, value)
            colors.append(color)

        return colors or [ColorPalette.DEFAULT_COLOR]


def _compute_global_slice_ranges(device_coord, sharding_info):
    """Compute global tensor slice ranges for a specific device coordinate."""
    tensor_shape = list(sharding_info.tensor.shape)
    slice_ranges = []
    distribution_coord = sharding_info.reverse_coord_mapper(device_coord)

    for tensor_dim, dim_size in enumerate(tensor_shape):
        if tensor_dim in sharding_info.dim_to_axis:
            axis = sharding_info.dim_to_axis[tensor_dim]
            try:
                part_index = int(distribution_coord[axis]) if axis < distribution_coord.dims() else 0
                if (
                    axis in sharding_info.axis_offsets
                    and axis in sharding_info.axis_sizes
                    and part_index < len(sharding_info.axis_offsets[axis])
                ):
                    start = sharding_info.axis_offsets[axis][part_index]
                    size = sharding_info.axis_sizes[axis][part_index]
                    end = start + size
                    slice_ranges.append((start, end))
                else:
                    slice_ranges.append((0, dim_size))
            except (IndexError, TypeError):
                slice_ranges.append((0, dim_size))
        else:
            slice_ranges.append((0, dim_size))

    return slice_ranges


def _create_shard_annotation_text(device_id, device_coord, sharding_info):
    """Create annotation text showing shard information for a device."""
    if device_id is None:
        return ""

    shard_index = sharding_info.device_to_shard_map.get(device_id)
    if shard_index is None:
        if sharding_info.tensor.storage_type() == ttnn.StorageType.HOST:
            try:
                coord_to_shard = {mc: i for i, mc in enumerate(sharding_info.mesh_coords)}
                shard_index = coord_to_shard.get(device_coord)
            except (IndexError, ValueError):
                pass

    if shard_index is None or shard_index >= len(sharding_info.shards):
        return ""

    shard = sharding_info.shards[shard_index]
    shape_str = str(list(shard.shape))
    dtype_str = str(shard.dtype).split(".")[-1]
    layout_str = str(shard.layout).split(".")[-1]

    slice_ranges = _compute_global_slice_ranges(device_coord, sharding_info)
    slice_strs = [
        f"{start}:{end}" if not (start == 0 and end == sharding_info.global_tensor_shape[tensor_dim]) else ":"
        for tensor_dim, (start, end) in enumerate(slice_ranges)
    ]
    slice_header = f"[{', '.join(slice_strs)}]"

    return f"{slice_header}\nShape: {shape_str}\nDtype: {dtype_str}\nLayout: {layout_str}"


def _create_replication_color_mapping(sharding_info):
    """Create color mapping for devices with identical tensor slices."""
    slice_key_to_group = {}
    device_to_group = {}
    group_index = 0

    if sharding_info.tensor.storage_type() == ttnn.StorageType.HOST:
        coord_to_device = {ttnn.MeshCoordinate(mc[0], mc[1]): i for i, mc in enumerate(sharding_info.mesh_coords)}

        for coord, device_id in coord_to_device.items():
            slice_ranges = _compute_global_slice_ranges(coord, sharding_info)
            slice_key = tuple(slice_ranges)

            if slice_key not in slice_key_to_group:
                slice_key_to_group[slice_key] = group_index
                group_index += 1
            device_to_group[device_id] = slice_key_to_group[slice_key]
    else:
        mesh_device = sharding_info.tensor.device()
        view = mesh_device.get_view()

        # Only assign colors to devices that actually have tensor shards
        for device_id in sharding_info.device_to_shard_map.keys():
            coord = None
            for test_coord in ttnn.MeshCoordinateRange(mesh_device.shape):
                if not view.is_local(test_coord):
                    continue
                if mesh_device.get_device_id(test_coord) == device_id:
                    coord = test_coord
                    break

            if coord is not None:
                slice_ranges = _compute_global_slice_ranges(coord, sharding_info)
                slice_key = tuple(slice_ranges)

                if slice_key not in slice_key_to_group:
                    slice_key_to_group[slice_key] = group_index
                    group_index += 1
                device_to_group[device_id] = slice_key_to_group[slice_key]

    num_groups = len(slice_key_to_group)
    colors = ColorPalette.generate_muted_colors(num_groups)

    device_to_style = {}
    for device_id, group_id in device_to_group.items():
        color = colors[group_id % len(colors)]
        device_to_style[device_id] = f"on {color}"

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
                is_local = (
                    view.is_local(coord) if storage_type == ttnn.StorageType.DEVICE else host_buffer.is_local(coord)
                )
                if storage_type == ttnn.StorageType.DEVICE:
                    locality = "Local\n" if is_local else "Remote\n"
                    device_id = mesh_device.get_device_id(coord) if is_local else None
                    device_id_str = f"Dev. ID: {device_id}\n" if is_local else "Unknown ID\n"
                else:
                    locality = "Local\n" if is_local else "Remote\n"
                    device_id = row_idx * cols + col_idx
                    device_id_str = ""

                locality = "" if fully_local else locality
                coords = f"({row_idx}, {col_idx})\n"
                if annotate_cell and device_id is not None:
                    try:
                        annotation = annotate_cell(device_id, coord)
                    except TypeError:
                        annotation = annotate_cell(device_id)
                else:
                    annotation = ""

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

    Args:
        tensor: The tensor to visualize
    """
    from rich.console import Console
    from rich.panel import Panel
    from loguru import logger

    console = Console()

    try:
        shards = ttnn.get_device_tensors(tensor)
        sharding_info = TensorShardingInfo(tensor, shards)
        placement_panel = Panel(str(sharding_info.placements), title="Placement Configuration", border_style="blue")
        device_to_style = _create_replication_color_mapping(sharding_info)

        if tensor.storage_type() == ttnn.StorageType.HOST:
            try:
                host_buffer = tensor.host_buffer()
                rows, cols = host_buffer.shape()
            except Exception:
                rows, cols = sharding_info.mesh_shape

            coord_to_device_id = {
                (int(mc[0]), int(mc[1])): host_idx for host_idx, mc in enumerate(sharding_info.mesh_coords)
            }

            def style_host_cell(host_linear_id, unused_coord=None):
                try:
                    r = int(host_linear_id // cols)
                    c = int(host_linear_id % cols)
                    device_id = coord_to_device_id.get((r, c))
                    return device_to_style.get(device_id) if device_id is not None else None
                except Exception:
                    return None

            def annotate_host_cell(host_linear_id, host_coord):
                try:
                    r, c = int(host_coord[0]), int(host_coord[1])
                    device_id = coord_to_device_id.get((r, c))
                    if device_id is None:
                        return ""

                    distribution_coord = ttnn.MeshCoordinate(r, c)
                    return _create_shard_annotation_text(device_id, distribution_coord, sharding_info)
                except Exception:
                    return ""

            annotate_func = annotate_host_cell
            style_func = style_host_cell
        else:

            def annotate_with_shard_info(device_id, distribution_coord=None):
                return _create_shard_annotation_text(device_id, distribution_coord, sharding_info)

            def style_device_cell(device_id):
                return device_to_style.get(device_id)

            annotate_func = annotate_with_shard_info
            style_func = style_device_cell

        mesh_table = _get_rich_table(
            tensor.device(),
            tensor,
            annotate_cell=annotate_func,
            style_cell=style_func,
            storage_type=tensor.storage_type(),
        )

        console.print(placement_panel)
        console.print(mesh_table)

    except Exception as e:
        logger.error(f"Error visualizing tensor: {e}")


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

        # TODO: Remove shape indexing workaround after exposing subscripts in nanobind
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


# Temporary stubs to accommodate migration of Python-based sharding / concatenation to C++.
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
