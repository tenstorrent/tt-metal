# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
UnifiedKernelBuilder - Python wrapper for ProgramDescriptor generation.

Provides an MPI-like API for writing kernels that automatically handles:
- 3-way kernel split (Reader/Compute/Writer) for local-only operations
- Role-aware multicast/unicast with automatic semaphore management
- Cross-core synchronization primitives
"""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import ttnn
from .primitives import Role, McastGroup, BufferMode


def _get_tile_size_bytes(data_format: ttnn.DataType) -> int:
    """Calculate tile size in bytes for a given data format."""
    # Standard tile is 32x32
    # Map common data formats to bytes per element
    format_to_bytes = {
        ttnn.bfloat16: 2,
        ttnn.float32: 4,
        ttnn.int32: 4,
        ttnn.uint32: 4,
    }

    bytes_per_element = format_to_bytes.get(data_format, 2)  # Default to 2 (bfloat16)
    return 32 * 32 * bytes_per_element  # 32x32 tile


# CBFormatDescriptor accepts DataType directly, not DataFormat enum
# So we can just pass the data_format through


class UnifiedKernelBuilder:
    """
    Builder for creating ProgramDescriptor from unified kernel source.

    Example:
        builder = (
            UnifiedKernelBuilder("my_kernel.cpp")
            .add_tensor("in0", input_tensor)
            .add_tensor("out", output_tensor)
            .set_core_grid(all_cores)
        )
        program = builder.build(device)
    """

    def __init__(self, kernel_source: Union[str, Path], math_fidelity: Optional[ttnn.MathFidelity] = None):
        """
        Initialize the builder.

        Args:
            kernel_source: Path to unified kernel source file or inline source code string
            math_fidelity: Math fidelity for compute kernel (defaults to HiFi4)
        """
        self.kernel_source = kernel_source
        self.buffers: Dict[str, Tuple[Optional[ttnn.Tensor], ttnn.DataType]] = {}
        self.mcast_groups: Dict[str, McastGroup] = {}
        self.semaphores: Dict[str, int] = {}  # name -> semaphore_id
        self.core_grid: Optional[ttnn.CoreRangeSet] = None
        self.device: Optional[ttnn.Device] = None

        # Track compile-time, runtime, and common runtime arguments
        self.compile_time_args_: List[Tuple[str, int]] = []  # (name, value) pairs
        self.runtime_args_: List[Tuple[str, int]] = []  # (name, value) pairs
        self.common_runtime_args_: List[Tuple[str, int]] = []  # (name, value) pairs

        # Math config for compute kernel
        self.math_fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
        self.fp32_dest_acc_en = False
        self.dst_full_sync_en = False
        self.unpack_to_dest_mode: List[ttnn.UnpackToDestMode] = []
        self.bfp8_pack_precise = False
        self.math_approx_mode = False

        # Generated tile constants (for compile-time generated tiles)
        self.generated_tile_constants_: List[Dict] = []  # List of {name, data_format, generator_code}

        # Track next available semaphore ID
        self._next_semaphore_id = 0

        # Track next available CB index
        self._next_cb_index = 0

    def add_tensor(
        self,
        name: str,
        tensor: ttnn.Tensor,
        mode: BufferMode = BufferMode.DOUBLE,
    ) -> "UnifiedKernelBuilder":
        """
        Add a tensor (input or output).

        Args:
            name: Tensor name used in kernel (becomes accessible as `name` variable)
            tensor: The tensor to add
            mode: Buffer mode - DOUBLE (default, better perf) or SINGLE (saves L1)
        """
        self.buffers[name] = (tensor, tensor.dtype, mode)
        return self

    def add_buffer(
        self,
        name: str,
        tensor: Optional[ttnn.Tensor] = None,
        data_format: Optional[ttnn.DataType] = None,
        mode: BufferMode = BufferMode.DOUBLE,
    ) -> "UnifiedKernelBuilder":
        """
        Add a buffer (input or output tensor). Prefer add_tensor() for simplicity.

        Args:
            name: Buffer name used in kernel
            tensor: Optional tensor (if None, buffer is intermediate)
            data_format: Data format for the buffer (inferred from tensor if not provided)
            mode: Buffer mode - DOUBLE (default, better perf) or SINGLE (saves L1)
        """
        if tensor is None and data_format is None:
            raise ValueError("Either tensor or data_format must be provided")

        if data_format is None:
            data_format = tensor.dtype

        self.buffers[name] = (tensor, data_format, mode)
        return self

    def add_mcast_group(
        self,
        name: str,
        receivers: ttnn.CoreRangeSet,
        sender: Optional[ttnn.CoreCoord] = None,
        noc: Optional[ttnn.NOC] = None,
    ) -> "UnifiedKernelBuilder":
        """
        Add a multicast group.

        Args:
            name: Group name used in kernel
            receivers: Core range set for receivers
            sender: Optional sender core (if None, uses first core in receivers)
            noc: Optional NOC to use (defaults to NOC_1)
        """
        if sender is None:
            # Use first core from receivers as sender
            receiver_ranges = list(receivers.ranges())
            if not receiver_ranges:
                raise ValueError("Receivers must contain at least one core")
            sender = receiver_ranges[0].start

        group = McastGroup(name, sender, receivers, noc)
        self.mcast_groups[name] = group
        return self

    def add_semaphore(self, name: str, initial_value: int = 0) -> "UnifiedKernelBuilder":
        """
        Add a semaphore for synchronization.

        Args:
            name: Semaphore name
            initial_value: Initial semaphore value
        """
        self.semaphores[name] = initial_value
        return self

    def add_compile_time_arg(self, name: str, value: int) -> "UnifiedKernelBuilder":
        """
        Add a compile-time argument.

        Args:
            name: Argument name (used in INIT_ARGUMENTS)
            value: Argument value
        """
        self.compile_time_args_.append((name, value))
        return self

    def add_runtime_arg(self, name: str, value: int) -> "UnifiedKernelBuilder":
        """
        Add a runtime argument.

        Args:
            name: Argument name (used in INIT_ARGUMENTS)
            value: Argument value
        """
        self.runtime_args_.append((name, value))
        return self

    def add_common_runtime_arg(self, name: str, value: int) -> "UnifiedKernelBuilder":
        """
        Add a common runtime argument.

        Args:
            name: Argument name (used in INIT_ARGUMENTS)
            value: Argument value
        """
        self.common_runtime_args_.append((name, value))
        return self

    def add_generated_tile_constant(
        self, name: str, data_format: ttnn.DataType, generator_code: str
    ) -> "UnifiedKernelBuilder":
        """
        Add a generated tile constant (compile-time generated tile).

        Args:
            name: Constant name
            data_format: Data format for the tile
            generator_code: C++ code to generate the tile (executed in reader kernel)
        """
        self.generated_tile_constants_.append(
            {
                "name": name,
                "data_format": data_format,
                "generator_code": generator_code,
            }
        )
        return self

    def get_runtime_arg_idx(self, name: str) -> int:
        """Get the runtime argument index by name."""
        for idx, (arg_name, _) in enumerate(self.runtime_args_):
            if arg_name == name:
                return idx
        raise ValueError(f"Runtime argument '{name}' not found")

    def buffer_addresses_start_runtime_arg_idx(self) -> int:
        """Get the starting index for buffer addresses in runtime args."""
        return len(self.runtime_args_)

    def set_core_grid(self, core_grid: ttnn.CoreRangeSet) -> "UnifiedKernelBuilder":
        """Set the core grid for kernel execution."""
        self.core_grid = core_grid
        return self

    def build(self, device: ttnn.Device) -> ttnn.ProgramDescriptor:
        """
        Build the ProgramDescriptor.

        Args:
            device: Device to build the program for

        Returns:
            ProgramDescriptor ready for generic_op
        """
        self.device = device

        if self.core_grid is None:
            raise ValueError("Core grid must be set before building")

        # Determine if we need multicast or just local ops
        has_mcast = len(self.mcast_groups) > 0

        if has_mcast:
            return self._build_mcast_program()
        else:
            return self._build_local_program()

    def _compute_n_tiles(self) -> Optional[int]:
        """Compute total number of tiles from the first tensor."""
        for name, (tensor, data_format, mode) in self.buffers.items():
            if tensor is not None:
                # Get tensor volume and compute tiles
                # Tile size is 32x32
                TILE_HW = 32 * 32
                volume = tensor.volume()
                return volume // TILE_HW
        return None

    def _compute_grid_size(self) -> Tuple[int, int]:
        """Compute grid dimensions from core_grid."""
        if self.core_grid is None:
            return (1, 1)

        max_x, max_y = 0, 0
        for core_range in self.core_grid.ranges():
            max_x = max(max_x, core_range.end.x + 1)
            max_y = max(max_y, core_range.end.y + 1)
        return (max_x, max_y)

    def _generate_init_arguments(self) -> str:
        """Generate INIT_ARGUMENTS define string matching C++ builder pattern."""
        init_parts = []

        # Auto-compute grid_x and grid_y from core_grid
        grid_x, grid_y = self._compute_grid_size()
        init_parts.append(f"constexpr uint32_t grid_x = {grid_x};")
        init_parts.append(f"constexpr uint32_t grid_y = {grid_y};")
        init_parts.append(f"constexpr uint32_t num_cores = {grid_x * grid_y};")

        # Auto-compute n_tiles if not explicitly provided
        has_n_tiles = any(name == "n_tiles" for name, _ in self.compile_time_args_)
        if not has_n_tiles:
            n_tiles = self._compute_n_tiles()
            if n_tiles is not None:
                init_parts.append(f"constexpr uint32_t n_tiles = {n_tiles};")

        # Compile-time args
        for name, value in self.compile_time_args_:
            init_parts.append(f"constexpr uint32_t {name} = {value};")

        # Runtime args
        for idx, (name, _) in enumerate(self.runtime_args_):
            init_parts.append(f"const uint32_t {name} = get_arg_val<uint32_t>({idx});")

        # Common runtime args
        for idx, (name, _) in enumerate(self.common_runtime_args_):
            init_parts.append(f"const uint32_t {name} = get_common_arg_val<uint32_t>({idx});")

        return " ".join(init_parts)

    def _create_runtime_args_for_cores(
        self, core_ranges: ttnn.CoreRangeSet, args: List[int]
    ) -> List[Tuple[ttnn.CoreCoord, List[int]]]:
        """Create runtime args list with args for each core in the range set."""
        runtime_args_list = []
        for core_range in core_ranges.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    runtime_args_list.append((ttnn.CoreCoord(x, y), list(args)))
        return runtime_args_list

    def _build_local_program(self) -> ttnn.ProgramDescriptor:
        """Build program for local-only operations (3-way split)."""
        kernels = []
        cbs = []
        semaphores = []

        # Build compile-time args list (will be appended to as we process buffers)
        compile_time_args: List[int] = []

        # Add compile-time args
        for _, value in self.compile_time_args_:
            compile_time_args.append(value)

        # Create circular buffers for each buffer and track TensorAccessorArgs
        cb_descriptors = {}
        buffer_runtime_arg_indices = {}  # buffer_name -> runtime_arg_index

        for idx, (name, (tensor, data_format, mode)) in enumerate(self.buffers.items()):
            cb_index = idx  # Use buffer index as CB index
            self._next_cb_index = max(self._next_cb_index, cb_index + 1)

            tile_size = _get_tile_size_bytes(data_format)
            num_buffers = int(mode)  # BufferMode.SINGLE=1, BufferMode.DOUBLE=2
            cb_total_size = num_buffers * tile_size
            cb_page_size = tile_size

            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=data_format,
                page_size=cb_page_size,
            )

            cb_descriptor = ttnn.CBDescriptor(
                total_size=cb_total_size,
                core_ranges=self.core_grid,
                format_descriptors=[cb_format],
            )

            cb_descriptors[name] = (cb_descriptor, cb_index)
            cbs.append(cb_descriptor)

            # Track TensorAccessorArgs offset (before appending)
            cta_offset = len(compile_time_args)

            # Append TensorAccessorArgs if tensor is provided
            if tensor is not None:
                # Get buffer from tensor and create TensorAccessorArgs
                # Note: We'll need to handle this through ttnn API
                # For now, we'll add a placeholder - the actual TensorAccessorArgs
                # should be created from the buffer
                buffer_addr = tensor.buffer().address() if hasattr(tensor, "buffer") else 0
                page_size_bytes = tensor.buffer().page_size() if hasattr(tensor, "buffer") else tile_size
            else:
                buffer_addr = 0
                page_size_bytes = tile_size

            # Store runtime arg index for buffer address
            buffer_runtime_arg_indices[name] = len(self.runtime_args_) + len(buffer_runtime_arg_indices)

        # Add circular buffers for generated tile constants
        for constant_idx, constant in enumerate(self.generated_tile_constants_):
            cb_index = len(self.buffers) + constant_idx
            tile_size = _get_tile_size_bytes(constant["data_format"])

            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=constant["data_format"],
                page_size=tile_size,
            )

            cb_descriptor = ttnn.CBDescriptor(
                total_size=tile_size,
                core_ranges=self.core_grid,
                format_descriptors=[cb_format],
            )

            cbs.append(cb_descriptor)

        # Generate INIT_ARGUMENTS with buffer setup
        init_base = self._generate_init_arguments()
        init_parts = [init_base]

        # Add buffer setup to INIT_ARGUMENTS
        compile_time_args_with_buffers = list(compile_time_args)
        # Extract just the values from runtime_args_ (which are tuples of (name, value))
        runtime_args_with_buffers = [value for _, value in self.runtime_args_]
        # Extract just the values from common_runtime_args_ (which are tuples of (name, value))
        common_runtime_args_with_buffers = [value for _, value in self.common_runtime_args_]

        # Build INIT_ARGUMENTS separately for:
        # - Data movement kernels (BRISC/NCRISC): need TensorAccessorArgs and TensorAccessor
        # - Compute kernels (TRISC): only need CB indices, no TensorAccessor
        init_dataflow_parts = list(init_parts)  # Copy compile-time and runtime args
        init_compute_parts_only = list(init_parts)  # Copy for compute

        for idx, (name, (tensor, data_format, _mode)) in enumerate(self.buffers.items()):
            cb_index = idx
            cta_offset = len(compile_time_args_with_buffers)
            page_size_bytes = _get_tile_size_bytes(data_format)

            # Compute tensor shape info
            TILE_HW = 32 * 32
            if tensor is not None:
                volume = tensor.volume()
                n_tiles = volume // TILE_HW
                shape = tensor.shape
                # For 4D tensor [N, C, H, W], compute tiles in each dimension
                # H and W are tiled (32x32), N and C are batch/channel
                shape_h = shape[-2] if len(shape) >= 2 else 1
                shape_w = shape[-1] if len(shape) >= 1 else 1
                tiles_h = shape_h // 32
                tiles_w = shape_w // 32
            else:
                n_tiles = 0
                tiles_h = 0
                tiles_w = 0

            # Append TensorAccessorArgs compile-time args for this tensor
            if tensor is not None:
                accessor_args = ttnn.TensorAccessorArgs(tensor)
                tensor_ct_args = accessor_args.get_compile_time_args()
                compile_time_args_with_buffers.extend(tensor_ct_args)

            runtime_arg_idx = len(runtime_args_with_buffers)
            # Get actual buffer address from tensor
            buffer_addr = tensor.buffer_address() if tensor is not None else 0
            runtime_args_with_buffers.append(buffer_addr)

            # Compute only gets CB index, dummy placeholders, and tensor info
            init_compute_parts_only.append(f"constexpr auto {name}_cb = {cb_index};")
            init_compute_parts_only.append(f"constexpr uint32_t {name}_page_size_bytes = {page_size_bytes};")
            # Tensor shape info - available on all processors
            init_compute_parts_only.append(f"constexpr uint32_t {name}_n_tiles = {n_tiles};")
            init_compute_parts_only.append(f"constexpr uint32_t {name}_tiles_h = {tiles_h};")
            init_compute_parts_only.append(f"constexpr uint32_t {name}_tiles_w = {tiles_w};")
            # Dummy tensor placeholder - read_tile_impl ignores it on TRISC
            init_compute_parts_only.append(f"constexpr int {name} = 0;")

            # Data movement gets full TensorAccessor setup plus tensor info
            init_dataflow_parts.append(f"constexpr auto {name}_cb = {cb_index};")
            init_dataflow_parts.append(f"const uint32_t {name}_addr = get_arg_val<uint32_t>({runtime_arg_idx});")
            init_dataflow_parts.append(f"constexpr auto {name}_args = TensorAccessorArgs<{cta_offset}>();")
            init_dataflow_parts.append(f"constexpr uint32_t {name}_page_size_bytes = {page_size_bytes};")
            # Tensor shape info - available on all processors
            init_dataflow_parts.append(f"constexpr uint32_t {name}_n_tiles = {n_tiles};")
            init_dataflow_parts.append(f"constexpr uint32_t {name}_tiles_h = {tiles_h};")
            init_dataflow_parts.append(f"constexpr uint32_t {name}_tiles_w = {tiles_w};")
            init_dataflow_parts.append(
                f"const auto {name} = TensorAccessor({name}_args, {name}_addr, {name}_page_size_bytes);"
            )

        # Add generated tile constants
        for constant_idx, constant in enumerate(self.generated_tile_constants_):
            cb_index = len(self.buffers) + constant_idx
            init_compute_parts_only.append(f"constexpr auto {constant['name']}_cb = {cb_index};")
            init_compute_parts_only.append(
                f"constexpr auto {constant['name']} = ConstantTile({constant['name']}_cb, 0);"
            )
            init_dataflow_parts.append(f"constexpr auto {constant['name']}_cb = {cb_index};")
            init_dataflow_parts.append(f"constexpr auto {constant['name']} = ConstantTile({constant['name']}_cb, 0);")

        init_dataflow = " ".join(init_dataflow_parts)

        # Generate reader-specific and compute-specific additions
        init_reader_extra = []
        init_compute_extra = []

        for constant in self.generated_tile_constants_:
            init_reader_extra.append(constant["generator_code"] + ";")
            init_compute_extra.append(f"cb_wait_front({constant['name']}_cb, 1);")

        init_reader = init_dataflow + (" " + " ".join(init_reader_extra) if init_reader_extra else "")
        init_writer = init_dataflow  # Writer uses same as dataflow base
        init_compute = " ".join(init_compute_parts_only) + (
            " " + " ".join(init_compute_extra) if init_compute_extra else ""
        )

        # Read kernel source
        if isinstance(self.kernel_source, (str, Path)):
            kernel_str = str(self.kernel_source)
            # Check if it looks like inline source code (contains newlines or braces)
            if "\n" in kernel_str or "{" in kernel_str or "#include" in kernel_str:
                kernel_source_code = kernel_str
            else:
                # Try as file path
                source_path = Path(kernel_str)
                try:
                    if source_path.exists():
                        kernel_source_code = source_path.read_text()
                    else:
                        # Assume it's inline source code
                        kernel_source_code = kernel_str
                except OSError:
                    # Path too long or other error - treat as inline source
                    kernel_source_code = kernel_str
        else:
            kernel_source_code = str(self.kernel_source)

        # Auto-wrap kernel source with boilerplate if not already present
        # This allows user to write just the kernel body without includes/KERNEL_MAIN/INIT_ARGUMENTS
        needs_wrap = "KERNEL_MAIN" not in kernel_source_code
        if needs_wrap:
            kernel_source_code = f"""#include "unified_common.h"

KERNEL_MAIN {{
    INIT_ARGUMENTS

{kernel_source_code}
}}
"""

        # Generate defines (as list of tuples)
        base_defines = [
            ("TOTAL_NUM_CIRCULAR_BUFFERS", str(len(self.buffers))),
        ]

        reader_defines = [("INIT_ARGUMENTS", init_reader)] + base_defines
        writer_defines = [("INIT_ARGUMENTS", init_writer)] + base_defines
        compute_defines = [("INIT_ARGUMENTS", init_compute)] + base_defines

        # Create runtime args for all cores
        runtime_args_per_core = self._create_runtime_args_for_cores(self.core_grid, runtime_args_with_buffers)

        # Generate 3 kernel descriptors using proper config types
        # Reader kernel (BRISC) - automatically sets COMPILE_FOR_BRISC
        reader_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source_code,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.core_grid,
            compile_time_args=compile_time_args_with_buffers,
            named_compile_time_args=[],  # Empty for local ops
            defines=reader_defines,
            runtime_args=runtime_args_per_core,
            common_runtime_args=common_runtime_args_with_buffers,
            config=ttnn.ReaderConfigDescriptor(),
        )
        kernels.append(reader_kernel)

        # Compute kernel (TRISC) - automatically sets COMPILE_FOR_TRISC
        compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=self.math_fidelity,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            dst_full_sync_en=self.dst_full_sync_en,
            bfp8_pack_precise=self.bfp8_pack_precise,
            math_approx_mode=self.math_approx_mode,
        )
        # Set unpack_to_dest_mode as a property (not in constructor)
        if self.unpack_to_dest_mode:
            compute_config.unpack_to_dest_mode = self.unpack_to_dest_mode

        compute_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source_code,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.core_grid,
            compile_time_args=compile_time_args_with_buffers,
            named_compile_time_args=[],  # Empty for local ops
            defines=compute_defines,
            runtime_args=runtime_args_per_core,
            common_runtime_args=common_runtime_args_with_buffers,
            config=compute_config,
        )
        kernels.append(compute_kernel)

        # Writer kernel (NCRISC) - automatically sets COMPILE_FOR_NCRISC
        writer_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source_code,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.core_grid,
            compile_time_args=compile_time_args_with_buffers,
            named_compile_time_args=[],  # Empty for local ops
            defines=writer_defines,
            runtime_args=runtime_args_per_core,
            common_runtime_args=common_runtime_args_with_buffers,
            config=ttnn.WriterConfigDescriptor(),
        )
        kernels.append(writer_kernel)

        return ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[],
            cbs=cbs,
        )

    def _build_mcast_program(self) -> ttnn.ProgramDescriptor:
        """Build program for multicast operations."""
        kernels = []
        cbs = []
        semaphore_descriptors = []

        # Create circular buffers
        cb_descriptors = {}
        for name, (tensor, data_format, mode) in self.buffers.items():
            cb_index = self._next_cb_index
            self._next_cb_index += 1

            tile_size = _get_tile_size_bytes(data_format)
            num_buffers = int(mode)  # BufferMode.SINGLE=1, BufferMode.DOUBLE=2
            cb_total_size = num_buffers * tile_size
            cb_page_size = tile_size

            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=data_format,
                page_size=cb_page_size,
            )

            cb_descriptor = ttnn.CBDescriptor(
                total_size=cb_total_size,
                core_ranges=self.core_grid,
                format_descriptors=[cb_format],
            )

            cb_descriptors[name] = (cb_descriptor, cb_index)
            cbs.append(cb_descriptor)

        # Create semaphores for each mcast group
        all_cores = self.core_grid
        for mcast_group in self.mcast_groups.values():
            # Merge sender and receiver cores
            sender_range = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_group.sender, mcast_group.sender)])
            all_cores = all_cores.merge(sender_range)

        mcast_semaphores = {}
        for name, group in self.mcast_groups.items():
            sender_sem_id = self._next_semaphore_id
            self._next_semaphore_id += 1
            receiver_sem_id = self._next_semaphore_id
            self._next_semaphore_id += 1

            sender_sem = ttnn.SemaphoreDescriptor(
                id=sender_sem_id,
                core_ranges=all_cores,
                initial_value=0,
            )
            receiver_sem = ttnn.SemaphoreDescriptor(
                id=receiver_sem_id,
                core_ranges=all_cores,
                initial_value=0,
            )

            semaphore_descriptors.extend([sender_sem, receiver_sem])
            mcast_semaphores[name] = (sender_sem_id, receiver_sem_id)

        # Read kernel source
        if isinstance(self.kernel_source, (str, Path)):
            source_path = Path(self.kernel_source)
            if source_path.exists():
                kernel_source_code = source_path.read_text()
            else:
                kernel_source_code = str(self.kernel_source)
        else:
            kernel_source_code = str(self.kernel_source)

        # Generate kernels for sender and receiver roles
        for group_name, group in self.mcast_groups.items():
            sender_sem_id, receiver_sem_id = mcast_semaphores[group_name]

            # Get NOC coordinates
            receiver_ranges = list(group.receivers.ranges())
            if not receiver_ranges:
                continue

            first_receiver = receiver_ranges[0].start
            last_receiver = receiver_ranges[-1].end
            receiver_start_noc = self.device.worker_core_from_logical_core(first_receiver)
            receiver_end_noc = self.device.worker_core_from_logical_core(last_receiver)

            # Sender kernel
            sender_named_compile_args = [
                (f"{group_name}_dest_noc_start_x", receiver_start_noc.x),
                (f"{group_name}_dest_noc_start_y", receiver_start_noc.y),
                (f"{group_name}_dest_noc_end_x", receiver_end_noc.x),
                (f"{group_name}_dest_noc_end_y", receiver_end_noc.y),
                (f"{group_name}_num_cores", group.num_dests),
                (f"{group_name}_loopback", 1 if group.is_part_of_receiver_grid else 0),
                (f"{group_name}_is_part_of_receiver_grid", 1 if group.is_part_of_receiver_grid else 0),
                (f"{group_name}_data_sender_semaphore", sender_sem_id),
                (f"{group_name}_data_receiver_semaphore", receiver_sem_id),
            ]

            sender_defines = [
                ("MY_ROLE", str(Role.MCAST_SENDER.value)),
                ("MCAST_SENDER", "1"),
            ]

            # Add buffer defines
            for name, (_, cb_index) in cb_descriptors.items():
                sender_defines.append((f"CB_{name.upper()}", str(cb_index)))

            sender_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_source_code,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(group.sender, group.sender)]),
                named_compile_time_args=sender_named_compile_args,
                defines=sender_defines,
                runtime_args=[],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=group.noc,
                ),
            )
            kernels.append(sender_kernel)

            # Receiver kernel
            receiver_defines = [
                ("MY_ROLE", str(Role.MCAST_RECEIVER.value)),
                ("MCAST_RECEIVER", "1"),
                (f"{group_name}_data_receiver_semaphore", str(receiver_sem_id)),
            ]

            # Add buffer defines
            for name, (_, cb_index) in cb_descriptors.items():
                receiver_defines.append((f"CB_{name.upper()}", str(cb_index)))

            receiver_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_source_code,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=group.receivers,
                compile_time_args=[receiver_sem_id],
                defines=receiver_defines,
                runtime_args=[],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_0 if group.noc == ttnn.NOC.NOC_1 else ttnn.NOC.NOC_1,
                ),
            )
            kernels.append(receiver_kernel)

        return ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=semaphore_descriptors,
            cbs=cbs,
        )
