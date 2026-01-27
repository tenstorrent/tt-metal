# GDB Pretty Printers for TT-Metal
# Copyright (C) 2026
#
# This file implements GDB pretty printers for TT-Metal types.

import gdb
import gdb.printing


class ShapePrinter:
    """Pretty printer for tt::tt_metal::Shape

    Shape inherits from ShapeBase which contains a SmallVector<uint32_t> named 'value_'
    that stores the dimensions of the tensor.
    """

    def __init__(self, val):
        """Initialize the printer with a gdb.Value representing a Shape object."""
        self.val = val

    def to_string(self):
        """Return a string representation of the Shape.

        Format: Shape([d1, d2, ..., dn], rank=n, volume=v)
        where d1..dn are dimensions, n is the rank, and v is the total volume.
        """
        try:
            # Access the value_ member (SmallVector<uint32_t>)
            value_vec = self.val["value_"]

            # Cast to SmallVectorBase to access size and data pointer
            base_type = gdb.lookup_type("ttsl::detail::llvm::SmallVectorBase<unsigned int>")
            vec_base = value_vec.cast(base_type)

            # Get the size (number of dimensions)
            size = int(vec_base["Size"])

            if size == 0:
                return "Shape([])"

            # Get the data pointer
            begin_ptr = vec_base["BeginX"]
            uint32_ptr_type = gdb.lookup_type("uint32_t").pointer()
            data_ptr = begin_ptr.cast(uint32_ptr_type)

            # Extract all dimensions
            dims = []
            for i in range(size):
                dims.append(int(data_ptr[i]))

            # Calculate total volume
            volume = 1
            for dim in dims:
                volume *= dim

            # Format the output
            dims_str = ", ".join(str(d) for d in dims)
            return f"Shape([{dims_str}], rank={size}, volume={volume})"

        except gdb.error as e:
            return f"Shape(<inaccessible: {e}>)"
        except Exception as e:
            return f"Shape(<error: {e}>)"

    def display_hint(self):
        """Return a display hint for GDB.

        This tells GDB how to format the output.
        """
        return "array"


def build_ttmetal_pretty_printer():
    """Build and return the pretty printer collection for TT-Metal types."""
    pp = gdb.printing.RegexpCollectionPrettyPrinter("tt-metal")

    # Register Shape printer
    # Matches: tt::tt_metal::Shape
    pp.add_printer("Shape", "^tt::tt_metal::Shape$", ShapePrinter)

    return pp


def register_ttmetal_printers(obj=None):
    """Register TT-Metal pretty printers with GDB.

    Args:
        obj: The objfile to register printers for. If None, registers globally.
    """
    gdb.printing.register_pretty_printer(obj, build_ttmetal_pretty_printer(), replace=True)


# Auto-register printers when this script is loaded
register_ttmetal_printers()

print("TT-Metal GDB pretty printers loaded successfully")
