import networkx as nx
import os
import xlsxwriter  # Library for writing Excel files
from tracer_backend import OperationGraph
from tracer_backend_utils import WrappedOperation, Operation, AttrsBase, PlaceholderTensor
from pytorch_graph_utils import format_file_with_black
from dataclasses import asdict


class PytorchExcelGraph:
    def __init__(self, operation_graph: OperationGraph):
        self.graph = operation_graph.graph
        self.row_header = [
            "Unique Name",
            "Function Call Name",
            "Attrs",
            "Input shapes",
            "Input dtypes",
            "Output shapes",
            "Output dtypes",
            "Ops",
            "Params",
            "Inputs",
            "Outputs",
        ]
        self.int_fields = ["Ops", "Params"]

    def generate_row(self, node):
        """
        Generate a row for a given node in the graph.
        """
        operation = self.graph.nodes[node].get("operation")
        result = {header: None for header in self.row_header}
        assert isinstance(operation, Operation), "Expected operation to be an instance of Operation"
        result["Unique Name"] = operation.unique_name
        result["Function Call Name"] = operation.function_call_name
        if operation.meta_data:
            if operation.meta_data.meta and isinstance(operation.meta_data.meta, dict):
                result["Input shapes"] = operation.meta_data.meta.get("i_shapes", [])
                result["Input dtypes"] = operation.meta_data.meta.get("i_dtypes", [])
                result["Output shapes"] = operation.meta_data.meta.get("o_shapes", [])
                result["Output dtypes"] = operation.meta_data.meta.get("o_dtypes", [])
            if operation.meta_data.res is not None:
                output = []
                if isinstance(operation.meta_data.res, (list, tuple)):
                    output = [op.name for op in operation.meta_data.res if isinstance(op, PlaceholderTensor)]
                else:
                    output = (
                        operation.meta_data.res.name if isinstance(operation.meta_data.res, PlaceholderTensor) else None
                    )
                result["Outputs"] = output
        result["Inputs"] = [input_op.name for input_op in operation.args if isinstance(input_op, PlaceholderTensor)]
        if isinstance(operation, WrappedOperation):
            # Extract relevant attributes from the operation
            result["Attrs"] = asdict(operation.attrs) if isinstance(operation.attrs, AttrsBase) else None
            result["Ops"] = operation.ops
            result["Params"] = operation.params
        else:
            result["Ops"] = -1
            result["Params"] = -1
        return result

    def dump_to_excel_file(self, file_path: str):
        """
        Dump the graph data to an Excel file.
        """
        # Create an Excel workbook and worksheet
        workbook = xlsxwriter.Workbook(file_path)
        worksheet = workbook.add_worksheet()

        for col, header in enumerate(self.row_header):
            worksheet.write(0, col, header)

        # Write data rows
        for row_idx, node_id in enumerate(list(nx.topological_sort(self.graph))[::-1]):
            row_data = self.generate_row(node_id)
            if row_data:
                for col_idx, header in enumerate(self.row_header):
                    if header in self.int_fields and row_data[header] is not None:
                        worksheet.write(row_idx + 1, col_idx, int(row_data[header]))
                    else:
                        worksheet.write(row_idx + 1, col_idx, str(row_data[header]))

        # Close the workbook
        workbook.close()
        print(f"Generated excel sheet report to {os.path.abspath(file_path)}")
