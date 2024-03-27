import dataclasses
import sqlite3

import ttnn


@dataclasses.dataclass
class Operation:
    operation_id: int
    name: str
    buffers: list


@dataclasses.dataclass
class BufferPage:
    operation_id: int
    address: int
    device_id: int
    core_y: int
    core_x: int
    page_index: int
    page_address: int
    page_size: int
    buffer_type: int


def display(operations):
    for operation in operations.values():
        print(operation.name)
        operation.buffers.sort(key=lambda x: (x.device_id, x.core_y, x.core_x))
        for buffer in operation.buffers:
            print(f"\t{buffer}")
        print()


def main():
    sqlite_connection = sqlite3.connect(ttnn.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations")
    operations = {}
    for row in cursor.fetchall():
        operation = Operation(*row, [])
        operations[operation.operation_id] = operation

    cursor.execute("SELECT * FROM buffers")
    for row in cursor.fetchall():
        buffer_page = BufferPage(*row)
        operations[buffer_page.operation_id].buffers.append(buffer_page)

    display(operations)

    sqlite_connection.close()


if __name__ == "__main__":
    main()
