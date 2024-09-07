# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from jinja2 import Environment, FileSystemLoader
import re


def get_template_dir():
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if not tt_metal_home:
        raise EnvironmentError("TT_METAL_HOME environment variable is not set.")
    return os.path.join(tt_metal_home, "ttnn/tools/op_generator/templates")


TEMPLATE_DIR = get_template_dir()


def render_template(template_path, context):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_path)
    return template.render(context)


def write_file(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(content)


def to_snake_case(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def generate_structure(operation_name, operations_folder, category=None, include_program_factory=False):
    templates = {
        "device/op.cpp": "device/op.cpp.j2",
        "device/op.hpp": "device/op.hpp.j2",
        "hpp": "hpp.j2",
        "pybind.hpp": "pybind.hpp.j2",
    }

    snake_operation_name = to_snake_case(operation_name)
    if category:
        operation_dir = os.path.join(operations_folder, category, snake_operation_name)
    else:
        operation_dir = os.path.join(operations_folder, snake_operation_name)

    try:
        os.makedirs(operation_dir, exist_ok=True)

        context = {
            "operation_name": operation_name,
            "snake_operation_name": snake_operation_name,
            "category": category if category else "",
            "full_namespace": f"ttnn::operations::{category}" if category else "ttnn::operations",
        }

        for file_key, template_name in templates.items():
            file_parts = file_key.split("/")
            if file_key == "device/op.cpp":
                file_parts[-1] = f"{snake_operation_name}_op.cpp"
            elif file_key == "device/op.hpp":
                file_parts[-1] = f"{snake_operation_name}_op.hpp"
            elif file_key == "hpp":
                file_parts[-1] = f"{snake_operation_name}.hpp"
            elif file_key == "pybind.hpp":
                file_parts[-1] = f"{snake_operation_name}_pybind.hpp"
            file_path = os.path.join(operation_dir, *file_parts)

            template_content = render_template(template_name, context)

            write_file(file_path, template_content)
            print(f"Created {file_path}")

        print("Don't forget to add generated files to your CMakeLists.txt and root PyBind file.")
    except Exception as e:
        print(f"Error occurred: {e}")
        if os.path.exists(operation_dir):
            shutil.rmtree(operation_dir)
            print(f"Cleaned up {operation_dir}")


if __name__ == "__main__":
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if not tt_metal_home:
        print("Error: TT_METAL_HOME environment variable is not set.")
        exit(1)

    operations_folder = os.path.join(tt_metal_home, "ttnn/cpp/ttnn/operations")

    operation_name = input("Enter the operation name (enter SuperFastAdd to get ttnn::super_fast_add): ").strip()
    category = input("Enter the category name (or leave blank if none): ").strip()

    generate_structure(operation_name, operations_folder, category if category else None)
