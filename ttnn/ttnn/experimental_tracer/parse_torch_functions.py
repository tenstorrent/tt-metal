import ast
import sys
import re

import json
import requests
import sseclient

import re
from typing import Optional, Tuple
from pytorch_graph_utils import format_file_with_black
import asyncio
import logging
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


logging.getLogger("mcp.client.streamable_http").setLevel(logging.ERROR)

SERVER_URL = "https://mcp.deepwiki.com/mcp"


def deepwiki_query(question):
    final_result = None

    async def main_async():
        nonlocal final_result
        transport = StreamableHttpTransport(SERVER_URL)
        client = Client(transport=transport)
        async with client:
            repo = "tenstorrent/tt-metal"
            result = await client.call_tool("ask_question", {"repoName": repo, "question": question})
            try:
                final_result = result.content[0].text
            except (IndexError, AttributeError):
                final_result = str(result)

    try:
        asyncio.run(main_async())
    except:
        return None
    return final_result


def parse_explanation_and_function(text: str) -> Optional[Tuple[str, str]]:
    """
    Extracts the explanation and function code from AI output using the
    BEGIN/END markers. Returns (explanation, function_code) if found,
    otherwise None.
    """

    exp_match = re.search(r"BEGIN EXPLANATION\s+(.*?)\s+END EXPLANATION", text, re.DOTALL | re.IGNORECASE)
    func_match = re.search(r"BEGIN FUNCTION\s*(?:python)?\s*(.*?)\s*\s*END FUNCTION", text, re.DOTALL | re.IGNORECASE)

    if not exp_match or not func_match:
        return None

    explanation = exp_match.group(1).strip()
    function_code = func_match.group(1).strip()

    function_code = function_code.rstrip("`").strip()  # Remove trailing backticks if present
    function_code = function_code.rstrip("\n").strip()  # Remove trailing newlines if present
    function_code = function_code.lstrip("```python").strip()  # Remove leading 'python' if present
    function_code = function_code.lstrip("\n").strip()  # Remove leading newlines if present
    function_code = "\n".join(
        [
            line
            for line in function_code.split("\n")
            if not line.strip().startswith("#")
            and line.strip() != ""
            and not line.strip().startswith("import")
            and not line.strip().startswith("from")
        ]
    )
    if function_code.split("def")[0] != "":
        return None
    if function_code.count("def") != 1:
        return None
    if function_code.count("return") != 1:
        return None
    if function_code.split("\n")[-1].strip() == "":
        if "return" not in function_code.split("\n")[-2]:
            return None
    explanation = explanation.replace('"""', "").replace("\n    ", "\n")
    explanation = explanation.replace("<cite/>", "").replace("<cite>", "")
    explanation = explanation.replace("\n", "\n    ").strip()  # add a tab after every line
    return explanation, function_code


def get_single_chat_response(message: str, api_key: str, pipeline_id: int) -> str | None:
    """
    Sends a message to RunLLM's streaming chat API, collects all 'generation_in_progress'
    chunks, and returns the concatenated result once streaming is complete.
    Returns None on any error.
    """
    url = f"https://api.runllm.com/api/pipeline/{pipeline_id}/chat"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload = {"message": message}

    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as resp:
            if resp.status_code != 200:
                return None

            client = sseclient.SSEClient(resp)
            output = ""
            for event in client.events():
                try:
                    chunk = json.loads(event.data)
                    if chunk.get("chunk_type") == "generation_in_progress":
                        output += chunk.get("content", "")
                except (ValueError, json.JSONDecodeError):
                    return None  # Parsing failed

            return output if output else None

    except requests.RequestException:
        return None


def extract_composite_functions(filename):
    with open(filename, "r") as f:
        src = f.read()
    tree = ast.parse(src)
    with open(filename, "r") as f:
        lines = f.readlines()

    composite_funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (
            re.match(r"COMPOSITE_\w+", node.name) or re.match(r"main", node.name)
        ):
            # Extract the function source code using line numbers
            start = node.lineno - 1
            # node.end_lineno is available in Python 3.8+
            end = node.end_lineno if hasattr(node, "end_lineno") else node.body[-1].lineno
            func_str = "".join(lines[start:end])
            composite_funcs.append(func_str)
    return composite_funcs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_torch_functions.py clustered_graph.py")
        sys.exit(1)
    filename = sys.argv[1]
    funcs = extract_composite_functions(filename)
    failed_funcs = [(func, 0) for func in funcs if "torch.ops.aten." in func and "torch." in func]
    remaining_funcs = [func for func in funcs if (func, 0) not in failed_funcs]
    ttnn_functions = {func: ("", func) for func in remaining_funcs}
    while failed_funcs:
        print(f"{len(failed_funcs)} functions remaining to process...")
        func, num_times = failed_funcs[0]
        failed_funcs.remove((func, num_times))
        if num_times >= 5:
            print(f"Tried to processes {func} 5 times. Giving up...")
            ttnn_functions[func] = ("", func)
            continue
        message = f"""
Extracted composite function:
```python
{func}
```

You must produce exactly two clearly delimited sections:

1. EXPLANATION SECTION
Write a very concise explanation of how the extracted function works, including a breakdown of each step, inputs, outputs, and any important implementation details. Make the explanation have multiple lines so it can fit in a python docstring.
Wrap this section between the exact markers (on their own lines):
BEGIN EXPLANATION
... your explanation here ...
END EXPLANATION

2. FUNCTION SECTION
Write a new TTNN equivalent Python function that:
- Has the same name, parameters, and return type as the original function.
- Does NOT use any PyTorch operations.
- Uses only the TTNN API.
- Is self-contained (no additional imports or dependencies).
- Is valid, executable Python code.

Wrap this section between the exact markers (on their own lines):
BEGIN FUNCTION
```python
# Your TTNN equivalent function code here
END FUNCTION

Do NOT include any text, commentary, or formatting outside these markers. Make your explanation minimal. I only want the EXPLANATION SECTION and the FUNCTION SECTION. Everything else is not important. make the FUNCTION SECTION runnable. In the FUNCTION SECTION, have the least amount of comments (e.g. # <text>) as necessary to generate the best FUNCTION SECTION result.
    """
        # response = get_single_chat_response(message, "llm_yalEHGlxSUmdG8BcBpF6oQ==", 1115)
        response = deepwiki_query(message)
        if response is None:
            failed_funcs.append((func, num_times + 1))  # Re-add to failed list if response is None
            print("Failed to get a valid response. Retrying...")
        else:
            result = parse_explanation_and_function(response)
            if result is None:
                failed_funcs.append((func, num_times + 1))  # Re-add to failed list if parsing failed
                print("Failed to parse explanation and function from the response. Trying again...")
                continue
            explanation, new_func = result
            ttnn_functions[func] = (explanation, new_func)
            print("Explanation:")
            print(explanation)
            print("\nFunction:")
            print(new_func)
            print("\n\n")
    with open("ttnn_composite_functions.py", "w") as f:
        f.write("# Auto-generated TTNN composite functions\n\n")
        f.write("import ttnn\n")
        for func in funcs:
            explanation, func_code = ttnn_functions[func]
            # Write explanation as a multiline comment
            if len(explanation.strip()) > 0:
                explanation_comment = f'    """\n    {explanation}\n    """\n'
                fn_def = func_code.split(":")
                fn_def[1] = "\n" + explanation_comment + fn_def[1]  # Insert explanation before the function body
                func_code = ":".join(fn_def)
            # Write the function code
            f.write(func_code + "\n\n")
    format_file_with_black("ttnn_composite_functions.py")
    print(
        f"Extracted {len(ttnn_functions) - len(remaining_funcs)} translated function out of a total of {len(ttnn_functions)} functions to ttnn_composite_functions.py"
    )
