## TTNN

TTNN is a Python library that provides a launching point for learning the APIs within ``TT-METAL``.
The TTNN library assumes the user is familiar with [PyTorch](https://pytorch.org) and provides
operations that easily translate PyTorch tensors to and from ``ttnn.Tensor``(s).   This library is an application programming interface intended for Tenstorrent device operations with a primary dependency on the Python libray tt_lib within the tt_eager subproject.  This code has been tested with PyTorch 1.13.

We trust that this library will be a valuable guide to helping you on your journey to take full advantage of our devices!

#### Please learn the API using our Jupyter Notebook tutorials
* There is a collection of tutorials written with Jupyter Notebooks to help you ramp up your skillset for using `tt-metal`. These
notebooks can be found under https://github.com/tenstorrent-metal/tt-metal/tree/main/ttnn/tutorials.
* These tutorials assume you already have a machine set up with either a grayskull or wormhole device available and that you have successfully
followed the instructions for [installing and building the software](https://github.com/tenstorrent-metal/tt-metal/blob/main/README.md).
* From within the `ttnn/tutorials` directory, launch the notebooks with: `jupyter lab --no-browser --port=8888`


#### Here are a few key differences with the operations we currently support compared to their equivalents in PyTorch
* shape
    * Unlike PyTorch, the shape class maintains both the intended shape and the shape including padding.
    * When moving between row major and tile layout via the ttnn.to_layout(...) function, the last two dimensions are automatically padded and unpadded as necessary.
* matmul
    * The tensors must be moved to device before the operation can be done.
    * The last two dimensions must be a multiple of 32 if the multiplication is to work on the device.  For example, a tensor with shape (1, 1, 3, 4) would not be successfully multiplied to another tensor with tile layout (ttnn.TILE_LAYOUT).  For these reason the to_layout function will automatically add padding to the last two dimensions so that the height and width dimensions are a multiples of 32 and ready for a device multiplication.
    * Results from a matmul will not have the exact same precision. The order of operations are likely to be different even when comparing dytypes of bfloat16.  Instead of PyTorch allclose, we often use a pearson correlation coefficient to verify the results.
    * The dot product is not fully supported yet and unfortunately returns a Tensor with a shape.
* add
    * The tensors must be moved to device before the operation can be done.
* subtract.
    * Broadcasting is only suported in the last two dimensions (ie the height and width dimensions).
    * The tensors must be moved to device before the operation can be done.
* reshape
    * The last two dimensions in the reshape must be a multiple of 32 when using the tensor on a device.
    * When converting a tt tensor to a PyTorch tensor, the 3rd and 4th dimensions must be a multiple of 32 or the operation will default to using the PyTorch implementation.
* transpose
    * There is no transpose method.  Please use permute instead.
* permute
    * When using the ttnn library, the operation requires the first parameter to be the tensor and the next to be the new order of dimensions within a parenthesis.

#### Frequently asked questions
* What if my device hangs?
    * Try resetting the board on the command line with: `tt-smi -tr all`
* What types are supported on device?
    * We currently support ttnn.bfloat16, ttnn.bfloat8 and ttnn.uint32.
* What shapes are supported on device?
    * The last dimension of the shape multiplied by the number of bytes of the sizeof the dataype must be a multiple of four.  For example, ttnn.bloat16 would need to have the last dimension be even.
    * TODO : address ttnn.bfloat8_b and how mantissa is stored per tile
    * TODO : address converting from int in data type for torch tensors to ttnn.uint32
    * TODO : mention how ttnn.blfloat32 is not supported on device
* Is slicing available?
    * Slicing is supported.  At the moment this feature falls back to using PyTorch slicing on the host.
    * Example:
        * tensor1 = ttnn.from_torch(torch.randn(3,3))
        * print(tensor1[:1])
* Why are the results from operations like add and matmul not the same precision and require a pearson correlation coefficient comparison?
    * Results for operations are different because the order of floating point operations is different between CPU and the TT device.  A similiar issue would arise when comparing cpu and gpu operations.
* How do I create a tensor of all zeros that is not on device and the height and width do not have to be multiples of 32?
    * Use PyTorch to achieve this.
        * tensor = ttnn.from_torch(torch.zeros(3,3))
        * print(tensor)
* How do I dump the logs of operations from the TT device?
    * You can add one or both of these environment variables
        *   `export TT_METAL_LOGGER_TYPES=Op`
        *   `export TT_METAL_LOGGER_LEVEL=DEBUG`
    * For the location of the operations use the following environment variable
        * `export OPERATION_HISTORY_CSV=<filename>`


#### How to debug from python and C++ at the same time within vscode
* `export CONFIG=debug` within your virtual environment and run `make build`
* Allow your process to attach to the debugger.
    * `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
    * When done you can reverse this with... `echo 1 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
* Add a task in launch.json for debugging your python code.  For example, you might have something like this below...
    ```
    {
      "name": "Python: test_matmul",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/build/python_env/bin/pytest",
      "args": [
        "${workspaceFolder}/tests/ttnn/test_matmul.py",
        "-k",
        "test_matmul"
      ],
      "justMyCode": false,
      "stopOnEntry": false,
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "internalConsoleOptions": "openOnSessionStart",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    ```
* Additionally, add the following to launch.json (note that the name "C++: Attach to Python" will be used to find this task in the next step)
    ```
    {
      "name": "C++: Attach to Python",
      "type": "cppdbg",
      "request": "attach",
      "program": "${workspaceFolder}/build/python_env/bin/python3",
      "MIMode": "gdb",
      "miDebuggerPath": "gdb",
      "processId": "296907"
    },
    ```
* Wherever you want to debug the python code, add the function call update_process_id() from utils_for_testing.py
    * When you run your python test in the vscode debugger (for example, launching the test_matmul above), this method will update the "processId" of the task "C++: Attach to Python" in the .vscode/launch.json
    * Use a breakpoint in your python code to halt execution to give you time to attach the C++ debugger.
    * Run the "C++: Attach to Python" which will now have the processId updated that was being used by the debugger.
    * Remember to have a breakpoint in the C++ code that you want to debug as well.
    * Finally, continue from the python breakpoint to begin debugging your C++ calls.
    * Note, the function update_process_id() is defined in utils_for_testing.py as...
        ```
        def update_process_id():
            print(f"Debugging PID: {os.getpid()}")
            cwd = os.getcwd()
            launch_json_path = os.path.join(cwd, ".vscode", "launch.json")
            with open(launch_json_path, "r") as f:
                launch_data = json.load(f)

            for config in launch_data.get("configurations", []):
                if config.get("name") == "C++: Attach to Python":
                    config["processId"] = str(os.getpid())
                    break

            with open(launch_json_path, "w") as f:
                json.dump(launch_data, f, indent=4)

        ```
