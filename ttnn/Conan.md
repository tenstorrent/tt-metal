# Building TT-NN with Conan

This document provides instructions for building **TT-NN** (Tenstorrent Neural Network library) using the [Conan package manager](https://conan.io/). Conan simplifies dependency management and provides a standardized way to build and package the library.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installing Conan](#installing-conan)
- [Building TT-NN with Conan](#building-tt-nn-with-conan)
  - [Quick Start](#quick-start)
  - [Build Options](#build-options)
  - [Build Types](#build-types)
- [Using TT-NN as a Conan Dependency](#using-tt-nn-as-a-conan-dependency)
- [Testing the Conan Package](#testing-the-conan-package)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

[Conan cheatsheet](https://media.jfrog.com/wp-content/uploads/2021/05/25084434/conan-cheatsheet.pdf)

Before building with Conan, ensure you have:

1. **Git** (with submodules cloned):
   ```bash
   git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
   cd tt-metal
   ```

2. **Python 3.10** with pip installed

3. **System dependencies** as outlined in [INSTALLING.md](../INSTALLING.md):
   - Linux OS (Ubuntu 22.04 recommended)
   - Hardware setup completed (if running on Tenstorrent hardware)

---

## Installing Conan

Install Conan version 2.x according to official [documentation](https://docs.conan.io/2/installation.html):

Verify the installation:

```bash
conan --version
```

Configure your Conan profile (first time only):

```bash
conan profile detect --force
```

This creates a default profile based on your system configuration. You can view it with:

```bash
conan profile show default
```

---

## Building TT-NN with Conan

### Quick Start

To build TT-NN with default settings:

```bash
# Create the package locally
conan create . --build=missing

# Or build in the source tree for development
conan install . --build=missing
conan build .
```

The `--build=missing` flag tells Conan to build any dependencies that aren't already available in your Conan cache.

### Build Options

The TT-NN Conan recipe supports several options that can be configured:

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `shared` | `[True, False]` | `True` | Build as shared library |
| `build_examples` | `[True, False]` | `False` | Build programming examples |
| `build_tests` | `[True, False]` | `False` | Build tests |
| `enable_distributed` | `[True, False]` | `True` | Enable multihost distributed compute |
| `enable_profiler` | `[True, False]` | `False` | Enable Tracy profiler support |

**Example with custom options:**

```bash
conan create . \
  --build=missing \
  -o shared=True \
  -o enable_profiler=True
```

### Build Types

You can specify different build types using the `-s` (settings) flag:

```bash
# Release build (optimized)
conan create . -s build_type=Release --build=missing

# Debug build (with debug symbols)
conan create . -s build_type=Debug --build=missing

# Release with debug info (default)
conan create . -s build_type=RelWithDebInfo --build=missing
```

---

## Using TT-NN as a Conan Dependency

Once built, you can use TT-NN as a dependency in your own Conan-based projects.

### In your project's `conanfile.txt`:

```ini
[requires]
tt-nn/[>=0.60.0]

[generators]
CMakeDeps
CMakeToolchain
```

### In your project's `conanfile.py`:

```python
from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

class MyProjectConan(ConanFile):
    name = "my-project"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("tt-nn/[>=0.60.0]")

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
```

### In your project's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.25)
project(MyProject)

find_package(tt-nn REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE tt-nn::tt-nn)
```

Then install dependencies and build:

```bash
conan install . --build=missing
conan build .
```

---

## Testing the Conan Package

The repository includes a test package to verify the Conan build works correctly:

```bash
# Test the package after creating it
conan create . --build=missing

# Or test manually
conan test test_package "tt-nn/[>=0.62.0]"
```

This will:
1. Build a small test application that links against TT-NN
2. Verify that the headers and libraries are accessible
3. Run the test executable

---

## Advanced Usage

### Using a Custom Build Folder

By default, Conan uses `.conan-build` as the build folder to avoid conflicts with existing build directories. You can customize this:

```bash
conan install . --output-folder=my-custom-build --build=missing
conan build . --output-folder=my-custom-build
```

### Editable Mode (Development)

For active development, you can use editable mode to work with the source directly:

```bash
# Make the package editable
conan editable add . tt-nn/0.60.0

# Install dependencies
conan install . --output-folder=.conan-build --build=missing

# Build
conan build . --output-folder=.conan-build

# When done developing
conan editable remove tt-nn/0.60.0
```

### Specifying Compiler

To use a specific compiler:

```bash
conan create . \
  -s compiler=clang \
  -s compiler.version=17 \
  --build=missing
```

Or with gcc:

```bash
conan create . \
  -s compiler=gcc \
  -s compiler.version=12 \
  --build=missing
```

### Building Without Python Bindings

The Conan recipe builds TT-NN as a C++ library only (Python bindings are disabled by default in the Conan recipe). If you need Python bindings, use the standard build process described in [INSTALLING.md](../INSTALLING.md).

---

## Troubleshooting

### Issue: `Missing submodules` error

**Solution:** Ensure all git submodules are initialized and updated:

```bash
git submodule update --init --recursive
```

### Issue: Build fails with compiler errors

**Solution:** Ensure you're using a supported compiler:
- GCC 12 or later
- Clang 17 or later

Update your Conan profile:

```bash
conan profile detect --force
```

### Issue: Out of memory during build

**Solution:** The project uses unity builds by default to speed up compilation. If you run out of memory:

1. Reduce parallel jobs:
   ```bash
   conan create . --build=missing -c tools.cmake.cmake_program:cmake_jobs=2
   ```

2. Or disable unity builds (modify `conanfile.py` and set `tc.variables["TT_UNITY_BUILDS"] = False`)


---

## Conan Recipe Details

The TT-NN Conan recipe (`conanfile.py`) is located at the root of the repository. Key characteristics:

- **Package Name:** `tt-nn`
- **Version:** Automatically derived from git tags
- **License:** Apache-2.0
- **Build System:** CMake with Ninja generator
- **Dependencies:** Managed via CMake's CPM if not installed by conan (most dependencies are fetched during build)

### Environment Variables Set by Conan

When consuming the package, Conan automatically sets:

- `TT_METAL_RUNTIME_ROOT` - Points to the package installation directory

### Exported Libraries

The package exports the following libraries:

- `tt_stl`
- `tt-nn`
- `tt_metal`

And provides include directories for headers and third-party dependencies.

---

## Additional Resources

- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
- [TT-Metalium Programming Guide](../METALIUM_GUIDE.md)
- [Standard Installation Guide](../INSTALLING.md)
- [Conan Documentation](https://docs.conan.io/)
- [Contributing Guidelines](../CONTRIBUTING.md)

---

## Support

For issues related to:
- **TT-NN functionality:** Open an issue on the [tt-metal GitHub repository](https://github.com/tenstorrent/tt-metal/issues)
- **Conan build process:** Check this document's troubleshooting section or open an issue with the `[Conan]` tag
- **General questions:** Join the [Tenstorrent Discord](https://discord.gg/tvhGzHQwaj)
