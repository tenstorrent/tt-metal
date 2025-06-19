# Install

These instructions will guide you through the installation of Tenstorrent system tools and drivers, followed by the installation of TT-Metalium and TT-NN.

> [!IMPORTANT]
>
> If you are using a release version of this software, check installation instructions packaged with it.
> You can find them in either the release assets for that version, or in the source files for that [version tag](https://github.com/tenstorrent/tt-metal/tags).

## Prerequisites:

### 1: Set Up the Hardware
- Follow the instructions for the Tenstorrent device you are using at: [Hardware Setup](https://docs.tenstorrent.com)

---

### 2: Install Driver & Firmware

Note the current compatibility matrix:

| Device               | OS              | Python   | Driver (TT-KMD)    | Firmware (TT-Flash)                        | TT-SMI                | TT-Topology                    |
|----------------------|-----------------|----------|--------------------|--------------------------------------------|-----------------------|--------------------------------|
| Galaxy (Wormhole 4U) | Ubuntu 22.04    | 3.10     | v1.33 or above     | fw_pack-80.10.1.0                          | v2.2.3 or lower       | v1.1.3, `mesh` config          |
| Galaxy (Wormhole 6U) | Ubuntu 22.04    | 3.10     | v1.33 or above     | fw_pack-80.17.0.0 (v80.17.0.0)             | v3.0.15 or above      | N/A                            |
| Wormhole             | Ubuntu 22.04    | 3.10     | v1.33 or above     | fw_pack-80.17.0.0 (v80.17.0.0)             | v3.0.12 or above      | N/A                            |
| T3000 (Wormhole)     | Ubuntu 22.04    | 3.10     | v1.33 or above     | fw_pack-80.17.0.0 (v80.17.0.0)             | v3.0.12 or above      | v1.2.5 or above, `mesh` config |
| Blackhole            | Ubuntu 22.04    | 3.10     | v1.33 or above     | fw_pack-80.18.0.0 (v80.18.0.0)             | v3.0.12 or above      | N/A                            |

#### Install System-level Dependencies
```
wget https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/{install_dependencies.sh,tt_metal/sfpi-version.sh}
chmod a+x install_dependencies.sh
sudo ./install_dependencies.sh
```

---

#### Install the Driver (TT-KMD)
- DKMS must be installed:

| OS                     | Command                                            |
|------------------------|----------------------------------------------------|
| Ubuntu / Debian        | ```apt install dkms```                             |
| Fedora                 | ```dnf install dkms```                             |
| Enterprise Linux Based | ```dnf install epel-release && dnf install dkms``` |

- Install the latest TT-KMD version:
```
git clone https://github.com/tenstorrent/tt-kmd.git
cd tt-kmd
sudo dkms add .
sudo dkms install "tenstorrent/$(./tools/current-version)"
sudo modprobe tenstorrent
cd ..
```

- For more information visit Tenstorrents [TT-KMD GitHub repository](https://github.com/tenstorrent/tt-kmd).

---

#### Update Device TT-Firmware with TT-Flash


- Install TT-Flash:

```
pip install git+https://github.com/tenstorrent/tt-flash.git
```

- Update TT-Firmware:

  - First, set the appropriate TT-Firmware version per device:

  | Device                        | Command                                                    |
  |-------------------------------|------------------------------------------------------------|
  | Blackhole                     | ```fw_tag=v80.18.0.0 fw_pack=fw_pack-80.18.0.0.fwbundle``` |
  | Galaxy (6U) / Wormhole / T300 | ```fw_tag=v80.17.0.0 fw_pack=fw_pack-80.17.0.0.fwbundle``` |

  - Then Download and install TT-Firmware:

  ```
  wget https://github.com/tenstorrent/tt-firmware/raw/refs/tags/$fw_tag/$fw_pack
  tt-flash flash --fw-tar $fw_pack
  ```

- For more information visit Tenstorrent's [TT-Firmware GitHub Repository](https://github.com/tenstorrent/tt-firmware) and [TT-Flash Github Repository](https://github.com/tenstorrent/tt-flash).

---

#### Install System Management Interface (TT-SMI)
- Install Tenstorrent Software Management Interface (TT-SMI) according to the table above. We will use a specific version here as an example:
```
pip install git+https://github.com/tenstorrent/tt-smi@v3.0.12
```

- Verify System Configuration

Once hardware and system software are installed, verify that the system has been configured correctly.

  - Run the TT-SMI utility:
  ```
  tt-smi
  ```
  A display with device information, telemetry, and firmware will appear:<br>

![image](https://docs.tenstorrent.com/_images/tt_smi.png)
<br>
  If the tool runs without error, your system has been configured correctly.

- For more information, visit Tenstorrent's [TT-SMI GitHub repository](https://github.com/tenstorrent/tt-smi).

---

#### (Optional) Multi-Card Configuration (TT-Topology)

> [!CAUTION]
> Be sure to align the topology version with the compatible version in the table above for your particular configuration.

- For TT-Loudbox or TT-QuietBox systems, visit Tenstorrent's [TT-Topology README](https://github.com/tenstorrent/tt-topology/blob/main/README.md).

---

### TT-NN / TT-Metalium Installation

#### There are three options for installing TT-Metalium:

- [Option 1: From Source](#option-1-from-source)

  Installing from source gets developers closer to the metal and the source code.

- [Option 2: From Docker Release Image](#option-2-from-docker-release-image)

  Installing from Docker Release Image is the quickest way to access our APIs and to start running AI models.

- [Option 3: From Wheel](#option-3-from-wheel)

  Install from wheel as an alternative method to get quick access to our APIs and to running AI models.

---

### Option 1: From Source
Install from source if you are a developer who wants to be close to the metal and the source code. Recommended for running the demo models.

#### Step 1. Clone the Repository:

```sh
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
```

#### Step 2. Invoke our Build Scripts:

```
./build_metal.sh
```

- (recommended) For an out-of-the-box virtual environment to use, execute:
```
./create_venv.sh
source python_env/bin/activate
```

- (optional) Software dependencies for profiling use:
  - Download and install [Doxygen](https://www.doxygen.nl/download.html), (v1.9 or higher, but less than v1.10)

- Continue to [You Are All Set!](#you-are-all-set)

---

### Option 2: From Docker Release Image
Installing from Docker Release Image is the quickest way to access our APIs and to start running AI models.

Download the latest Docker release from our [Docker registry](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64) page

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
docker run -it --rm -v /dev/hugepages-1G:/dev/hugepages-1G --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc bash
```

- For more information on the Docker Release Images, visit our [Docker registry page](https://github.com/orgs/tenstorrent/packages?q=tt-metalium-ubuntu&tab=packages&q=tt-metalium-ubuntu-22.04-release-amd64).

- You are all set! Try some [TT-NN Basic Examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) next.

---

### Option 3: From Wheel
Install from wheel for quick access to our APIs and to get an AI model running

#### Step 1. Download and Install the Latest Wheel:

- Navigate to our [releases page](https://github.com/tenstorrent/tt-metal/releases/latest) and download the latest wheel file for the Tenstorrent card architecture you have installed.

- Install the wheel using your Python environment manager of choice. For example, to install with `pip`:

  ```sh
  pip install <wheel_file.whl>
  ```

#### Step 2. (For models users only) Set Up Environment for Models:

To try our pre-built models in `models/`, you must:

  - Install their required dependencies
  - Set appropriate environment variables
  - Set the CPU performance governor to ensure high performance on the host

- This is done by executing the following:
  ```sh
  export PYTHONPATH=$(pwd)
  pip install -r tt_metal/python_env/requirements-dev.txt
  sudo apt-get install cpufrequtils
  sudo cpupower frequency-set -g performance
  ```

---

### You are All Set!

#### To verify your installation (for source or wheel installation only), try executing a programming example:

- First, set the following environment variables:

  - Run the appropriate command for the Tenstorrent card you have installed:

  | Card             | Command                              |
  |------------------|--------------------------------------|
  | Grayskull        | ```export ARCH_NAME=grayskull```     |
  | Wormhole         | ```export ARCH_NAME=wormhole_b0```   |
  | Blackhole        | ```export ARCH_NAME=blackhole```     |

  - Run:
  ```
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd)
  ```

- Then, try running a programming example:
  ```
  python3 -m ttnn.examples.usage.run_op_on_device
  ```

- For more programming examples to try, visit Tenstorrent's [TT-NN Basic Examples Page](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples) or get started with [Simple Kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html)

---

### Interested in Contributing?
- For more information on development and contributing, visit Tenstorrent's [CONTRIBUTING.md page](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md).
