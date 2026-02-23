#!/usr/bin/env python3
"""4-stage blitz-decode pipeline bridge using PipelineBlock.

Stage-0 (mesh_id == 0) bridges the C++ inference server via SHM while
all 4 stages relay data through D2D sockets on a single BH Galaxy.

Launch with:
    tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml \\
        ttnn/ttnn/distributed/ttrun_hello_world.py
"""

import mmap
import os
import signal
import struct
import sys

import torch
import ttnn

from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import ttnn_dtype_from_torch_dtype

# Pipeline sizing: 64-byte end-to-end pages.
# embedding_dim=32 bfloat16 elements == 64 bytes per row, matching the H2D token page size
# so the same page size flows unchanged through H2D → D2D → D2H.
EMBEDDING_DIM = 32
EMBEDDING_VOCAB = 131072        # covers max token id (~125836) in fixed_reply_sequence
EMBEDDING_DTYPE = torch.bfloat16
EMBEDDING_SIZE_BYTES = EMBEDDING_DIM * 2   # 32 × 2 = 64 bytes
TOKEN_SIZE_BYTES = 64                       # H2D page: task_id (36 B) + token_id (8 B) + pad
FIFO_SIZE = 128                             # 2 × page size for minimal in-flight buffering
PIPELINE_CORE = ttnn.CoreCoord(0, 0)
FABRIC_MAX_PAYLOAD = 7168

# SHM layout – must match TtRunDeviceBackend (C++)
_PAGE_SIZE = 64
_NUM_SLOTS = 1024
_CHANNEL_HEADER = 64
_CHANNEL_SIZE = _CHANNEL_HEADER + _NUM_SLOTS * _PAGE_SIZE
_SHM_SIZE = 2 * _CHANNEL_SIZE
_C2P_READ_POS = 0
_C2P_WRITE_POS = 4
_C2P_SLOTS_OFF = _CHANNEL_HEADER
_P2C_READ_POS = _CHANNEL_SIZE
_P2C_WRITE_POS = _CHANNEL_SIZE + 4
_P2C_SLOTS_OFF = _CHANNEL_SIZE + _CHANNEL_HEADER

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _rank():
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _shm_recv_from_cpp(buf):
    r = struct.unpack_from("<I", buf, _C2P_READ_POS)[0]
    while not _shutdown:
        w = struct.unpack_from("<I", buf, _C2P_WRITE_POS)[0]
        if r != w:
            break
    if _shutdown:
        return None
    slot_off = _C2P_SLOTS_OFF + (r % _NUM_SLOTS) * _PAGE_SIZE
    data = bytes(buf[slot_off : slot_off + _PAGE_SIZE])
    struct.pack_into("<I", buf, _C2P_READ_POS, r + 1)
    return data


def _shm_send_to_cpp(buf, payload):
    while not _shutdown:
        r = struct.unpack_from("<I", buf, _P2C_READ_POS)[0]
        w = struct.unpack_from("<I", buf, _P2C_WRITE_POS)[0]
        if (w - r) % (1 << 32) < _NUM_SLOTS:
            break
    if _shutdown:
        return
    slot_off = _P2C_SLOTS_OFF + (w % _NUM_SLOTS) * _PAGE_SIZE
    buf[slot_off : slot_off + _PAGE_SIZE] = payload
    struct.pack_into("<I", buf, _P2C_WRITE_POS, w + 1)


def _open_shm(shm_name):
    path = f"/dev/shm/{shm_name}"
    if not os.path.exists(path):
        return None
    fd = os.open(path, os.O_RDWR)
    try:
        return mmap.mmap(fd, _SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)


def _open_mesh_device():
    fabric_router_config = ttnn._ttnn.fabric.FabricRouterConfig()
    fabric_router_config.max_packet_payload_size_bytes = FABRIC_MAX_PAYLOAD
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))


def _create_pipeline_block(mesh_device):
    if mesh_device.get_system_mesh_id() == 0:
        torch_embedding = torch.randn(
            (1, 1, EMBEDDING_VOCAB, EMBEDDING_DIM), dtype=EMBEDDING_DTYPE
        )
        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn_dtype_from_torch_dtype(EMBEDDING_DTYPE),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(
            embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE,
            FIFO_SIZE,             # upstream d2d socket fifo size
            FIFO_SIZE,             # downstream d2d socket fifo size
            EMBEDDING_SIZE_BYTES,  # upstream d2d socket page size (= d2h page size)
            EMBEDDING_SIZE_BYTES,  # downstream d2d socket page size (= embedding row size)
            h2d_socket_fifo_size=FIFO_SIZE,
            d2h_socket_fifo_size=FIFO_SIZE,
            d2h_socket_page_size=EMBEDDING_SIZE_BYTES,
            embedding_tensor=embedding_tensor,
        )
    return PipelineBlock(
        mesh_device,
        PIPELINE_CORE,
        FIFO_SIZE,
        FIFO_SIZE,
        EMBEDDING_SIZE_BYTES,
        EMBEDDING_SIZE_BYTES,
    )


def _shm_pipeline_bridge(pipeline_block, shm_buf, log):
    """Read token pages from C++, push through the pipeline, echo input page back.

    The input page is echoed back unchanged (task_id + token_id preserved) so the
    C++ server can match results to requests. The pipeline round-trip still exercises
    all H2D, D2D, and D2H socket paths end-to-end.
    """
    token_elems = TOKEN_SIZE_BYTES // 4
    while not _shutdown:
        page = _shm_recv_from_cpp(shm_buf)
        if page is None:
            break

        token_ints = struct.unpack_from(f"<{token_elems}I", page)
        input_tensor = ttnn.from_torch(
            torch.tensor(token_ints, dtype=torch.uint32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pipeline_block.write_token(input_tensor)

        output_tensor = ttnn.from_torch(
            torch.zeros(1, EMBEDDING_DIM, dtype=EMBEDDING_DTYPE),
            dtype=ttnn_dtype_from_torch_dtype(EMBEDDING_DTYPE),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pipeline_block.read_output(output_tensor)

        # Echo the original page back so C++ task_id / token_id matching still works.
        _shm_send_to_cpp(shm_buf, page)

    log.write("Rank 0: SHM pipeline bridge exiting\n")
    log.flush()


def main():
    log_dir = "/tmp/tt-run/logs"
    os.makedirs(log_dir, exist_ok=True)
    pid = os.getpid()
    rank = _rank()
    log_path = f"{log_dir}/ttrun_hello_world_{pid}_{rank}.log"
    try:
        log = open(log_path, "a", encoding="utf-8")
    except OSError as e:
        print(f"ttrun_hello_world: could not open {log_path}: {e}", file=sys.stderr)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    with log:
        log.write(f"Rank {rank}: opening 4×2 mesh device\n")
        log.flush()

        try:
            mesh_device = _open_mesh_device()
        except Exception as e:
            log.write(f"Rank {rank}: failed to open mesh device: {e}\n")
            sys.exit(1)

        ttnn.enable_asynchronous_slow_dispatch(mesh_device)
        my_mesh_id = mesh_device.get_system_mesh_id()
        log.write(f"Rank {rank}: mesh_id={my_mesh_id}\n")
        log.flush()

        try:
            pipeline_block = _create_pipeline_block(mesh_device)
            pipeline_block.run()
            log.write(f"Rank {rank}: pipeline running\n")
            log.flush()

            if pipeline_block.is_first_pipeline_stage():
                shm_name = os.environ.get("TT_IPC_SHM")
                if shm_name:
                    shm_buf = _open_shm(shm_name)
                    if shm_buf:
                        log.write("Rank 0: entering SHM pipeline bridge\n")
                        log.flush()
                        try:
                            _shm_pipeline_bridge(pipeline_block, shm_buf, log)
                        finally:
                            shm_buf.close()
                    else:
                        log.write(f"Rank 0: SHM region '{shm_name}' not found\n")
                else:
                    log.write("Rank 0: TT_IPC_SHM not set – pipeline up, terminating\n")
                log.flush()

            pipeline_block.terminate()
        finally:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

        log.write(f"Rank {rank}: done\n")


if __name__ == "__main__":
    main()
