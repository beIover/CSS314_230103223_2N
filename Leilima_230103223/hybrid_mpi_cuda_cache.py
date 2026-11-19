
import os
import sys
import json
import time
import math
import hashlib
import platform
import subprocess
import ctypes

import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64


# ============================================
# GLOBAL CONFIGURATION
# ============================================

GITHUB_USERNAME = "beIover"
STUDENT_ID = "230103223"

GLOBAL_NX = 512
GLOBAL_NY = 512

LOCAL_NX = 256
LOCAL_NY = 256

PADDED_NX = LOCAL_NX + 2  # 258
PADDED_NY = LOCAL_NY + 2  # 258

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile

STENCIL_STEPS = 300
SHM_ITERS = 40_000_000

LAMBDA_DAMP = 0.15

ALIGNMENT = 64
NUMBER_OF_SLOTS = 64


# ============================================
# HARDWARE INFORMATION
# ============================================

def get_gpu_uuid():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader"
        ],
        encoding="utf-8"
    )
    return output.strip().splitlines()[0].strip()


def detect_cpu_line_size():
    path = (
        "/sys/devices/system/cpu/cpu0/cache/"
        "index0/coherency_line_size"
    )
    if os.path.exists(path):
        with open(path, "r") as file:
            return int(file.read().strip())
    return 64


def get_cpu_signature():
    model = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as file:
            for line in file:
                if "model name" in line:
                    model = line.split(":", 1)[1].strip()
                    break
    return (
        f"{platform.machine()}::"
        f"{model}::"
        f"{os.cpu_count()}"
    )


# ============================================
# TASK 1: HYBRID MPI CONTEXT
# ============================================

def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks, got {size}")

    # 1. Cartesian topology
    dims = (2, 2)
    periods = (False, False)
    cart_comm = comm.Create_cart(dims=dims, periods=periods, reorder=False)
    coords = cart_comm.Get_coords(rank)
    up, down = cart_comm.Shift(0, 1)
    left, right = cart_comm.Shift(1, 1)

    neighbors = {
        "up": up,
        "down": down,
        "left": left,
        "right": right
    }

    # 2. GPU initialization
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)

    # 3. CPU information
    cpu_sig = get_cpu_signature()
    line_bytes = detect_cpu_line_size()

    # 4. Deterministic seed
    seed_input = f"{github_user.strip()}::{cpu_sig}::{gpu_uuid}"
    digest = hashlib.sha256(seed_input.encode("utf-8")).digest()
    instance_seed = int.from_bytes(digest[:8], byteorder="big")

    # 5. Shared-memory communicator
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()

    if node_size != 4:
        raise RuntimeError(f"Expected 4 ranks on one node, got {node_size}")

    # 6. Allocate 64-byte-aligned shared memory window
    dtype = np.dtype(np.uint64)
    required_bytes = NUMBER_OF_SLOTS * dtype.itemsize
    allocation_size = required_bytes + ALIGNMENT - 1

    win = MPI.Win.Allocate_shared(
        allocation_size if node_rank == 0 else 0,
        disp_unit=1,
        comm=node_comm
    )

    shared_buffer, item_size = win.Shared_query(0)
    if item_size < 1:
        raise RuntimeError(f"Invalid shared memory displacement unit: {item_size}")

    base_address = ctypes.addressof(ctypes.c_char.from_buffer(shared_buffer))
    offset = (-base_address) % ALIGNMENT

    if offset + required_bytes > len(shared_buffer):
        raise RuntimeError("Insufficient shared memory for aligned allocation")

    shm_view = np.ndarray(
        shape=(NUMBER_OF_SLOTS,),
        dtype=dtype,
        buffer=shared_buffer,
        offset=offset
    )

    address = shm_view.ctypes.data
    if address % ALIGNMENT != 0:
        raise RuntimeError(f"Shared memory is not 64-byte aligned: {address % ALIGNMENT}")

    if node_rank == 0:
        shm_view[:] = 0
    node_comm.Barrier()

    return {
        "comm": comm,
        "rank": rank,
        "size": size,
        "cart_comm": cart_comm,
        "coords": coords,
        "neighbors": neighbors,
        "node_comm": node_comm,
        "node_rank": node_rank,
        "node_size": node_size,
        "win": win,
        "shm_view": shm_view,
        "shared_buffer": shared_buffer,
        "shared_offset": offset,
        "shared_base_address": base_address,
        "gpu_uuid": gpu_uuid,
        "cpu_sig": cpu_sig,
        "line_bytes": line_bytes,
        "instance_seed": instance_seed
    }


# ============================================
# TASK 2: GPU FIELD INITIALIZATION & HALO EXCHANGE
# ============================================

@cuda.jit
def initialize_fields_kernel(field, global_x, global_y):
    x, y = cuda.grid(2)
    if x < LOCAL_NX and y < LOCAL_NY:
        gx = global_x + x
        gy = global_y + y
        field[y + 1, x + 1] = gy * 512.0 + gx


def initialize_fields(context):
    coords = context["coords"]
    global_y = coords[0] * LOCAL_NY
    global_x = coords[1] * LOCAL_NX

    field = cuda.to_device(
        np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64)
    )
    initialize_fields_kernel[(16, 16), (16, 16)](field, global_x, global_y)
    cuda.synchronize()
    return field


@cuda.jit
def pack_halos_kernel(field, send_up, send_down, send_left, send_right):
    idx = cuda.grid(1)
    if idx < LOCAL_NX:
        send_up[idx] = field[1, idx + 1]
        send_down[idx] = field[LOCAL_NY, idx + 1]
        send_left[idx] = field[idx + 1, 1]
        send_right[idx] = field[idx + 1, LOCAL_NX]


@cuda.jit
def unpack_halos_kernel(
    field,
    recv_up,
    recv_down,
    recv_left,
    recv_right,
    has_up,
    has_down,
    has_left,
    has_right,
):
    idx = cuda.grid(1)
    if idx < LOCAL_NX:
        if has_up:
            field[0, idx + 1] = recv_up[idx]
        if has_down:
            field[LOCAL_NY + 1, idx + 1] = recv_down[idx]
        if has_left:
            field[idx + 1, 0] = recv_left[idx]
        if has_right:
            field[idx + 1, LOCAL_NX + 1] = recv_right[idx]


def exchange_halos(context, field, buffers):
    cart_comm = context["cart_comm"]
    neighbors = context["neighbors"]

    pack_halos_kernel[(4,), (64,)](
        field,
        buffers["d_send_up"],
        buffers["d_send_down"],
        buffers["d_send_left"],
        buffers["d_send_right"],
    )
    cuda.synchronize()

    buffers["d_send_up"].copy_to_host(buffers["h_send_up"])
    buffers["d_send_down"].copy_to_host(buffers["h_send_down"])
    buffers["d_send_left"].copy_to_host(buffers["h_send_left"])
    buffers["d_send_right"].copy_to_host(buffers["h_send_right"])

    buffers["h_recv_up"].fill(0.0)
    buffers["h_recv_down"].fill(0.0)
    buffers["h_recv_left"].fill(0.0)
    buffers["h_recv_right"].fill(0.0)

    reqs = []
    if neighbors["up"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_up"], source=neighbors["up"], tag=101))
        reqs.append(cart_comm.Isend(buffers["h_send_up"], dest=neighbors["up"], tag=102))
    if neighbors["down"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_down"], source=neighbors["down"], tag=102))
        reqs.append(cart_comm.Isend(buffers["h_send_down"], dest=neighbors["down"], tag=101))
    if neighbors["left"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_left"], source=neighbors["left"], tag=103))
        reqs.append(cart_comm.Isend(buffers["h_send_left"], dest=neighbors["left"], tag=104))
    if neighbors["right"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_right"], source=neighbors["right"], tag=104))
        reqs.append(cart_comm.Isend(buffers["h_send_right"], dest=neighbors["right"], tag=103))

    if reqs:
        MPI.Request.Waitall(reqs)

    buffers["d_recv_up"].copy_to_device(buffers["h_recv_up"])
    buffers["d_recv_down"].copy_to_device(buffers["h_recv_down"])
    buffers["d_recv_left"].copy_to_device(buffers["h_recv_left"])
    buffers["d_recv_right"].copy_to_device(buffers["h_recv_right"])

    unpack_halos_kernel[(4,), (64,)](
        field,
        buffers["d_recv_up"],
        buffers["d_recv_down"],
        buffers["d_recv_left"],
        buffers["d_recv_right"],
        neighbors["up"] != MPI.PROC_NULL,
        neighbors["down"] != MPI.PROC_NULL,
        neighbors["left"] != MPI.PROC_NULL,
        neighbors["right"] != MPI.PROC_NULL,
    )
    cuda.synchronize()


def validate_halos(context, field):
    host_field = field.copy_to_host()
    coords = context["coords"]
    neighbors = context["neighbors"]
    gy0 = coords[0] * LOCAL_NY
    gx0 = coords[1] * LOCAL_NX

    if neighbors["up"] == MPI.PROC_NULL:
        expected_up = np.zeros(LOCAL_NX, dtype=np.float64)
    else:
        expected_up = (gy0 - 1) * 512.0 + (gx0 + np.arange(LOCAL_NX, dtype=np.float64))
    if not np.allclose(host_field[0, 1 : LOCAL_NX + 1], expected_up):
        return False

    if neighbors["down"] == MPI.PROC_NULL:
        expected_down = np.zeros(LOCAL_NX, dtype=np.float64)
    else:
        expected_down = (gy0 + LOCAL_NY) * 512.0 + (gx0 + np.arange(LOCAL_NX, dtype=np.float64))
    if not np.allclose(host_field[LOCAL_NY + 1, 1 : LOCAL_NX + 1], expected_down):
        return False

    if neighbors["left"] == MPI.PROC_NULL:
        expected_left = np.zeros(LOCAL_NY, dtype=np.float64)
    else:
        expected_left = (gy0 + np.arange(LOCAL_NY, dtype=np.float64)) * 512.0 + (gx0 - 1)
    if not np.allclose(host_field[1 : LOCAL_NY + 1, 0], expected_left):
        return False

    if neighbors["right"] == MPI.PROC_NULL:
        expected_right = np.zeros(LOCAL_NY, dtype=np.float64)
    else:
        expected_right = (gy0 + np.arange(LOCAL_NY, dtype=np.float64)) * 512.0 + (gx0 + LOCAL_NX)
    if not np.allclose(host_field[1 : LOCAL_NY + 1, LOCAL_NX + 1], expected_right):
        return False

    return True


def run_task2(context):
    field = initialize_fields(context)

    buffers = {
        "d_send_up": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_down": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_left": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_right": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_up": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_down": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_left": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_right": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "h_send_up": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_down": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_left": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_right": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_up": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_down": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_left": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_right": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
    }

    exchange_halos(context, field, buffers)
    local_ok = validate_halos(context, field)
    all_ok = context["cart_comm"].allreduce(local_ok, op=MPI.LAND)

    if context["rank"] == 0 and all_ok:
        print("TASK 2: PASSED", flush=True)

    return field, bool(all_ok)


# ==============================================================================
# TASK 3: 2D Shared-Memory Tiled Stencil & Dual-Stream Pipeline
# ==============================================================================

def build_physics_fields(context):
    """
    Populates rank-local (258, 258) float64 arrays u, kappa, and S using global
    coordinates without allocating a global 512x512 array.
    """
    row_tile, col_tile = context["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    rng = np.random.default_rng(int(context["instance_seed"]) + context["rank"])
    phase_shift = float(rng.uniform(0.0, 0.25 * math.pi))

    gy0 = row_tile * LOCAL_NY
    gx0 = col_tile * LOCAL_NX

    ly = np.arange(PADDED_NY, dtype=np.float64)[:, None]
    lx = np.arange(PADDED_NX, dtype=np.float64)[None, :]
    x = (gx0 + lx) * dx
    y = (gy0 + ly) * dy

    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    u_host = (
        np.exp(-45.0 * r2)
        + 0.25 * np.sin(2.0 * math.pi * x + phase_shift) * np.sin(2.0 * math.pi * y)
    )
    kappa_host = 0.70 + 0.30 * np.sin(3.0 * math.pi * x) * np.cos(3.0 * math.pi * y)
    src_host = 0.05 * np.sin(math.pi * x) * np.sin(math.pi * y)

    if row_tile == 0:
        u_host[0, :] = 0.0
    if row_tile == 1:
        u_host[PADDED_NY - 1, :] = 0.0
    if col_tile == 0:
        u_host[:, 0] = 0.0
    if col_tile == 1:
        u_host[:, PADDED_NX - 1] = 0.0

    local_kmax = float(np.max(kappa_host))
    global_kmax = context["cart_comm"].allreduce(local_kmax, op=MPI.MAX)
    dt_max = (dx * dx) / (4.0 * global_kmax)
    dt = 0.5 * dt_max

    return u_host, kappa_host, src_host, dx, dt, global_kmax


@cuda.jit
def tiled_stencil_kernel(u_in, u_out, kappa, src, dx_inv2, dt, lam, y_start, y_end, x_start, x_end):
    """
    16x16 threadblock stencil using an 18x18 shared-memory tile with cuda.syncthreads().
    Updates cells in [y_start..y_end, x_start..x_end] (inclusive).
    """
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    gx = cuda.blockIdx.x * TILE_DIM + tx + 1
    gy = cuda.blockIdx.y * TILE_DIM + ty + 1

    sx = tx + 1
    sy = ty + 1

    if gy < PADDED_NY and gx < PADDED_NX:
        tile[sy, sx] = u_in[gy, gx]

    if ty == 0 and gy - 1 >= 0 and gx < PADDED_NX:
        tile[0, sx] = u_in[gy - 1, gx]
    if ty == TILE_DIM - 1 and gy + 1 < PADDED_NY and gx < PADDED_NX:
        tile[SHARED_DIM - 1, sx] = u_in[gy + 1, gx]
    if tx == 0 and gx - 1 >= 0 and gy < PADDED_NY:
        tile[sy, 0] = u_in[gy, gx - 1]
    if tx == TILE_DIM - 1 and gx + 1 < PADDED_NX and gy < PADDED_NY:
        tile[sy, SHARED_DIM - 1] = u_in[gy, gx + 1]

    cuda.syncthreads()

    if y_start <= gy <= y_end and x_start <= gx <= x_end:
        c = tile[sy, sx]
        lap = (
            tile[sy - 1, sx]
            + tile[sy + 1, sx]
            + tile[sy, sx - 1]
            + tile[sy, sx + 1]
            - 4.0 * c
        ) * dx_inv2
        rhs = kappa[gy, gx] * lap - lam * c + src[gy, gx]
        u_out[gy, gx] = c + dt * rhs


def _exchange_mpi_nonblocking(cart_comm, neighbors, buffers, tag_base):
    buffers["h_recv_up"].fill(0.0)
    buffers["h_recv_down"].fill(0.0)
    buffers["h_recv_left"].fill(0.0)
    buffers["h_recv_right"].fill(0.0)

    reqs = []
    if neighbors["up"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_up"], source=neighbors["up"], tag=tag_base + 1))
        reqs.append(cart_comm.Isend(buffers["h_send_up"], dest=neighbors["up"], tag=tag_base + 2))
    if neighbors["down"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_down"], source=neighbors["down"], tag=tag_base + 2))
        reqs.append(cart_comm.Isend(buffers["h_send_down"], dest=neighbors["down"], tag=tag_base + 1))
    if neighbors["left"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_left"], source=neighbors["left"], tag=tag_base + 3))
        reqs.append(cart_comm.Isend(buffers["h_send_left"], dest=neighbors["left"], tag=tag_base + 4))
    if neighbors["right"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(buffers["h_recv_right"], source=neighbors["right"], tag=tag_base + 4))
        reqs.append(cart_comm.Isend(buffers["h_send_right"], dest=neighbors["right"], tag=tag_base + 3))
    if reqs:
        MPI.Request.Waitall(reqs)


def step_overlapped(
    context, d_u0, d_u1, d_kappa, d_src, buffers,
    stream_interior, stream_boundary, dx_inv2, dt, tag_base
):
    cart_comm = context["cart_comm"]
    neighbors = context["neighbors"]
    grid_2d = (LOCAL_NX // TILE_DIM, LOCAL_NY // TILE_DIM)
    block_2d = (TILE_DIM, TILE_DIM)

    # 1. Compute deep interior [2..255, 2..255] asynchronously on stream_interior
    tiled_stencil_kernel[grid_2d, block_2d, stream_interior](
        d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP,
        2, LOCAL_NY - 1, 2, LOCAL_NX - 1
    )

    # 2. Pack halos and transfer D2H asynchronously on stream_boundary
    pack_halos_kernel[(4,), (64,), stream_boundary](
        d_u0,
        buffers["d_send_up"],
        buffers["d_send_down"],
        buffers["d_send_left"],
        buffers["d_send_right"],
    )
    buffers["d_send_up"].copy_to_host(buffers["h_send_up"], stream=stream_boundary)
    buffers["d_send_down"].copy_to_host(buffers["h_send_down"], stream=stream_boundary)
    buffers["d_send_left"].copy_to_host(buffers["h_send_left"], stream=stream_boundary)
    buffers["d_send_right"].copy_to_host(buffers["h_send_right"], stream=stream_boundary)
    stream_boundary.synchronize()

    # 3. Non-blocking MPI halo exchange while stream_interior executes on GPU
    _exchange_mpi_nonblocking(cart_comm, neighbors, buffers, tag_base)

    # 4. Async H2D + unpack + 4 outer border strips on stream_boundary
    buffers["d_recv_up"].copy_to_device(buffers["h_recv_up"], stream=stream_boundary)
    buffers["d_recv_down"].copy_to_device(buffers["h_recv_down"], stream=stream_boundary)
    buffers["d_recv_left"].copy_to_device(buffers["h_recv_left"], stream=stream_boundary)
    buffers["d_recv_right"].copy_to_device(buffers["h_recv_right"], stream=stream_boundary)

    unpack_halos_kernel[(4,), (64,), stream_boundary](
        d_u0,
        buffers["d_recv_up"],
        buffers["d_recv_down"],
        buffers["d_recv_left"],
        buffers["d_recv_right"],
        neighbors["up"] != MPI.PROC_NULL,
        neighbors["down"] != MPI.PROC_NULL,
        neighbors["left"] != MPI.PROC_NULL,
        neighbors["right"] != MPI.PROC_NULL,
    )

    # Top strip (y = 1, x = 1..256)
    tiled_stencil_kernel[grid_2d, block_2d, stream_boundary](
        d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP,
        1, 1, 1, LOCAL_NX
    )
    # Bottom strip (y = 256, x = 1..256)
    tiled_stencil_kernel[grid_2d, block_2d, stream_boundary](
        d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP,
        LOCAL_NY, LOCAL_NY, 1, LOCAL_NX
    )
    # Left strip (y = 2..255, x = 1)
    tiled_stencil_kernel[grid_2d, block_2d, stream_boundary](
        d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP,
        2, LOCAL_NY - 1, 1, 1
    )
    # Right strip (y = 2..255, x = 256)
    tiled_stencil_kernel[grid_2d, block_2d, stream_boundary](
        d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP,
        2, LOCAL_NY - 1, LOCAL_NX, LOCAL_NX
    )
    cuda.synchronize()


def run_stencil_solver(context, u_init, kappa_h, src_h, dx, dt, overlapped: bool):
    cart_comm = context["cart_comm"]

    d_u0 = cuda.to_device(u_init)
    d_u1 = cuda.to_device(u_init)
    d_kappa = cuda.to_device(kappa_h)
    d_src = cuda.to_device(src_h)

    buffers = {
        "d_send_up": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_down": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_left": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_right": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_up": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_down": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_left": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_right": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "h_send_up": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_down": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_left": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_right": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_up": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_down": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_left": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_right": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
    }

    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()
    grid_2d = (LOCAL_NX // TILE_DIM, LOCAL_NY // TILE_DIM)
    block_2d = (TILE_DIM, TILE_DIM)
    dx_inv2 = 1.0 / (dx * dx)
    energy_history = {}

    def record_energy(step_idx, d_field):
        host_u = d_field.copy_to_host()
        interior = host_u[1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]
        local_e = float(np.sum(interior * interior) * (dx * dx))
        global_e = cart_comm.allreduce(local_e, op=MPI.SUM)
        energy_history[str(step_idx)] = global_e

    # Warmup JIT compilation
    tiled_stencil_kernel[grid_2d, block_2d](
        d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP, 1, LOCAL_NY, 1, LOCAL_NX
    )
    cuda.synchronize()
    cart_comm.Barrier()

    record_energy(0, d_u0)
    t0 = time.perf_counter()

    for step in range(1, STENCIL_STEPS + 1):
        tag_base = (step % 1000) * 10
        if not overlapped:
            exchange_halos(context, d_u0, buffers)
            tiled_stencil_kernel[grid_2d, block_2d](
                d_u0, d_u1, d_kappa, d_src, dx_inv2, dt, LAMBDA_DAMP,
                1, LOCAL_NY, 1, LOCAL_NX
            )
            cuda.synchronize()
        else:
            step_overlapped(
                context, d_u0, d_u1, d_kappa, d_src, buffers,
                stream_interior, stream_boundary, dx_inv2, dt, tag_base
            )

        d_u0, d_u1 = d_u1, d_u0
        if step in (100, 200, 300):
            record_energy(step, d_u0)

    cart_comm.Barrier()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    final_u = d_u0.copy_to_host()[1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]
    return final_u, elapsed_ms, energy_history


# ==============================================================================
# TASK 4: Multi-Rank Host Cache-Line False Sharing Sweep
# ==============================================================================

@njit(nogil=True)
def host_shm_accumulate(shm_arr, slot_idx, iters):
    """
    Performs 40,000,000 in-place read-modify-write updates directly on
    shm_arr[slot_idx] without register hoisting across iterations.
    """
    for _ in range(iters):
        shm_arr[slot_idx] = shm_arr[slot_idx] + uint64(1)


def benchmark_shm_strides(context: dict) -> dict:
    node_comm = context["node_comm"]
    rank = context["rank"]
    shm_view = context["shm_view"]

    # JIT warmup
    shm_view[rank * 16] = 0
    host_shm_accumulate(shm_view, rank * 16, 1000)
    node_comm.Barrier()

    results = {}
    for stride in (1, 8, 16):
        slot_idx = rank * stride
        shm_view[slot_idx] = 0
        node_comm.Barrier()

        t0 = time.perf_counter()
        host_shm_accumulate(shm_view, slot_idx, SHM_ITERS)
        node_comm.Barrier()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert int(shm_view[slot_idx]) == SHM_ITERS, (
            f"Counter mismatch on rank {rank}, stride {stride}: {shm_view[slot_idx]}"
        )
        max_ms = node_comm.allreduce(elapsed_ms, op=MPI.MAX)
        results[f"stride_{stride}"] = {
            "stride_words": stride,
            "byte_spacing": stride * 8,
            "wall_ms": round(max_ms, 3),
            "ns_per_rmw": round((max_ms * 1e6) / SHM_ITERS, 3),
        }

    s1 = results["stride_1"]["wall_ms"]
    s8 = results["stride_8"]["wall_ms"]
    s16 = results["stride_16"]["wall_ms"]
    results["false_sharing_slowdown_vs_64B"] = round(s1 / s8, 3)
    results["false_sharing_slowdown_vs_128B"] = round(s1 / s16, 3)
    results["prefetcher_pairing_ratio_64B_vs_128B"] = round(s8 / s16, 3)
    return results


# ==============================================================================
# TASK 5: Cross-Tier Verification, Assembly Audits & Manifest Export
# ==============================================================================

def run_assembly_audits() -> dict:
    ptx_dict = tiled_stencil_kernel.inspect_asm()
    ptx_code = next(iter(ptx_dict.values()))
    st_shared_count = ptx_code.count("st.shared.f64")
    ld_shared_count = ptx_code.count("ld.shared.f64")
    bar_sync_count = ptx_code.count("bar.sync")

    assert st_shared_count > 0 and ld_shared_count > 0 and bar_sync_count > 0, (
        "PTX audit failed: expected st.shared.f64, ld.shared.f64, and bar.sync"
    )

    llvm_dict = host_shm_accumulate.inspect_llvm()
    llvm_ir = next(iter(llvm_dict.values()))
    load_count = llvm_ir.count("load i64")
    store_count = llvm_ir.count("store i64")
    add_count = llvm_ir.count("add i64")

    return {
        "ptx_audit": {
            "st.shared.f64": st_shared_count,
            "ld.shared.f64": ld_shared_count,
            "bar.sync": bar_sync_count,
        },
        "llvm_ir_audit": {
            "load_i64_instructions": load_count,
            "store_i64_instructions": store_count,
            "add_i64_instructions": add_count,
        },
    }


def main():
    context = init_hybrid_context(GITHUB_USERNAME)
    rank = context["rank"]
    cart_comm = context["cart_comm"]
    node_comm = context["node_comm"]
    node_rank = context["node_rank"]

    # Task 1 Verification
    context["shm_view"][node_rank] = rank
    node_comm.Barrier()
    shared_values = [int(context["shm_view"][i]) for i in range(4)]
    if shared_values != [0, 1, 2, 3]:
        raise RuntimeError(f"TASK 1 FAILED: {shared_values}")
    if rank == 0:
        print("TASK 1: PASSED", flush=True)

    # Task 2 Verification
    context["comm"].Barrier()
    _, task2_passed = run_task2(context)
    context["comm"].Barrier()
    if not task2_passed:
        raise RuntimeError("TASK 2 FAILED")

    # Task 3: Dual-Stream vs Serialized 2D Shared-Memory Stencil Solver
    u_init, kappa_h, src_h, dx, dt, global_kmax = build_physics_fields(context)
    u_ser, ser_ms, energy_ser = run_stencil_solver(
        context, u_init, kappa_h, src_h, dx, dt, overlapped=False
    )
    u_ovl, ovl_ms, energy_ovl = run_stencil_solver(
        context, u_init, kappa_h, src_h, dx, dt, overlapped=True
    )

    local_max_diff = float(np.max(np.abs(u_ser - u_ovl)))
    global_max_diff = cart_comm.allreduce(local_max_diff, op=MPI.MAX)
    assert global_max_diff < 1e-12, f"TASK 3 FAILED: max_diff={global_max_diff}"
    if rank == 0:
        print(f"TASK 3: PASSED (max_diff={global_max_diff:.3e})", flush=True)

    # Task 4: Multi-Rank Host Cache-Line False Sharing Sweep
    shm_bench = benchmark_shm_strides(context)
    if rank == 0:
        print("TASK 4: PASSED", flush=True)

    # Task 5: Assembly Audits & Manifest Export
    if rank == 0:
        audits = run_assembly_audits()
        dt_max = (dx * dx) / (4.0 * global_kmax)
        manifest = {
            "student_id": STUDENT_ID,
            "github_username": GITHUB_USERNAME,
            "gpu_uuid": context["gpu_uuid"],
            "cpu_signature": context["cpu_sig"],
            "cpu_cache_line_bytes": int(context["line_bytes"]),
            "instance_seed": int(context["instance_seed"]),
            "shm_aligned_base_addr": hex(context["shm_view"].ctypes.data),
            "grid_config": {
                "global_interior": [GLOBAL_NX, GLOBAL_NY],
                "rank_interior": [LOCAL_NX, LOCAL_NY],
                "rank_padded": [PADDED_NX, PADDED_NY],
                "tile_dim": TILE_DIM,
                "shared_dim": SHARED_DIM,
                "steps": STENCIL_STEPS,
            },
            "cfl_stability": {
                "dx": dx,
                "kappa_max": global_kmax,
                "dt_max": dt_max,
                "dt_used": dt,
                "cfl_number": (global_kmax * dt) / (dx * dx),
            },
            "stencil_telemetry": {
                "gpu_serialized_ms": round(ser_ms, 3),
                "gpu_overlapped_ms": round(ovl_ms, 3),
                "overlap_speedup": round(ser_ms / ovl_ms, 3),
                "cross_mode_max_diff": global_max_diff,
                "global_energy_overlapped": energy_ovl,
            },
            "shm_coherency_telemetry": shm_bench,
            "assembly_audits": audits,
        }

        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("TASK 5: PASSED", flush=True)
        print("=" * 72)
        print("HYBRID LAB MANIFEST (hybrid_lab_manifest.json)")
        print("=" * 72)
        print(json.dumps(manifest, indent=2), flush=True)

    context["comm"].Barrier()
    context["win"].Free()
    context["cart_comm"].Free()
    context["node_comm"].Free()


if __name__ == "__main__":
    main()
