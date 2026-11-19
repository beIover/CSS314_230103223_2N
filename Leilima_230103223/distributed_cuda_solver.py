import os
import sys
import json
import time
import math
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15

def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8"
    ).strip().splitlines()[0].strip()
    return out

def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")
    
    cuda.select_device(0)
    
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)
    
    dims = MPI.Compute_dims(size, [2, 2])
    cart_comm = comm.Create_cart(dims, periods=[False, False], reorder=False)
    
    up_nbr, down_nbr = cart_comm.Shift(0, 1)
    left_nbr, right_nbr = cart_comm.Shift(1, 1)
    rank_row, rank_col = cart_comm.Get_coords(rank)
    
    return cart_comm, rank_row, rank_col, up_nbr, down_nbr, left_nbr, right_nbr, instance_seed, gpu_uuid

# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ==============================================================================
def initialize_subdomains(rank_row, rank_col, seed):
    nx, ny = GLOBAL_NX // 2, GLOBAL_NY // 2
    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    kappa = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    S = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    
    # Детерминированная генерация параметров из seed
    rng = np.random.default_rng(seed)
    c1_x, c1_y = rng.uniform(0.25, 0.75, 2)
    c2_x, c2_y = rng.uniform(0.25, 0.75, 2)
    amp1, amp2 = rng.uniform(0.5, 1.5, 2)
    
    for r in range(ny + 2):
        gy = rank_row * ny + r
        y = gy * dy
        for c in range(nx + 2):
            gx = rank_col * nx + c
            x = gx * dx
            
            # Пространственно неоднородная теплопроводность kappa in [0.08, 0.24]
            val_k = 0.16 + 0.08 * math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y)
            kappa[r, c] = min(max(val_k, 0.08), 0.24)
            
            # Начальные гауссовы импульсы и источник S(x,y)
            d1 = (x - c1_x)**2 + (y - c1_y)**2
            d2 = (x - c2_x)**2 + (y - c2_y)**2
            u[r, c] = amp1 * math.exp(-d1 / 0.01) + amp2 * math.exp(-d2 / 0.015)
            S[r, c] = 0.5 * amp1 * math.exp(-d1 / 0.005)
            
    # Граничные условия Дирихле (u = 0 на внешних границах)
    if rank_row == 0:
        u[0, :] = 0.0
    if rank_row == 1:
        u[ny + 1, :] = 0.0
    if rank_col == 0:
        u[:, 0] = 0.0
    if rank_col == 1:
        u[:, nx + 1] = 0.0

    return (
        cuda.to_device(u),
        cuda.to_device(u.copy()),
        cuda.to_device(kappa),
        cuda.to_device(S),
        float(np.max(kappa))
    )

# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels
# ==============================================================================
@cuda.jit
def pack_halo_kernel(u, top, bottom, left, right, nx, ny):
    idx = cuda.grid(1)
    if idx < nx:
        top[idx] = u[1, idx + 1]
        bottom[idx] = u[ny, idx + 1]
    if idx < ny:
        left[idx] = u[idx + 1, 1]
        right[idx] = u[idx + 1, nx]

@cuda.jit
def unpack_halo_kernel(u, top, bottom, left, right, nx, ny):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = top[idx]
        u[ny + 1, idx + 1] = bottom[idx]
    if idx < ny:
        u[idx + 1, 0] = left[idx]
        u[idx + 1, nx + 1] = right[idx]

# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil Kernel
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_next, kappa, S, r_start, r_end, c_start, c_end, dx, dy, dt):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    i = by * TILE_DIM + ty + 1
    j = bx * TILE_DIM + tx + 1

    shmem = cuda.shared.array((18, 18), dtype=float64)
    
    # Загрузка центральных ячеек и 1-ячеечного гало
    if i <= 256 and j <= 256:
        shmem[ty + 1, tx + 1] = u[i, j]
        if tx == 0:
            shmem[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1:
            shmem[ty + 1, TILE_DIM + 1] = u[i, j + 1]
        if ty == 0:
            shmem[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1:
            shmem[TILE_DIM + 1, tx + 1] = u[i + 1, j]
            
    cuda.syncthreads()

    # Вычисление только внутри заданного bounding-box
    if r_start <= i <= r_end and c_start <= j <= c_end:
        u_c = shmem[ty + 1, tx + 1]
        u_l = shmem[ty + 1, tx]
        u_r = shmem[ty + 1, tx + 2]
        u_u = shmem[ty, tx + 1]
        u_d = shmem[ty + 2, tx + 1]
        
        k_c = kappa[i, j]
        k_iph = 0.5 * (k_c + kappa[i, j + 1])
        k_imh = 0.5 * (k_c + kappa[i, j - 1])
        k_jph = 0.5 * (k_c + kappa[i + 1, j])
        k_jmh = 0.5 * (k_c + kappa[i - 1, j])
        
        flux_x = (k_iph * (u_r - u_c) - k_imh * (u_c - u_l)) / (dx * dx)
        flux_y = (k_jph * (u_d - u_c) - k_jmh * (u_c - u_u)) / (dy * dy)
        
        u_next[i, j] = u_c + dt * (flux_x + flux_y - LAMBDA_DAMP * (u_c**3) + S[i, j])

# ==============================================================================
# TASK 5: Dual-Stream Orchestrator & Comparative Benchmark
# ==============================================================================
def run_solver(cart_comm, u_init, kappa_d, S_d, nbrs, dx, dy, dt, overlapped=True):
    up, down, left, right = nbrs
    nx, ny = 256, 256
    u_d = cuda.to_device(u_init.copy_to_host())
    u_next_d = cuda.to_device(u_init.copy_to_host())
    
    # Буферы на GPU и pinned-буферы на CPU
    d_send_top, d_send_bot = cuda.device_array(nx, dtype=np.float64), cuda.device_array(nx, dtype=np.float64)
    d_send_left, d_send_right = cuda.device_array(ny, dtype=np.float64), cuda.device_array(ny, dtype=np.float64)
    d_recv_top, d_recv_bot = cuda.device_array(nx, dtype=np.float64), cuda.device_array(nx, dtype=np.float64)
    d_recv_left, d_recv_right = cuda.device_array(ny, dtype=np.float64), cuda.device_array(ny, dtype=np.float64)
    
    h_send_top, h_send_bot = cuda.pinned_array(nx, dtype=np.float64), cuda.pinned_array(nx, dtype=np.float64)
    h_send_left, h_send_right = cuda.pinned_array(ny, dtype=np.float64), cuda.pinned_array(ny, dtype=np.float64)
    h_recv_top, h_recv_bot = cuda.pinned_array(nx, dtype=np.float64), cuda.pinned_array(nx, dtype=np.float64)
    h_recv_left, h_recv_right = cuda.pinned_array(ny, dtype=np.float64), cuda.pinned_array(ny, dtype=np.float64)
    
    stream_int = cuda.stream()
    stream_bnd = cuda.stream() if overlapped else stream_int
    
    grid_2d = (16, 16)
    block_2d = (16, 16)
    
    cart_comm.Barrier()
    t0 = time.perf_counter()
    
    for step in range(STEPS):
        # Stage A: Pack & Async D2H
        pack_halo_kernel[4, 64, stream_bnd](u_d, d_send_top, d_send_bot, d_send_left, d_send_right, nx, ny)
        d_send_top.copy_to_host(h_send_top, stream=stream_bnd)
        d_send_bot.copy_to_host(h_send_bot, stream=stream_bnd)
        d_send_left.copy_to_host(h_send_left, stream=stream_bnd)
        d_send_right.copy_to_host(h_send_right, stream=stream_bnd)
        
        if overlapped:
            # Stage B: Non-blocking MPI + Concurrent Interior Compute
            stream_bnd.synchronize()
            reqs = [
                cart_comm.Irecv(h_recv_top, source=up, tag=1),
                cart_comm.Irecv(h_recv_bot, source=down, tag=0),
                cart_comm.Irecv(h_recv_left, source=left, tag=3),
                cart_comm.Irecv(h_recv_right, source=right, tag=2),
                cart_comm.Isend(h_send_top, dest=up, tag=0),
                cart_comm.Isend(h_send_bot, dest=down, tag=1),
                cart_comm.Isend(h_send_left, dest=left, tag=2),
                cart_comm.Isend(h_send_right, dest=right, tag=3),
            ]
            # Считаем глубокую внутреннюю область [2..255, 2..255] пока идет обмен по сети
            tiled_stencil_kernel[grid_2d, block_2d, stream_int](u_d, u_next_d, kappa_d, S_d, 2, 255, 2, 255, dx, dy, dt)
            
            # Stage C: Wait MPI, Async H2D & Boundary Ring Stencil
            MPI.Request.Waitall(reqs)
            if up != MPI.PROC_NULL: d_recv_top.copy_to_device(h_recv_top, stream=stream_bnd)
            if down != MPI.PROC_NULL: d_recv_bot.copy_to_device(h_recv_bot, stream=stream_bnd)
            if left != MPI.PROC_NULL: d_recv_left.copy_to_device(h_recv_left, stream=stream_bnd)
            if right != MPI.PROC_NULL: d_recv_right.copy_to_device(h_recv_right, stream=stream_bnd)
            
            unpack_halo_kernel[4, 64, stream_bnd](u_d, d_recv_top, d_recv_bot, d_recv_left, d_recv_right, nx, ny)
            
            # Восстановление внешних границ Дирихле (u=0)
            if up == MPI.PROC_NULL: u_d[0, :] = 0.0
            if down == MPI.PROC_NULL: u_d[ny+1, :] = 0.0
            if left == MPI.PROC_NULL: u_d[:, 0] = 0.0
            if right == MPI.PROC_NULL: u_d[:, nx+1] = 0.0
            
            # Обновляем 4 граничных кольца
            tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](u_d, u_next_d, kappa_d, S_d, 1, 1, 1, 256, dx, dy, dt)
            tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](u_d, u_next_d, kappa_d, S_d, 256, 256, 1, 256, dx, dy, dt)
            tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](u_d, u_next_d, kappa_d, S_d, 2, 255, 1, 1, dx, dy, dt)
            tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](u_d, u_next_d, kappa_d, S_d, 2, 255, 256, 256, dx, dy, dt)
            
            stream_int.synchronize()
            stream_bnd.synchronize()
        else:
            # Serialized режим (без перекрытия)
            stream_bnd.synchronize()
            reqs = [
                cart_comm.Irecv(h_recv_top, source=up, tag=1),
                cart_comm.Irecv(h_recv_bot, source=down, tag=0),
                cart_comm.Irecv(h_recv_left, source=left, tag=3),
                cart_comm.Irecv(h_recv_right, source=right, tag=2),
                cart_comm.Isend(h_send_top, dest=up, tag=0),
                cart_comm.Isend(h_send_bot, dest=down, tag=1),
                cart_comm.Isend(h_send_left, dest=left, tag=2),
                cart_comm.Isend(h_send_right, dest=right, tag=3),
            ]
            MPI.Request.Waitall(reqs)
            if up != MPI.PROC_NULL: d_recv_top.copy_to_device(h_recv_top, stream=stream_bnd)
            if down != MPI.PROC_NULL: d_recv_bot.copy_to_device(h_recv_bot, stream=stream_bnd)
            if left != MPI.PROC_NULL: d_recv_left.copy_to_device(h_recv_left, stream=stream_bnd)
            if right != MPI.PROC_NULL: d_recv_right.copy_to_device(h_recv_right, stream=stream_bnd)
            unpack_halo_kernel[4, 64, stream_bnd](u_d, d_recv_top, d_recv_bot, d_recv_left, d_recv_right, nx, ny)
            if up == MPI.PROC_NULL: u_d[0, :] = 0.0
            if down == MPI.PROC_NULL: u_d[ny+1, :] = 0.0
            if left == MPI.PROC_NULL: u_d[:, 0] = 0.0
            if right == MPI.PROC_NULL: u_d[:, nx+1] = 0.0
            tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](u_d, u_next_d, kappa_d, S_d, 1, 256, 1, 256, dx, dy, dt)
            stream_bnd.synchronize()
            
        u_d, u_next_d = u_next_d, u_d

        if step % 100 == 0 or step == STEPS - 1:
            host_u = u_d.copy_to_host()[1:ny+1, 1:nx+1]
            local_energy = float(np.sum(host_u) * dx * dy)
            global_energy = cart_comm.allreduce(local_energy, op=MPI.SUM)
            if cart_comm.Get_rank() == 0 and overlapped and (step % 100 == 0):
                print(f"[Overlapped] Step {step:03d} | Global Thermal Energy E = {global_energy:.8f}")

    cart_comm.Barrier()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return u_d.copy_to_host()[1:ny+1, 1:nx+1], elapsed_ms

if __name__ == "__main__":
    github_user = sys.argv[1]
    cart_comm, r_row, r_col, up, down, left, right, seed, gpu_uuid = init_distributed_context(github_user)
    rank = cart_comm.Get_rank()
    
    u_init, _, kappa_d, S_d, local_k_max = initialize_subdomains(r_row, r_col, seed)
    global_k_max = cart_comm.allreduce(local_k_max, op=MPI.MAX)
    
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    dt_max = (dx * dx) / (4.0 * global_k_max)
    dt = 0.8 * dt_max  # Безопасный шаг с запасом по CFL
    
    nbrs = (up, down, left, right)
    
    # Прогрев JIT-компиляции Numba
    _ = run_solver(cart_comm, u_init, kappa_d, S_d, nbrs, dx, dy, dt, overlapped=True)
    
    # Основные прогоны
    state_ovl, overlapped_ms = run_solver(cart_comm, u_init, kappa_d, S_d, nbrs, dx, dy, dt, overlapped=True)
    state_ser, serialized_ms = run_solver(cart_comm, u_init, kappa_d, S_d, nbrs, dx, dy, dt, overlapped=False)
    
    local_diff = float(np.max(np.abs(state_ovl - state_ser)))
    parity_max_diff = cart_comm.allreduce(local_diff, op=MPI.MAX)
    
    # Аудит PTX инструкций ядра
    ptx_dict = tiled_stencil_kernel.inspect_asm()
    ptx_code = next(iter(ptx_dict.values()))
    st_shared_cnt = ptx_code.count("st.shared")
    ld_shared_cnt = ptx_code.count("ld.shared")
    bar_sync_cnt = ptx_code.count("bar.sync")
    ptx_passed = (st_shared_cnt > 0) and (ld_shared_cnt > 0) and (bar_sync_cnt > 0)
    
    if rank == 0:
        print("\n=== BENCHMARK & AUDIT SUMMARY ===")
        print(f"MPI Ranks: 4 (2x2 Cartesian Grid)")
        print(f"GPU UUID: {gpu_uuid} | Seed: {seed}")
        print(f"Max Diffusivity (kappa_max): {global_k_max:.6f} | dt_max: {dt_max:.4e} | dt_used: {dt:.4e}")
        print(f"Serialized Time: {serialized_ms:.2f} ms | Overlapped Time: {overlapped_ms:.2f} ms")
        print(f"Parity Max Diff: {parity_max_diff:.4e} (Valid: {parity_max_diff < 1e-12})")
        print(f"PTX Counts -> st.shared: {st_shared_cnt}, ld.shared: {ld_shared_cnt}, bar.sync: {bar_sync_cnt}")
        
        manifest = {
            "GITHUB_USERNAME": github_user,
            "gpu_uuid": gpu_uuid,
            "instance_seed": seed,
            "mpi_ranks": 4,
            "parity_max_diff": parity_max_diff,
            "ptx_audit_passed": ptx_passed,
            "ptx_counts": {
                "st.shared": st_shared_cnt,
                "ld.shared": ld_shared_cnt,
                "bar.sync": bar_sync_cnt
            },
            "metrics": {
                "serialized_ms": serialized_ms,
                "overlapped_ms": overlapped_ms,
                "kappa_max": global_k_max,
                "dt_max": dt_max,
                "dt_used": dt
            }
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("Generated submission_manifest.json successfully!")
