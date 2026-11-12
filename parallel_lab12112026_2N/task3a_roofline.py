import multiprocessing as mp
from multiprocessing import shared_memory
import platform
import subprocess
import time
from harness import run_benchmark

STUDENT_ID = 230103223
BUF_BYTES = 96_000_000  # 96 MB per buffer (192 MB total DRAM footprint)
FLOPS_PER_WORKER = 80_000_000  # 8.0 x 10^7 FMA FLOPs per worker


def detect_physical_cores() -> int:
  system = platform.system()
  try:
    if system == "Darwin":
      out = subprocess.check_output(
          ["sysctl", "-n", "hw.physicalcpu"], text=True
      ).strip()
      return max(1, int(out))
    elif system == "Linux":
      pairs = set()
      phys_id, core_id = None, None
      with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
        for line in f:
          if line.startswith("physical id"):
            phys_id = line.split(":")[1].strip()
          elif line.startswith("core id"):
            core_id = line.split(":")[1].strip()
          elif line.strip() == "" and core_id is not None:
            pairs.add((phys_id or "0", core_id))
            phys_id, core_id = None, None
      if pairs:
        return len(pairs)
    elif system == "Windows":
      out = subprocess.check_output(
          ["wmic", "cpu", "get", "NumberOfCores"], text=True
      )
      vals = [int(x) for x in out.split() if x.isdigit()]
      if vals:
        return sum(vals)
  except Exception:
    pass
  return max(1, mp.cpu_count() // 2)


def bw_persistent_worker(
    shm_a_name: str,
    shm_b_name: str,
    start: int,
    end: int,
    total_trials: int,
    start_barrier,
    end_barrier,
) -> None:
  shm_a = shared_memory.SharedMemory(name=shm_a_name)
  shm_b = shared_memory.SharedMemory(name=shm_b_name)
  mv_a = shm_a.buf
  mv_b = shm_b.buf
  try:
    for _ in range(total_trials):
      start_barrier.wait()
      mv_a[start:end] = mv_b[start:end]
      end_barrier.wait()
  finally:
    mv_a.release()
    mv_b.release()
    shm_a.close()
    shm_b.close()


def measure_dram_bandwidth(
    p_phys: int, warmup_runs: int = 3, timed_runs: int = 9
) -> tuple[float, float]:
  """Returns (median_ms, bw_sustained_gbps)."""
  shm_a = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
  shm_b = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
  try:
    # Fault in physical pages before timing (sliced to exact BUF_BYTES for macOS 16KB page rounding)
    shm_a.buf[:BUF_BYTES] = b"\x00" * BUF_BYTES
    shm_b.buf[:BUF_BYTES] = b"\x5a" * BUF_BYTES

    total_trials = warmup_runs + timed_runs
    start_barrier = mp.Barrier(p_phys + 1)
    end_barrier = mp.Barrier(p_phys + 1)

    base = BUF_BYTES // p_phys
    rem = BUF_BYTES % p_phys
    procs = []
    cur = 0
    for t in range(p_phys):
      chunk = base + (1 if t < rem else 0)
      p = mp.Process(
          target=bw_persistent_worker,
          args=(
              shm_a.name,
              shm_b.name,
              cur,
              cur + chunk,
              total_trials,
              start_barrier,
              end_barrier,
          ),
      )
      p.start()
      procs.append(p)
      cur += chunk

    def trial() -> None:
      start_barrier.wait()
      end_barrier.wait()

    stats = run_benchmark(trial, warmup_runs=warmup_runs, timed_runs=timed_runs)
    for p in procs:
      p.join()

    # 1 read + 1 write per byte = 2 * BUF_BYTES bytes of DRAM traffic
    bw_gbps = (2.0 * BUF_BYTES) / ((stats.median_ms / 1e3) * 1e9)
    return stats.median_ms, bw_gbps
  finally:
    shm_a.close()
    shm_a.unlink()
    shm_b.close()
    shm_b.unlink()


def fma_unrolled_worker(iters: int) -> float:
  """16-way unrolled scalar FMA loop: 16 * 2 = 32 FLOPs per iteration."""
  a = 1.0000001
  b = 0.0000001
  x0 = 1.0
  x1 = 1.1
  x2 = 1.2
  x3 = 1.3
  x4 = 1.4
  x5 = 1.5
  x6 = 1.6
  x7 = 1.7
  x8 = 1.8
  x9 = 1.9
  x10 = 2.0
  x11 = 2.1
  x12 = 2.2
  x13 = 2.3
  x14 = 2.4
  x15 = 2.5

  for _ in range(iters):
    x0 = x0 * a + b
    x1 = x1 * a + b
    x2 = x2 * a + b
    x3 = x3 * a + b
    x4 = x4 * a + b
    x5 = x5 * a + b
    x6 = x6 * a + b
    x7 = x7 * a + b
    x8 = x8 * a + b
    x9 = x9 * a + b
    x10 = x10 * a + b
    x11 = x11 * a + b
    x12 = x12 * a + b
    x13 = x13 * a + b
    x14 = x14 * a + b
    x15 = x15 * a + b

  return (
      x0
      + x1
      + x2
      + x3
      + x4
      + x5
      + x6
      + x7
      + x8
      + x9
      + x10
      + x11
      + x12
      + x13
      + x14
      + x15
  )


def measure_compute_ceiling(
    p_phys: int, warmup_runs: int = 2, timed_runs: int = 7
) -> tuple[float, float]:
  """Returns (median_ms, p_sustained_gflops)."""
  iters_per_worker = FLOPS_PER_WORKER // 32
  total_flops = p_phys * (iters_per_worker * 32)
  args = [iters_per_worker] * p_phys

  with mp.Pool(processes=p_phys) as pool:

    def trial() -> None:
      _ = pool.map(fma_unrolled_worker, args)

    stats = run_benchmark(trial, warmup_runs=warmup_runs, timed_runs=timed_runs)

  p_gflops = total_flops / ((stats.median_ms / 1e3) * 1e9)
  return stats.median_ms, p_gflops


def run_roofline_characterization(p_phys: int) -> tuple[float, float, float]:
  bw_ms, bw_gbps = measure_dram_bandwidth(p_phys)
  comp_ms, p_gflops = measure_compute_ceiling(p_phys)
  ridge_point = p_gflops / bw_gbps

  print(f"=== PART 3A: Empirical Hardware Roofline (P_phys = {p_phys}) ===")
  print(f"DRAM Copy Median Time        : {bw_ms:.3f} ms (192 MB traffic)")
  print(f"Sustained DRAM Bandwidth     : {bw_gbps:.4f} GB/s")
  print(
      f"Unrolled FMA Median Time     : {comp_ms:.3f} ms"
      f" ({p_phys * FLOPS_PER_WORKER / 1e6:.0f} MFLOPs)"
  )
  print(f"Sustained Compute Ceiling    : {p_gflops:.4f} GFLOP/s")
  print(f"Empirical Ridge Point (I*)   : {ridge_point:.6f} FLOPs/Byte\n")
  return bw_gbps, p_gflops, ridge_point


if __name__ == "__main__":
  p_phys = detect_physical_cores()
  run_roofline_characterization(p_phys)