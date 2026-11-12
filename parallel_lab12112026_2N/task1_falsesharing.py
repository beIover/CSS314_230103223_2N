import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import platform
import subprocess
from harness import emit_verify_tag, run_benchmark

STUDENT_ID = 230103223
TOTAL_OPS = 24_000_000  # M = 2.4 x 10^7 total operations
SHM_SIZE = 4096  # 512 x 8-byte uint64 words


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
      phys_id = None
      core_id = None
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


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
  shm = shared_memory.SharedMemory(name=shm_name)
  arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
  acc = arr[slot_idx]
  mask = 0xFFFFFFFFFFFFFFFF
  mult = 6364136223846793005
  for i in range(iters):
    acc = (acc * mult + (i | 1)) & mask
    arr[slot_idx] = acc  # Direct hardware store to shared L1 cache line
  del arr  # Release buffer export before closing SharedMemory
  shm.close()


def benchmark_config(p: int, stride: int):
  max_slot = (p - 1) * stride
  alloc_size = max(SHM_SIZE, (max_slot + 1) * 8)
  shm = shared_memory.SharedMemory(create=True, size=alloc_size)

  base_iters = TOTAL_OPS // p
  rem = TOTAL_OPS % p
  work_splits = [base_iters + (1 if t < rem else 0) for t in range(p)]

  def trial() -> None:
    shm.buf[:alloc_size] = b"\x00" * alloc_size
    procs = [
        mp.Process(
            target=fs_worker,
            args=(shm.name, t * stride, work_splits[t]),
        )
        for t in range(p)
    ]
    for proc in procs:
      proc.start()
    for proc in procs:
      proc.join()

  try:
    return run_benchmark(trial, warmup_runs=2, timed_runs=7)
  finally:
    shm.close()
    shm.unlink()


def build_process_list(p_log: int) -> list[int]:
  procs = []
  p = 1
  while p <= p_log:
    procs.append(p)
    p *= 2
  if procs[-1] != p_log:
    procs.append(p_log)
  return procs


def main() -> None:
  p_log = mp.cpu_count()
  p_phys = detect_physical_cores()
  print(
      f"Detected Architecture: P_phys = {p_phys}, P_log = {p_log},"
      f" M = {TOTAL_OPS:,}\n"
  )

  # 1. Stride Sweep at Max Physical Cores (p = P_phys)
  strides = [1, 2, 4, 8, 16, 32]
  print(f"=== TABLE 1A: Stride Sensitivity at p = P_phys ({p_phys} cores) ===")
  print(
      f"{'Stride':>6} | {'Offset (B)':>10} | {'Median (ms)':>11} |"
      f" {'IQR (ms)':>9} | {'Throughput (Mops/s)':>19}"
  )
  print("-" * 67)

  med_s1_pphys = 0.0
  med_s16_pphys = 0.0
  for s in strides:
    stats = benchmark_config(p_phys, s)
    offset_bytes = s * 8
    mops_per_sec = (TOTAL_OPS / 1e6) / (stats.median_ms / 1e3)
    if s == 1:
      med_s1_pphys = stats.median_ms
    elif s == 16:
      med_s16_pphys = stats.median_ms
    print(
        f"{s:>6} | {offset_bytes:>10} | {stats.median_ms:>11.3f} |"
        f" {stats.iqr_ms:>9.3f} | {mops_per_sec:>19.2f}"
    )

  # 2. Process Scaling Sweep: stride = 1 vs stride = 16
  proc_counts = build_process_list(p_log)
  print("\n=== TABLE 1B: Karp-Flatt Diagnostic Series ===")
  for s in (1, 16):
    print(f"\n--- Configuration: stride = {s} ({s * 8} bytes separation) ---")
    print(
        f"{'p':>4} | {'Median (ms)':>11} | {'IQR (ms)':>9} | {'Speedup S_p':>11}"
        f" | {'Efficiency E_p':>14} | {'Karp-Flatt e(p)':>15}"
    )
    print("-" * 78)
    t1 = None
    for p in proc_counts:
      stats = benchmark_config(p, s)
      if p == 1:
        t1 = stats.median_ms
      sp = t1 / stats.median_ms
      ep = sp / p
      if p >= 2:
        kf = ((1.0 / sp) - (1.0 / p)) / (1.0 - (1.0 / p))
        kf_str = f"{kf:.5f}"
      else:
        kf_str = "N/A"
      print(
          f"{p:>4} | {stats.median_ms:>11.3f} | {stats.iqr_ms:>9.3f} |"
          f" {sp:>11.4f} | {ep:>14.4f} | {kf_str:>15}"
      )

  print("\n" + emit_verify_tag("PHASE1", STUDENT_ID, med_s1_pphys, med_s16_pphys))


if __name__ == "__main__":
  main()