import math
import multiprocessing as mp
from multiprocessing.pool import Pool
import platform
import subprocess
import time
from harness import emit_verify_tag, run_benchmark

STUDENT_ID = 230103223
N_STRONG = 1_600_000  # 1.6 x 10^6 elements
N0_WEAK = 250_000  # 2.5 x 10^5 elements per worker


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


def run_stage_a(n_elements: int) -> float:
  n_serial = int(0.05 * n_elements)
  acc = 1.0
  sin = math.sin
  sqrt = math.sqrt
  for i in range(1, n_serial + 1):
    x = i * 1e-6 + 0.5
    v = sin(x)
    v = sqrt(abs(v) + 1.0)
    v = sin(v)
    v = sqrt(v + 1.0)
    v = sin(v)
    v = sqrt(v + 1.0)
    v = sin(v)
    v = sqrt(v + 1.0)
    v = sin(v)
    v = sqrt(v + 1.0)
    v = sin(v)
    v = sqrt(v + 1.0)
    acc += v
  return acc


def stage_b_worker(bounds: tuple[int, int]) -> float:
  start_idx, end_idx = bounds
  acc = 0.0
  sin = math.sin
  cos = math.cos
  for i in range(start_idx, end_idx):
    x = (i & 1023) * 0.001 + 0.1
    v = x
    for k in range(35):
      v = sin(v) * 0.999 + cos(x + k * 0.01) * 0.001
    acc += v
  return acc


def make_chunks(n_elements: int, p: int) -> list[tuple[int, int]]:
  base = n_elements // p
  rem = n_elements % p
  chunks = []
  cur = 0
  for t in range(p):
    size = base + (1 if t < rem else 0)
    chunks.append((cur, cur + size))
    cur += size
  return chunks


def benchmark_pipeline(
    pool: Pool,
    n_elements: int,
    p: int,
    warmup_runs: int = 3,
    timed_runs: int = 9,
):
  chunks = make_chunks(n_elements, p)
  stage_records: list[tuple[float, float]] = []

  def trial() -> None:
    t0 = time.perf_counter_ns()
    _ = run_stage_a(n_elements)
    t_mid = time.perf_counter_ns()
    _ = pool.map(stage_b_worker, chunks)
    t1 = time.perf_counter_ns()
    stage_records.append(((t1 - t0) / 1e6, (t_mid - t0) / 1e6))

  stats = run_benchmark(trial, warmup_runs=warmup_runs, timed_runs=timed_runs)
  timed_only = sorted(stage_records[warmup_runs:], key=lambda r: r[0])
  median_stage_a_ms = timed_only[len(timed_only) // 2][1]
  return stats, median_stage_a_ms


def main() -> None:
  p_log = mp.cpu_count()
  p_phys = detect_physical_cores()
  print(f"Detected Architecture: P_phys = {p_phys}, P_log = {p_log}\n")

  # Experiment 2A: Strong Scaling Sweep (Fixed N = 1.6 x 10^6)
  print(f"=== TABLE 2A: Strong Scaling (Fixed N = {N_STRONG:,}) ===")
  print(
      f"{'p':>3} | {'T_p (ms)':>10} | {'IQR (ms)':>9} | {'T_A (ms)':>9} |"
      f" {'S_p^strong':>10} | {'E_p^strong':>10} | {'Marginal dS':>11}"
  )
  print("-" * 79)

  strong_results = {}
  t1_strong = 0.0
  t_stage_a_1 = 0.0
  prev_sp = 0.0

  for p in range(1, p_log + 1):
    with mp.Pool(processes=p) as pool:
      stats, t_a_ms = benchmark_pipeline(pool, N_STRONG, p)
    if p == 1:
      t1_strong = stats.median_ms
      t_stage_a_1 = t_a_ms
    sp = t1_strong / stats.median_ms
    ep = sp / p
    marginal = sp - prev_sp if p > 1 else 1.0
    prev_sp = sp
    strong_results[p] = (stats.median_ms, sp, ep, marginal)
    print(
        f"{p:>3} | {stats.median_ms:>10.2f} | {stats.iqr_ms:>9.2f} |"
        f" {t_a_ms:>9.2f} | {sp:>10.4f} | {ep:>10.4f} | {marginal:>11.4f}"
    )

  # Experiment 2B: Weak Scaling Sweep (N(p) = p * 2.5 x 10^5)
  print(f"\n=== TABLE 2B: Weak Scaling (N_0 = {N0_WEAK:,} per worker) ===")
  print(
      f"{'p':>3} | {'N(p)':>10} | {'T_p (ms)':>10} | {'IQR (ms)':>9} |"
      f" {'alpha':>8} | {'E_p^weak':>10} | {'S_p^Gustafson':>13}"
  )
  print("-" * 80)

  weak_results = {}
  t1_weak = 0.0

  for p in range(1, p_log + 1):
    n_p = p * N0_WEAK
    with mp.Pool(processes=p) as pool:
      stats, t_a_ms = benchmark_pipeline(pool, n_p, p)
    if p == 1:
      t1_weak = stats.median_ms
    alpha = t_a_ms / stats.median_ms
    ep_weak = t1_weak / stats.median_ms
    s_gustafson = p - alpha * (p - 1)
    weak_results[p] = (stats.median_ms, alpha, ep_weak, s_gustafson)
    print(
        f"{p:>3} | {n_p:>10,} | {stats.median_ms:>10.2f} | {stats.iqr_ms:>9.2f}"
        f" | {alpha:>8.4f} | {ep_weak:>10.4f} | {s_gustafson:>13.4f}"
    )

  # Analytical Summary & Verification Hash
  s_serial = t_stage_a_1 / t1_strong
  s_inf = 1.0 / s_serial
  s_amdahl_plog = 1.0 / (s_serial + (1.0 - s_serial) / p_log)
  s_strong_plog = strong_results[p_log][1]
  s_gustafson_plog = weak_results[p_log][3]

  print("\n=== SECTION 3.3: Analytical Summary ===")
  print(
      f"Measured serial fraction s = T_A(1)/T_1^strong : {s_serial:.5f}"
      f" ({s_serial * 100:.2f}%)"
  )
  print(f"Amdahl theoretical max speedup S_inf = 1/s     : {s_inf:.4f}x")
  print(f"Amdahl theoretical S(P_log={p_log})                : {s_amdahl_plog:.4f}x")
  print(f"Observed Strong Speedup S(P_log={p_log})           : {s_strong_plog:.4f}x")
  print(
      f"Observed Gustafson Speedup S_G(P_log={p_log})      :"
      f" {s_gustafson_plog:.4f}x"
  )

  if p_log > p_phys and p_phys >= 2:
    ds_before = strong_results[p_phys][3]
    ds_after = strong_results[p_phys + 1][3]
    pct_drop = ((ds_before - ds_after) / ds_before) * 100.0
    print(
        f"Marginal speedup at p={p_phys} -> p={p_phys + 1}         :"
        f" {ds_before:.4f} -> {ds_after:.4f} ({pct_drop:.2f}% drop)"
    )

  print(
      "\n" + emit_verify_tag("PHASE2", STUDENT_ID, s_strong_plog, s_gustafson_plog)
  )


if __name__ == "__main__":
  main()