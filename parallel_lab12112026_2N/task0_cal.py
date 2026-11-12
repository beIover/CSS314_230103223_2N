import multiprocessing as mp
import time
from harness import emit_verify_tag


STUDENT_ID = 230103223
NUM_READS = 2_000_000


def measure_clock_latency_ns(iterations: int = NUM_READS) -> float:
  # Brief warm-up to stabilize CPU frequency and interpreter loop
  for _ in range(10_000):
    time.perf_counter_ns()

  t0 = time.perf_counter_ns()
  for _ in range(iterations):
    time.perf_counter_ns()
  t1 = time.perf_counter_ns()

  return (t1 - t0) / iterations


def main() -> None:
  p_log = mp.cpu_count()
  avg_latency_ns = measure_clock_latency_ns(NUM_READS)

  print(f"Logical CPUs (P_log)      : {p_log}")
  print(f"Clock reads               : {NUM_READS:,}")
  print(f"Average call latency (ns) : {avg_latency_ns:.4f} ns")
  print(emit_verify_tag("PHASE0", STUDENT_ID, avg_latency_ns, float(p_log)))


if __name__ == "__main__":
  main()