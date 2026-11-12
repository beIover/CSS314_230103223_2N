
import multiprocessing as mp
import math
import time

from harness import emit_verify_tag


# ==========================================
# CONFIGURATION
# ==========================================

STUDENT_ID = 230103223

STRONG_N = 1_200_000
WEAK_N0 = 200_000

PROCESS_COUNTS = range(1, mp.cpu_count() + 1)

RUNS = 3


# ==========================================
# STAGE A: SERIAL WORKLOAD
# 10 sin/sqrt operations per element
# ==========================================

def stage_a(n):

    count = max(1, int(n * 0.05))

    result = 0.0

    for i in range(count):

        x = (i % 1000 + 1) / 1000.0

        for _ in range(5):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0)

        result += x

    return result


# ==========================================
# STAGE B: PARALLEL WORKER
# 30 operations per element
# ==========================================

def stage_b_worker(bounds):

    start, end = bounds

    result = 0.0

    for i in range(start, end):

        x = (i % 1000 + 1) / 1000.0

        for _ in range(10):

            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0)
            x = x * 0.999999 + 0.000001

        result += x

    return result


# ==========================================
# CREATE PARTITIONS
# ==========================================

def create_chunks(n, p):

    return [
        (
            i * n // p,
            (i + 1) * n // p
        )
        for i in range(p)
    ]


# ==========================================
# RUN ONE PIPELINE TRIAL
# ==========================================

def run_trial(pool, n, p):

    start = time.perf_counter_ns()

    # Stage A runs in the parent process.
    stage_a_start = time.perf_counter_ns()

    result_a = stage_a(n)

    stage_a_time = (
        time.perf_counter_ns() - stage_a_start
    ) / 1e6

    # Stage B runs in persistent worker processes.
    chunks = create_chunks(n, p)

    results_b = pool.map(
        stage_b_worker,
        chunks
    )

    result_b = sum(results_b)

    total_time = (
        time.perf_counter_ns() - start
    ) / 1e6

    # Keep both results used.
    checksum = result_a + result_b

    return total_time, stage_a_time, checksum


# ==========================================
# MEASURE WITH PERSISTENT POOL
# ==========================================

def measure_case(n, p):

    total_samples = []
    stage_a_samples = []

    # Pool is created outside the timed trials.
    with mp.Pool(processes=p) as pool:

        for _ in range(RUNS):

            total, stage_a_ms, checksum = run_trial(
                pool, n, p
            )

            total_samples.append(total)
            stage_a_samples.append(stage_a_ms)

    total_samples.sort()
    stage_a_samples.sort()

    median_total = total_samples[RUNS // 2]
    median_stage_a = stage_a_samples[RUNS // 2]

    return median_total, median_stage_a


# ==========================================
# PHASE 2 MAIN
# ==========================================

def main():

    print("\nPHASE 2: STRONG AND WEAK SCALING")
    print("================================")

    print("Student ID:", STUDENT_ID)
    print("Logical cores:", mp.cpu_count())
    print("Strong N:", STRONG_N)
    print("Weak N0:", WEAK_N0)
    print("Measurement runs:", RUNS)

    processes = list(PROCESS_COUNTS)

    # --------------------------------------
    # STRONG SCALING
    # --------------------------------------

    print("\n1. STRONG SCALING")
    print("-----------------")

    strong_results = {}

    print(
        f"{'P':>4} {'N':>12} {'Stage A ms':>12} "
        f"{'Total ms':>12} {'Speedup':>10} "
        f"{'Efficiency':>12}"
    )

    for p in processes:

        total_ms, stage_a_ms = measure_case(
            STRONG_N, p
        )

        strong_results[p] = {
            "total": total_ms,
            "stage_a": stage_a_ms
        }

        if p == 1:
            strong_baseline = total_ms

        speedup = strong_baseline / total_ms
        efficiency = speedup / p

        print(
            f"{p:>4} "
            f"{STRONG_N:>12} "
            f"{stage_a_ms:>12.3f} "
            f"{total_ms:>12.3f} "
            f"{speedup:>10.3f} "
            f"{efficiency:>12.3f}"
        )

    # Amdahl's Law
    serial_fraction = (
        strong_results[1]["stage_a"]
        / strong_results[1]["total"]
    )

    amdahl_cap = 1.0 / serial_fraction

    strong_speedup_log = (
        strong_baseline
        / strong_results[mp.cpu_count()]["total"]
    )

    print("\nAmdahl serial fraction:", f"{serial_fraction:.6f}")
    print("Amdahl theoretical cap:", f"{amdahl_cap:.4f}")
    print(
        "Observed speedup at P_log:",
        f"{strong_speedup_log:.4f}"
    )

    # --------------------------------------
    # WEAK SCALING
    # --------------------------------------

    print("\n2. WEAK SCALING")
    print("---------------")

    weak_results = {}

    print(
        f"{'P':>4} {'N':>12} {'Stage A ms':>12} "
        f"{'Total ms':>12} {'Efficiency':>12} "
        f"{'Gustafson':>12}"
    )

    for p in processes:

        n = p * WEAK_N0

        total_ms, stage_a_ms = measure_case(n, p)

        weak_results[p] = {
            "total": total_ms,
            "stage_a": stage_a_ms
        }

        if p == 1:
            weak_baseline = total_ms

        weak_efficiency = weak_baseline / total_ms

        alpha = stage_a_ms / total_ms

        gustafson = p - alpha * (p - 1)

        print(
            f"{p:>4} "
            f"{n:>12} "
            f"{stage_a_ms:>12.3f} "
            f"{total_ms:>12.3f} "
            f"{weak_efficiency:>12.3f} "
            f"{gustafson:>12.4f}"
        )

    # --------------------------------------
    # GUSTAFSON AND VERIFICATION
    # --------------------------------------

    p_log = mp.cpu_count()

    weak_log = weak_results[p_log]

    alpha_log = (
        weak_log["stage_a"] / weak_log["total"]
    )

    gustafson_log = (
        p_log - alpha_log * (p_log - 1)
    )

    print("\n3. ARCHITECTURAL ANALYSIS")
    print("------------------------")

    print("Physical cores: 4")
    print("Logical cores:", p_log)

    print(
        "Strong speedup at P_log:",
        f"{strong_speedup_log:.4f}"
    )

    print(
        "Amdahl theoretical limit:",
        f"{amdahl_cap:.4f}"
    )

    print(
        "Gustafson scaled speedup at P_log:",
        f"{gustafson_log:.4f}"
    )

    print("\n4. VERIFICATION")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup_log,
            gustafson_log
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

