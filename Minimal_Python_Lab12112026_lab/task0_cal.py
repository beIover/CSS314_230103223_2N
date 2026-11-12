
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103223
ITERATIONS = 1_000_000


def main():
    logical_cores = mp.cpu_count()

    start = time.perf_counter_ns()

    for _ in range(ITERATIONS):
        time.perf_counter_ns()

    elapsed = time.perf_counter_ns() - start

    average_latency = elapsed / ITERATIONS

    print("PHASE 0: HARDWARE TELEMETRY")
    print("---------------------------")
    print("Student ID:", STUDENT_ID)
    print("Logical cores:", logical_cores)
    print("Total time (ns):", elapsed)
    print("Average call latency (ns):", average_latency)

    print(
        emit_verify_tag(
            "PHASE0",
            STUDENT_ID,
            average_latency,
            elapsed / 1e6
        )
    )


if __name__ == "__main__":
    main()

