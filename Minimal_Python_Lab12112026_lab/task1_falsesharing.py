
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103223
M = 16_000_000
P_PHYS = 4
P_LOG = mp.cpu_count()

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]
RUNS = 3

MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005


# Worker function must be at module level.
def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * MULTIPLIER + (i | 1)
        ) & MASK

        # Write on every iteration to exercise shared cache lines.
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_once(p, stride):
    shm = shared_memory.SharedMemory(
        create=True,
        size=4096
    )

    processes = []

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(p):
            arr[i * stride] = 1

        del arr

        iters = M // p

        start = time.perf_counter_ns()

        for i in range(p):
            process = mp.Process(
                target=fs_worker,
                args=(shm.name, i * stride, iters)
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) / 1e6

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        return elapsed_ms

    finally:
        shm.close()
        shm.unlink()


def measure(p, stride):
    stats = run_benchmark(
        lambda: run_once(p, stride),
        warmup=0,
        runs=RUNS
    )

    throughput = M / (stats.median_ms * 1000)

    return stats, throughput


def karp_flatt(speedup, p):
    if p <= 1 or speedup <= 0:
        return None

    return (
        (1 / speedup - 1 / p)
        / (1 - 1 / p)
    )


def main():

    print("\nPHASE 1: FALSE SHARING")
    print("======================")

    print("\n1. STRIDE SWEEP")
    print("Processes:", P_PHYS)

    stride_results = {}

    print(
        f"{'Stride':>8} {'Bytes':>8} "
        f"{'Median ms':>12} {'IQR ms':>10} "
        f"{'Mops/s':>12}"
    )

    for stride in STRIDES:

        stats, throughput = measure(P_PHYS, stride)

        stride_results[stride] = stats.median_ms

        print(
            f"{stride:>8} "
            f"{stride * 8:>8} "
            f"{stats.median_ms:>12.3f} "
            f"{stats.iqr_ms:>10.3f} "
            f"{throughput:>12.3f}"
        )

    print("\n2. PROCESS SCALING")

    scaling_results = {}

    for stride in [1, 16]:

        print(f"\nStride = {stride}")
        print(
            f"{'P':>5} {'Median ms':>12} "
            f"{'Speedup':>12} {'Efficiency':>12} "
            f"{'Karp-Flatt':>12}"
        )

        scaling_results[stride] = {}

        baseline = None

        for p in PROCESS_COUNTS:

            stats, throughput = measure(p, stride)

            if p == 1:
                baseline = stats.median_ms

            speedup = baseline / stats.median_ms
            efficiency = speedup / p

            e = karp_flatt(speedup, p)

            scaling_results[stride][p] = stats.median_ms

            e_text = "-" if e is None else f"{e:.5f}"

            print(
                f"{p:>5} "
                f"{stats.median_ms:>12.3f} "
                f"{speedup:>12.3f} "
                f"{efficiency:>12.3f} "
                f"{e_text:>12}"
            )

    # Verification uses measurements at P_PHYS.
    time_stride1 = scaling_results[1][P_PHYS]
    time_stride16 = scaling_results[16][P_PHYS]

    print("\n3. VERIFICATION")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            time_stride1,
            time_stride16
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

