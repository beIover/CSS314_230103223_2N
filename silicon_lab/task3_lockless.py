import threading
import time


TOTAL_OPS = 2_000_000
NUM_THREADS = 4


def lockless_bench():

    local_results = [0] * NUM_THREADS

    def worker(thread_id):

        local_count = 0

        for _ in range(TOTAL_OPS // NUM_THREADS):
            local_count += 1

        local_results[thread_id] = local_count

    threads = [
        threading.Thread(
            target=worker,
            args=(i,)
        )
        for i in range(NUM_THREADS)
    ]

    start = time.perf_counter()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    total = sum(local_results)

    elapsed = time.perf_counter() - start

    return total, elapsed


if __name__ == "__main__":

    value, elapsed = lockless_bench()

    print(f"Lockless Value = {value:,}")
    print(f"Expected Value = {TOTAL_OPS:,}")
    print(f"Execution Time = {elapsed:.4f}s")
