# Monte Carlo Pi - Java Thread Benchmark

## Description

This project demonstrates the effects of data races, synchronization,
and local reduction using native Java threads.

The program approximates pi using the Monte Carlo method.

Pi is calculated as:

pi = 4 * hits / total

## Part 1 - Data Race

50,000,000 random points are generated using 4 Java threads.

All threads update the shared variable:

`static long totalHits = 0;`

using:

`totalHits++;`

This creates a data race because the increment is not atomic.

The program is executed 5 times and the resulting values of pi are recorded.

## Part 2 - Synchronization

The race condition is fixed using synchronized access to the shared counter.

The execution time is compared with a single-threaded version.

## Part 3 - Local Reduction

Each thread maintains its own local hit counter.

The partial counters are added together after all threads finish.

The benchmark is performed with:

1, 2, 4, 8, 16, and 32 threads.

## Results

### Part 1

| Run | Pi |
|---|---:|
| 1 | 0.9671291200 |
| 2 | 0.8710220000 |
| 3 | 1.0111489600 |
| 4 | 0.8568056800 |
| 5 | 0.8516904000 |

### Part 2

| Version | Runtime (ms) |
|---|---:|
| Single thread | 378.42 |
| Synchronized 4 threads | 2358.63 |

*(Note: Synchronized version took 6.23x longer to run than the single-threaded version).*

### Part 3

| Threads | Runtime (ms) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 1118.03 | 1.00x | 100.00% |
| 2 | 587.59 | 1.90x | 95.14% |
| 4 | 283.85 | 3.94x | 98.47% |
| 8 | 224.91 | 4.97x | 62.14% |
| 16 | 224.67 | 4.98x | 31.10% |
| 32 | 225.96 | 4.95x | 15.46% |

## Questions

### 1. Why didn't 16 threads run twice as fast as 8 threads?

Increasing the number of threads does not guarantee proportional performance improvement.
The computer has a limited number of physical and logical CPU resources (likely 8 physical cores on this machine).
Additional threads introduce scheduling and context-switching overhead.
After the available hardware threads are fully utilized, additional threads compete
for the same resources and therefore provide diminishing performance improvements.

### 2. Why was the synchronized version slower than one thread?

The synchronized version requires threads to acquire a lock whenever they update
the shared counter. This creates lock contention because many threads repeatedly
try to access the same variable. The time spent waiting for synchronization and passing the lock between cores over the memory bus is significantly greater than the performance benefit of using multiple threads, resulting in ~95% of CPU cycles being wasted on memory bus locks.
