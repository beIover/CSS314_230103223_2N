# Task 1: Strong Scaling vs. Weak Scaling

## 1. Introduction

Parallel computing uses multiple processors to execute parts of a program simultaneously.

There are two important models for understanding parallel performance:

- Amdahl's Law (1967)
- Gustafson's Law (1988)

Both models explain how performance changes when the number of processors increases. However, they use different assumptions about workload size.

## 2. Amdahl's Law (Strong Scaling)

Amdahl's Law describes strong scaling.

Strong scaling means that the total problem size remains constant while the number of processors increases.

The main assumption is that the serial part of a program cannot be parallelized.

### Mathematical formula

S(P) = 1 / [s + (1-s)/P]

Where:

- S(P) = speedup
- P = number of processors
- s = serial fraction of the program
- (1-s) = parallel fraction of the program

### Explanation

When more processors are added, the parallel part becomes faster. However, the serial part still executes sequentially.

Therefore, the serial portion limits the maximum possible speedup.

For example, if 10% of a program is serial, even an infinite number of processors cannot produce a speedup greater than 10.

### Main characteristics

- Problem size remains constant.
- Execution time decreases as processors increase.
- Serial execution limits performance.
- Additional processors eventually provide smaller improvements.

## 3. Gustafson's Law (Weak Scaling)

Gustafson's Law describes weak scaling.

Weak scaling means that the problem size increases proportionally with the number of processors.

The main assumption is that more processors allow us to solve larger problems in approximately the same execution time.

### Mathematical formula

S(P) = P - α(P-1)

Where:

- S(P) = scaled speedup
- P = number of processors
- α = serial fraction measured during parallel execution
- (1-α) = parallel fraction

### Explanation

Unlike Amdahl's Law, Gustafson's Law considers that the workload can grow when more processors become available.

The serial portion remains relatively small while the parallel workload increases.

This makes parallel computing useful for solving larger scientific and computational problems.

### Main characteristics

- Problem size increases with processor count.
- Execution time can remain approximately constant.
- Larger workloads can benefit from additional processors.
- Speedup can increase almost linearly under ideal conditions.

## 4. Comparison of Amdahl's Law and Gustafson's Law

| Characteristic | Amdahl's Law | Gustafson's Law |
|---|---|---|
| Scaling type | Strong scaling | Weak scaling |
| Problem size | Constant | Increases |
| Processor count | Increases | Increases |
| Main limitation | Serial fraction | Serial fraction |
| Main objective | Reduce execution time | Solve larger problems |
| Workload | Fixed | Scaled |
| Performance | Limited by serial work | Can approach linear speedup |

## 5. Fundamental Assumptions

Amdahl's Law assumes that the total workload remains constant.

When the processor count increases, the parallel portion is divided among more processors. However, the serial portion cannot be divided.

Gustafson's Law assumes that the workload grows as more processors become available.

Instead of only reducing the execution time of a fixed problem, the system uses additional processors to perform more computational work.

The main difference is how the problem size behaves when processor count increases.

## 6. Why Do Their Predictions Differ?

The two laws predict different performance because they describe different computing situations.

Amdahl's Law focuses on solving the same problem faster. Its speedup is limited by the serial portion of the program.

Gustafson's Law focuses on solving larger problems using more processors. It shows that additional processing power can be used to increase the amount of useful computational work.

For this reason, Amdahl's Law is more restrictive when evaluating fixed-size workloads, while Gustafson's Law is useful for evaluating large-scale parallel applications.

## 7. Conclusion

Amdahl's Law and Gustafson's Law are both important models in parallel computing.

Amdahl's Law explains the limitations of strong scaling when the problem size remains constant.

Gustafson's Law explains the potential of weak scaling when the problem size increases with processor count.

Understanding both models helps software engineers choose appropriate parallel computing strategies and evaluate the performance of distributed systems.
