# Task 2: Formal Derivation of Gustafson's Scaled Speedup

## 1. Introduction

Gustafson's Law explains how parallel computing performance improves when the problem size increases with the number of processors.

Unlike Amdahl's Law, which assumes a fixed problem size, Gustafson's Law considers scaled workloads.

The objective is to derive the mathematical formula for scaled speedup.

## 2. Definitions and Assumptions

Let:

- \(P\) = number of processors.
- \(T_P\) = total execution time using \(P\) processors.
- \(\alpha\) = serial fraction of the parallel execution time.
- \(1-\alpha\) = parallel fraction of the parallel execution time.

We normalize the parallel execution time to 1 unit.

Therefore:

\[
T_P = 1
\]

The serial portion takes:

\[
T_s = \alpha
\]

The parallel portion takes:

\[
T_{parallel} = 1-\alpha
\]

## 3. Derivation Step by Step

**Step 1: Calculate the serial execution time.**

The serial portion cannot be divided among processors.

\[
T_s = \alpha
\]

**Step 2: Calculate the equivalent single-processor parallel execution time.**

If the parallel portion were executed using only one processor, it would take \(P\) times longer than when distributed across \(P\) processors.

Therefore:

\[
T_{parallel,1}=P(1-\alpha)
\]

**Step 3: Calculate the total single-processor execution time.**

The total time for one processor is the sum of the serial and parallel portions:

\[
T_1 = \alpha + P(1-\alpha)
\]

**Step 4: Calculate scaled speedup.**

Speedup is the ratio of single-processor execution time to parallel execution time:

\[
S(P)=\frac{T_1}{T_P}
\]

Since the normalized parallel execution time is 1:

\[
S(P)=\alpha+P(1-\alpha)
\]

**Step 5: Simplify the equation.**

Expand the brackets:

\[
S(P)=\alpha+P-P\alpha
\]

Rearrange:

\[
\boxed{S(P)=P-\alpha(P-1)}
\]

This is Gustafson's Law.

## 4. Alternative Formula

If \(f\) represents the parallel fraction, then:

\[
f=1-\alpha
\]

Substituting into the original formula:

\[
S(P)=P-(1-f)(P-1)
\]

Simplifying:

\[
\boxed{S(P)=1+f(P-1)}
\]

Both formulas represent the same scaled-speedup model.

## 5. Boundary Conditions

The formula can be checked using three important cases.

**Case 1: One processor**

When \(P=1\):

\[
S(1)=1-\alpha(0)=1
\]

Speedup is 1 because no parallel acceleration is possible with one processor.

**Case 2: No serial work**

When \(\alpha=0\):

\[
S(P)=P
\]

This represents ideal linear speedup.

**Case 3: Entire workload is serial**

When \(\alpha=1\):

\[
S(P)=P-(P-1)=1
\]

Adding processors provides no speedup.

## 6. Illustrative Numerical Example

Assume:

- Number of processors: \(P=8\)
- Serial fraction: \(\alpha=0.05\)

Using Gustafson's formula:

\[
S(8)=8-0.05(8-1)
\]

\[
S(8)=8-0.35
\]

\[
\boxed{S(8)=7.65}
\]

The theoretical scaled speedup is 7.65.

*Note: This is an illustrative calculation, not an empirical measurement from the student's computer.*

## 7. Comparison with Amdahl's Law

Amdahl's Law:

\[
S_A(P)=\frac{1}{\alpha+(1-\alpha)/P}
\]

Gustafson's Law:

\[
S_G(P)=P-\alpha(P-1)
\]

The main difference is the workload assumption.

- Amdahl's Law assumes a fixed problem size.
- Gustafson's Law assumes the problem size grows with the number of processors.

Amdahl's Law emphasizes the limitations of parallelization, while Gustafson's Law demonstrates the potential benefits of scaling larger workloads.

## 8. Conclusion

Gustafson's Law demonstrates that parallel performance can increase almost linearly when the workload grows with the number of available processors.

The derived formula is:

\[
\boxed{S(P)=P-\alpha(P-1)}
\]

It shows that the serial fraction limits speedup, but a sufficiently small serial fraction allows parallel systems to process significantly larger problems efficiently.
