# Task 11: Sun-Ni Memory-Bounded Speedup vs. Gustafson's Model

## 1. Executive Overview & Paradigm Shift

In 1991, Xian-He Sun and Lionel M. Ni introduced the **Memory-Bounded Speedup** model to bridge the gap between Amdahl's Law (fixed-size / compute-bound) and Gustafson's Law (fixed-time / linear workload scaling). 

Sun and Ni observed that in real-world High-Performance Computing (HPC) architectures, the primary physical constraint on scaling is **available node memory**, not wall-clock time execution limits or uniprocessor baseline size.

---

## 2. Comparative Mathematical Formulations

Let:
* $s$: Serial fraction of total workload on 1 core.
* $p$: Parallel fraction ($s + p = 1$).
* $P$: Number of processing nodes.
* $G(P)$: Workload expansion factor resulting from a $P$-fold increase in system memory capacity.

### Model Spectrum Comparison

1. **Amdahl's Law (Fixed-Size / $G(P) = 1$):**
   $$S_A(P) = \frac{s + p}{s + \frac{p}{P}} = \frac{1}{s + \frac{1-s}{P}}$$

2. **Gustafson's Law (Fixed-Time / $G(P) = P$):**
   $$S_G(P) = \frac{s + p \cdot P}{s + p} = s + p \cdot P = P - \alpha(P - 1)$$

3. **Sun-Ni Memory-Bounded Model ($G(P)$ variable):**
   $$S_{SN}(P) = \frac{s + p \cdot G(P)}{s + \frac{p \cdot G(P)}{P}}$$

---

## 3. The Function $G(P)$ and Algorithmic Complexity

The function $G(P)$ represents how computational work grows when problem memory size scales $P$-fold:

* If memory footprint scales as $M(N) = O(N^a)$ and computational complexity scales as $W(N) = O(N^b)$, then:
  $$G(P) = P^{b/a}$$

### Case 1: $b = a$ (Linear Compute relative to Memory)
* **Examples:** Vector addition, image processing ($O(N)$ work, $O(N)$ memory).
* $G(P) = P^{1} = P$.
* **Sun-Ni collapses directly to Gustafson's Law:** $S_{SN}(P) = S_G(P)$.

### Case 2: $b > a$ (Compute grows faster than Memory)
* **Examples:** Dense Matrix Multiplication ($O(N^3)$ work, $O(N^2)$ memory $\implies b/a = 3/2 = 1.5$), N-body direct solver ($O(N^2)$ work, $O(N)$ memory $\implies b/a = 2$).
* $G(P) = P^{1.5} > P$.
* **Gustafson's Law UNDERESTIMATES achievable speedup:** Sun-Ni yields higher speedups because parallel compute work expands faster than linear processor count ($G(P) > P$).

### Case 3: $b < a$ (Compute grows slower than Memory)
* **Examples:** Sparse lookup tables, memory-bandwidth bounded search algorithms.
* $G(P) < P$.
* **Gustafson's Law OVERESTIMATES achievable speedup:** Gustafson assumes $O(P)$ compute expansion, but memory constraints limit actual computational density below $P$.

---

## 4. Conditions of Overestimation and Underestimation

| Regime | Mathematical Condition | Real-World Example | Gustafson's Accuracy |
| :--- | :---: | :--- | :--- |
| **Super-Linear Work Scaling** | $G(P) > P$ ($b > a$) | Dense GEMM ($G(P) = P^{1.5}$) | **Underestimates** throughput |
| **Linear Work Scaling** | $G(P) = P$ ($b = a$) | Monte Carlo integration | **Accurately models** throughput |
| **Sub-Linear Work Scaling** | $G(P) < P$ ($b < a$) | Memory-bound graph traversal | **Overestimates** throughput |

---

## 5. Critical Synthesis

Sun-Ni's Memory-Bounded model generalizes parallel performance theory:
* **Amdahl's Law** is a special boundary condition where $G(P) = 1$.
* **Gustafson's Law** is a special boundary condition where $G(P) = P$.
* **Sun-Ni's Model** accurately reflects real hardware by grounding scaling in physical RAM limits ($G(P) = P^{b/a}$), explaining why algorithms with high compute-to-memory ratios (like dense linear algebra) achieve higher speedups than predicted by Gustafson's Law alone.
