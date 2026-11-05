# Task 03: The Serial Fraction Invariance Paradox and Frame of Reference

## 1. Frame of Reference Comparison

The fundamental divergence between Amdahl's Law and Gustafson's Law stems from the **frame of reference** used to measure the serial fraction of a workload.

| Parameter | Model | Frame of Reference | Workload Scale |
| :--- | :--- | :--- | :--- |
| **$s$ (Amdahl)** | Fixed-Size (Strong Scaling) | Measured on a **uniprocessor (1 core)** prior to parallelization. | Workload $W$ is fixed ($W = 1$). |
| **$\alpha$ (Gustafson)** | Fixed-Time (Weak Scaling) | Measured on the **parallel system ($P$ cores)** during execution. | Workload $W(P)$ scales with $P$. |

---

## 2. Mathematical Relationship Between $s$ and $\alpha$

Let the total execution time on $P$ processors under Gustafson's model be normalized to 1:
$$T_P = s' + p' = 1$$

Where:
* $s' = \alpha$ is the observed serial time on $P$ cores.
* $p' = 1 - \alpha$ is the observed parallel time on $P$ cores.

The total sequential workload (time on 1 core) required to compute this expanded problem is:
$$T_1 = s' + p' \cdot P = \alpha + (1 - \alpha)P$$

If we express Amdahl's uniprocessor serial fraction $s$ in terms of Gustafson's observed parallel fraction $\alpha$:
$$s = \frac{\text{Serial Work}}{\text{Total Work}} = \frac{s'}{s' + p' \cdot P} = \frac{\alpha}{\alpha + (1 - \alpha)P}$$

---

## 3. Analytical Resolution of the Paradox

### The Paradox
Amdahl's Law predicts that if an application has even a small serial fraction (e.g., $s = 0.05$), speedup is bounded by $1/s = 20\times$, regardless of how many thousands of cores are added. This suggests that serial overhead inevitably throttles massive parallelism.

### The Resolution
Measuring the serial fraction $\alpha$ on the parallel machine resolves this paradox through two key observations:

1. **Workload Expansion ($W \propto P$):** As processor count $P$ increases, the problem size expands so that the parallel execution time remains constant. The parallel work grows as $O(P)$, whereas serial work (e.g., startup overhead, parameter broadcast) remains constant $O(1)$ or grows slowly $O(\log P)$.
2. **Relative Dilution of Serial Overhead:** Because total parallel work $W_p(P) = p' \cdot P$ grows linearly with $P$, the serial fraction measured on the parallel system becomes:
   $$\alpha(P) = \frac{W_s}{W_s + W_p(P) / P} \approx \text{const}$$
   Because $W_p(P)$ expands to fill available memory/compute on $P$ cores, $W_s$ becomes an infinitesimally small fraction of the *total accumulated work* across all $P$ cores.

---

## 4. Summary Conclusion

Amdahl measures serial overhead relative to a **tiny, fixed workload**, making serial time dominate as $P \to \infty$. Gustafson measures serial overhead relative to a **massively scaled workload**, where parallel work expands linearly with $P$. 

Measuring $\alpha$ on the parallel system accurately reflects that memory and problem size scale alongside hardware capacity, proving that serial overhead does not inherently cap massive speedups in weak scaling workloads.
