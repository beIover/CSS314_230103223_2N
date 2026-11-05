# Task 06: Parametric Speedup Across Processor Densities

## 1. Problem Formulation

Given a fixed serial time fraction $\alpha = 0.04$ ($4\%$) measured on a parallel machine, calculate Gustafson's scaled speedup $S(P)$ and parallel efficiency $E(P)$ across scaling processor topologies:
$$P \in \{16, 64, 256, 1024, 4096, 16384\}$$

### Mathematical Formulas
* **Scaled Speedup:**
  $$S(P) = P - \alpha(P - 1) = (1 - \alpha)P + \alpha = 0.96P + 0.04$$

* **Parallel Efficiency:**
  $$E(P) = \frac{S(P)}{P} = 1 - \alpha + \frac{\alpha}{P} = 0.96 + \frac{0.04}{P}$$

---

## 2. Parametric Calculation Table

| Processor Count ($P$) | Parallel Fraction ($1-\alpha$) | Serial Fraction ($\alpha$) | Scaled Speedup $S(P)$ | Efficiency $E(P)$ | Efficiency (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.96 | 0.04 | 1.0000 | 1.0000 | 100.00% |
| **16** | 0.96 | 0.04 | 15.4000 | 0.9625 | 96.25% |
| **64** | 0.96 | 0.04 | 61.4800 | 0.9606 | 96.06% |
| **256** | 0.96 | 0.04 | 245.8000 | 0.9602 | 96.02% |
| **1024** | 0.96 | 0.04 | 983.0800 | 0.9600 | 96.00% |
| **4096** | 0.96 | 0.04 | 3932.2000 | 0.9600 | 96.00% |
| **16384** | 0.96 | 0.04 | 15728.6800 | 0.9600 | 96.00% |

---

## 3. Evaluation of Efficiency Decay Behavior

1. **Slow Asymptotic Decay:**
   As processor density increases exponentially ($16 \to 16,384$), efficiency drops by only **$0.25\%$** (from $96.25\%$ down to $96.00\%$).

2. **Asymptotic Lower Bound:**
   As $P \to \infty$, the term $\frac{\alpha}{P} \to 0$. The efficiency stabilizes directly at:
   $$\lim_{P \to \infty} E(P) = 1 - \alpha = 0.96 \quad (96.00\%)$$

3. **Practical HPC Implication:**
   Under weak scaling assumptions, expanding cluster size by a factor of $1,000\times$ causes negligible loss in hardware efficiency, confirming that weak scaling effectively bypasses Amdahl's hardware utilization limit.
