# Task 08: Cross-Over Point Determination (Amdahl vs. Gustafson)

## 1. Problem Formulation & Setup

We evaluate a parallel workload on a processor architecture with $P = 128$ cores, comparing predictions under a static serial overhead fraction of $8\%$ ($s = \alpha = 0.08$):

* **Amdahl's Law Baseline:** Fixed-size workload (Strong Scaling) where $s = 0.08$ is measured on 1 core.
* **Gustafson's Law Baseline:** Scaled workload (Weak Scaling) where $\alpha = 0.08$ is measured on the parallel machine.

---

## 2. Step-by-Step Mathematical Calculations

### 2.1 Amdahl's Law Speedup Calculation
$$S_{\text{Amdahl}}(P) = \frac{1}{s + \frac{1 - s}{P}}$$

Substituting $s = 0.08$ and $P = 128$:
$$S_{\text{Amdahl}}(128) = \frac{1}{0.08 + \frac{1 - 0.08}{128}} = \frac{1}{0.08 + \frac{0.92}{128}}$$

$$\frac{0.92}{128} = 0.0071875$$

$$S_{\text{Amdahl}}(128) = \frac{1}{0.08 + 0.0071875} = \frac{1}{0.0871875} \approx 11.4695\times$$

### 2.2 Gustafson's Scaled Speedup Calculation
$$S_{\text{Gustafson}}(P) = P - \alpha(P - 1)$$

Substituting $\alpha = 0.08$ and $P = 128$:
$$S_{\text{Gustafson}}(128) = 128 - 0.08 \times (128 - 1) = 128 - 0.08 \times 127$$

$$0.08 \times 127 = 10.16$$

$$S_{\text{Gustafson}}(128) = 128 - 10.16 = 117.8400\times$$

---

## 3. Discrepancy & Variance Analysis

| Metric | Amdahl's Law ($S_A$) | Gustafson's Law ($S_G$) | Absolute Variance ($\Delta S$) |
| :--- | :---: | :---: | :---: |
| **Predicted Speedup ($P=128$)** | **$11.47\times$** | **$117.84\times$** | **$+106.37\times$** |
| **Parallel Efficiency ($E(128)$)** | **$8.96\%$** | **$92.06\%$** | **$+83.10\%$** |

### Variance Ratios
* **Speedup Expansion Ratio ($S_G / S_A$):**
  $$\frac{S_{\text{Gustafson}}}{S_{\text{Amdahl}}} = \frac{117.84}{11.469534} \approx 10.2742\times$$
* **Percentage Increase over Amdahl:**
  $$\frac{S_G - S_A}{S_A} \times 100\% = \frac{106.3705}{11.4695} \times 100\% \approx 927.42\%$$
* **Relative Model Discrepancy w.r.t. Gustafson:**
  $$\frac{S_G - S_A}{S_G} \times 100\% = \frac{106.3705}{117.84} \times 100\% \approx 90.27\%$$

---

## 4. Architectural Analysis: Why the Models Diverge

1. **Fixed Problem Size Ceiling (Amdahl):**
   Amdahl keeps total problem size constant while splitting it into $128$ smaller chunks. As $P \to \infty$, execution time is choked by the fixed $8\%$ serial portion, capping maximum speedup at $1 / s = 12.5\times$ regardless of how many cores are added.
   
2. **Dynamic Workload Scaling (Gustafson):**
   Gustafson assumes problem size grows by $128\times$ to fully occupy all $128$ cores in the same total execution window. The total parallel work expands proportionally ($O(P)$), while serial overhead remains constant ($O(1)$), allowing speedup to scale linearly to $117.84\times$.
