# Task 09: Multi-Component Workload Partitioning Analysis

## 1. Workload Phase Breakdown

A scientific simulation pipeline is split into three execution phases measured on the parallel system:

| Phase | Description | Parallel System Time Fraction | Classification |
| :--- | :--- | :---: | :--- |
| **Phase A** | Initial disk I/O and parameter broadcast | $2\%$ ($0.02$) | Serial Overhead ($s_A$) |
| **Phase B** | Boundary message-passing synchronization | $6\%$ ($0.06$) | Serial Overhead ($s_B$) |
| **Phase C** | Core domain flux computation | $92\%$ ($0.92$) | Parallel Compute ($p$) |

### Aggregate Serial Fraction ($\alpha$)
The aggregate serial fraction is the sum of non-parallelizable overheads:
$$\alpha = s_A + s_B = 0.02 + 0.06 = 0.08 \quad (8\%)$$

The parallel fraction is:
$$p = 1 - \alpha = 0.92 \quad (92\%)$$

---

## 2. Scaled Speedup and Efficiency Equations

Under Gustafson's Law:
$$S(P) = P - \alpha(P - 1) = 0.92 \cdot P + 0.08$$

Parallel efficiency is:
$$E(P) = \frac{S(P)}{P} = 0.92 + \frac{0.08}{P}$$

---

## 3. Numeric Calculations

### 3.1 Evaluation for $P = 512$ Cores
* **Scaled Speedup:**
  $$S(512) = 0.92 \times 512 + 0.08 = 471.04 + 0.08 = 471.1200\times$$

* **Parallel Efficiency:**
  $$E(512) = \frac{471.12}{512} = 0.92015625 \quad (92.02\%)$$

### 3.2 Evaluation for $P = 2,048$ Cores
* **Scaled Speedup:**
  $$S(2048) = 0.92 \times 2048 + 0.08 = 1884.16 + 0.08 = 1884.2400\times$$

* **Parallel Efficiency:**
  $$E(2048) = \frac{1884.24}{2048} = 0.9200390625 \quad (92.00\%)$$

---

## 4. Phase Breakdown & Scaling Summary Table

| Core Count ($P$) | Serial I/O Overhead | Sync Overhead | Parallel Compute Work | Scaled Speedup $S(P)$ | Efficiency $E(P)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | $0.02 \times 512 = 10.24$ | $0.06 \times 512 = 30.72$ | $0.92 \times 512 = 471.04$ | **$471.12\times$** | **$92.02\%$** |
| **2048** | $0.02 \times 2048 = 40.96$ | $0.06 \times 2048 = 122.88$ | $0.92 \times 2048 = 1884.16$ | **$1884.24\times$** | **$92.00\%$** |

---

## 5. Architectural Recommendations

1. **Dominant Bottleneck Identification:**
   Boundary synchronization (Phase B) constitutes **$75\%$** of the total serial fraction ($6\%$ out of $8\%$). Optimization efforts must prioritize non-blocking MPI communication and compute-communication overlap rather than storage I/O.
   
2. **Asymptotic Scaling:**
   Even as core count scales $4\times$ (from $512$ to $2,048$), parallel efficiency drops by less than $0.02\%$, stabilizing near the compute fraction baseline of $92\%$.

