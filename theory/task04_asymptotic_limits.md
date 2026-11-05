# Task 04: Asymptotic Analysis of Parallel Efficiency

## 1. Mathematical Formulation

Parallel efficiency $E(P)$ measures how effectively a parallel system utilizes $P$ processing units compared to a single processor:
$$E(P) = \frac{S(P)}{P}$$

Where $S(P)$ is the speedup achieved on $P$ processors.

---

## 2. Gustafson's Model: Asymptotic Efficiency Limit

Under Gustafson's Law, scaled speedup is defined as:
$$S_G(P) = P - \alpha(P - 1) = (1 - \alpha)P + \alpha$$

Where $\alpha \in [0, 1]$ represents the serial time fraction observed on the parallel system.

### Derivation of Efficiency Formula
Substituting $S_G(P)$ into the efficiency equation:
$$E_G(P) = \frac{(1 - \alpha)P + \alpha}{P} = (1 - \alpha) + \frac{\alpha}{P}$$

### Asymptotic Limit Proof as $P \to \infty$
Taking the limit of $E_G(P)$ as $P$ approaches infinity:
$$\lim_{P \to \infty} E_G(P) = \lim_{P \to \infty} \left( (1 - \alpha) + \frac{\alpha}{P} \right)$$

Since $\lim_{P \to \infty} \frac{\alpha}{P} = 0$:
$$\lim_{P \to \infty} E_G(P) = 1 - \alpha$$

### Physical Interpretation
Under Gustafson's weak scaling model, as the processor count $P$ expands to infinity, efficiency **does not degrade to zero**. Instead, it asymptotes directly to the parallel fraction $p = 1 - \alpha$. If an application has a 5% serial fraction ($\alpha = 0.05$), parallel efficiency will never drop below $95\%$, regardless of core count.

---

## 3. Amdahl's Model: Asymptotic Efficiency Limit

Under Amdahl's Law, speedup for a fixed-size problem is defined as:
$$S_A(P) = \frac{1}{s + \frac{1 - s}{P}}$$

Where $s \in (0, 1]$ is the uniprocessor serial fraction.

### Derivation of Efficiency Formula
$$E_A(P) = \frac{S_A(P)}{P} = \frac{1}{s \cdot P + (1 - s)}$$

### Asymptotic Limit Proof as $P \to \infty$
Taking the limit of $E_A(P)$ for any non-zero serial fraction $s > 0$:
$$\lim_{P \to \infty} E_A(P) = \lim_{P \to \infty} \frac{1}{s \cdot P + (1 - s)} = 0$$

### Physical Interpretation
Under Amdahl's strong scaling model, parallel efficiency continuously decays toward **zero** as $P$ increases. The fixed serial bottleneck $s$ dominates total execution time, leading to severe processor idle time and hardware waste.

---

## 4. Analytical Comparison & Summary Table

| Model | Speedup Formula $S(P)$ | Efficiency Formula $E(P)$ | Asymptotic Limit $\lim_{P \to \infty} E(P)$ |
| :--- | :--- | :--- | :--- |
| **Amdahl (Fixed-Size)** | $\frac{1}{s + \frac{1 - s}{P}}$ | $\frac{1}{s \cdot P + 1 - s}$ | **$0$** (Efficiency collapses) |
| **Gustafson (Fixed-Time)** | $P - \alpha(P - 1)$ | $1 - \alpha + \frac{\alpha}{P}$ | **$1 - \alpha$** (Sustained efficiency) |

### Key Conclusion
* **Amdahl's Law** demonstrates that adding computing power to a **fixed workload** yields diminishing returns, driving parallel efficiency down to $0$.
* **Gustafson's Law** proves that by scaling the workload proportionally with processor count $P$, systems sustain a constant efficiency baseline of $1 - \alpha$, justifying massive parallel scaling (HPC/supercomputing clusters).
