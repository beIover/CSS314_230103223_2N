# Lab Performance Synthesis & Hardware Profile — Student ID: 230103223

## Hardware Profile
- **Host Machine:** MacBook Pro (Apple Silicon / macOS)
- **Available Cores:** Verified via Python execution core inspection.

---

## Empirical Results Summary Table

| Experiment Mode | Flat-Park Metric & Observations |
| :--- | :--- |
| **A: Stubborn Serial Tail** (`py.laires_a_pxe`) | $e(2) \approx 0.12$, $e(4) \approx 0.13$ — Stable algorithmic serial fraction (~12%). |
| **B: Lock Contention** (`py.noitnetnoc_b_pxe`) | $e(4) \gg e(2)$ and $e > 1.5$ — Catastrophic cache thrashing & lock contention. |
| **C: Privatization Fix** (`py.dexif_c_pxe`) | $e \le 0.03$ — Near-linear speedup achieved via local thread-private accumulation. |

---
- **Git Branch:** `230103223`
- **Status:** All experiments completed, verified, and pushed.
