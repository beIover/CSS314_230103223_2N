# Task 10: Memory Growth Rate vs. Scaled Speedup

## 1. Mathematical Formulation of Weak Scaling Memory Limits

Under pure weak scaling, aggregate problem workload and total memory footprint scale linearly with processor count $P$:
$$W(P) = P \cdot W(1)$$
$$M(P) = c \cdot P$$

Where:
* $M(P)$ is the total aggregate memory required across $P$ nodes.
* $c = M_{\text{node}} = 32\text{ GB} = 32 \times 2^{30}\text{ bytes} = 34,359,738,368\text{ bytes}$ is the strict per-node memory capacity limit.

---

## 2. 3D Grid Dimension Scaling Laws

For a 3D grid-based solver of aggregate global dimension $N \times N \times N$, the memory footprint scales cubically with global grid size:
$$M(N) = \mu \cdot N^3$$

Where $\mu$ represents the memory required per grid point (e.g., $\mu = 8\text{ bytes}$ for a single double-precision float per grid cell).

Equating aggregate memory capacity to physical cluster memory:
$$\mu \cdot N(P)^3 = P \cdot M_{\text{node}}$$

Isolating global grid dimension $N(P)$:
$$N(P) = \left( \frac{M_{\text{node}}}{\mu} \right)^{1/3} \cdot P^{1/3} = N(1) \cdot P^{1/3}$$

Similarly, local grid dimension $N_{\text{local}}$ assigned to each individual node remains constant under weak scaling:
$$N_{\text{local}} = \left( \frac{M_{\text{node}}}{\mu} \right)^{1/3} = \text{constant}$$

$$N_{\text{global}}(P) = N_{\text{local}} \cdot P^{1/3}$$

---

## 3. Capacity Limit Calculations

Assuming standard double-precision storage ($\mu = 8\text{ bytes/cell}$) and $M_{\text{node}} = 32\text{ GiB}$:

### Per-Node Maximum Local Grid Dimension ($N_{\text{local}}$)
$$N_{\text{local}} = \sqrt[3]{\frac{34,359,738,368}{8}} = \sqrt[3]{4,294,967,296} \approx 1,625.4987$$

$$\text{Max integer } N_{\text{local}} = 1,625 \text{ grid points per dimension per node}$$

---

### 3.1 Evaluation for $P = 64$ Nodes

* **Total Aggregate Memory:** 
  $$M(64) = 64 \times 32\text{ GB} = 2,048\text{ GB} \quad (2\text{ TB})$$
* **Global Grid Dimension ($N_{\text{global}}$):**
  $$N(64) = N_{\text{local}} \cdot 64^{1/3} = 1,625.4987 \times 4 = 6,501.9947$$
* **Max Global Grid Size ($N \times N \times N$):**
  $$\mathbf{N_{64} = 6,501} \quad (6,501 \times 6,501 \times 6,501 \approx 2.75 \times 10^{11} \text{ grid points})$$

---

### 3.2 Evaluation for $P = 512$ Nodes

* **Total Aggregate Memory:** 
  $$M(512) = 512 \times 32\text{ GB} = 16,384\text{ GB} \quad (16\text{ TB})$$
* **Global Grid Dimension ($N_{\text{global}}$):**
  $$N(512) = N_{\text{local}} \cdot 512^{1/3} = 1,625.4987 \times 8 = 13,003.9894$$
* **Max Global Grid Size ($N \times N \times N$):**
  $$\mathbf{N_{512} = 13,003} \quad (13,003 \times 13,003 \times 13,003 \approx 2.20 \times 10^{12} \text{ grid points})$$

---

## 4. Summary Table

| Metric | Single Node ($P=1$) | Cluster ($P=64$) | Cluster ($P=512$) | Scaling Exponent |
| :--- | :---: | :---: | :---: | :---: |
| **Total Memory Cap** | $32\text{ GB}$ | $2,048\text{ GB}$ | $16,384\text{ GB}$ | $O(P)$ |
| **Local Grid Dim ($N_{\text{local}}$)** | $1,625$ | $1,625$ | $1,625$ | $O(1)$ |
| **Global Grid Dim ($N_{\text{global}}$)** | $1,625$ | **$6,501$** | **$13,003$** | $O(P^{1/3})$ |
| **Total Grid Cells ($N^3$)** | $4.29 \times 10^9$ | $2.75 \times 10^{11}$ | $2.20 \times 10^{12}$ | $O(P)$ |

---

## 5. Architectural Synthesis

1. **Sublinear Dimension Growth:**
   Doubling processor count increases spatial grid resolution by a factor of only $2^{1/3} \approx 1.26\times$. To double spatial resolution $N$ in 3D simulations, hardware memory and node counts must expand by $2^3 = 8\times$.
2. **Memory Ceiling Compliance:**
   As long as per-node problem dimension $N_{\text{local}} \le 1,625$, memory footprint remains strictly within the $32\text{ GB}$ physical threshold, enabling linear weak scaling across arbitrary cluster sizes.
