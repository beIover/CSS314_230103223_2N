# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS
import os
# Force native OpenMP runtime (libgomp / iomp)
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import math, time, hashlib, json
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id

STUDENT_NAME = "Leilima Akat"       # <-- Впиши сюда свое полное имя и фамилию (например, "Akerke Surname")
STUDENT_ID   = 230103223      # <-- Твой Student ID

assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a valid numeric STUDENT_ID!"
assert (os.cpu_count() or 1) >= 4, "Local multi-core CPU (>=4 logical cores) required; no Colab!"

# Trigger OpenMP layer initialization and verify backend
@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc

_ = _omp_probe()
assert nb.threading_layer() == "omp", f"Expected 'omp' layer, got {nb.threading_layer()}"

rng = np.random.default_rng(STUDENT_ID)

# Student-specific OpenMP parameters
T1_N        = 4_000_000 + (STUDENT_ID % 199) * 1024
T1_BINS     = 48 + (STUDENT_ID % 17)
T2_ITERS    = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE  = 8 + (STUDENT_ID % 5) * 8   # >= 8 int64 slots (=> 64 bytes cache-line isolation)
T3_ROWS     = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD    = 37 + (STUDENT_ID % 19)
GRID_H      = 384 + (STUDENT_ID % 4) * 32
GRID_W      = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR   = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

t1_data     = rng.integers(0, T1_BINS, size=T1_N, dtype=np.int32)
t1_weights  = rng.uniform(0.25, 2.50, size=T1_N).astype(np.float64)

MAX_LAB_THREADS = 8 if (os.cpu_count() or 1) >= 8 else (os.cpu_count() or 4)
THREAD_COUNTS = [t for t in [1, 2, 4, 8] if t <= MAX_LAB_THREADS]

print(f"Backend: {nb.threading_layer()} | Logical Cores: {os.cpu_count()} | Student: {STUDENT_ID}")
print("-" * 70)


# ==============================================================================
# TASK 1: RACE-CONDITION FORENSICS & THREAD-PRIVATE HISTOGRAM REDUCTION
# ==============================================================================
@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1                  # Data race across OpenMP threads
        w_sums[b] += weights[i]       # Non-associative floating-point race
    return hist, w_sums

@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    max_threads = get_num_threads()
    n_chunks = 8  # Фиксированное число блоков для бит-идентичности на 1, 2, 4, 8 потоках
    priv_hist = np.zeros((n_chunks, num_bins), dtype=np.int64)
    priv_wsums = np.zeros((n_chunks, num_bins), dtype=np.float64)
    worker_hits = np.zeros(max_threads, dtype=np.int64)

    n = indices.shape[0]
    chunk_size = (n + n_chunks - 1) // n_chunks

    for c in prange(n_chunks):
        tid = get_thread_id()
        start = c * chunk_size
        end = min(start + chunk_size, n)
        for i in range(start, end):
            b = indices[i]
            priv_hist[c, b] += 1
            priv_wsums[c, b] += weights[i]
            worker_hits[tid] += 1

    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for c in range(n_chunks):
        for b in range(num_bins):
            hist[b] += priv_hist[c, b]
            w_sums[b] += priv_wsums[c, b]

    return hist, w_sums, worker_hits

# 1. Empirical Race Demonstration (10 consecutive runs on 8 threads)
set_num_threads(1)
ref_hist, ref_wsums = buggy_hist_omp(t1_data, t1_weights, T1_BINS)

set_num_threads(MAX_LAB_THREADS)
lost_counts = []
max_discrepancies = []
for run in range(10):
    b_hist, b_wsums = buggy_hist_omp(t1_data, t1_weights, T1_BINS)
    lost_counts.append(int(T1_N - np.sum(b_hist)))
    max_discrepancies.append(int(np.max(np.abs(ref_hist - b_hist))))

print(f"[Task 1] Buggy lost counts over 10 runs: {lost_counts}")
print(f"[Task 1] Max per-bin discrepancy vs 1-thread ref: {max(max_discrepancies)}")

# 2. Verify bit-identical deterministic output across 1, 2, 4, 8 threads
t1_hist, t1_wsums = None, None
for tc in THREAD_COUNTS:
    set_num_threads(tc)
    h, w, hits = safe_hist_omp(t1_data, t1_weights, T1_BINS)
    if t1_hist is None:
        t1_hist, t1_wsums = h, w
    else:
        assert np.array_equal(h, t1_hist), f"Mismatch in hist at {tc} threads"
        assert np.array_equal(w, t1_wsums), f"Mismatch in w_sums at {tc} threads"
    if tc == MAX_LAB_THREADS:
        print(f"[Task 1] Worker hits across {tc} threads: {hits.tolist()}")

DIAGNOSIS_T1 = (
    "IEEE 754 floating-point addition is non-associative because every intermediate addition rounds "
    "the significand to 53 bits of precision. Even if integer updates were protected by hardware atomic "
    "instructions, dynamic OpenMP thread scheduling alters the exact sequence in which partial weights "
    "arrive at each histogram bin. Changing the grouping order of numbers with different magnitudes shifts "
    "rounding errors in the least significant bits, yielding slightly different final sums across runs. "
    "Using thread-private 2D buffers followed by a fixed sequential reduction enforces a strict accumulation "
    "tree and guarantees bit-identical results across thread counts."
)
print("-" * 70)


# ==============================================================================
# TASK 2: ELIMINATING CACHE-LINE FALSE SHARING IN OPENMP
# ==============================================================================
@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid] ^= (x >> 17)   # Writes directly to shared cache line every iteration!
    return state

@njit(parallel=True)
def padded_state_omp(iters, seed_val, stride):
    n_threads = get_num_threads()
    state = np.zeros((n_threads, stride), dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid, 0] ^= (x >> 17)  # Isolated in separate 64-byte cache lines
    return state

# Warm-up compilation
set_num_threads(1)
_ = unpadded_state_omp(100, STUDENT_ID)
_ = padded_state_omp(100, STUDENT_ID, PAD_STRIDE)

unpadded_times = {}
padded_times = {}
t2_state = None

for tc in THREAD_COUNTS:
    set_num_threads(tc)
    
    t0 = time.perf_counter()
    u_res = unpadded_state_omp(T2_ITERS, STUDENT_ID)
    unpadded_times[tc] = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_res = padded_state_omp(T2_ITERS, STUDENT_ID, PAD_STRIDE)
    padded_times[tc] = time.perf_counter() - t0

    assert np.array_equal(u_res, p_res[:, 0]), f"State mismatch at {tc} threads!"
    if tc == MAX_LAB_THREADS:
        t2_state = p_res[:, 0].copy()

speedup_t2 = unpadded_times[MAX_LAB_THREADS] / padded_times[MAX_LAB_THREADS]
print(f"[Task 2] Unpadded times (s): { {k: round(v, 4) for k, v in unpadded_times.items()} }")
print(f"[Task 2] Padded times (s):   { {k: round(v, 4) for k, v in padded_times.items()} }")
print(f"[Task 2] Speedup at {MAX_LAB_THREADS} threads: {speedup_t2:.2f}x")

DIAGNOSIS_T2 = (
    "In the unpadded kernel, adjacent 8-byte int64 slots reside within the same 64-byte L1/L2 cache line. "
    "Under the MESI cache coherence protocol, each core executing a store must acquire Exclusive or Modified "
    "ownership of that cache line, broadcast an invalidation signal to all peer cores, and flush dirty data "
    "across the interconnect. This cache-line ping-pong serializes memory writes and degrades wall-clock scaling "
    "as more threads contend for the line. Stride-padding each thread's state by at least 64 bytes places every "
    "worker's active slot on a distinct cache line, eliminating coherence invalidations."
)
print("-" * 70)


# ==============================================================================
# TASK 3: IRREGULAR WORKLOAD SKEW & DYNAMIC CHUNK-STEALING SCHEDULER
# ==============================================================================
work_units = np.array([40 + ((r * 7919 + STUDENT_ID) % SKEW_MOD) ** 2 for r in range(T3_ROWS)], dtype=np.int64)

@njit(parallel=True)
def static_skew_omp(w_units, stride):
    n_rows = w_units.shape[0]
    n_threads = get_num_threads()
    out = np.zeros(n_rows, dtype=np.float64)
    ops_padded = np.zeros((n_threads, stride), dtype=np.int64)

    for r in prange(n_rows):
        tid = get_thread_id()
        steps = w_units[r]
        acc = 0.0
        for k in range(steps):
            x = 0.001 * (k + 1) + 0.0001 * r
            acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
        out[r] = acc
        ops_padded[tid, 0] += steps

    return out, ops_padded[:, 0]

@njit(parallel=True)
def balanced_skew_omp(w_units, stride):
    n_rows = w_units.shape[0]
    n_threads = get_num_threads()
    out = np.zeros(n_rows, dtype=np.float64)
    ops_padded = np.zeros((n_threads, stride), dtype=np.int64)

    # Stride-interleaved worker pool eliminates contiguous skew clusters without lock contention
    for worker in prange(n_threads):
        tid = get_thread_id()
        local_ops = 0
        for r in range(worker, n_rows, n_threads):
            steps = w_units[r]
            acc = 0.0
            for k in range(steps):
                x = 0.001 * (k + 1) + 0.0001 * r
                acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
            out[r] = acc
            local_ops += steps
        ops_padded[tid, 0] = local_ops

    return out, ops_padded[:, 0]

# Warm-up
set_num_threads(MAX_LAB_THREADS)
_ = static_skew_omp(work_units[:64], PAD_STRIDE)
_ = balanced_skew_omp(work_units[:64], PAD_STRIDE)

t0 = time.perf_counter()
out_static, ops_static = static_skew_omp(work_units, PAD_STRIDE)
time_static = time.perf_counter() - t0

t0 = time.perf_counter()
out_balanced, ops_balanced = balanced_skew_omp(work_units, PAD_STRIDE)
time_balanced = time.perf_counter() - t0

imbalance_static = float(np.max(ops_static) / np.min(ops_static[ops_static > 0]))
imbalance_balanced = float(np.max(ops_balanced) / np.min(ops_balanced[ops_balanced > 0]))
max_err_t3 = float(np.max(np.abs(out_static - out_balanced)))

assert max_err_t3 < 1e-11, f"Task 3 precision error too high: {max_err_t3}"
assert imbalance_balanced < 1.15, f"Task 3 imbalance too high: {imbalance_balanced}"

t3_out = out_balanced.copy()
print(f"[Task 3] Static time: {time_static:.4f}s | Imbalance ratio: {imbalance_static:.3f}x")
print(f"[Task 3] Balanced time: {time_balanced:.4f}s | Imbalance ratio: {imbalance_balanced:.3f}x")

DIAGNOSIS_T3 = (
    "Default prange scheduling partitions loop iterations into large contiguous blocks, causing threads "
    "assigned to dense clusters of high-trip-count rows to become stragglers while other cores sit idle "
    "at the implicit barrier. By explicitly launching n_threads OpenMP workers and assigning rows via a "
    "deterministic cyclic stride-interleaved distribution (r = worker, worker + n_threads, ...), heavy "
    "and light rows are evenly dispersed across every core with zero atomic cursor contention. This keeps "
    "total operations balanced across threads and drives the load imbalance ratio well below 1.15x."
)
print("-" * 70)


# ==============================================================================
# TASK 4: PARALLEL RED-BLACK SOR WAVEFRONT PDE SOLVER
# ==============================================================================
def init_grid():
    U = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    U[0, :]  = 25.0 * np.cos(np.linspace(0, np.pi / 2, GRID_W))
    U[-1, :] = -50.0
    U[:, 0]  = 100.0
    U[:, -1] = 75.0 * np.sin(np.linspace(0, np.pi, GRID_H))
    return U

@njit(parallel=True)
def sor_red_black_step(U, omega, stride):
    H, W = U.shape
    n_threads = get_num_threads()
    max_deltas = np.zeros((n_threads, stride), dtype=np.float64)

    # Red Sweep: (r + c) % 2 == 0
    for r in prange(1, H - 1):
        tid = get_thread_id()
        local_max = 0.0
        c_start = 1 if (r + 1) % 2 == 0 else 2
        for c in range(c_start, W - 1, 2):
            old_val = U[r, c]
            new_val = (1.0 - omega) * old_val + 0.25 * omega * (
                U[r - 1, c] + U[r + 1, c] + U[r, c - 1] + U[r, c + 1]
            )
            U[r, c] = new_val
            diff = abs(new_val - old_val)
            if diff > local_max:
                local_max = diff
        if local_max > max_deltas[tid, 0]:
            max_deltas[tid, 0] = local_max

    # Black Sweep: (r + c) % 2 == 1 (implicit OpenMP barrier separates sweeps)
    for r in prange(1, H - 1):
        tid = get_thread_id()
        local_max = max_deltas[tid, 0]
        c_start = 1 if (r + 1) % 2 == 1 else 2
        for c in range(c_start, W - 1, 2):
            old_val = U[r, c]
            new_val = (1.0 - omega) * old_val + 0.25 * omega * (
                U[r - 1, c] + U[r + 1, c] + U[r, c - 1] + U[r, c + 1]
            )
            U[r, c] = new_val
            diff = abs(new_val - old_val)
            if diff > local_max:
                local_max = diff
        max_deltas[tid, 0] = local_max

    step_max = 0.0
    for t in range(n_threads):
        if max_deltas[t, 0] > step_max:
            step_max = max_deltas[t, 0]
    return step_max

def solve_pde_omp(sweeps=600):
    U = init_grid()
    final_delta = 0.0
    for _ in range(sweeps):
        final_delta = sor_red_black_step(U, OMEGA_SOR, PAD_STRIDE)
    return U, final_delta

# Warm-up
set_num_threads(1)
_ = solve_pde_omp(2)

pde_times = {}
ref_grid = None
t4_grid = None

for tc in THREAD_COUNTS:
    set_num_threads(tc)
    t0 = time.perf_counter()
    grid_res, res_delta = solve_pde_omp(600)
    pde_times[tc] = time.perf_counter() - t0

    if tc == 1:
        ref_grid = grid_res.copy()
    else:
        max_diff = float(np.max(np.abs(grid_res - ref_grid)))
        assert max_diff < 1e-9, f"SOR divergence at {tc} threads: {max_diff}"

    if tc == MAX_LAB_THREADS:
        t4_grid = grid_res.copy()

print(f"[Task 4] SOR 600-sweep times (s): { {k: round(v, 4) for k, v in pde_times.items()} }")
print(f"[Task 4] Speedup at {MAX_LAB_THREADS} threads: {pde_times[1] / pde_times[MAX_LAB_THREADS]:.2f}x")
print("-" * 70)


# ==============================================================================
# FINAL SECTION: DETERMINISTIC VERIFICATION & TELEMETRY BUNDLE
# ==============================================================================
set_num_threads(8 if (os.cpu_count() or 1) >= 8 else os.cpu_count())

for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
    assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"

checksum_blob = (
    f"{STUDENT_ID}|"
    f"{int(np.sum(t1_hist))}|{float(np.sum(t1_wsums)):.4f}|"
    f"{int(np.bitwise_xor.reduce(t2_state))}|"
    f"{float(np.sum(t3_out)):.6f}|"
    f"{float(np.sum(t4_grid)):.6f}"
)
lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()

bundle = {
    "instructor": "Instructor - Sufyan bin Uzayr",
    "student_name": STUDENT_NAME,
    "student_id": STUDENT_ID,
    "omp_layer": nb.threading_layer(),
    "cpu_cores": os.cpu_count(),
    "verification_digest": lab_digest,
}
print("=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
print(json.dumps(bundle, indent=2))