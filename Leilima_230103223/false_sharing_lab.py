import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "beIover"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

def detect_cache_line_size() -> int:
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.cachelinesize"],
            encoding="utf-8"
        )
        return int(out.strip())
    return 64

# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Binding
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, \
        "Set GIT_USERNAME to your real Git handle before running."
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"
    
    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity")
    if affinity_supported:
        try:
            os.sched_setaffinity(0, set(range(NUM_THREADS)))
        except Exception:
            affinity_supported = False

    return {
        "git_username": git_user.strip(),
        "os_platform": sys.platform,
        "cpu_signature": cpu_sig,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported
    }

# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel (In-Place Volatile Memory Stores)
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    val = uint64(seed)
    for i in range(iters):
        # Зависимость по данным вынуждает CPU делать реальную запись store i64 в память
        val = (val ^ (val << uint64(13))) + uint64(i)
        val = (val ^ (val >> uint64(7)))
        buffer[slot] = (buffer[slot] + (val & uint64(1)) + uint64(1)) ^ (val >> uint64(63))

# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    val = uint64(seed)
    acc = buffer[slot]
    for i in range(iters):
        val = (val ^ (val << uint64(13))) + uint64(i)
        val = (val ^ (val >> uint64(7)))
        acc = (acc + (val & uint64(1)) + uint64(1)) ^ (val >> uint64(63))
    buffer[slot] = acc

# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(kernel_fn, stride: int, base_seed: int):
    raw = np.zeros(NUM_THREADS * stride + 16, dtype=np.uint64)
    offset = (64 - (raw.ctypes.data % 64)) // 8
    if offset == 8:
        offset = 0
    buffer = raw[offset : offset + NUM_THREADS * stride]
    buffer[:] = 0

    barrier = threading.Barrier(NUM_THREADS)
    threads = []

    def thread_entry(tid):
        slot = tid * stride
        t_seed = np.uint64((base_seed ^ (tid * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)
        barrier.wait()
        kernel_fn(buffer, np.int64(slot), np.int64(ITERATIONS_PER_THREAD), t_seed)

    for tid in range(NUM_THREADS):
        t = threading.Thread(target=thread_entry, args=(tid,))
        threads.append(t)

    ru_start = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    t0 = time.perf_counter_ns()

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed_ns = time.perf_counter_ns() - t0
    ru_end = resource.getrusage(resource.RUSAGE_SELF) if resource else None

    vol_ctx = (ru_end.ru_nvcsw - ru_start.ru_nvcsw) if resource else 0
    invol_ctx = (ru_end.ru_nivcsw - ru_start.ru_nivcsw) if resource else 0

    final_values = [int(buffer[tid * stride]) for tid in range(NUM_THREADS)]
    return {
        "elapsed_ns": elapsed_ns,
        "ns_per_op": elapsed_ns / (NUM_THREADS * ITERATIONS_PER_THREAD),
        "voluntary_ctx_switches": vol_ctx,
        "involuntary_ctx_switches": invol_ctx,
        "final_values": final_values
    }

def extract_native_llvm(llvm_ir: str, func_name: str) -> str:
    """Извлекает нативную LLVM-функцию, исключая cpython-обёртку."""
    blocks = llvm_ir.split("\ndefine ")
    for b in blocks:
        first_line = b.splitlines()[0] if b else ""
        if func_name in first_line and "cpython" not in first_line:
            return b
    return ""

# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def audit_llvm_and_parity(ctx: dict):
    dummy = np.zeros(64, dtype=np.uint64)
    worker_strided_accumulate(dummy, np.int64(0), np.int64(10), np.uint64(123))
    worker_private_accumulate(dummy, np.int64(0), np.int64(10), np.uint64(123))

    llvm_strided = next(iter(worker_strided_accumulate.inspect_llvm().values()))
    llvm_private = next(iter(worker_private_accumulate.inspect_llvm().values()))

    native_strided = extract_native_llvm(llvm_strided, "worker_strided_accumulate")
    native_private = extract_native_llvm(llvm_private, "worker_private_accumulate")

    strided_i64_stores = native_strided.count("store i64")
    private_i64_stores = native_private.count("store i64")
    has_pyobject = ("PyObject" in native_strided) or ("PyObject" in native_private)

    strides = [1, 4, 8, 16]
    results = {}
    for s in strides:
        print(f"Running stride={s} ({s * 8} bytes)...")
        results[f"stride_{s}"] = run_stride_benchmark(worker_strided_accumulate, s, ctx["instance_seed"])

    print("Running privatized register baseline (stride=1)...")
    privatized_res = run_stride_benchmark(worker_private_accumulate, 1, ctx["instance_seed"])

    ref_vals = privatized_res["final_values"]
    parity_ok = all(results[f"stride_{s}"]["final_values"] == ref_vals for s in strides)
    llvm_ok = (strided_i64_stores > 0) and (private_i64_stores > 0) and (not has_pyobject)
    verification_passed = parity_ok and llvm_ok

    slowdown_1_vs_8 = results["stride_1"]["elapsed_ns"] / results["stride_8"]["elapsed_ns"]

    print("\n=== CACHE COHERENCY BENCHMARK SUMMARY ===")
    print(f"Git User: {ctx['git_username']} | OS: {ctx['os_platform']} | Line Size: {ctx['cache_line_size']} B")
    print(f"Instance Seed: {ctx['instance_seed']}")
    for s in strides:
        r = results[f"stride_{s}"]
        print(f"Stride {s:2d} ({s*8:3d}B): {r['elapsed_ns']/1e6:8.2f} ms | {r['ns_per_op']:.4f} ns/op | ctx(vol/invol): {r['voluntary_ctx_switches']}/{r['involuntary_ctx_switches']}")
    print(f"Privatized   : {privatized_res['elapsed_ns']/1e6:8.2f} ms | {privatized_res['ns_per_op']:.4f} ns/op")
    print(f"Slowdown Factor (Stride 1 / Stride 8): {slowdown_1_vs_8:.2f}x")
    print(f"LLVM IR i64 stores -> strided: {strided_i64_stores}, private: {private_i64_stores} | PyObject fallback: {has_pyobject}")
    print(f"Bit-exact State Parity: {parity_ok} | Verification Passed: {verification_passed}")

    manifest = {
        "git_username": ctx["git_username"],
        "system_context": ctx,
        "verification_passed": verification_passed,
        "parity_exact": parity_ok,
        "llvm_audit": {
            "strided_store_i64_count": strided_i64_stores,
            "private_store_i64_count": private_i64_stores,
            "pyobject_fallback": has_pyobject
        },
        "benchmarks": {
            **results,
            "privatized_baseline": privatized_res,
            "slowdown_stride1_vs_stride8": slowdown_1_vs_8
        }
    }

    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nSaved cache_lab_manifest.json successfully!")

if __name__ == "__main__":
    context = init_system_context(GIT_USERNAME)
    audit_llvm_and_parity(context)
