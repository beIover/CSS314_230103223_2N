import array
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import emit_verify_tag, run_benchmark
from task3a_roofline import STUDENT_ID, detect_physical_cores, run_roofline_characterization

N = 384
MAT_ELEMS = N * N
MAT_BYTES = MAT_ELEMS * 8  # double-precision (float64)
TOTAL_GEMM_FLOPS = 2.0 * (N**3)
TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def make_row_ranges(n_rows: int, p_phys: int) -> list[tuple[int, int]]:
  base = n_rows // p_phys
  rem = n_rows % p_phys
  ranges = []
  cur = 0
  for t in range(p_phys):
    size = base + (1 if t < rem else 0)
    ranges.append((cur, cur + size))
    cur += size
  return ranges


def gemm_v1_worker(args: tuple[str, str, str, int, int, int]) -> None:
  """Variant 1: Naive i-j-k parallel GEMM (stride-N traversal on B)."""
  shm_a_name, shm_b_name, shm_c_name, r_start, r_end, n = args
  shm_a = shared_memory.SharedMemory(name=shm_a_name)
  shm_b = shared_memory.SharedMemory(name=shm_b_name)
  shm_c = shared_memory.SharedMemory(name=shm_c_name)
  A = memoryview(shm_a.buf).cast("d")
  B = memoryview(shm_b.buf).cast("d")
  C = memoryview(shm_c.buf).cast("d")
  try:
    for i in range(r_start, r_end):
      for j in range(n):
        dot = 0.0
        for k in range(n):
          dot += A[i * n + k] * B[k * n + j]
        C[i * n + j] = dot
  finally:
    A.release()
    B.release()
    C.release()
    shm_a.close()
    shm_b.close()
    shm_c.close()


def gemm_v2_worker(args: tuple[str, str, str, int, int, int]) -> None:
  """Variant 2: Loop-reordered i-k-j parallel GEMM (unit-stride B & C)."""
  shm_a_name, shm_b_name, shm_c_name, r_start, r_end, n = args
  shm_a = shared_memory.SharedMemory(name=shm_a_name)
  shm_b = shared_memory.SharedMemory(name=shm_b_name)
  shm_c = shared_memory.SharedMemory(name=shm_c_name)
  A = memoryview(shm_a.buf).cast("d")
  B = memoryview(shm_b.buf).cast("d")
  C = memoryview(shm_c.buf).cast("d")
  try:
    for i in range(r_start, r_end):
      i_off = i * n
      for k in range(n):
        a_ik = A[i_off + k]
        k_off = k * n
        for j in range(n):
          C[i_off + j] += a_ik * B[k_off + j]
  finally:
    A.release()
    B.release()
    C.release()
    shm_a.close()
    shm_b.close()
    shm_c.close()


def gemm_v3_worker(args: tuple[str, str, str, int, int, int, int]) -> None:
  """Variant 3: 6-loop cache-blocked (i,k,j) GEMM with tile slice hoisting."""
  shm_a_name, shm_b_name, shm_c_name, r_start, r_end, n, bs = args
  shm_a = shared_memory.SharedMemory(name=shm_a_name)
  shm_b = shared_memory.SharedMemory(name=shm_b_name)
  shm_c = shared_memory.SharedMemory(name=shm_c_name)
  A = memoryview(shm_a.buf).cast("d")
  B = memoryview(shm_b.buf).cast("d")
  C = memoryview(shm_c.buf).cast("d")
  try:
    for ii in range(r_start, r_end, bs):
      i_end = ii + bs if ii + bs < r_end else r_end
      for kk in range(0, n, bs):
        k_end = kk + bs if kk + bs < n else n
        for jj in range(0, n, bs):
          j_end = jj + bs if jj + bs < n else n
          tile_len = j_end - jj
          # Hoist B tile rows once per (kk, jj) block
          b_rows = [
              B[k * n + jj : k * n + j_end] for k in range(kk, k_end)
          ]
          for i in range(ii, i_end):
            i_off = i * n
            c_tile = C[i_off + jj : i_off + j_end]
            for idx_k, k in enumerate(range(kk, k_end)):
              a_ik = A[i_off + k]
              b_tile = b_rows[idx_k]
              for j in range(tile_len):
                c_tile[j] += a_ik * b_tile[j]
            c_tile.release()
          for b_row in b_rows:
            b_row.release()
  finally:
    A.release()
    B.release()
    C.release()
    shm_a.close()
    shm_b.close()
    shm_c.close()


def main() -> None:
  p_phys = detect_physical_cores()
  bw_gbps, p_gflops, ridge_point = run_roofline_characterization(p_phys)

  shm_a = shared_memory.SharedMemory(create=True, size=MAT_BYTES)
  shm_b = shared_memory.SharedMemory(create=True, size=MAT_BYTES)
  shm_c = shared_memory.SharedMemory(create=True, size=MAT_BYTES)
  zero_bytes = b"\x00" * MAT_BYTES

  try:
    # Deterministic matrix initialization
    init_a = array.array(
        "d", [((i % 97) - 48) * 0.01 for i in range(MAT_ELEMS)]
    )
    init_b = array.array(
        "d", [((i % 89) - 44) * 0.01 for i in range(MAT_ELEMS)]
    )
    shm_a.buf[:] = init_a.tobytes()
    shm_b.buf[:] = init_b.tobytes()

    row_ranges = make_row_ranges(N, p_phys)

    with mp.Pool(processes=p_phys) as pool:
      # -------------------------------------------------------
      # Variant 1: Naive i-j-k Parallel GEMM
      # -------------------------------------------------------
      v1_args = [
          (shm_a.name, shm_b.name, shm_c.name, rs, re, N)
          for rs, re in row_ranges
      ]

      def trial_v1() -> None:
        shm_c.buf[:] = zero_bytes
        pool.map(gemm_v1_worker, v1_args)

      stats_v1 = run_benchmark(trial_v1, warmup_runs=1, timed_runs=5)
      gflops_v1 = TOTAL_GEMM_FLOPS / ((stats_v1.median_ms / 1e3) * 1e9)

      # -------------------------------------------------------
      # Variant 2: Loop-Reordered i-k-j Parallel GEMM
      # -------------------------------------------------------
      v2_args = [
          (shm_a.name, shm_b.name, shm_c.name, rs, re, N)
          for rs, re in row_ranges
      ]

      def trial_v2() -> None:
        shm_c.buf[:] = zero_bytes
        pool.map(gemm_v2_worker, v2_args)

      stats_v2 = run_benchmark(trial_v2, warmup_runs=2, timed_runs=7)
      gflops_v2 = TOTAL_GEMM_FLOPS / ((stats_v2.median_ms / 1e3) * 1e9)

      # Save Variant 2 reference output for numerical verification
      ref_c = array.array("d")
      ref_c.frombytes(bytes(shm_c.buf))

      # -------------------------------------------------------
      # Variant 3: 6-Loop Blocked GEMM across 10 Tile Sizes
      # -------------------------------------------------------
      print(f"=== PART 3B: Parallel GEMM Optimization Sweep (N = {N}, P_phys = {p_phys}) ===")
      print(
          f"{'Variant / BS':>14} | {'W_tile (KB)':>11} | {'Median (ms)':>11} |"
          f" {'IQR (ms)':>9} | {'GFLOP/s':>10} | {'Max Err':>10}"
      )
      print("-" * 79)
      print(
          f"{'V1 (i-j-k)':>14} | {'N/A':>11} | {stats_v1.median_ms:>11.2f} |"
          f" {stats_v1.iqr_ms:>9.2f} | {gflops_v1:>10.4f} | {'N/A':>10}"
      )
      print(
          f"{'V2 (i-k-j)':>14} | {'N/A':>11} | {stats_v2.median_ms:>11.2f} |"
          f" {stats_v2.iqr_ms:>9.2f} | {gflops_v2:>10.4f} | {'0.00e+00':>10}"
      )

      best_bs = TILE_SIZES[0]
      best_gflops_v3 = -1.0

      for bs in TILE_SIZES:
        v3_args = [
            (shm_a.name, shm_b.name, shm_c.name, rs, re, N, bs)
            for rs, re in row_ranges
        ]

        def trial_v3() -> None:
          shm_c.buf[:] = zero_bytes
          pool.map(gemm_v3_worker, v3_args)

        stats_v3 = run_benchmark(trial_v3, warmup_runs=2, timed_runs=7)
        gflops_v3 = TOTAL_GEMM_FLOPS / ((stats_v3.median_ms / 1e3) * 1e9)

        # Verify numerical equivalence against Variant 2
        test_c = array.array("d")
        test_c.frombytes(bytes(shm_c.buf))
        max_err = max(abs(a - b) for a, b in zip(test_c, ref_c))
        assert max_err < 1e-9, f"Numerical mismatch at BS={bs}: {max_err}"

        w_tile_kb = (3 * bs * bs * 8) / 1024.0
        print(
            f"{f'V3 (BS={bs})':>14} | {w_tile_kb:>11.2f} |"
            f" {stats_v3.median_ms:>11.2f} | {stats_v3.iqr_ms:>9.2f} |"
            f" {gflops_v3:>10.4f} | {max_err:>10.2e}"
        )

        if gflops_v3 > best_gflops_v3:
          best_gflops_v3 = gflops_v3
          best_bs = bs

    # -------------------------------------------------------
    # Part 3C: Roofline Efficiency Mapping
    # -------------------------------------------------------
    w_opt_bytes = 3 * best_bs * best_bs * 8
    i_naive = 0.125
    i_reordered = 2.0 / 24.0  # 1 load B + 1 load/store C = 24 B per 2 FLOPs
    i_tiled = best_bs / 12.0

    ceil_v1 = min(p_gflops, i_naive * bw_gbps)
    ceil_v2 = min(p_gflops, i_reordered * bw_gbps)
    ceil_v3 = min(p_gflops, i_tiled * bw_gbps)

    print("\n=== PART 3C: Roofline & Cache Capacity Analysis ===")
    print(f"Optimal Tile Size BS*               : {best_bs}")
    print(
        f"Optimal Tile Working Set W_tile(BS*): {w_opt_bytes:,} B"
        f" ({w_opt_bytes / 1024.0:.2f} KB)"
    )
    print(
        f"Variant 1 (I={i_naive:.4f} F/B)        : {gflops_v1:.4f} /"
        f" {ceil_v1:.4f} GFLOP/s ({(gflops_v1 / ceil_v1) * 100:.2f}% of Roofline)"
    )
    print(
        f"Variant 2 (I={i_reordered:.4f} F/B)        : {gflops_v2:.4f} /"
        f" {ceil_v2:.4f} GFLOP/s ({(gflops_v2 / ceil_v2) * 100:.2f}% of Roofline)"
    )
    print(
        f"Variant 3 BS* (I={i_tiled:.4f} F/B)    : {best_gflops_v3:.4f} /"
        f" {ceil_v3:.4f} GFLOP/s"
        f" ({(best_gflops_v3 / ceil_v3) * 100:.2f}% of Roofline)"
    )

    print(
        "\n" + emit_verify_tag("PHASE3", STUDENT_ID, gflops_v1, best_gflops_v3)
    )
  finally:
    shm_a.close()
    shm_a.unlink()
    shm_b.close()
    shm_b.unlink()
    shm_c.close()
    shm_c.unlink()


if __name__ == "__main__":
  main()