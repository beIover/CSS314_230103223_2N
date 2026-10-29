import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

public class MonteCarloPiBenchmark {

    static final long PART1_ITERATIONS = 50_000_000L;
    static final long PART3_ITERATIONS = 100_000_000L;

    // =========================================================
    // PART 1
    // Shared variable with a DATA RACE
    // =========================================================

    static long totalHits = 0;

    static class RaceWorker extends Thread {
        private final long iterations;

        RaceWorker(long iterations) {
            this.iterations = iterations;
        }

        @Override
        public void run() {
            for (long i = 0; i < iterations; i++) {
                double x = ThreadLocalRandom.current().nextDouble();
                double y = ThreadLocalRandom.current().nextDouble();

                if (x * x + y * y <= 1.0) {
                    totalHits++;
                }
            }
        }
    }

    static double runPart1() throws InterruptedException {
        totalHits = 0;

        long iterationsPerThread = PART1_ITERATIONS / 4;

        Thread[] threads = new Thread[4];

        long start = System.nanoTime();

        for (int i = 0; i < 4; i++) {
            threads[i] = new RaceWorker(iterationsPerThread);
            threads[i].start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        long end = System.nanoTime();

        double pi = 4.0 * totalHits / PART1_ITERATIONS;

        System.out.printf(
                "Part 1: hits = %,d, pi = %.10f, time = %.2f ms%n",
                totalHits,
                pi,
                (end - start) / 1_000_000.0
        );

        return pi;
    }


    // =========================================================
    // PART 2
    // synchronized increment
    // =========================================================

    static long synchronizedHits = 0;

    static synchronized void incrementSynchronizedHits() {
        synchronizedHits++;
    }

    static class SyncWorker extends Thread {
        private final long iterations;

        SyncWorker(long iterations) {
            this.iterations = iterations;
        }

        @Override
        public void run() {
            for (long i = 0; i < iterations; i++) {
                double x = ThreadLocalRandom.current().nextDouble();
                double y = ThreadLocalRandom.current().nextDouble();

                if (x * x + y * y <= 1.0) {
                    incrementSynchronizedHits();
                }
            }
        }
    }

    static class SingleWorker extends Thread {
        private final long iterations;

        SingleWorker(long iterations) {
            this.iterations = iterations;
        }

        @Override
        public void run() {
            long localHits = 0;

            for (long i = 0; i < iterations; i++) {
                double x = ThreadLocalRandom.current().nextDouble();
                double y = ThreadLocalRandom.current().nextDouble();

                if (x * x + y * y <= 1.0) {
                    localHits++;
                }
            }

            synchronizedHits = localHits;
        }
    }

    static double runPart2SingleThread() throws InterruptedException {
        synchronizedHits = 0;

        long start = System.nanoTime();

        Thread thread = new SingleWorker(PART1_ITERATIONS);
        thread.start();
        thread.join();

        long end = System.nanoTime();

        double pi = 4.0 * synchronizedHits / PART1_ITERATIONS;
        double timeMs = (end - start) / 1_000_000.0;

        System.out.printf(
                "Part 2 - 1 thread: pi = %.10f, time = %.2f ms%n",
                pi,
                timeMs
        );

        return timeMs;
    }

    static double runPart2Synchronized() throws InterruptedException {
        synchronizedHits = 0;

        long iterationsPerThread = PART1_ITERATIONS / 4;

        Thread[] threads = new Thread[4];

        long start = System.nanoTime();

        for (int i = 0; i < 4; i++) {
            threads[i] = new SyncWorker(iterationsPerThread);
            threads[i].start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        long end = System.nanoTime();

        double pi = 4.0 * synchronizedHits / PART1_ITERATIONS;
        double timeMs = (end - start) / 1_000_000.0;

        System.out.printf(
                "Part 2 - synchronized 4 threads: pi = %.10f, time = %.2f ms%n",
                pi,
                timeMs
        );

        return timeMs;
    }


    // =========================================================
    // PART 3
    // Local counter / reduction
    // =========================================================

    static class ReductionWorker extends Thread {
        private final long iterations;
        private long localHits;

        ReductionWorker(long iterations) {
            this.iterations = iterations;
            this.localHits = 0;
        }

        @Override
        public void run() {
            for (long i = 0; i < iterations; i++) {
                double x = ThreadLocalRandom.current().nextDouble();
                double y = ThreadLocalRandom.current().nextDouble();

                if (x * x + y * y <= 1.0) {
                    localHits++;
                }
            }
        }

        long getLocalHits() {
            return localHits;
        }
    }

    static class BenchmarkResult {
        double pi;
        double timeMs;

        BenchmarkResult(double pi, double timeMs) {
            this.pi = pi;
            this.timeMs = timeMs;
        }
    }

    static BenchmarkResult runReduction(int threadCount)
            throws InterruptedException {

        Thread[] threads = new Thread[threadCount];

        long iterationsPerThread = PART3_ITERATIONS / threadCount;

        long start = System.nanoTime();

        for (int i = 0; i < threadCount; i++) {
            threads[i] = new ReductionWorker(iterationsPerThread);
            threads[i].start();
        }

        long total = 0;

        for (Thread thread : threads) {
            thread.join();

            ReductionWorker worker = (ReductionWorker) thread;
            total += worker.getLocalHits();
        }

        long end = System.nanoTime();

        double pi = 4.0 * total / PART3_ITERATIONS;
        double timeMs = (end - start) / 1_000_000.0;

        return new BenchmarkResult(pi, timeMs);
    }


    // =========================================================
    // MAIN
    // =========================================================

    public static void main(String[] args) throws Exception {

        System.out.println("==========================================");
        System.out.println("MONTE CARLO PI - JAVA THREAD BENCHMARK");
        System.out.println("==========================================");

        System.out.println();
        System.out.println("PART 1: DATA RACE");
        System.out.println("------------------------------------------");

        for (int i = 1; i <= 5; i++) {
            System.out.println("Run " + i);
            runPart1();
        }


        System.out.println();
        System.out.println("PART 2: SYNCHRONIZATION");
        System.out.println("------------------------------------------");

        double singleTime = runPart2SingleThread();
        double syncTime = runPart2Synchronized();

        System.out.printf(
                "Synchronized / Single-thread runtime: %.2fx%n",
                syncTime / singleTime
        );


        System.out.println();
        System.out.println("PART 3: LOCAL REDUCTION");
        System.out.println("------------------------------------------");

        int[] threadCounts = {1, 2, 4, 8, 16, 32};

        double baseline = 0;

        System.out.printf(
                "%-10s %-15s %-15s %-15s %-15s%n",
                "Threads",
                "Runtime(ms)",
                "Pi",
                "Speedup",
                "Efficiency"
        );

        for (int threadCount : threadCounts) {

            BenchmarkResult result = runReduction(threadCount);

            if (threadCount == 1) {
                baseline = result.timeMs;
            }

            double speedup = baseline / result.timeMs;
            double efficiency = speedup / threadCount * 100.0;

            System.out.printf(
                    "%-10d %-15.2f %-15.10f %-15.2fx %-14.2f%%%n",
                    threadCount,
                    result.timeMs,
                    result.pi,
                    speedup,
                    efficiency
            );
        }

        System.out.println();
        System.out.println("Benchmark finished.");
    }
}
