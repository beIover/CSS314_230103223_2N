import java.util.concurrent.ThreadLocalRandom;

public class MonteCarloPiBenchmarkpart2 {

    // Defined based on the 50,000,000 iterations mentioned in your assignment
    static final long PART1_ITERATIONS = 50_000_000;
    
    // =========================================================
    // PART 2
    // Synchronization
    // =========================================================

    static long synchronizedHits = 0;

    // This method is synchronized.
    // Only one thread can execute this method at a time.
    static synchronized void incrementSynchronizedHits() {
        synchronizedHits++;
    }

    // ---------------------------------------------------------
    // Worker for the synchronized 4-thread version
    // ---------------------------------------------------------
    static class SyncWorker extends Thread {

        private final long iterations;

        SyncWorker(long iterations) {
            this.iterations = iterations;
        }

        @Override
        public void run() {
            for (long i = 0; i < iterations; i++) {
                // Generate a random point inside 1 x 1 square
                double x = ThreadLocalRandom.current().nextDouble();
                double y = ThreadLocalRandom.current().nextDouble();

                // Check whether point is inside quarter of a circle
                if (x * x + y * y <= 1.0) {
                    // Safe because the method is synchronized
                    incrementSynchronizedHits();
                }
            }
        }
    }

    // ---------------------------------------------------------
    // Worker for the single-threaded version
    // ---------------------------------------------------------
    static class SingleWorker extends Thread {

        private final long iterations;

        SingleWorker(long iterations) {
            this.iterations = iterations;
        }

        @Override
        public void run() {
            // Local variable.
            // No synchronization is needed.
            long localHits = 0;

            for (long i = 0; i < iterations; i++) {
                // Generate random point
                double x = ThreadLocalRandom.current().nextDouble();
                double y = ThreadLocalRandom.current().nextDouble();

                // Check whether point is inside quarter circle
                if (x * x + y * y <= 1.0) {
                    localHits++;
                }
            }

            // Save final result
            synchronizedHits = localHits;
        }
    }

    // ---------------------------------------------------------
    // Run the single-threaded version
    // ---------------------------------------------------------
    static double runPart2SingleThread() throws InterruptedException {
        synchronizedHits = 0;

        // Start timer
        long start = System.nanoTime();

        // Create one thread
        Thread thread = new SingleWorker(PART1_ITERATIONS);

        // Start thread
        thread.start();

        // Wait until it finishes
        thread.join();

        // Stop timer
        long end = System.nanoTime();

        // Calculate pi
        double pi = 4.0 * synchronizedHits / PART1_ITERATIONS;

        // Convert nanoseconds to milliseconds
        double timeMs = (end - start) / 1_000_000.0;

        System.out.printf("Part 2 - 1 thread: pi = %.10f, time = %.2f ms%n", pi, timeMs);

        return timeMs;
    }

    // ---------------------------------------------------------
    // Run the synchronized 4-thread version
    // ---------------------------------------------------------
    static double runPart2Synchronized() throws InterruptedException {
        synchronizedHits = 0;

        // Divide work between 4 threads
        long iterationsPerThread = PART1_ITERATIONS / 4;
        Thread[] threads = new Thread[4];

        // Start timer
        long start = System.nanoTime();

        // Create and start 4 threads
        for (int i = 0; i < 4; i++) {
            threads[i] = new SyncWorker(iterationsPerThread);
            threads[i].start();
        }

        // Wait for all 4 threads
        for (Thread thread : threads) {
            thread.join();
        }

        // Stop timer
        long end = System.nanoTime();

        // Calculate pi
        double pi = 4.0 * synchronizedHits / PART1_ITERATIONS;

        // Convert to milliseconds
        double timeMs = (end - start) / 1_000_000.0;

        System.out.printf("Part 2 - synchronized 4 threads: pi = %.10f, time = %.2f ms%n", pi, timeMs);

        return timeMs;
    }

    // ---------------------------------------------------------
    // Main Method to trigger the runs
    // ---------------------------------------------------------
    public static void main(String[] args) throws InterruptedException {
        System.out.println("PART 2: SYNCHRONIZATION");
        System.out.println("------------------------------------------");
        
        double timeSingle = runPart2SingleThread();
        double timeSync = runPart2Synchronized();
        
        System.out.printf("Synchronized / Single-thread runtime: %.2fx%n", (timeSync / timeSingle));
    }
}