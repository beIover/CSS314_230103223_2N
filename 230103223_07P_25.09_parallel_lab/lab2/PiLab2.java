import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class PiLab2 {
    private static final long STEPS = 100_000_000L;
    private static final double STEP_SIZE = 1.0 / STEPS;

    public static void main(String[] args) {
        String mode = args.length > 0 ? args[0] : "reduction";
        int threads = args.length > 1 ? Integer.parseInt(args[1]) : 4;

        switch (mode) {
            case "race" -> runRace(threads);
            case "critical" -> runCritical(threads);
            case "reduction" -> runReduction(threads);
            default -> System.out.println("Unknown mode");
        }
    }

    // Variant A: Naive Race Condition
    private static void runRace(int threads) {
        double[] sharedSum = new double[1];
        long startTime = System.nanoTime();
        ForkJoinPool pool = new ForkJoinPool(threads);

        try {
            pool.submit(() -> {
                IntStream.range(0, threads).parallel().forEach(t -> {
                    long chunk = STEPS / threads;
                    long start = t * chunk;
                    long end = (t == threads - 1) ? STEPS : start + chunk;

                    for (long i = start; i < end; i++) {
                        double x = (i + 0.5) * STEP_SIZE;
                        // Race condition: гонка потоков
                        sharedSum[0] += 4.0 / (1.0 + x * x);
                    }
                });
            }).join();
        } finally {
            pool.shutdown();
        }

        long endTime = System.nanoTime();
        double pi = sharedSum[0] * STEP_SIZE;
        double timeMs = (endTime - startTime) / 1_000_000.0;
        System.out.printf("RACE,P=%d,TimeMs=%.2f,Pi=%.10f,Err=%.10f%n",
                threads, timeMs, pi, Math.abs(pi - Math.PI));
    }

    // Variant B: Critical Section
    private static void runCritical(int threads) {
        double[] sharedSum = new double[1];
        Object lock = new Object();
        long startTime = System.nanoTime();
        ForkJoinPool pool = new ForkJoinPool(threads);

        try {
            pool.submit(() -> {
                IntStream.range(0, threads).parallel().forEach(t -> {
                    long chunk = STEPS / threads;
                    long start = t * chunk;
                    long end = (t == threads - 1) ? STEPS : start + chunk;

                    for (long i = start; i < end; i++) {
                        double x = (i + 0.5) * STEP_SIZE;
                        double term = 4.0 / (1.0 + x * x);
                        synchronized (lock) {
                            sharedSum[0] += term;
                        }
                    }
                });
            }).join();
        } finally {
            pool.shutdown();
        }

        long endTime = System.nanoTime();
        double pi = sharedSum[0] * STEP_SIZE;
        double timeMs = (endTime - startTime) / 1_000_000.0;
        System.out.printf("CRITICAL,P=%d,TimeMs=%.2f,Pi=%.10f,Err=%.10f%n",
                threads, timeMs, pi, Math.abs(pi - Math.PI));
    }

    // Variant C: Parallel Reduction
    private static void runReduction(int threads) {
        long startTime = System.nanoTime();
        ForkJoinPool pool = new ForkJoinPool(threads);
        double pi;

        try {
            double totalSum = pool.submit(() ->
                IntStream.range(0, threads).parallel().mapToDouble(t -> {
                    long chunk = STEPS / threads;
                    long start = t * chunk;
                    long end = (t == threads - 1) ? STEPS : start + chunk;
                    double localSum = 0;

                    for (long i = start; i < end; i++) {
                        double x = (i + 0.5) * STEP_SIZE;
                        localSum += 4.0 / (1.0 + x * x);
                    }
                    return localSum;
                }).sum()
            ).join();

            pi = totalSum * STEP_SIZE;
        } finally {
            pool.shutdown();
        }

        long endTime = System.nanoTime();
        double timeMs = (endTime - startTime) / 1_000_000.0;
        System.out.printf("%.2f", timeMs);
    }
}
