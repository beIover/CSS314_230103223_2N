// --- Java 17+: Fork-Join Team Emulation ---
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class ForkJoinLab1 {
    public static void main(String[] args) {
        int targetThreads = 4;
        System.out.println("Master thread starting. Spawning thread team of size: " + targetThreads);

        // Эмуляция #pragma omp parallel num_threads(targetThreads)
        ForkJoinPool customPool = new ForkJoinPool(targetThreads);

        try {
            customPool.submit(() -> {
                // Параллельный поток эмулирует распределение потоков в OpenMP
                IntStream.range(0, targetThreads).parallel().forEach(idx -> {
                    long osTid = Thread.currentThread().threadId();
                    String threadName = Thread.currentThread().getName();
                    boolean isMaster = (idx == 0);

                    // Небольшая задержка для наглядности планирования ОС
                    try {
                        Thread.sleep(10 * (idx % 3));
                    } catch (InterruptedException ignored) {}

                    System.out.printf("[%s] Logical Rank: %d | Worker Thread: %s (OS ID: %d)%n",
                            isMaster ? "Master" : "Worker", idx, threadName, osTid);
                });
            }).join(); // Аналог неявного барьера (implicit barrier)
        } finally {
            customPool.shutdown();
        }

        System.out.println("Parallel region closed. Execution returned to master thread.");
    }
}
