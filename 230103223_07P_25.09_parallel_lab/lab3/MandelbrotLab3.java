import java.util.concurrent.*;
import java.util.ArrayList;
import java.util.List;

public class MandelbrotLab3 {
    private static final int WIDTH = 800;
    private static final int HEIGHT = 800;
    private static final int MAX_ITER = 500;

    public static void main(String[] args) {
        int threads = args.length > 0 ? Integer.parseInt(args[0]) : 4;
        int chunkSize = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        String mode = args.length > 2 ? args[2] : "dynamic";

        long startTime = System.nanoTime();
        ForkJoinPool pool = new ForkJoinPool(threads);

        try {
            if ("static".equalsIgnoreCase(mode)) {
                // Статическое распределение блоков
                int rowsPerThread = HEIGHT / threads;
                pool.submit(() -> {
                    java.util.stream.IntStream.range(0, threads).parallel().forEach(t -> {
                        int startRow = t * rowsPerThread;
                        int endRow = (t == threads - 1) ? HEIGHT : startRow + rowsPerThread;
                        for (int r = startRow; r < endRow; r++) {
                            computeRow(r);
                        }
                    });
                }).join();
            } else {
                // Динамическое распределение (Dynamic / Work-Stealing)
                List<RecursiveAction> tasks = new ArrayList<>();
                for (int r = 0; r < HEIGHT; r += chunkSize) {
                    int startRow = r;
                    int endRow = Math.min(HEIGHT, r + chunkSize);
                    tasks.add(new RecursiveAction() {
                        @Override
                        protected void compute() {
                            for (int row = startRow; row < endRow; row++) {
                                computeRow(row);
                            }
                        }
                    });
                }
                pool.submit(() -> ForkJoinTask.invokeAll(tasks)).join();
            }
        } finally {
            pool.shutdown();
        }

        long endTime = System.nanoTime();
        double timeMs = (endTime - startTime) / 1_000_000.0;
        System.out.printf("%.2f", timeMs);
    }

    private static void computeRow(int row) {
        double c_im = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;
        for (int col = 0; col < WIDTH; col++) {
            double c_re = (col - WIDTH / 2.0) * 4.0 / WIDTH;
            double x = 0, y = 0;
            int iter = 0;
            while (x * x + y * y <= 4 && iter < MAX_ITER) {
                double x_new = x * x - y * y + c_re;
                y = 2 * x * y + c_im;
                x = x_new;
                iter++;
            }
        }
    }
}
