package ultron;

import java.util.Arrays;
import java.util.Random;

/**
 * Allocation-free token sampler for realtime generation on low-end devices.
 * Reuses internal buffers; one Sampler per generation thread.
 */
public class Sampler {
    private final double[] sc;
    private final float[] sorted;
    private final Random rnd;

    public Sampler(int vocab, long seed) {
        sc = new double[vocab];
        sorted = new float[vocab];
        rnd = new Random(seed);
    }

    /** temperature <= 0.01 = greedy. topK <= 0 = no cutoff. */
    public int sample(float[] logits, int vocab, float temperature, int topK) {
        if (temperature <= 0.01f) {
            int best = 0;
            for (int v = 1; v < vocab; v++) if (logits[v] > logits[best]) best = v;
            return best;
        }
        double max = -1e30;
        for (int v = 0; v < vocab; v++) {
            double s = logits[v] / temperature;
            sc[v] = s;
            if (s > max) max = s;
        }
        if (topK > 0 && topK < vocab) {
            System.arraycopy(logits, 0, sorted, 0, vocab);
            Arrays.sort(sorted, 0, vocab);
            float thresh = sorted[vocab - topK];
            for (int v = 0; v < vocab; v++) if (logits[v] < thresh) sc[v] = -1e30;
        }
        double sum = 0;
        for (int v = 0; v < vocab; v++) {
            double e = Math.exp(sc[v] - max);
            sc[v] = e;
            sum += e;
        }
        double u = rnd.nextDouble() * sum, acc = 0;
        for (int v = 0; v < vocab; v++) {
            acc += sc[v];
            if (acc >= u) return v;
        }
        return vocab - 1;
    }
}
