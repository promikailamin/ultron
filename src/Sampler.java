package ultron;

import java.util.Arrays;
import java.util.Random;

/**
 * Allocation-free token sampler for realtime generation on low-end devices.
 * Reuses internal buffers; one Sampler per generation thread.
 *
 * Supports temperature, top-k, top-p (nucleus) sampling and a repetition
 * penalty over the tokens generated so far.
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
        return sample(logits, vocab, temperature, topK, 1.0f, null, 0, 0, 1.0f);
    }

    /**
     * Sample one token.
     *
     * @param topP        nucleus probability (1.0 disables top-p)
     * @param recent      generated token ids used for the repetition penalty
     * @param recentFrom  first index in recent
     * @param recentCount number of recent tokens to inspect
     * @param repPenalty  > 1 penalises repeated tokens; 1.0 disables it
     */
    public int sample(float[] logits, int vocab, float temperature, int topK, float topP,
                      int[] recent, int recentFrom, int recentCount, float repPenalty) {
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

        // Repetition penalty: lower the probability of tokens already produced.
        if (repPenalty > 1.0f && recent != null && recentCount > 0) {
            int from = Math.max(0, recentFrom + recentCount - 64);
            for (int i = from; i < recentFrom + recentCount; i++) {
                int id = recent[i];
                if (id < 0 || id >= vocab) continue;
                if (sc[id] > 0) sc[id] /= repPenalty;
                else sc[id] *= repPenalty;
                if (sc[id] > max) max = sc[id];
            }
        }

        // Convert to probabilities.
        double sum = 0;
        for (int v = 0; v < vocab; v++) {
            double e = Math.exp(sc[v] - max);
            sc[v] = e;
            sum += e;
        }

        // Top-k first, then top-p on the remaining mass. Both filters are
        // applied so the sample stays inside the high-probability head.
        boolean filtered = false;
        if (topK > 0 && topK < vocab) {
            for (int v = 0; v < vocab; v++) sorted[v] = (float) sc[v];
            Arrays.sort(sorted, 0, vocab);
            float thresh = sorted[vocab - topK];
            for (int v = 0; v < vocab; v++) {
                if (sc[v] < thresh) {
                    sum -= sc[v];
                    sc[v] = 0;
                    filtered = true;
                }
            }
        }
        if (topP > 0.01f && topP < 1.0f) {
            for (int v = 0; v < vocab; v++) sorted[v] = (float) sc[v];
            Arrays.sort(sorted, 0, vocab);
            double target = sum * topP;
            double acc = 0;
            double cutoff = Double.MAX_VALUE;
            for (int i = vocab - 1; i >= 0; i--) {
                double p = sorted[i];
                if (p <= 0) break;
                acc += p;
                cutoff = p;
                if (acc >= target) break;
            }
            for (int v = 0; v < vocab; v++) {
                if (sc[v] < cutoff) {
                    sum -= sc[v];
                    sc[v] = 0;
                    filtered = true;
                }
            }
        }

        if (filtered && sum <= 0) {
            // Defensive fallback: never crash on an empty nucleus.
            return vocab - 1;
        }
        double u = rnd.nextDouble() * sum, acc = 0;
        for (int v = 0; v < vocab; v++) {
            acc += sc[v];
            if (acc >= u) return v;
        }
        return vocab - 1;
    }
}
