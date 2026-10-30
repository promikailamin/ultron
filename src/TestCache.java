package ultron;

import java.util.Random;

/** Verifies incremental KV-cached inference matches full forward pass. */
public class TestCache {
    public static void main(String[] args) throws Exception {
        Tokenizer tok;
        try (java.io.InputStream is = Resources.open("ultron-vocab.txt", "data/ultron-vocab.txt")) {
            tok = Tokenizer.load(is);
        }
        Ultron m;
        try (java.io.InputStream is = Resources.open("ultron.bin", "out/ultron.bin")) {
            m = Ultron.load(is);
        }
        Random r = new Random(9);

        int S = m.T;
        int[] ids = new int[S];
        for (int i = 0; i < S; i++) ids[i] = r.nextInt(m.V);

        Ultron.Act act = new Ultron.Act(m.T, m.D, m.F, m.L, m.H, m.V);
        float[] full = m.forwardLast(ids, S, act);

        Ultron.KVCache c = m.newCache();
        float[] inc = null;
        for (int i = 0; i < S; i++) inc = m.step(ids[i], c);

        double maxDiff = 0;
        int worst = 0;
        for (int v = 0; v < m.V; v++) {
            double diff = Math.abs(full[v] - inc[v]);
            if (diff > maxDiff) { maxDiff = diff; worst = v; }
        }
        System.out.printf("max |full - incremental| = %.6f (at token %d: %.4f vs %.4f)%n",
                maxDiff, worst, full[worst], inc[worst]);

        // window-shift behavior: feed 2*T tokens, ensure no crash and len stays <= T
        c.clear();
        for (int i = 0; i < 2 * m.T; i++) m.step(r.nextInt(m.V), c);
        System.out.printf("after %d tokens: cache len=%d (max %d)%n", 2 * m.T, c.len, m.T);

        if (maxDiff < 5e-3 && c.len <= m.T) System.out.println("CACHE TEST PASSED");
        else System.out.println("CACHE TEST FAILED");
    }
}
