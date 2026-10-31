package ultron;

import java.util.Locale;

/**
 * Dumps raw logits from the Java forward pass so train_py/parity_test.py can
 * compare them with PyTorch. Output is one "position token logit" per line.
 */
public class LogitDump {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: ultron.LogitDump <prompt> [positions]");
            System.exit(2);
        }
        String prompt = args[0];
        int positions = args.length > 1 ? Integer.parseInt(args[1]) : 10;

        Tokenizer tok = Tokenizer.loadDefault();
        Ultron m;
        try (java.io.InputStream is = Resources.open("ultron.bin", "out/ultron.bin")) {
            m = Ultron.load(is);
        }
        if (m.V != tok.V) throw new RuntimeException("vocab/model mismatch: " + m.V + " vs " + tok.V);

        int[] ids = tok.encode(prompt);
        if (ids.length == 0) ids = new int[]{tok.stoi.get(Tokenizer.PAR)};
        if (ids.length > m.T) ids = java.util.Arrays.copyOfRange(ids, ids.length - m.T, ids.length);
        Ultron.Act act = new Ultron.Act(m.T, m.D, m.F, m.L, m.H, m.V);
        float[] logits = m.forwardAll(ids, ids.length, act);

        int n = Math.min(Math.max(1, positions), ids.length);
        System.out.printf(Locale.ROOT, "# config %d %d %d %d %d %d%n",
                m.V, m.D, m.L, m.F, m.T, m.H);
        System.out.printf(Locale.ROOT, "# tokens %d%n", ids.length);
        for (int t = 0; t < n; t++) {
            int off = t * m.V;
            for (int v = 0; v < m.V; v++) {
                System.out.printf(Locale.ROOT, "%d %d %.9g%n", t, v, logits[off + v]);
            }
        }
    }
}
