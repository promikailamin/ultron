package ultron;

import java.util.Random;

/** Finite-difference check for answer-only (masked) training. */
public class TestMask {
    public static void main(String[] args) {
        int V = 29, D = 6, L = 1, F = 12, T = 6, H = 2;
        Ultron m = new Ultron(V, D, L, F, T, H);
        m.init(4242);
        Random r = new Random(4242);

        int S = T;
        int[] ids = new int[S + 1];
        for (int i = 0; i < ids.length; i++) ids[i] = r.nextInt(V);
        boolean[] mask = new boolean[S];
        for (int i = 0; i < S; i++) mask[i] = i >= S / 2;

        Ultron.Act act = new Ultron.Act(T, D, F, L, H, V);
        java.util.Arrays.fill(m.g, 0);
        double loss = m.trainExample(ids, S, act, m.g, mask);
        System.out.printf("masked loss=%.6f params=%d%n", loss, m.paramCount());

        float[] base = m.p.clone();
        double eps = args.length > 0 ? Double.parseDouble(args[0]) : 1e-2;
        double dot = 0, na = 0, nn = 0, maxRel = 0;
        int significant = 0, worst = -1;
        for (int i = 0; i < m.p.length; i++) {
            m.p[i] = base[i] + (float) eps;
            double lp = m.trainExample(ids, S, act, null, mask);
            m.p[i] = base[i] - (float) eps;
            double lm = m.trainExample(ids, S, act, null, mask);
            m.p[i] = base[i];
            double num = (lp - lm) / (2 * eps);
            double a = m.g[i];
            dot += a * num;
            na += a * a;
            nn += num * num;
            if (Math.abs(a) + Math.abs(num) < 5e-3) continue;
            significant++;
            double den = Math.abs(a) + Math.abs(num);
            double rel = Math.abs(a - num) / den;
            if (rel > maxRel) { maxRel = rel; worst = i; }
        }
        double cos = dot / (Math.sqrt(na) * Math.sqrt(nn) + 1e-30);
        System.out.printf("masked cosine(analytic, numeric) = %.6f%n", cos);
        System.out.printf("significant params checked = %d, max rel err = %.4f at %d%n",
                significant, maxRel, worst);
        if (cos > 0.999 && maxRel < 0.10) System.out.println("MASK TEST PASSED");
        else System.out.println("MASK TEST FAILED");
    }
}
