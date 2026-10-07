package ultron;

import java.util.Random;

public class GradCheck {
    public static void main(String[] args) {
        int V = 47, D = 8, L = 2, F = 16, T = 8, H = 2;
        Ultron m = new Ultron(V, D, L, F, T, H);
        Random r = new Random(12345);
        m.init(12345);

        int S = T;
        int[] ids = new int[S + 1];
        for (int i = 0; i < ids.length; i++) ids[i] = r.nextInt(V);

        Ultron.Act act = new Ultron.Act(T, D, F, L, H, V);
        java.util.Arrays.fill(m.g, 0);
        double loss = m.trainExample(ids, S, act, m.g);
        System.out.printf("loss=%.6f params=%d%n", loss, m.paramCount());

        // clone params so perturbation does not disturb analytic run
        float[] base = m.p.clone();
        double[] num = new double[m.p.length];
        float[] save = new float[m.p.length];

        double eps = args.length > 0 ? Double.parseDouble(args[0]) : 1e-2;
        // perturb in groups is unsafe; do every param (model is tiny)
        for (int i = 0; i < m.p.length; i++) {
            System.arraycopy(m.p, 0, save, 0, m.p.length);
            m.p[i] = base[i] + (float) eps;
            double lp = m.trainExample(ids, S, act, null);
            System.arraycopy(save, 0, m.p, 0, m.p.length);
            m.p[i] = base[i] - (float) eps;
            double lm = m.trainExample(ids, S, act, null);
            System.arraycopy(save, 0, m.p, 0, m.p.length);
            num[i] = (lp - lm) / (2 * eps);
            if (i % 200 == 0) System.err.print(".");
        }
        System.arraycopy(base, 0, m.p, 0, m.p.length);

        double dot = 0, na = 0, nn = 0, maxRel = 0;
        int worst = -1;
        int significant = 0;
        for (int i = 0; i < m.p.length; i++) {
            double a = m.g[i];
            dot += a * num[i];
            na += a * a;
            nn += num[i] * num[i];
            if (Math.abs(a) + Math.abs(num[i]) < 5e-3) continue; // skip noise-level grads
            significant++;
            double den = Math.abs(a) + Math.abs(num[i]);
            double rel = Math.abs(a - num[i]) / den;
            if (rel > maxRel) { maxRel = rel; worst = i; }
        }
        double cos = dot / (Math.sqrt(na) * Math.sqrt(nn) + 1e-30);
        System.out.printf("cosine(analytic, numeric) = %.6f%n", cos);
        System.out.printf("significant params checked = %d, max rel err = %.4f at %d (a=%.6g n=%.6g)%n",
                significant, maxRel, worst, worst >= 0 ? m.g[worst] : 0, worst >= 0 ? num[worst] : 0);
        if (cos > 0.999 && maxRel < 0.10) System.out.println("GRADCHECK PASSED");
        else System.out.println("GRADCHECK FAILED");
    }
}
