package ultron;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Train {
    static final int VOCAB = 8192, D = 96, L = 5, F = 384, CTX = 64, HEADS = 4;
    static final String[] CORPUS = {
            "data/pp.txt", "data/moby.txt", "data/alice.txt", "data/sherlock.txt",
            "data/frank.txt", "data/jane.txt", "data/tom.txt",
            "data/wp.txt", "data/ge.txt", "data/drac.txt", "data/mc.txt"
    };

    static class IntList {
        int[] a = new int[1 << 20];
        int n = 0;
        void add(int x) {
            if (n == a.length) a = Arrays.copyOf(a, a.length * 2);
            a[n++] = x;
        }
    }

    public static void main(String[] args) throws Exception {
        int steps = args.length > 0 ? Integer.parseInt(args[0]) : 5000;
        int batch = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        double lr0 = args.length > 2 ? Double.parseDouble(args[2]) : 1e-3;
        boolean resume = args.length > 3 && args[3].equals("resume");

        long t0 = System.currentTimeMillis();

        Tokenizer tok;
        if (Files.exists(Paths.get("data/ultron-vocab.txt"))) {
            tok = Tokenizer.load("data/ultron-vocab.txt");
            System.out.println("loaded vocab: " + tok.V + " tokens");
        } else {
            tok = Tokenizer.build(Arrays.asList(CORPUS), VOCAB);
            tok.save("data/ultron-vocab.txt");
            System.out.println("built vocab: " + tok.V + " tokens");
        }
        int PAR = tok.stoi.get(Tokenizer.PAR), EOS = tok.stoi.get(Tokenizer.EOS);

        IntList buf = new IntList();
        long chars = 0;
        for (String f : CORPUS) {
            String txt = Tokenizer.normalize(new String(Files.readAllBytes(Paths.get(f)), "UTF-8"));
            chars += txt.length();
            int[] pc = tok.encode(txt);
            buf.add(PAR);
            for (int x : pc) buf.add(x);
            buf.add(EOS);
        }
        int[] data = Arrays.copyOf(buf.a, buf.n);
        System.out.printf("corpus: %.2f M chars -> %d tokens, vocab=%d%n",
                chars / 1e6, data.length, tok.V);

        Ultron m = resume ? Ultron.load("out/ultron.bin") : new Ultron(tok.V, D, L, F, CTX, HEADS);
        if (!resume) m.init(42);
        System.out.printf("ULTRON model: V=%d d=%d layers=%d ffn=%d ctx=%d heads=%d%n",
                m.V, m.D, m.L, m.F, m.T, m.H);
        System.out.printf("parameters: %,d%n", m.paramCount());

        int nT = Math.min(Runtime.getRuntime().availableProcessors(), batch);
        float[][] tg = new float[nT][];
        Ultron.Act[] acts = new Ultron.Act[nT];
        Random[] rngs = new Random[nT];
        double[] tls = new double[nT];
        int[] tcnt = new int[nT];
        for (int i = 0; i < nT; i++) {
            tg[i] = new float[m.p.length];
            acts[i] = new Ultron.Act(CTX, D, F, L, HEADS, m.V);
            rngs[i] = new Random(777 + i);
        }

        // Adam state
        float[] am = new float[m.p.length];
        float[] av = new float[m.p.length];
        int warmup = Math.min(200, Math.max(50, steps / 20));

        double bestPpl = Double.MAX_VALUE;
        int startStep = 0;

        for (int step = startStep; step <= steps; step++) {
            for (int i = 0; i < nT; i++) Arrays.fill(tg[i], 0);
            final int stepF = step;
            Thread[] th = new Thread[nT];
            for (int i = 0; i < nT; i++) {
                final int ti = i, lo = i * batch / nT, hi = (i + 1) * batch / nT;
                th[i] = new Thread(() -> {
                    double ls = 0;
                    int cnt = 0;
                    for (int b = lo; b < hi; b++) {
                        int s = rngs[ti].nextInt(data.length - CTX - 1);
                        int[] win = Arrays.copyOfRange(data, s, s + CTX + 1);
                        ls += m.trainExample(win, CTX, acts[ti], tg[ti]);
                        cnt += CTX;
                    }
                    tls[ti] = ls;
                    tcnt[ti] = cnt;
                });
                th[i].start();
            }
            for (Thread t : th) {
                try { t.join(); } catch (InterruptedException e) { throw new RuntimeException(e); }
            }

            double loss = 0;
            int cnt = 0;
            Arrays.fill(m.g, 0);
            for (int i = 0; i < nT; i++) loss += tls[i];
            for (int i = 0; i < nT; i++) cnt += tcnt[i];
            for (int i = 0; i < nT; i++)
                for (int j = 0; j < m.p.length; j++) m.g[j] += tg[i][j] * ((float) tcnt[i] / cnt);
            loss /= batch;

            // grad clip
            double sq = 0;
            for (float x : m.g) sq += (double) x * x;
            double gnorm = Math.sqrt(sq);
            float clip = 1.0f;
            if (gnorm > 1.0) clip = (float) (1.0 / gnorm);

            // lr schedule: warmup + cosine
            double lr;
            if (step < warmup) lr = lr0 * (step + 1) / warmup;
            else {
                double prog = (double) (step - warmup) / Math.max(1, steps - warmup);
                double fin = 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * prog));
                lr = lr0 * fin;
            }

            // adam
            double bc1 = 1 - Math.pow(0.9, step + 1);
            double bc2 = 1 - Math.pow(0.95, step + 1);
            for (int j = 0; j < m.p.length; j++) {
                float gj = m.g[j] * clip;
                float mj = (float) (0.9 * am[j] + 0.1 * gj);
                float vj = (float) (0.95 * av[j] + 0.05 * gj * gj);
                am[j] = mj; av[j] = vj;
                double mh = mj / bc1, vh = vj / bc2;
                m.p[j] -= (float) (lr * mh / (Math.sqrt(vh) + 1e-8));
            }

            if (step % 20 == 0 || step == steps) {
                double ppl = Math.exp(loss);
                long el = System.currentTimeMillis() - t0;
                double tokPerS = (double) cnt * Math.max(1, step) / (el / 1000.0);
                System.out.printf("step %5d/%d  loss %.4f  ppl %.1f  lr %.2e  gnorm %.2f  %.0f tok/s  %ds%n",
                        step, steps, loss, ppl, lr, gnorm, tokPerS, el / 1000);
                if (Double.isFinite(ppl) && ppl < bestPpl) bestPpl = ppl;
            }
            if (step > 0 && step % 50 == 0 && step < steps) {
                m.save("out/ultron.bin");
                System.out.println("  checkpoint saved (" + step + ")");
            }
        }

        m.save("out/ultron.bin");
        System.out.printf("done. best ppl %.1f. model saved to out/ultron.bin%n", bestPpl);
    }
}
