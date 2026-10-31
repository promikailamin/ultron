package ultron;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Experimental dialogue-only fine tuning against an existing checkpoint. */
public class DialogueTrain {
    static final int CTX = 64;

    static class Example {
        final int[] ids;
        final boolean[] mask;
        Example(int[] ids, boolean[] mask) { this.ids = ids; this.mask = mask; }
    }

    static Example makeExample(Tokenizer tok, String q, String a) {
        int[] pids = tok.encode("user: " + q + "\nultron:");
        int[] aids = tok.encode(a);
        int firstAnswer = 1 + pids.length;
        int maxAnswer = CTX - firstAnswer;
        if (maxAnswer < 6) return null;
        if (aids.length > maxAnswer) aids = Arrays.copyOf(aids, maxAnswer);
        int S = firstAnswer + aids.length;
        int[] ids = new int[S + 1];
        ids[0] = tok.stoi.get(Tokenizer.PAR);
        System.arraycopy(pids, 0, ids, 1, pids.length);
        System.arraycopy(aids, 0, ids, firstAnswer, aids.length);
        ids[S] = tok.stoi.get(Tokenizer.EOS);
        boolean[] mask = new boolean[S];
        for (int t = 0; t < S; t++) mask[t] = (t + 1) >= firstAnswer;
        return new Example(ids, mask);
    }

    public static void main(String[] args) throws Exception {
        int steps = Integer.parseInt(args[0]);
        int batch = Integer.parseInt(args[1]);
        double lr0 = Double.parseDouble(args[2]);
        String out = args.length > 3 ? args[3] : "out/ultron-exp.bin";
        String corpus = args.length > 4 ? args[4] : "data/chat.txt,data/mc_qa.txt";
        double lmFrac = args.length > 5 ? Double.parseDouble(args[5]) : 0.0;
        boolean fresh = args.length > 6 && args[6].equals("fresh");
        String initPath = args.length > 7 ? args[7] : "out/ultron.bin";

        Tokenizer tok = Tokenizer.load("data/ultron-vocab.txt");
        Ultron m;
        if (fresh) { m = new Ultron(tok.V, 96, 5, 384, CTX, 4); m.init(42); }
        else m = Ultron.load(initPath);
        if (m.V != tok.V || m.T != CTX) throw new RuntimeException("incompatible checkpoint");

        ArrayList<Example> dialogues = new ArrayList<>();
        for (String file : corpus.split(",")) {
            if (!Files.exists(Paths.get(file))) continue;
            try (BufferedReader br = Files.newBufferedReader(Paths.get(file))) {
                String line, question = null;
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("user: ")) question = line.substring(6).trim();
                    else if (line.startsWith("ultron: ") && question != null) {
                        Example ex = makeExample(tok, question, line.substring(8).trim());
                        if (ex != null) dialogues.add(ex);
                        question = null;
                    }
                }
            }
        }
        System.out.println("dialogues: " + dialogues.size());

        int[] lmData = new int[0];
        if (lmFrac > 0) {
            StringBuilder sb = new StringBuilder();
            for (String f : new String[]{"data/grammar.txt", "data/chat.txt", "data/mc_qa.txt"}) {
                if (Files.exists(Paths.get(f))) sb.append(new String(Files.readAllBytes(Paths.get(f)), "UTF-8")).append("\n\n");
            }
            lmData = tok.encode(sb.toString());
        }

        final int[] lm = lmData;
        int nT = Math.min(Runtime.getRuntime().availableProcessors(), batch);
        float[][] tg = new float[nT][];
        Ultron.Act[] acts = new Ultron.Act[nT];
        Random[] rngs = new Random[nT];
        double[] tls = new double[nT];
        int[] tcnt = new int[nT];
        for (int i = 0; i < nT; i++) {
            tg[i] = new float[m.p.length];
            acts[i] = new Ultron.Act(CTX, m.D, m.F, m.L, m.H, m.V);
            rngs[i] = new Random(999 + i);
        }
        float[] am = new float[m.p.length], av = new float[m.p.length];
        int warmup = Math.min(200, Math.max(50, steps / 20));
        long t0 = System.currentTimeMillis();
        for (int step = 0; step <= steps; step++) {
            for (int i = 0; i < nT; i++) Arrays.fill(tg[i], 0);
            Thread[] th = new Thread[nT];
            for (int i = 0; i < nT; i++) {
                final int ti = i, lo = i * batch / nT, hi = (i + 1) * batch / nT;
                th[i] = new Thread(() -> {
                    double ls = 0; int cnt = 0;
                    for (int b = lo; b < hi; b++) {
                        boolean useLm = lm.length > CTX + 1 && rngs[ti].nextDouble() < lmFrac;
                        if (!useLm && !dialogues.isEmpty()) {
                            Example ex = dialogues.get(rngs[ti].nextInt(dialogues.size()));
                            int s = ex.ids.length - 1;
                            ls += m.trainExample(ex.ids, s, acts[ti], tg[ti], ex.mask);
                            cnt += s;
                        } else if (lm.length > CTX + 1) {
                            int s = rngs[ti].nextInt(lm.length - CTX - 1);
                            int[] win = Arrays.copyOfRange(lm, s, s + CTX + 1);
                            ls += m.trainExample(win, CTX, acts[ti], tg[ti]);
                            cnt += CTX;
                        }
                    }
                    tls[ti] = ls; tcnt[ti] = cnt;
                });
                th[i].start();
            }
            for (Thread t : th) t.join();
            double loss = 0; int cnt = 0;
            Arrays.fill(m.g, 0);
            for (int i = 0; i < nT; i++) loss += tls[i];
            for (int i = 0; i < nT; i++) cnt += tcnt[i];
            for (int i = 0; i < nT; i++)
                for (int j = 0; j < m.p.length; j++) m.g[j] += tg[i][j] * ((float) tcnt[i] / cnt);
            loss /= batch;
            double sq = 0;
            for (float x : m.g) sq += (double) x * x;
            double gnorm = Math.sqrt(sq);
            float clip = gnorm > 1.0 ? (float) (1.0 / gnorm) : 1f;
            double lr;
            if (step < warmup) lr = lr0 * (step + 1) / warmup;
            else {
                double prog = (double) (step - warmup) / Math.max(1, steps - warmup);
                lr = lr0 * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * prog)));
            }
            double bc1 = 1 - Math.pow(0.9, step + 1), bc2 = 1 - Math.pow(0.95, step + 1);
            for (int j = 0; j < m.p.length; j++) {
                float gj = m.g[j] * clip;
                float mj = (float) (0.9 * am[j] + 0.1 * gj);
                float vj = (float) (0.95 * av[j] + 0.05 * gj * gj);
                am[j] = mj; av[j] = vj;
                double mh = mj / bc1, vh = vj / bc2;
                m.p[j] -= (float) (lr * mh / (Math.sqrt(vh) + 1e-8));
            }
            if (step % 20 == 0 || step == steps) {
                long el = System.currentTimeMillis() - t0;
                System.out.printf("step %5d/%d loss %.4f ppl %.1f lr %.2e gnorm %.2f %.0f tok/s %ds%n",
                        step, steps, loss, Math.exp(loss), lr, gnorm,
                        (double) cnt * Math.max(1, step) / (el / 1000.0), el / 1000);
            }
            if (step > 0 && step % 200 == 0 && step < steps) {
                m.save(out);
                System.out.println("  saved " + out + " (" + step + ")");
            }
        }
        m.save(out);
        System.out.println("saved " + out);
    }
}
