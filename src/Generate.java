package ultron;

import java.util.Arrays;

public class Generate {
    public static void main(String[] args) throws Exception {
        String prompt = args.length > 0 ? args[0] : "";
        int maxNew = args.length > 1 ? Integer.parseInt(args[1]) : 100;
        float temp = args.length > 2 ? Float.parseFloat(args[2]) : 0.8f;
        int topk = args.length > 3 ? Integer.parseInt(args[3]) : 40;
        long seed = args.length > 4 ? Long.parseLong(args[4]) : 2026;

        Tokenizer tok = Tokenizer.load("data/ultron-vocab.txt");
        Ultron m = Ultron.load("out/ultron.bin");
        if (m.V != tok.V) throw new RuntimeException("vocab/model mismatch: " + m.V + " vs " + tok.V);

        Ultron.KVCache cache = m.newCache();
        Sampler sampler = new Sampler(m.V, seed);
        int PAR = tok.stoi.get(Tokenizer.PAR), EOS = tok.stoi.get(Tokenizer.EOS);

        int[] ctx;
        if (prompt.isEmpty()) ctx = new int[]{PAR};
        else ctx = tok.encode(prompt);
        int shown = ctx.length;

        // prefill: feed the prompt token by token (same path as generation)
        float[] logits = null;
        for (int i = 0; i < ctx.length; i++) logits = m.step(ctx[i], cache);

        long t0 = System.nanoTime();
        int generated = 0;
        for (int n = 0; n < maxNew; n++) {
            int next = sampler.sample(logits, m.V, temp, topk);
            ctx = Arrays.copyOf(ctx, ctx.length + 1);
            ctx[ctx.length - 1] = next;
            generated++;
            if (next == EOS) break;
            logits = m.step(next, cache);
        }
        double secs = (System.nanoTime() - t0) / 1e9;

        System.out.println("---");
        String cont = tok.decode(ctx, shown, ctx.length);
        if (prompt.isEmpty()) System.out.println(cont.isEmpty() ? tok.decode(ctx, 0, ctx.length) : cont);
        else {
            if (!cont.isEmpty() && Character.isLetterOrDigit(cont.charAt(0))) cont = " " + cont;
            System.out.println(prompt + cont);
        }
        System.out.println("---");
        System.out.printf("%d tokens in %.2fs (%.1f tok/s), kv-cache positions=%d/%d%n",
                generated, secs, generated / Math.max(secs, 1e-9), cache.len, m.T);
    }
}
