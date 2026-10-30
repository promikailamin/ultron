package ultron;

import java.io.*;
import java.nio.file.*;
import java.util.Random;

public class Ultron {
    public final int V, D, L, F, T, H, HD;
    public final float[] p;
    public final float[] g;
    final int[] lb;
    final int[][] lo;
    final int wteO, wpeO, lnfO, lay;

    private static final int LN1G = 0, LN1B = 1, WQ = 2, BQ = 3, WK = 4, BK = 5,
            WV = 6, BV = 7, WO = 8, BO = 9, LN2G = 10, LN2B = 11,
            W1 = 12, B1 = 13, W2 = 14, B2 = 15;

    public Ultron(int V, int D, int L, int F, int T, int H) {
        this.V = V; this.D = D; this.L = L; this.F = F; this.T = T; this.H = H;
        this.HD = D / H;
        int off = 0;
        wteO = off; off += V * D;
        wpeO = off; off += T * D;
        lay = 4 * D * D + 9 * D + 2 * D * F + F;
        lb = new int[L];
        lo = new int[L][16];
        for (int l = 0; l < L; l++) {
            lb[l] = off;
            int o = 0;
            int[] f = lo[l];
            f[LN1G] = o; o += D; f[LN1B] = o; o += D;
            f[WQ] = o; o += D * D; f[BQ] = o; o += D;
            f[WK] = o; o += D * D; f[BK] = o; o += D;
            f[WV] = o; o += D * D; f[BV] = o; o += D;
            f[WO] = o; o += D * D; f[BO] = o; o += D;
            f[LN2G] = o; o += D; f[LN2B] = o; o += D;
            f[W1] = o; o += D * F; f[B1] = o; o += F;
            f[W2] = o; o += F * D; f[B2] = o; o += D;
            off += lay;
        }
        lnfO = off; off += 2 * D;
        p = new float[off];
        g = new float[off];
    }

    public int paramCount() { return p.length; }

    public void init(long seed) {
        Random r = new Random(seed);
        for (int i = 0; i < p.length; i++) p[i] = (float) (r.nextGaussian() * 0.02);
        for (int l = 0; l < L; l++) {
            int[] f = lo[l], b = lb;
            java.util.Arrays.fill(p, b[l] + f[LN1G], b[l] + f[LN1G] + D, 1f);
            java.util.Arrays.fill(p, b[l] + f[LN2G], b[l] + f[LN2G] + D, 1f);
            java.util.Arrays.fill(p, b[l] + f[BQ], b[l] + f[BQ] + D, 0f);
            java.util.Arrays.fill(p, b[l] + f[BK], b[l] + f[BK] + D, 0f);
            java.util.Arrays.fill(p, b[l] + f[BV], b[l] + f[BV] + D, 0f);
            java.util.Arrays.fill(p, b[l] + f[BO], b[l] + f[BO] + D, 0f);
            java.util.Arrays.fill(p, b[l] + f[B1], b[l] + f[B1] + F, 0f);
            java.util.Arrays.fill(p, b[l] + f[B2], b[l] + f[B2] + D, 0f);
        }
        java.util.Arrays.fill(p, lnfO, lnfO + D, 1f);
        java.util.Arrays.fill(p, lnfO + D, lnfO + 2 * D, 0f);
    }

    // ---------------- activation state ----------------
    public static class Act {
        final int T, D, F, L, H, V;
        final float[][] xin, xatt, ln1o, ln2o, q, k, v, ao;
        final float[][] m1, r1, m2, r2;
        final float[] probs;    // [L][T][H][T]
        final float[][] fpre, fpost;
        final float[] xfin, h, mf, rf;
        final float[] logits;   // [T][V]
        final float[] x, ln, tmp, dh, dx, dbuf, dattn, dln1, dln2, dq, dk, dv, dpre, dfpost, scr;

        Act(int T, int D, int F, int L, int H, int V) {
            this.T = T; this.D = D; this.F = F; this.L = L; this.H = H; this.V = V;
            xin = new float[L][]; xatt = new float[L][]; ln1o = new float[L][];
            ln2o = new float[L][]; q = new float[L][]; k = new float[L][];
            v = new float[L][]; ao = new float[L][];
            m1 = new float[L][]; r1 = new float[L][];
            m2 = new float[L][]; r2 = new float[L][];
            fpre = new float[L][]; fpost = new float[L][];
            for (int l = 0; l < L; l++) {
                xin[l] = new float[T * D]; xatt[l] = new float[T * D];
                ln1o[l] = new float[T * D]; ln2o[l] = new float[T * D];
                q[l] = new float[T * D]; k[l] = new float[T * D];
                v[l] = new float[T * D]; ao[l] = new float[T * D];
                m1[l] = new float[T]; r1[l] = new float[T];
                m2[l] = new float[T]; r2[l] = new float[T];
                fpre[l] = new float[T * F]; fpost[l] = new float[T * F];
            }
            probs = new float[L * T * H * T];
            xfin = new float[T * D]; h = new float[T * D];
            mf = new float[T]; rf = new float[T];
            logits = new float[T * V];
            x = new float[T * D]; ln = new float[T * D]; tmp = new float[T * D];
            dh = new float[T * D]; dx = new float[T * D]; dbuf = new float[T * D];
            dattn = new float[T * D]; dln1 = new float[T * D]; dln2 = new float[T * D];
            dq = new float[T * D]; dk = new float[T * D]; dv = new float[T * D];
            dpre = new float[T * F]; dfpost = new float[T * F]; scr = new float[T];
        }
    }

    // ---------------- linear algebra ----------------
    // C[n][m] = A[n][k] * W[k][m] + b[m]  (C overwritten)
    static void mm(float[] C, int co, float[] A, int ao, float[] W, int wo,
                   int n, int k, int m, float[] b, int bo) {
        for (int i = 0; i < n; i++) {
            int ci = co + i * m, ai = ao + i * k;
            if (b != null) System.arraycopy(b, bo, C, ci, m);
            else java.util.Arrays.fill(C, ci, ci + m, 0f);
            for (int r = 0; r < k; r++) {
                float av = A[ai + r];
                if (av == 0f) continue;
                int wr = wo + r * m;
                for (int j = 0; j < m; j++) C[ci + j] += av * W[wr + j];
            }
        }
    }

    // dA[n][k] += dC[n][m] * W^T ; dW[k][m] += A^T * dC ; db[m] += colsum(dC)
    // dA, dC must be distinct buffers. dA must be pre-zeroed if accumulation not wanted.
    static void mmb(float[] dA, int dao, float[] dC, int dco, float[] A, int ao,
                    float[] W, int wo, int n, int k, int m,
                    float[] db, int dbo, float[] dW, int dwo) {
        for (int i = 0; i < n; i++) {
            int ci = dco + i * m, ai = ao + i * k, dai = dao + i * k;
            for (int r = 0; r < k; r++) {
                float av = A[ai + r], acc = 0f;
                int wr = wo + r * m;
                for (int j = 0; j < m; j++) {
                    float dv = dC[ci + j];
                    acc += dv * W[wr + j];
                    dW[dwo + r * m + j] += av * dv;
                }
                dA[dai + r] += acc;
            }
            if (db != null)
                for (int j = 0; j < m; j++) db[dbo + j] += dC[ci + j];
        }
    }

    // ---------------- layernorm ----------------
    static void lnFwd(float[] out, int oo, float[] in, int io, float[] gg, int go,
                      float[] bb, int bo, int n, int d, float[] mean, float[] rstd, int ni) {
        final float eps = 1e-5f;
        for (int i = 0; i < n; i++) {
            int ii = io + i * d, oi = oo + i * d;
            float mu = 0f;
            for (int j = 0; j < d; j++) mu += in[ii + j];
            mu /= d;
            float var = 0f;
            for (int j = 0; j < d; j++) { float z = in[ii + j] - mu; var += z * z; }
            var /= d;
            float rs = (float) (1.0 / Math.sqrt(var + eps));
            mean[ni + i] = mu; rstd[ni + i] = rs;
            for (int j = 0; j < d; j++)
                out[oi + j] = (in[ii + j] - mu) * rs * gg[go + j] + bb[bo + j];
        }
    }

    // dIn += layernormBackward(dOut); also accumulates dg, db
    static void lnBwd(float[] dIn, int dio, float[] dOut, int doo, float[] in, int io,
                      float[] gg, int go, int n, int d, float[] mean, float[] rstd, int ni,
                      float[] dg, int dgo, float[] db, int dbo) {
        for (int i = 0; i < n; i++) {
            int ii = io + i * d, di = dio + i * d, doi = doo + i * d;
            float mu = mean[ni + i], rs = rstd[ni + i];
            float n1 = 0f, n2 = 0f;
            for (int j = 0; j < d; j++) {
                float xhat = (in[ii + j] - mu) * rs;
                float dn = dOut[doi + j] * gg[go + j];
                n1 += dn; n2 += dn * xhat;
                dg[dgo + j] += dOut[doi + j] * xhat;
                db[dbo + j] += dOut[doi + j];
            }
            n1 /= d; n2 /= d;
            for (int j = 0; j < d; j++) {
                float xhat = (in[ii + j] - mu) * rs;
                float dn = dOut[doi + j] * gg[go + j];
                dIn[di + j] += (dn - n1 - xhat * n2) * rs;
            }
        }
    }

    static float gelu(float x) {
        return 0.5f * x * (1f + (float) Math.tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    static float geluGrad(float x) {
        float u = 0.7978845608f * (x + 0.044715f * x * x * x);
        float t = (float) Math.tanh(u);
        float du = 0.7978845608f * (1f + 3f * 0.044715f * x * x);
        return 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * du;
    }

    // ---------------- forward body ----------------
    private void forwardBody(int[] ids, int S, Act act) {
        final int d = D, f = F, hd = HD;
        float scale = (float) (1.0 / Math.sqrt(hd));
        float[] x = act.x;

        for (int t = 0; t < S; t++) {
            int to = t * d, we = wteO + ids[t] * d, pe = wpeO + t * d;
            for (int j = 0; j < d; j++) x[to + j] = p[we + j] + p[pe + j];
        }

        for (int l = 0; l < L; l++) {
            int[] off = lo[l];
            int lb0 = lb[l];
            System.arraycopy(x, 0, act.xin[l], 0, S * d);

            lnFwd(act.ln1o[l], 0, x, 0, p, lb0 + off[LN1G], p, lb0 + off[LN1B],
                    S, d, act.m1[l], act.r1[l], 0);

            mm(act.q[l], 0, act.ln1o[l], 0, p, lb0 + off[WQ], S, d, d, p, lb0 + off[BQ]);
            mm(act.k[l], 0, act.ln1o[l], 0, p, lb0 + off[WK], S, d, d, p, lb0 + off[BK]);
            mm(act.v[l], 0, act.ln1o[l], 0, p, lb0 + off[WV], S, d, d, p, lb0 + off[BV]);

            float[] ao = act.ao[l];
            java.util.Arrays.fill(ao, 0, S * d, 0f);
            for (int t = 0; t < S; t++) {
                int qo = t * d;
                for (int hh = 0; hh < H; hh++) {
                    int qh = qo + hh * hd;
                    int pb = ((l * T + t) * H + hh) * T;
                    float max = -1e30f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        int kh = t2 * d + hh * hd;
                        float s = 0f;
                        for (int i = 0; i < hd; i++) s += act.q[l][qh + i] * act.k[l][kh + i];
                        s *= scale;
                        act.probs[pb + t2] = s;
                        if (s > max) max = s;
                    }
                    float sum = 0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float e = (float) Math.exp(act.probs[pb + t2] - max);
                        act.probs[pb + t2] = e;
                        sum += e;
                    }
                    float inv = 1f / sum;
                    for (int t2 = 0; t2 <= t; t2++) {
                        act.probs[pb + t2] *= inv;
                        float w = act.probs[pb + t2];
                        int vh = t2 * d + hh * hd;
                        for (int i = 0; i < hd; i++) ao[qh + i] += w * act.v[l][vh + i];
                    }
                }
            }

            mm(act.tmp, 0, ao, 0, p, lb0 + off[WO], S, d, d, p, lb0 + off[BO]);
            for (int i = 0; i < S * d; i++) x[i] += act.tmp[i];

            System.arraycopy(x, 0, act.xatt[l], 0, S * d);

            lnFwd(act.ln2o[l], 0, x, 0, p, lb0 + off[LN2G], p, lb0 + off[LN2B],
                    S, d, act.m2[l], act.r2[l], 0);

            mm(act.fpre[l], 0, act.ln2o[l], 0, p, lb0 + off[W1], S, d, f, p, lb0 + off[B1]);
            for (int i = 0; i < S * f; i++) act.fpost[l][i] = gelu(act.fpre[l][i]);
            mm(act.tmp, 0, act.fpost[l], 0, p, lb0 + off[W2], S, f, d, p, lb0 + off[B2]);
            for (int i = 0; i < S * d; i++) x[i] += act.tmp[i];
        }

        lnFwd(act.h, 0, x, 0, p, lnfO, p, lnfO + d, S, d, act.mf, act.rf, 0);
        System.arraycopy(x, 0, act.xfin, 0, S * d);
    }

    private void allLogits(int S, Act act) {
        for (int t = 0; t < S; t++) {
            int ho = t * D, loff = t * V;
            for (int v = 0; v < V; v++) {
                int we = wteO + v * D;
                float s = 0f;
                for (int j = 0; j < D; j++) s += act.h[ho + j] * p[we + j];
                act.logits[loff + v] = s;
            }
        }
    }

    /** training example: ids[0..S] window (S inputs, S targets). Returns mean CE loss. */
    public double trainExample(int[] ids, int S, Act act, float[] gr) {
        return trainExample(ids, S, act, gr, null);
    }

    /**
     * Masked training example. When mask != null, only positions where
     * mask[t] is true contribute to the loss. This lets the trainer apply
     * gradient to the assistant reply while keeping the user prompt fixed.
     */
    public double trainExample(int[] ids, int S, Act act, float[] gr, boolean[] mask) {
        final int d = D, f = F, hd = HD;
        forwardBody(ids, S, act);
        allLogits(S, act);
        float[] lg = act.logits;

        double loss = 0;
        int lossCount = 0;
        for (int t = 0; t < S; t++) {
            if (mask != null && !mask[t]) continue;
            int loff = t * V;
            float max = -1e30f;
            for (int v = 0; v < V; v++) if (lg[loff + v] > max) max = lg[loff + v];
            float sum = 0f;
            for (int v = 0; v < V; v++) {
                float e = (float) Math.exp(lg[loff + v] - max);
                lg[loff + v] = e;
                sum += e;
            }
            float inv = 1f / sum;
            for (int v = 0; v < V; v++) lg[loff + v] *= inv;
            int tgt = ids[t + 1];
            float pt = lg[loff + tgt];
            loss -= Math.log(Math.max(pt, 1e-12));
            for (int v = 0; v < V; v++)
                lg[loff + v] = lg[loff + v] - (v == tgt ? 1f : 0f);
            lossCount++;
        }
        if (lossCount == 0) {
            lossCount = 1;
        }
        loss /= lossCount;
        if (gr == null) return loss;

        // ---- backward ----
        float[] dh = act.dh;
        java.util.Arrays.fill(dh, 0, S * d, 0f);
        float invN = 1f / lossCount;
        for (int t = 0; t < S; t++) {
            if (mask != null && !mask[t]) continue;
            int ho = t * d, loff = t * V;
            for (int v = 0; v < V; v++) {
                float dl = lg[loff + v] * invN;
                if (dl == 0f) continue;
                int we = wteO + v * d;
                for (int j = 0; j < d; j++) {
                    gr[we + j] += dl * act.h[ho + j];
                    dh[ho + j] += dl * p[we + j];
                }
            }
        }

        float[] dx = act.dx;
        java.util.Arrays.fill(dx, 0, S * d, 0f);
        lnBwd(dx, 0, dh, 0, act.xfin, 0, p, lnfO, S, d, act.mf, act.rf, 0,
                gr, lnfO, gr, lnfO + d);

        float[] da = act.dbuf, dattn = act.dattn, dln1 = act.dln1, dln2 = act.dln2;
        float[] dq = act.dq, dk = act.dk, dv = act.dv, dpre = act.dpre,
                dfpost = act.dfpost, scr = act.scr;

        for (int l = L - 1; l >= 0; l--) {
            int[] off = lo[l];
            int lb0 = lb[l];

            // da = grad wrt x2 (post-attn residual); starts as grad wrt x3 (skip)
            System.arraycopy(dx, 0, da, 0, S * d);

            // ---- mlp backward: f2 = fpost @ W2 ----
            java.util.Arrays.fill(dfpost, 0, S * f, 0f);
            mmb(dfpost, 0, dx, 0, act.fpost[l], 0, p, lb0 + off[W2], S, f, d,
                    gr, lb0 + off[B2], gr, lb0 + off[W2]);
            for (int i = 0; i < S * f; i++)
                dpre[i] = dfpost[i] * geluGrad(act.fpre[l][i]);
            java.util.Arrays.fill(dln2, 0, S * d, 0f);
            mmb(dln2, 0, dpre, 0, act.ln2o[l], 0, p, lb0 + off[W1], S, d, f,
                    gr, lb0 + off[B1], gr, lb0 + off[W1]);
            lnBwd(da, 0, dln2, 0, act.xatt[l], 0, p, lb0 + off[LN2G], S, d,
                    act.m2[l], act.r2[l], 0, gr, lb0 + off[LN2G], gr, lb0 + off[LN2B]);

            // ---- attention backward: a = ao @ Wo + bo ----
            java.util.Arrays.fill(dattn, 0, S * d, 0f);
            mmb(dattn, 0, da, 0, act.ao[l], 0, p, lb0 + off[WO], S, d, d,
                    gr, lb0 + off[BO], gr, lb0 + off[WO]);

            java.util.Arrays.fill(dq, 0, S * d, 0f);
            java.util.Arrays.fill(dk, 0, S * d, 0f);
            java.util.Arrays.fill(dv, 0, S * d, 0f);
            float sc = (float) (1.0 / Math.sqrt(hd));
            for (int t = 0; t < S; t++) {
                int to = t * d;
                for (int hh = 0; hh < H; hh++) {
                    int qh = to + hh * hd;
                    int pb = ((l * T + t) * H + hh) * T;
                    float dot = 0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        int vh = t2 * d + hh * hd;
                        float s = 0f;
                        for (int i = 0; i < hd; i++) s += dattn[qh + i] * act.v[l][vh + i];
                        scr[t2] = s;
                        dot += s * act.probs[pb + t2];
                    }
                    for (int t2 = 0; t2 <= t; t2++) {
                        float w = act.probs[pb + t2];
                        int vh = t2 * d + hh * hd;
                        for (int i = 0; i < hd; i++) dv[vh + i] += w * dattn[qh + i];
                        float ds = w * (scr[t2] - dot);
                        if (ds == 0f) continue;
                        int kh = t2 * d + hh * hd;
                        float a = ds * sc;
                        for (int i = 0; i < hd; i++) {
                            dq[qh + i] += a * act.k[l][kh + i];
                            dk[kh + i] += a * act.q[l][qh + i];
                        }
                    }
                }
            }

            // ---- q/k/v backward into ln1 output grad ----
            java.util.Arrays.fill(dln1, 0, S * d, 0f);
            mmb(dln1, 0, dq, 0, act.ln1o[l], 0, p, lb0 + off[WQ], S, d, d,
                    gr, lb0 + off[BQ], gr, lb0 + off[WQ]);
            mmb(dln1, 0, dk, 0, act.ln1o[l], 0, p, lb0 + off[WK], S, d, d,
                    gr, lb0 + off[BK], gr, lb0 + off[WK]);
            mmb(dln1, 0, dv, 0, act.ln1o[l], 0, p, lb0 + off[WV], S, d, d,
                    gr, lb0 + off[BV], gr, lb0 + off[WV]);
            lnBwd(da, 0, dln1, 0, act.xin[l], 0, p, lb0 + off[LN1G], S, d,
                    act.m1[l], act.r1[l], 0, gr, lb0 + off[LN1G], gr, lb0 + off[LN1B]);

            // da is now grad wrt layer input x_l
            System.arraycopy(da, 0, dx, 0, S * d);
        }

        // scatter embedding gradients (tied input embeddings)
        for (int t = 0; t < S; t++) {
            int to = t * d, we = wteO + ids[t] * d, pe = wpeO + t * d;
            for (int j = 0; j < d; j++) {
                gr[we + j] += dx[to + j];
                gr[pe + j] += dx[to + j];
            }
        }
        return loss;
    }

    /** inference: computes logits for the last position only, stored at act.logits[0..V) */
    public float[] forwardLast(int[] ids, int S, Act act) {
        forwardBody(ids, S, act);
        int ho = (S - 1) * D;
        for (int v = 0; v < V; v++) {
            int we = wteO + v * D;
            float s = 0f;
            for (int j = 0; j < D; j++) s += act.h[ho + j] * p[we + j];
            act.logits[v] = s;
        }
        return act.logits;
    }

    // ---------------- incremental inference with KV cache (low-end friendly) ----------------
    // Reuses cached keys/values so each generated token costs one forward step instead of
    // re-running the whole context. All scratch lives in the cache: step() is allocation-free
    // and thread-safe as long as each thread uses its own KVCache.
    public static class KVCache {
        final int L, T, D, F, V;
        final float[] k, v;             // [L][T][D]
        final float[] x, ln1o, q, k1, v1, attn, ln2o, h;
        final float[] fa;               // [F]
        final float[] m1, r1, m2, r2, mf, rf;
        final float[] probScratch;      // [T]
        final float[] logits;           // [V]
        int len;

        KVCache(int L, int T, int D, int F, int V) {
            this.L = L; this.T = T; this.D = D; this.F = F; this.V = V;
            k = new float[L * T * D];
            v = new float[L * T * D];
            x = new float[D]; ln1o = new float[D]; q = new float[D];
            k1 = new float[D]; v1 = new float[D]; attn = new float[D];
            ln2o = new float[D]; h = new float[D];
            fa = new float[F];
            m1 = new float[1]; r1 = new float[1]; m2 = new float[1]; r2 = new float[1];
            mf = new float[1]; rf = new float[1];
            probScratch = new float[T];
            logits = new float[V];
        }

        public void clear() { len = 0; }

        // drop oldest token: slide window so positions restart at 0 (matches training windows)
        void shift() {
            int block = (T - 1) * D;
            for (int l = 0; l < L; l++) {
                System.arraycopy(k, (l * T + 1) * D, k, l * T * D, block);
                System.arraycopy(v, (l * T + 1) * D, v, l * T * D, block);
            }
            len = T - 1;
        }
    }

    public KVCache newCache() { return new KVCache(L, T, D, F, V); }

    /** Feed one token; returns logits for the next token (length V, owned by the cache). */
    public float[] step(int token, KVCache c) {
        final int d = D, f = F, hd = HD;
        final float scale = (float) (1.0 / Math.sqrt(hd));
        int pos = c.len;
        if (pos >= T) c.shift();
        pos = c.len;

        // embedding
        int we = wteO + token * d, pe = wpeO + pos * d;
        for (int j = 0; j < d; j++) c.x[j] = p[we + j] + p[pe + j];

        for (int l = 0; l < L; l++) {
            int[] off = lo[l];
            int lb0 = lb[l];

            lnFwd(c.ln1o, 0, c.x, 0, p, lb0 + off[LN1G], p, lb0 + off[LN1B],
                    1, d, c.m1, c.r1, 0);
            mm(c.q, 0, c.ln1o, 0, p, lb0 + off[WQ], 1, d, d, p, lb0 + off[BQ]);
            mm(c.k1, 0, c.ln1o, 0, p, lb0 + off[WK], 1, d, d, p, lb0 + off[BK]);
            mm(c.v1, 0, c.ln1o, 0, p, lb0 + off[WV], 1, d, d, p, lb0 + off[BV]);

            int slot = (l * T + pos) * d;
            System.arraycopy(c.k1, 0, c.k, slot, d);
            System.arraycopy(c.v1, 0, c.v, slot, d);

            int att = pos + 1;
            java.util.Arrays.fill(c.attn, 0, d, 0f);
            for (int hh = 0; hh < H; hh++) {
                int qh = hh * hd;
                float max = -1e30f;
                for (int t2 = 0; t2 < att; t2++) {
                    int kh = (l * T + t2) * d + qh;
                    float s = 0f;
                    for (int i = 0; i < hd; i++) s += c.q[qh + i] * c.k[kh + i];
                    s *= scale;
                    c.probScratch[t2] = s;
                    if (s > max) max = s;
                }
                float sum = 0f;
                for (int t2 = 0; t2 < att; t2++) {
                    float e = (float) Math.exp(c.probScratch[t2] - max);
                    c.probScratch[t2] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                for (int t2 = 0; t2 < att; t2++) {
                    float w = c.probScratch[t2] * inv;
                    int vh = (l * T + t2) * d + qh;
                    for (int i = 0; i < hd; i++) c.attn[qh + i] += w * c.v[vh + i];
                }
            }

            mm(c.fa, 0, c.attn, 0, p, lb0 + off[WO], 1, d, d, p, lb0 + off[BO]);
            for (int j = 0; j < d; j++) c.x[j] += c.fa[j];

            lnFwd(c.ln2o, 0, c.x, 0, p, lb0 + off[LN2G], p, lb0 + off[LN2B],
                    1, d, c.m2, c.r2, 0);
            mm(c.fa, 0, c.ln2o, 0, p, lb0 + off[W1], 1, d, f, p, lb0 + off[B1]);
            for (int j = 0; j < f; j++) c.fa[j] = gelu(c.fa[j]);
            mm(c.attn, 0, c.fa, 0, p, lb0 + off[W2], 1, f, d, p, lb0 + off[B2]);
            for (int j = 0; j < d; j++) c.x[j] += c.attn[j];
        }

        lnFwd(c.h, 0, c.x, 0, p, lnfO, p, lnfO + d, 1, d, c.mf, c.rf, 0);
        for (int vv = 0; vv < V; vv++) {
            int w = wteO + vv * d;
            float s = 0f;
            for (int j = 0; j < d; j++) s += c.h[j] * p[w + j];
            c.logits[vv] = s;
        }

        c.len = pos + 1;
        return c.logits;
    }

    // ---------------- persistence ----------------
    private static final int MAGIC = 0x554C5431; // "ULT1"

    /** Writes model to a stream; does NOT close the caller's stream. */
    public void save(OutputStream os) throws IOException {
        DataOutputStream o = new DataOutputStream(new BufferedOutputStream(os));
        o.writeInt(MAGIC);
        o.writeInt(V); o.writeInt(D); o.writeInt(L); o.writeInt(F); o.writeInt(T); o.writeInt(H);
        for (float x : p) o.writeFloat(x);
        o.flush();
    }

    public void save(String path) throws IOException {
        try (OutputStream os = new FileOutputStream(path)) { save(os); }
    }

    /** Reads model from a stream; does NOT close the caller's stream. */
    public static Ultron load(InputStream is) throws IOException {
        DataInputStream in = new DataInputStream(new BufferedInputStream(is));
        if (in.readInt() != MAGIC) throw new IOException("not an Ultron model file");
        int V = in.readInt(), D = in.readInt(), L = in.readInt(),
                F = in.readInt(), T = in.readInt(), H = in.readInt();
        Ultron m = new Ultron(V, D, L, F, T, H);
        for (int i = 0; i < m.p.length; i++) m.p[i] = in.readFloat();
        return m;
    }

    public static Ultron load(String path) throws IOException {
        try (InputStream is = new FileInputStream(path)) { return load(is); }
    }
}
