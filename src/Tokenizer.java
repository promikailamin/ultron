package ultron;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Tokenizer {
    public static final String UNK = "<unk>", EOS = "<eos>", PAR = "<p>";

    public int V;
    public String[] itos;
    public HashMap<String, Integer> stoi = new HashMap<>();
    private HashMap<String, Integer> mergeRanks;
    private HashMap<String, int[]> bpeCache = new HashMap<>();
    private boolean bpe;
    private static final String SPACE_MARK = "\u2581";
    private HashSet<String> noSpaceBefore = new HashSet<>(Arrays.asList(
            ",", ".", "!", "?", ":", ";", ")", "]", "}", "\"", "%", "…"));
    private HashSet<String> noSpaceAfter = new HashSet<>(Arrays.asList(
            "(", "[", "{", "\"", "-", "\u2014"));

    static boolean isWordChar(char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'';
    }

    static String normalize(String s) {
        int a = s.indexOf("*** START OF THE PROJECT GUTENBERG");
        if (a >= 0) {
            int nl = s.indexOf('\n', a);
            s = (nl >= 0) ? s.substring(nl + 1) : s;
        }
        int b = s.indexOf("*** END OF THE PROJECT GUTENBERG");
        if (b >= 0) s = s.substring(0, b);
        s = s.toLowerCase(Locale.ROOT);
        s = s.replace("\r\n", "\n").replace('\r', '\n');
        s = s.replaceAll("\\[illustration[^\\]]*\\]", " ");
        s = s.replaceAll("\\{\\{[^}]*\\}\\}", " ");
        return s;
    }

    static List<String> pieces(String text) {
        ArrayList<String> out = new ArrayList<>();
        int i = 0, n = text.length();
        while (i < n) {
            char c = text.charAt(i);
            if (c == ' ' || c == '\t' || c == '\n') {
                int j = i;
                int newlines = 0;
                while (j < n && (text.charAt(j) == ' ' || text.charAt(j) == '\t' || text.charAt(j) == '\n')) {
                    if (text.charAt(j) == '\n') newlines++;
                    j++;
                }
                if (newlines >= 2) out.add(PAR);
                i = j;
            } else if (isWordChar(c)) {
                int j = i;
                while (j < n && isWordChar(text.charAt(j))) j++;
                String w = text.substring(i, j);
                if (w.length() > 1) while (w.endsWith("'")) w = w.substring(0, w.length() - 1);
                if (w.length() > 1 && w.startsWith("'")) w = w.substring(1);
                if (w.length() > 0) out.add(w);
                i = j;
            } else {
                out.add(String.valueOf(c));
                i++;
            }
        }
        return out;
    }

    /** Loads the BPE tokenizer when its merges file exists, else the legacy vocab. */
    public static Tokenizer loadDefault() throws IOException {
        try (InputStream vocab = Resources.open("ultron-vocab.txt", "data/ultron-vocab.txt")) {
            if (Resources.exists("ultron-merges.txt", "data/ultron-merges.txt")) {
                try (InputStream merges = Resources.open("ultron-merges.txt", "data/ultron-merges.txt")) {
                    return loadBPE(vocab, merges);
                }
            }
            return load(vocab);
        }
    }

    public static Tokenizer build(List<String> files, int targetVocab) {
        HashMap<String, Integer> wc = new HashMap<>();
        ArrayList<String> order = new ArrayList<>();
        long total = 0;
        for (String f : files) {
            String txt;
            try {
                txt = normalize(new String(Files.readAllBytes(Paths.get(f)), "UTF-8"));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            for (String pc : pieces(txt)) {
                Integer c = wc.get(pc);
                if (c == null) { wc.put(pc, 1); order.add(pc); } else wc.put(pc, c + 1);
                total++;
            }
        }
        ArrayList<String> vocab = new ArrayList<>();
        vocab.add(UNK); vocab.add(EOS); vocab.add(PAR);
        for (char c = 'a'; c <= 'z'; c++) vocab.add(String.valueOf(c));
        for (char c = '0'; c <= '9'; c++) vocab.add(String.valueOf(c));
        HashSet<String> seen = new HashSet<>(vocab);

        ArrayList<String> punct = new ArrayList<>();
        ArrayList<String> words = new ArrayList<>();
        for (String s : wc.keySet()) {
            if (s.equals(UNK) || s.equals(EOS) || s.equals(PAR)) continue;
            if (s.length() == 1 && !seen.contains(s)) punct.add(s);
            else if (s.length() > 1) words.add(s);
        }
        punct.sort((x, y) -> wc.get(y) - wc.get(x));
        for (String s : punct) {
            if (vocab.size() >= targetVocab) break;
            vocab.add(s); seen.add(s);
        }
        words.sort((x, y) -> wc.get(y) - wc.get(x));
        for (String s : words) {
            if (vocab.size() >= targetVocab) break;
            vocab.add(s);
        }
        Tokenizer t = new Tokenizer();
        t.itos = vocab.toArray(new String[0]);
        t.V = t.itos.length;
        for (int i = 0; i < t.V; i++) t.stoi.put(t.itos[i], i);
        return t;
    }

    public int[] encode(String text) {
        if (bpe) return encodeBPE(text);
        List<String> pcs = pieces(normalize(text));
        int[] out = new int[Math.max(16, pcs.size() * 2)];
        int n = 0;
        for (int i = 0; i < pcs.size(); i++) {
            Integer id = stoi.get(pcs.get(i));
            if (id != null) {
                if (n == out.length) out = java.util.Arrays.copyOf(out, out.length * 2);
                out[n++] = id;
            } else {
                // unknown word: split into characters (all a-z0-9 are in vocab)
                String pc = pcs.get(i);
                for (int c = 0; c < pc.length(); c++) {
                    if (n == out.length) out = java.util.Arrays.copyOf(out, out.length * 2);
                    Integer ci = stoi.get(String.valueOf(pc.charAt(c)));
                    out[n++] = (ci != null) ? ci : 0;
                }
            }
        }
        return java.util.Arrays.copyOf(out, n);
    }

    public String decode(int[] ids, int from, int to) {
        if (bpe) return decodeBPE(ids, from, to);
        StringBuilder sb = new StringBuilder();
        String prev = "";
        boolean prevSet = false;
        for (int i = from; i < to; i++) {
            String tok = itos[ids[i]];
            if (tok.equals(PAR)) {
                while (sb.length() > 0 && sb.charAt(sb.length() - 1) == ' ') sb.setLength(sb.length() - 1);
                if (sb.length() > 0) sb.append("\n\n");
                prevSet = false;
                continue;
            }
            if (tok.equals(EOS) || tok.equals(UNK)) continue;
            boolean space = sb.length() > 0 && prevSet
                    && !noSpaceAfter.contains(prev) && !noSpaceBefore.contains(tok);
            // Consecutive single-character pieces come from an unknown word
            // that encode() split into characters. Join them back into a word
            // instead of printing "w o r d".
            if (space && prev.length() == 1 && tok.length() == 1
                    && Character.isLetterOrDigit(prev.charAt(0))
                    && Character.isLetterOrDigit(tok.charAt(0))) {
                space = false;
            }
            if (space) sb.append(' ');
            sb.append(tok);
            prev = tok;
            prevSet = true;
        }
        return sb.toString();
    }

    // ---------------- byte-pair encoding ----------------

    static List<String> bpePieces(String text) {
        String s = normalize(text);
        ArrayList<String> out = new ArrayList<>();
        int i = 0, n = s.length();
        boolean pendingSpace = true;
        while (i < n) {
            char c = s.charAt(i);
            if (c == ' ' || c == '\t' || c == '\n') {
                int j = i, newlines = 0;
                while (j < n && (s.charAt(j) == ' ' || s.charAt(j) == '\t' || s.charAt(j) == '\n')) {
                    if (s.charAt(j) == '\n') newlines++;
                    j++;
                }
                if (newlines >= 2) out.add(PAR);
                pendingSpace = true;
                i = j;
            } else if (isWordChar(c)) {
                int j = i;
                while (j < n && isWordChar(s.charAt(j))) j++;
                String w = s.substring(i, j);
                while (w.length() > 1 && w.endsWith("'")) w = w.substring(0, w.length() - 1);
                while (w.length() > 1 && w.startsWith("'")) w = w.substring(1);
                if (!w.isEmpty()) out.add((pendingSpace ? SPACE_MARK : "") + w);
                pendingSpace = false;
                i = j;
            } else {
                out.add((pendingSpace ? SPACE_MARK : "") + c);
                pendingSpace = false;
                i++;
            }
        }
        return out;
    }

    private int[] encodePiece(String piece) {
        int[] cached = bpeCache.get(piece);
        if (cached != null) return cached;
        ArrayList<String> syms = new ArrayList<>();
        for (int i = 0; i < piece.length(); i++) syms.add(String.valueOf(piece.charAt(i)));
        while (syms.size() > 1) {
            int bestRank = Integer.MAX_VALUE, bestIdx = -1;
            for (int i = 0; i + 1 < syms.size(); i++) {
                Integer rank = mergeRanks.get(syms.get(i) + "\t" + syms.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            String left = syms.get(bestIdx), right = syms.get(bestIdx + 1);
            String merged = left + right;
            ArrayList<String> next = new ArrayList<>();
            for (int i = 0; i < syms.size();) {
                if (i + 1 < syms.size() && syms.get(i).equals(left)
                        && syms.get(i + 1).equals(right)) {
                    next.add(merged);
                    i += 2;
                } else {
                    next.add(syms.get(i));
                    i++;
                }
            }
            syms = next;
        }
        int[] ids = new int[Math.max(8, syms.size() * 2)];
        int n = 0;
        for (String sym : syms) {
            Integer id = stoi.get(sym);
            if (id != null) {
                if (n == ids.length) ids = java.util.Arrays.copyOf(ids, ids.length * 2);
                ids[n++] = id;
            } else {
                for (int i = 0; i < sym.length(); i++) {
                    Integer ci = stoi.get(String.valueOf(sym.charAt(i)));
                    if (n == ids.length) ids = java.util.Arrays.copyOf(ids, Math.max(8, ids.length * 2));
                    ids[n++] = (ci != null) ? ci : 0;
                }
            }
        }
        ids = java.util.Arrays.copyOf(ids, n);
        bpeCache.put(piece, ids);
        return ids;
    }

    private int[] encodeBPE(String text) {
        ArrayList<Integer> out = new ArrayList<>();
        for (String piece : bpePieces(text)) {
            if (piece.equals(PAR)) {
                out.add(stoi.get(PAR));
            } else {
                for (int id : encodePiece(piece)) out.add(id);
            }
        }
        int[] ids = new int[out.size()];
        for (int i = 0; i < ids.length; i++) ids[i] = out.get(i);
        return ids;
    }

    private String decodeBPE(int[] ids, int from, int to) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; i < to; i++) {
            String tok = itos[ids[i]];
            if (tok.equals(PAR)) sb.append("\n\n");
            else if (!tok.equals(EOS) && !tok.equals(UNK)) sb.append(tok.replace(SPACE_MARK, " "));
        }
        int start = 0;
        while (start < sb.length() && sb.charAt(start) == ' ') start++;
        return sb.substring(start);
    }

    /** Writes vocab to a stream; does NOT close the caller's stream. */
    public void save(OutputStream os) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < V; i++) sb.append(i).append('\t').append(itos[i]).append('\n');
        os.write(sb.toString().getBytes("UTF-8"));
        os.flush();
    }

    public void save(String path) throws IOException {
        try (OutputStream os = new FileOutputStream(path)) { save(os); }
    }

    /** Reads a BPE vocab plus its ordered merge list. */
    public static Tokenizer loadBPE(InputStream vocab, InputStream merges) throws IOException {
        Tokenizer t = load(vocab);
        t.bpe = true;
        t.mergeRanks = new HashMap<>();
        BufferedReader br = new BufferedReader(new InputStreamReader(merges, "UTF-8"));
        String line;
        int rank = 0;
        while ((line = br.readLine()) != null) {
            if (line.isEmpty()) continue;
            int tab = line.indexOf('\t');
            if (tab < 0) continue;
            t.mergeRanks.put(line.substring(0, tab) + "\t" + line.substring(tab + 1), rank++);
        }
        return t;
    }

    /** Reads vocab from a stream; does NOT close the caller's stream. */
    public static Tokenizer load(InputStream is) throws IOException {
        Tokenizer t = new Tokenizer();
        BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
        java.util.ArrayList<String> lines = new java.util.ArrayList<>();
        String ln;
        while ((ln = br.readLine()) != null) lines.add(ln);
        t.itos = new String[lines.size()];
        for (String l : lines) {
            int tab = l.indexOf('\t');
            int id = Integer.parseInt(l.substring(0, tab));
            t.itos[id] = l.substring(tab + 1);
        }
        t.V = t.itos.length;
        t.stoi = new HashMap<>();
        for (int i = 0; i < t.V; i++) t.stoi.put(t.itos[i], i);
        return t;
    }

    public static Tokenizer load(String path) throws IOException {
        try (InputStream is = new FileInputStream(path)) { return load(is); }
    }
}
