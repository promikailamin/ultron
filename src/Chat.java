package ultron;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Ultron chat CLI.
 *
 * Answers are produced in three layers:
 *   1. the curated conversation knowledge base (data/chat_kb.tsv),
 *   2. a lightweight retrieval index over the English grammar and Minecraft
 *      corpora (data/grammar.txt, data/mc.txt),
 *   3. neural next-token generation from the trained transformer checkpoint.
 *
 * The retrieval layers keep answers grounded even though the neural model is
 * intentionally small and runs on low-end hardware.
 */
public class Chat {
    private static final Set<String> STOP = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "done", "how", "what", "who", "whom", "whose",
            "where", "when", "why", "which", "can", "could", "would", "should",
            "will", "shall", "may", "might", "must", "to", "of", "in", "on", "at",
            "for", "with", "about", "from", "by", "as", "it", "its", "this",
            "that", "these", "those", "i", "me", "my", "mine", "we", "our",
            "ours", "you", "your", "yours", "he", "him", "his", "she", "her",
            "hers", "they", "them", "their", "theirs", "and", "or", "but", "if",
            "then", "than", "so", "because", "have", "has", "had", "not", "no",
            "yes", "please", "tell", "give", "some", "any", "there", "here",
            "also", "very", "just", "really", "want", "need", "know", "like"));

    static class Entry {
        final String category, question, answer, norm;
        final Set<String> tokens;

        Entry(String category, String question, String answer) {
            this.category = category;
            this.question = question;
            this.answer = answer;
            this.norm = norm(question);
            this.tokens = tokens(question);
        }
    }

    static class Passage {
        final String text;
        final Set<String> tokens;

        Passage(String text, Set<String> tokens) {
            this.text = text;
            this.tokens = tokens;
        }
    }

    private final List<Entry> kb = new ArrayList<>();
    private final List<Passage> passages = new ArrayList<>();
    private final Map<String, List<Integer>> index = new HashMap<>();

    private Tokenizer tok;
    private Ultron model;
    private Ultron.KVCache cache;
    private Sampler sampler;
    private long seed = 2026;

    public static void main(String[] args) throws Exception {
        Chat chat = new Chat();
        chat.load();
        if (args.length > 0) {
            System.out.println(chat.reply(String.join(" ", args)));
            return;
        }
        System.out.println("Ultron chat - trained English + Minecraft Java Edition assistant");
        System.out.println("Type :help for commands, :quit to exit.");
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        while (true) {
            System.out.print("you> ");
            System.out.flush();
            String line = br.readLine();
            if (line == null) break;
            line = line.trim();
            if (line.isEmpty()) continue;
            if (line.equals(":quit") || line.equals(":exit") || line.equals(":q")) break;
            if (line.equals(":help")) {
                System.out.println("Commands: :help, :quit. Ask about English grammar, writing, or Minecraft.");
                continue;
            }
            System.out.println("ultron> " + chat.reply(line));
        }
    }

    void load() throws IOException {
        loadKb();
        loadPassages("data/grammar.txt");
        loadPassages("data/mc.txt");

        tok = Tokenizer.load("data/ultron-vocab.txt");
        model = Ultron.load("out/ultron.bin");
        if (model.V != tok.V) {
            throw new IOException("vocab/model mismatch: " + model.V + " vs " + tok.V);
        }
        cache = model.newCache();
        sampler = new Sampler(model.V, seed);
    }

    private void loadKb() throws IOException {
        Path p = Paths.get("data/chat_kb.tsv");
        if (!Files.exists(p)) return;
        for (String line : Files.readAllLines(p, StandardCharsets.UTF_8)) {
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\t", 3);
            if (parts.length < 3) continue;
            kb.add(new Entry(parts[0].trim(), parts[1].trim(), parts[2].trim()));
        }
    }

    private void loadPassages(String file) throws IOException {
        Path p = Paths.get(file);
        if (!Files.exists(p)) return;
        String text = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
        for (String block : text.split("\\n\\s*\\n")) {
            String clean = block.replaceAll("\\s+", " ").trim();
            if (clean.length() < 80 || clean.length() > 2400) continue;
            if (clean.startsWith("***") || clean.startsWith("=====")) continue;
            Set<String> kw = tokens(clean);
            if (kw.size() < 5) continue;
            int id = passages.size();
            passages.add(new Passage(clean, kw));
            for (String t : kw) index.computeIfAbsent(t, k -> new ArrayList<>()).add(id);
        }
    }

    String reply(String input) {
        String clean = input == null ? "" : input.trim();
        if (clean.isEmpty()) return "Ask me a question about English grammar, writing, or Minecraft Java Edition.";

        Entry best = bestEntry(clean);
        if (best != null) return best.answer;

        String grounded = retrieve(clean);
        if (grounded != null) return grounded;

        // A very small next-token model hallucinates on factual questions it
        // has never seen. Only let it speak freely for conversational text.
        if (!wantsFacts(clean)) {
            String generated = generate(clean);
            if (generated != null && looksUsable(generated)) return generated;
        }

        return "I am not sure about that yet. I know English grammar, writing, and Minecraft Java Edition. "
                + "Try asking something like: What is a noun? How do I make a crafting table? What is redstone?";
    }

    private static boolean wantsFacts(String s) {
        String n = norm(s);
        if (s.contains("?")) return true;
        for (String w : new String[]{"what", "who", "whom", "whose", "when", "where", "why",
                "which", "how", "explain", "define", "describe", "tell", "list", "compare"}) {
            if (n.equals(w) || n.startsWith(w + " ")) return true;
        }
        return false;
    }

    private Entry bestEntry(String input) {
        String ni = norm(input);
        Set<String> it = tokens(input);
        Entry best = null;
        double bestScore = 0;
        for (Entry e : kb) {
            double s;
            if (ni.equals(e.norm)) {
                s = 10000;
            } else if (ni.length() >= 4 && (ni.contains(e.norm) || e.norm.contains(ni))) {
                s = 5000 + Math.min(ni.length(), e.norm.length());
            } else {
                int inter = 0;
                for (String t : it) if (e.tokens.contains(t)) inter++;
                if (inter == 0) continue;
                int union = it.size() + e.tokens.size() - inter;
                double jaccard = (double) inter / Math.max(1, union);
                double coverage = (double) inter / Math.max(1, it.size());
                s = 100 * jaccard + 100 * coverage;
            }
            if (s > bestScore) {
                bestScore = s;
                best = e;
            }
        }
        return bestScore >= 70 ? best : null;
    }

    private String retrieve(String input) {
        if (passages.isEmpty()) return null;
        Set<String> qt = tokens(input);
        if (qt.isEmpty()) return null;

        int n = passages.size();
        HashMap<Integer, Double> scores = new HashMap<>();
        HashMap<Integer, Integer> hits = new HashMap<>();
        double totalWeight = 0;
        for (String t : qt) {
            List<Integer> ids = index.get(t);
            if (ids == null || ids.isEmpty()) continue;
            double idf = Math.log((n + 1.0) / (ids.size() + 1.0)) + 1.0;
            totalWeight += idf;
            for (int id : ids) {
                scores.merge(id, idf, Double::sum);
                hits.merge(id, 1, Integer::sum);
            }
        }
        if (totalWeight == 0) return null;

        int bestId = -1;
        double bestScore = 0;
        int bestHits = 0;
        for (Map.Entry<Integer, Double> e : scores.entrySet()) {
            int h = hits.getOrDefault(e.getKey(), 0);
            double coverage = e.getValue() / totalWeight;
            double s = coverage + 0.02 * Math.min(h, 6);
            int len = passages.get(e.getKey()).tokens.size();
            if (len > 0) s *= 1.0 / (1.0 + len / 500.0);
            if (h >= 2 && s > bestScore) {
                bestScore = s;
                bestId = e.getKey();
                bestHits = h;
            }
        }
        if (bestId < 0 || bestHits < 2 || bestScore < 0.28) return null;

        String answer = bestSentences(passages.get(bestId), qt);
        return answer == null ? null : "Here is what I know: " + answer;
    }

    private String bestSentences(Passage p, Set<String> qt) {
        String[] sentences = p.text.split("(?<=[.!?])\\s+");
        if (sentences.length == 0) return null;

        double bestScore = 0;
        int bestIdx = -1;
        for (int i = 0; i < sentences.length; i++) {
            String s = sentences[i].trim();
            if (s.length() < 25 || s.length() > 700) continue;
            if (s.startsWith("=") || s.startsWith("|") || s.startsWith("File:")) continue;
            Set<String> st = tokens(s);
            if (st.isEmpty()) continue;
            int inter = 0;
            for (String t : st) if (qt.contains(t)) inter++;
            double score = (double) inter / st.size() + 0.02 * inter;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        if (bestIdx < 0) {
            String fallback = p.text;
            return fallback.length() <= 400 ? fallback : fallback.substring(0, 397) + "...";
        }

        StringBuilder out = new StringBuilder(sentences[bestIdx].trim());
        if (bestIdx + 1 < sentences.length && out.length() < 320) {
            String next = sentences[bestIdx + 1].trim();
            if (next.length() > 20 && !next.startsWith("=") && out.length() + next.length() < 560) {
                out.append(' ').append(next);
            }
        }
        if (out.length() > 620) out.setLength(617);
        return out.toString();
    }

    private String generate(String input) {
        String prompt = "\n\nuser: " + input + "\nultron:";
        int[] ids = tok.encode(prompt);
        int promptLen = ids.length;
        cache.clear();
        float[] logits = null;
        for (int id : ids) logits = model.step(id, cache);

        int[] out = new int[promptLen + 90];
        System.arraycopy(ids, 0, out, 0, promptLen);
        int len = promptLen;
        int PAR = tok.stoi.get(Tokenizer.PAR);
        int EOS = tok.stoi.get(Tokenizer.EOS);
        sampler = new Sampler(model.V, seed++);

        for (int i = 0; i < 80 && len < out.length; i++) {
            int next = sampler.sample(logits, model.V, 0.65f, 40);
            if (next == PAR || next == EOS) break;
            out[len++] = next;
            logits = model.step(next, cache);
        }

        String text = tok.decode(out, promptLen, len).trim();
        int turn = text.toLowerCase(Locale.ROOT).indexOf("user:");
        if (turn >= 0) text = text.substring(0, turn).trim();
        if (text.endsWith(":")) text = text.substring(0, text.length() - 1).trim();
        return text;
    }

    private static boolean looksUsable(String s) {
        if (s == null) return false;
        s = s.trim();
        if (s.length() < 12) return false;
        String low = s.toLowerCase(Locale.ROOT);
        if (low.contains("ultron:") || low.contains("user:")) return false;

        // Reject degenerate loops such as "the the the the".
        String[] words = low.split("\\s+");
        HashMap<String, Integer> grams = new HashMap<>();
        int worst = 0;
        int singles = 0, letters = 0;
        for (String w : words) {
            if (w.length() == 1 && Character.isLetterOrDigit(w.charAt(0))) singles++;
            letters += w.replaceAll("[^a-z0-9]", "").length();
        }
        for (int i = 0; i + 3 < words.length; i++) {
            String g = words[i] + " " + words[i + 1] + " " + words[i + 2] + " " + words[i + 3];
            int c = grams.merge(g, 1, Integer::sum);
            if (c > worst) worst = c;
        }
        if (worst >= 4) return false;
        if (words.length >= 6 && singles * 4 > words.length) return false;
        double avgWordLength = words.length == 0 ? 0 : (double) letters / words.length;
        return !(words.length >= 8 && avgWordLength < 3.0);
    }

    static String norm(String s) {
        return s.toLowerCase(Locale.ROOT)
                .replaceAll("[^a-z0-9' ]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    static Set<String> tokens(String s) {
        LinkedHashSet<String> out = new LinkedHashSet<>();
        for (String raw : norm(s).split(" ")) {
            if (raw.isEmpty()) continue;
            String t = stem(raw);
            if (t.length() > 1 && !STOP.contains(t)) out.add(t);
        }
        if (out.isEmpty()) {
            for (String raw : norm(s).split(" ")) if (!raw.isEmpty()) out.add(raw);
        }
        return out;
    }

    static String stem(String w) {
        if (w.length() > 5 && w.endsWith("ing")) return w.substring(0, w.length() - 3);
        if (w.length() > 4 && w.endsWith("ies")) return w.substring(0, w.length() - 3) + "y";
        if (w.length() > 4 && w.endsWith("es")) return w.substring(0, w.length() - 2);
        if (w.length() > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.substring(0, w.length() - 1);
        if (w.length() > 4 && w.endsWith("ed")) return w.substring(0, w.length() - 2);
        return w;
    }
}
