package ultron;

import java.io.*;
import java.nio.charset.StandardCharsets;
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
            "also", "very", "just", "really", "want", "need", "know", "like",
            "get", "got", "gotten", "make", "made", "craft", "find", "use",
            "create", "build", "obtain", "give", "take", "put", "work",
            "works", "exist", "exists", "happen", "happens"));

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

    /** Receives decoded text as the neural model produces it. */
    interface TokenSink {
        void token(String text) throws IOException;
    }

    static class Generated {
        final String text;
        final List<String> pieces;

        Generated(String text, List<String> pieces) {
            this.text = text;
            this.pieces = pieces;
        }
    }

    private final List<Entry> kb = new ArrayList<>();
    private final List<Passage> passages = new ArrayList<>();
    private final Map<String, List<Integer>> index = new HashMap<>();

    private Tokenizer tok;
    private Ultron model;
    private Ultron.KVCache cache;
    private Sampler sampler;
    private Memory memory;
    private long seed = 2026;

    public static void main(String[] args) throws Exception {
        Chat chat = new Chat();
        chat.load();
        try {
            if (args.length > 0) {
                System.out.println(chat.converse(String.join(" ", args)));
                return;
            }
            chat.runInteractive();
        } finally {
            if (chat.memory != null) chat.memory.close();
        }
    }

    private void runInteractive() throws IOException {
        System.out.println("Ultron chat - trained English + Minecraft Java Edition assistant");
        System.out.println("Session: " + memory.sessionId() + "  (" + memory.directory() + ")");
        System.out.println("Type :help for commands, :quit to exit. Sessions are saved automatically.");
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
                System.out.println("Commands: :help, :quit, :new, :sessions, :load <id>, :memory, :forget <key|all>");
                System.out.println("Ask about English grammar, writing, Minecraft, or tell me about yourself.");
                continue;
            }
            if (line.equals(":new")) {
                System.out.println("Started session " + memory.newSession());
                continue;
            }
            if (line.equals(":sessions")) {
                List<String> ids = memory.listSessions(20);
                if (ids.isEmpty()) System.out.println("No saved sessions yet.");
                for (String id : ids) {
                    String mark = id.equals(memory.sessionId()) ? " *" : "";
                    System.out.println(id + mark);
                }
                continue;
            }
            if (line.startsWith(":load ")) {
                String id = line.substring(6).trim();
                try {
                    memory.loadSession(id);
                    System.out.println("Loaded session " + memory.sessionId());
                } catch (IOException e) {
                    System.out.println("Could not load session: " + e.getMessage());
                }
                continue;
            }
            if (line.equals(":memory")) {
                System.out.println(memory.summary());
                continue;
            }
            if (line.startsWith(":forget")) {
                String key = line.length() > 7 ? line.substring(7).trim() : "all";
                memory.forget(key);
                System.out.println("Memory updated.");
                continue;
            }
            System.out.print("ultron> ");
            System.out.flush();
            String answer = converse(line, piece -> {
                System.out.print(piece);
                System.out.flush();
            });
            if (answer == null || answer.isEmpty()) System.out.println();
            else System.out.println();
        }
    }

    private String converse(String input) throws IOException {
        return converse(input, null);
    }

    private String converse(String input, TokenSink sink) throws IOException {
        memory.append("user", input);
        String answer = reply(input, sink);
        if (sink != null && (answer == null || answer.isEmpty())) sink.token("I am not sure about that yet.");
        memory.append("ultron", answer);
        return answer;
    }

    void load() throws IOException {
        loadKb();
        loadPassages("grammar.txt", "data/grammar.txt");
        loadPassages("mc.txt", "data/mc.txt");
        loadQaPassages("mc_qa.txt", "data/mc_qa.txt");

        try (InputStream is = Resources.open("ultron-vocab.txt", "data/ultron-vocab.txt")) {
            tok = Tokenizer.load(is);
        }
        try (InputStream is = Resources.open("ultron.bin", "out/ultron.bin")) {
            model = Ultron.load(is);
        }
        if (model.V != tok.V) {
            throw new IOException("vocab/model mismatch: " + model.V + " vs " + tok.V);
        }
        cache = model.newCache();
        sampler = new Sampler(model.V, seed);
        memory = Memory.openDefault();
    }

    private void loadKb() throws IOException {
        if (!Resources.exists("chat_kb.tsv", "data/chat_kb.tsv")) return;
        try (BufferedReader br = Resources.reader("chat_kb.tsv", "data/chat_kb.tsv")) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\t", 3);
                if (parts.length < 3) continue;
                kb.add(new Entry(parts[0].trim(), parts[1].trim(), parts[2].trim()));
            }
        }
    }

    private void loadPassages(String resource, String fallback) throws IOException {
        if (!Resources.exists(resource, fallback)) return;
        String text = Resources.readText(resource, fallback);
        for (String block : text.split("\\n\\s*\\n")) {
            String clean = block.replaceAll("\\s+", " ").trim();
            if (clean.length() < 80 || clean.length() > 2400) continue;
            if (clean.startsWith("***") || clean.startsWith("=====")) continue;
            if (isWikiNoise(clean)) continue;
            Set<String> kw = tokens(clean);
            if (kw.size() < 5) continue;
            addPassage(clean, kw);
        }
    }

    /** Curated wiki Q&A gives retrieval a clean, short answer for each fact. */
    private void loadQaPassages(String resource, String fallback) throws IOException {
        if (!Resources.exists(resource, fallback)) return;
        try (BufferedReader br = Resources.reader(resource, fallback)) {
            String line;
            String question = null;
            StringBuilder answer = new StringBuilder();
            while ((line = br.readLine()) != null) {
                if (line.startsWith("user: ")) {
                    if (question != null) addQaPassage(question, answer.toString());
                    question = line.substring(6).trim();
                    answer.setLength(0);
                } else if (line.startsWith("ultron: ")) {
                    if (answer.length() > 0) answer.append(' ');
                    answer.append(line.substring(8).trim());
                } else if (line.trim().isEmpty()) {
                    if (question != null) addQaPassage(question, answer.toString());
                    question = null;
                    answer.setLength(0);
                }
            }
            if (question != null) addQaPassage(question, answer.toString());
        }
    }

    private void addQaPassage(String question, String answer) {
        String clean = answer.replaceAll("\\s+", " ").trim();
        if (clean.length() < 40 || clean.length() > 1800) return;
        if (isWikiNoise(clean)) return;
        Set<String> kw = tokens(question + " " + clean);
        if (kw.isEmpty()) return;
        addPassage(clean, kw);
    }

    private void addPassage(String text, Set<String> kw) {
        int id = passages.size();
        passages.add(new Passage(text, kw));
        for (String t : kw) index.computeIfAbsent(t, k -> new ArrayList<>()).add(id);
    }

    private static boolean isWikiNoise(String s) {
        String low = s.toLowerCase(Locale.ROOT);
        return low.contains("maintained on the bug tracker")
                || low.contains("issues relating to")
                || low.startsWith("navigation")
                || low.startsWith("references")
                || low.startsWith("gallery")
                || low.startsWith("data values")
                || low.startsWith("sounds")
                || low.startsWith("history")
                || low.startsWith("achievements")
                || low.startsWith("advancements")
                || low.startsWith("videos")
                || low.startsWith("see also")
                || low.contains("this article is about")
                || low.contains("this page is about");
    }

    String reply(String input) {
        return reply(input, null);
    }

    String reply(String input, TokenSink sink) {
        String clean = input == null ? "" : input.trim();
        if (clean.isEmpty()) {
            return emit(sink, "Ask me a question about English grammar, writing, or Minecraft Java Edition.");
        }

        Memory.Result learned = memory == null ? null : memory.learn(clean);
        boolean question = clean.contains("?") || wantsFacts(clean);
        if (learned != null && learned.changed && !question) {
            return emit(sink, learnedReply(learned));
        }
        String recalled = memory == null ? null : memory.recall(clean);
        if (recalled != null) return emit(sink, recalled);
        if (learned != null && learned.changed) return emit(sink, learnedReply(learned));

        // High-confidence FAQ answers stay exact. Everything else is written
        // by the transformer, one token at a time, from the prompt.
        Entry strong = strongEntry(clean);
        if (strong != null) return emit(sink, strong.answer);

        if (knownDomain(clean)) {
            Generated generated = generate(clean);
            if (generated != null && looksUsable(generated.text)
                    && looksRelevant(clean, generated.text)) {
                stream(generated.pieces, sink);
                return generated.text;
            }
        }

        Entry best = bestEntry(clean);
        if (best != null) return emit(sink, best.answer);

        String grounded = retrieve(clean);
        if (grounded != null) return emit(sink, grounded);

        return emit(sink, "I am not sure about that yet. I know English grammar, writing, and Minecraft Java Edition. "
                + "Try asking something like: What is a noun? How do I make a crafting table? What is redstone?");
    }

    /**
     * Emit a grounded (non-sampled) answer. The interactive UI still receives
     * it in small pieces so exact FAQ answers and neural replies share the
     * same token-streaming presentation.
     */
    private static String emit(TokenSink sink, String text) {
        if (sink != null && text != null) {
            try {
                String[] words = text.split(" ");
                for (int i = 0; i < words.length; i++) {
                    if (words[i].isEmpty()) continue;
                    sink.token(i == 0 ? words[i] : " " + words[i]);
                }
            } catch (IOException ignored) {
                // The interactive loop will notice a closed stdout on its own.
            }
        }
        return text;
    }

    private static void stream(List<String> pieces, TokenSink sink) {
        if (sink == null) return;
        boolean started = false;
        try {
            for (String piece : pieces) {
                if (!started) {
                    String trimmed = piece.replaceFirst("^\\s+", "");
                    if (trimmed.isEmpty()) continue;
                    piece = trimmed;
                    started = true;
                }
                sink.token(piece);
            }
        } catch (IOException ignored) {
            // Keep the already generated answer; the caller still logs it.
        }
    }

    /** True when at least one content word is in the model vocabulary. */
    private boolean knownDomain(String input) {
        if (tok == null) return false;
        for (String t : tokens(input)) {
            if (tok.stoi.containsKey(t)) return true;
        }
        for (String raw : norm(input).split(" ")) {
            if (raw.length() > 1 && !STOP.contains(raw) && tok.stoi.containsKey(raw)) return true;
        }
        return false;
    }

    private static boolean looksRelevant(String question, String answer) {
        Set<String> qt = tokens(question);
        if (qt.isEmpty()) return true;
        Set<String> at = tokens(answer);
        for (String t : qt) {
            if (at.contains(t)) return true;
            for (String a : at) {
                if (a.length() >= 4 && (a.contains(t) || t.contains(a))) return true;
            }
        }
        return false;
    }

    /** Only exact or substring FAQ matches are treated as high confidence. */
    private Entry strongEntry(String input) {
        String ni = norm(input);
        for (Entry e : kb) {
            if (ni.equals(e.norm)) return e;
            if (ni.length() >= 4 && (ni.contains(e.norm) || e.norm.contains(ni))) return e;
        }
        return null;
    }

    private static String learnedReply(Memory.Result learned) {
        if (learned.name != null) return "Nice to meet you, " + learned.name + ".";
        return "Got it. I will remember that.";
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
        int need = Math.min(2, qt.size());
        if (bestId < 0 || bestHits < need || bestScore < 0.28) return null;

        String answer = bestSentences(passages.get(bestId), qt);
        return answer == null ? null : answer;
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
            if (isWikiNoise(s)) continue;
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

    private Generated generate(String input) {
        StringBuilder prompt = new StringBuilder("\n\n");
        boolean appendedUser = false;
        if (memory != null) {
            String facts = memory.promptSummary();
            if (!facts.isEmpty()) prompt.append("ultron remembers: ").append(facts).append('\n');
            String history = memory.recentDialogue(6);
            if (!history.isEmpty()) {
                prompt.append(history).append('\n');
                appendedUser = "user".equals(memory.lastTurnRole());
            }
        }
        if (appendedUser) prompt.append("ultron:");
        else prompt.append("user: ").append(input).append("\nultron:");

        int[] ids = tok.encode(prompt.toString());
        if (ids.length > model.T - 1) {
            ids = Arrays.copyOfRange(ids, ids.length - (model.T - 1), ids.length);
        }
        int promptLen = ids.length;
        cache.clear();
        float[] logits = null;
        for (int id : ids) logits = model.step(id, cache);

        int[] out = new int[promptLen + 120];
        System.arraycopy(ids, 0, out, 0, promptLen);
        int len = promptLen;
        int PAR = tok.stoi.get(Tokenizer.PAR);
        int EOS = tok.stoi.get(Tokenizer.EOS);
        sampler = new Sampler(model.V, seed++);

        StringBuilder text = new StringBuilder();
        ArrayList<String> pieces = new ArrayList<>();
        String prev = "";
        boolean done = false;
        for (int i = 0; i < 110 && len < out.length && !done; i++) {
            int next = sampler.sample(logits, model.V, 0.72f, 48, 0.92f,
                    out, promptLen, len - promptLen, 1.12f);
            if (next == PAR || next == EOS) break;
            out[len++] = next;
            String cur = tok.decode(out, promptLen, len);
            String delta = cur.substring(Math.min(prev.length(), cur.length()));
            String candidate = text.toString() + delta;
            String low = candidate.toLowerCase(Locale.ROOT);
            int cut = low.indexOf("user:");
            if (cut < 0) cut = low.indexOf("ultron:");
            if (cut >= 0) {
                candidate = candidate.substring(0, cut);
                String extra = candidate.substring(text.length());
                if (!extra.isEmpty()) pieces.add(extra);
                text.setLength(0);
                text.append(candidate);
                done = true;
                break;
            }
            text.setLength(0);
            text.append(candidate);
            if (!delta.isEmpty()) pieces.add(delta);
            prev = cur;
            logits = model.step(next, cache);
        }

        String result = text.toString().trim();
        if (result.endsWith(":")) result = result.substring(0, result.length() - 1).trim();
        // Remove a leading space from the first streamed piece.
        if (!pieces.isEmpty()) {
            String first = pieces.get(0).replaceFirst("^\\s+", "");
            if (first.isEmpty()) pieces.remove(0);
            else pieces.set(0, first);
        }
        return new Generated(result, pieces);
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
