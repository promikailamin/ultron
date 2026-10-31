package ultron;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Ultron chat CLI.
 *
 * High-confidence questions retrieve a short passage from the curated,
 * intent-aware knowledge base and then generate a new answer with the
 * transformer checkpoint. The prompt is encoded, the network processes it,
 * and the answer is sampled one token at a time until the model emits its
 * end-of-sequence token. Follow-up questions such as "and how can i craft it"
 * inherit the previous topic. When the local checkpoint is not good enough yet,
 * a grounded composition fallback writes a fresh sentence from the passage; it
 * never replays the stored answer.
 */
public class Chat {
    /** Receives decoded text as the model produces it. */
    interface TokenSink {
        void token(String text) throws IOException;
    }

    private static final int MAX_ANSWER_TOKENS = 48;
    private static final int MIN_PROMPT_TOKENS = 12;
    private static final String FALLBACK =
            "I am not sure. I don't have that in my notes. Ask me about English grammar or Minecraft Java Edition.";
    private static final String NO_NOTES =
            "I am not sure. I don't have that in my notes.";

    private Tokenizer tok;
    private Ultron model;
    private Ultron.KVCache cache;
    private KnowledgeBase kb;
    private Memory memory;
    private String lastTopic;
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
        System.out.println("Ultron chat - generative English + Minecraft Java Edition model");
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
        String answer = reply(input, sink);
        memory.append("user", input);
        memory.append("ultron", answer);
        if (lastTopic != null && !lastTopic.isEmpty()) memory.append("topic", lastTopic);
        return answer;
    }

    void load() throws IOException {
        tok = Tokenizer.loadDefault();
        try (InputStream is = Resources.open("ultron.bin", "out/ultron.bin")) {
            model = Ultron.load(is);
        }
        if (model.V != tok.V) {
            throw new IOException("vocab/model mismatch: " + model.V + " vs " + tok.V);
        }
        cache = model.newCache();
        kb = KnowledgeBase.load();
        memory = Memory.openDefault();
    }

    /** Clears the pronoun topic between independent evaluation cases. */
    void resetTopic() {
        lastTopic = null;
    }

    String reply(String input) {
        return reply(input, null);
    }

    String reply(String input, TokenSink sink) {
        String clean = input == null ? "" : input.trim();
        if (clean.isEmpty()) {
            return emit(sink, "Ask me a question about English grammar, writing, or Minecraft Java Edition.");
        }

        String social = socialReply(clean);
        if (social != null) {
            stream(social, sink);
            return social;
        }

        Memory.Result learned = memory == null ? null : memory.learn(clean);
        if (learned != null && memory != null) {
            String remembered = memoryReply(clean, learned);
            if (remembered != null) {
                stream(remembered, sink);
                return remembered;
            }
        }

        String expanded = expandFollowup(clean);
        boolean followup = !expanded.equals(clean);
        if (!followup) {
            String topic = KnowledgeBase.extractTopic(clean);
            if (!topic.isEmpty()) lastTopic = topic;
        }

        String context = kb == null ? null : kb.retrieve(expanded, 2);
        if (context == null || context.isEmpty()) {
            String none = learned != null && learned.changed ? learnedReply(learned) : NO_NOTES;
            stream(none, sink);
            return none;
        }

        String best = null;
        for (int attempt = 0; attempt < 3; attempt++) {
            String generated = generate(expanded, attempt, context);
            if (generated != null && looksUsable(generated)
                    && (kb == null || kb.covers(expanded, generated))) {
                best = generated;
                break;
            }
        }

        if (best == null) {
            String composed = Paraphrase.compose(expanded, context, (int) (seed++ & 0x7fffffff));
            if (composed != null && looksUsable(composed)) {
                best = composed;
            }
        }

        if (best == null) best = NO_NOTES;
        stream(best, sink);
        return best;
    }

    /** Short, hardcoded conversational replies. No knowledge-base text here. */
    private static String socialReply(String input) {
        String q = KnowledgeBase.normalize(input);
        if (q.isEmpty()) return null;
        if (q.matches("^(?:hi|hello|hey|yo|hiya|good morning|good afternoon|good evening)(?: there)?$")) {
            return "Hello! I am Ultron. Ask me about English grammar, writing, or Minecraft Java Edition.";
        }
        if (q.equals("who are you") || q.equals("what are you")
                || q.equals("what is your name")) {
            return "I am Ultron, a small transformer language model written from scratch in Java.";
        }
        if (q.equals("what can you do")) {
            return "I can explain English grammar and Minecraft Java Edition, help with writing, and generate short answers with my neural language model.";
        }
        if (q.equals("how are you")) {
            return "I am running well and ready to help.";
        }
        if (q.equals("thank you") || q.equals("thanks")) {
            return "You are welcome.";
        }
        if (q.equals("bye") || q.equals("goodbye") || q.equals("good night")) {
            return "Goodbye! Have a great day.";
        }
        return null;
    }

    private String memoryReply(String input, Memory.Result learned) {
        String q = KnowledgeBase.normalize(input);
        if (learned.name != null) return "Nice to meet you, " + learned.name + ".";
        if (q.contains("what is my name") || q.contains("who am i")) {
            String name = memory == null ? "" : memory.fact("name");
            if (!name.isEmpty()) return "Your name is " + name + ".";
        }
        if (q.contains("what do i like") || q.contains("what do i love")) {
            String likes = memory == null ? "" : memory.fact("likes");
            if (!likes.isEmpty()) return "You told me you like " + likes + ".";
        }
        return learned.changed ? learnedReply(learned) : null;
    }

    /** Appends the active topic when the question uses a pronoun or "and". */
    private String expandFollowup(String input) {
        if (kb == null || !kb.isFollowup(input)) return input;
        String topic = currentTopic();
        if (topic == null || topic.isEmpty()) return input;
        return input + " " + topic;
    }

    private String currentTopic() {
        if (lastTopic != null && !lastTopic.isEmpty()) return lastTopic;
        if (memory == null) return "";
        for (String prior : memory.recentUserTexts(8)) {
            if (kb.isFollowup(prior)) continue;
            String topic = KnowledgeBase.extractTopic(prior);
            if (!topic.isEmpty()) {
                lastTopic = topic;
                return topic;
            }
        }
        return "";
    }

    /**
     * Runs the transformer. The prompt contains the current question plus a
     * short amount of dialogue and memory context; the answer is sampled token
     * by token from the network's next-token distribution.
     */
    private String generate(String input, int attempt, String context) {
        int budget = Math.max(MIN_PROMPT_TOKENS, model.T - MAX_ANSWER_TOKENS);
        String prompt = buildPrompt(input, context, budget);
        int[] ids = tok.encode(prompt);
        if (ids.length > budget) {
            ids = Arrays.copyOfRange(ids, ids.length - budget, ids.length);
        }
        int promptLen = ids.length;
        long baseSeed = seed++;

        cache.clear();
        float[] logits = null;
        for (int id : ids) logits = model.step(id, cache);

        int maxNew = Math.max(8, model.T - promptLen);
        int[] generated = new int[maxNew];
        int len = 0;
        int eos = tok.stoi.get(Tokenizer.EOS);
        int par = tok.stoi.get(Tokenizer.PAR);
        Sampler sampler = new Sampler(model.V, baseSeed + attempt * 7919L);

        float temperature = attempt == 0 ? 0.62f : 0.78f;
        int topK = attempt == 0 ? 40 : 60;

        String previous = "";
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < maxNew; i++) {
            int next = sampler.sample(logits, model.V, temperature, topK, 0.92f,
                    generated, 0, len, 1.2f);
            if (next == eos || next == par) break;
            generated[len++] = next;
            String decoded = tok.decode(generated, 0, len);
            if (decoded.length() > previous.length()) {
                text.append(decoded.substring(previous.length()));
                previous = decoded;
            }
            String low = text.toString().toLowerCase(Locale.ROOT);
            int cut = low.indexOf("user:");
            if (cut < 0) cut = low.indexOf("ultron:");
            if (cut >= 0) {
                text.setLength(cut);
                break;
            }
            if (len > 16 && len % 4 == 0 && hasRepeatedNgram(text.toString(), 4, 3)) break;
            logits = model.step(next, cache);
        }

        String result = postProcess(text.toString());
        if (result.length() < 3) return null;
        return result;
    }

    private static boolean hasRepeatedNgram(String text, int n, int limit) {
        String[] words = text.trim().toLowerCase(Locale.ROOT).split("\\s+");
        if (words.length < n * limit) return false;
        HashMap<String, Integer> counts = new HashMap<>();
        for (int i = 0; i + n <= words.length; i++) {
            StringBuilder gram = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) gram.append(' ');
                gram.append(words[i + j]);
            }
            if (counts.merge(gram.toString(), 1, Integer::sum) >= limit) return true;
        }
        return false;
    }

    private String buildPrompt(String input, String context, int budget) {
        StringBuilder prompt = new StringBuilder();
        if (context != null && !context.isEmpty()) {
            String trimmed = fitContext(context, input, budget);
            if (!trimmed.isEmpty()) prompt.append("context: ").append(trimmed).append('\n');
        }
        if (memory != null) {
            String facts = memory.promptSummary();
            if (!facts.isEmpty()) {
                prompt.append("ultron remembers: ").append(facts).append('\n');
            }
            String history = memory.recentDialogue(6);
            for (String line : history.split("\n")) {
                if (line.startsWith("user: ")) prompt.append(line).append('\n');
            }
        }
        prompt.append("user: ").append(input).append("\nultron:");
        return prompt.toString();
    }

    /** Keeps the question and trims the retrieved passage first. */
    private String fitContext(String context, String input, int budget) {
        String suffix = buildPrompt(input, "", budget);
        int suffixLen = tok.encode(suffix).length;
        int prefixLen = tok.encode("context: \n").length;
        int available = budget - suffixLen - prefixLen;
        if (available <= 8) return "";
        int[] ids = tok.encode(context);
        if (ids.length <= available) return context;
        String cut = tok.decode(ids, 0, available);
        int stop = Math.max(cut.lastIndexOf(". "), Math.max(cut.lastIndexOf("! "), cut.lastIndexOf("? ")));
        if (stop > 40) cut = cut.substring(0, stop + 1);
        else {
            int space = cut.lastIndexOf(' ');
            if (space > 0) cut = cut.substring(0, space);
        }
        return cut.trim();
    }

    private static String postProcess(String raw) {
        String text = raw.replaceAll("\\s+", " ").trim();
        text = text.replaceAll("^[\\s:;,.\\-]+", "");
        text = text.replaceAll("\\s+([,.!?;:])", "$1");
        text = text.replaceAll("([,.!?;:])(?=[A-Za-z])", "$1 ");
        text = text.replaceAll("\\bi\\b", "I");
        text = capitalizeSentences(text);
        if (!text.isEmpty()) {
            char last = text.charAt(text.length() - 1);
            if (Character.isLetterOrDigit(last) || last == '\'' || last == '"') {
                text += ".";
            }
        }
        return text.trim();
    }

    private static String capitalizeSentences(String s) {
        StringBuilder out = new StringBuilder(s.length());
        boolean start = true;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (start && Character.isLetter(c)) {
                out.append(Character.toUpperCase(c));
                start = false;
            } else {
                out.append(c);
                if (c == '.' || c == '!' || c == '?') start = true;
            }
        }
        return out.toString();
    }

    private static boolean looksUsable(String s) {
        if (s == null) return false;
        s = s.trim();
        if (s.length() < 8) return false;
        String low = s.toLowerCase(Locale.ROOT);
        if (low.contains("user:") || low.contains("ultron:")) return false;

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
        if (worst >= 3) return false;
        if (words.length >= 6 && singles * 4 > words.length) return false;
        double avgWordLength = words.length == 0 ? 0 : (double) letters / words.length;
        if (words.length >= 8 && avgWordLength < 3.0) return false;
        return letters >= 6;
    }

    private static String learnedReply(Memory.Result learned) {
        if (learned.name != null) return "Nice to meet you, " + learned.name + ".";
        return "Got it. I will remember that.";
    }

    private static String emit(TokenSink sink, String text) {
        stream(text, sink);
        return text;
    }

    private static void stream(String text, TokenSink sink) {
        if (sink == null || text == null) return;
        try {
            String[] words = text.split(" ");
            for (int i = 0; i < words.length; i++) {
                if (words[i].isEmpty()) continue;
                sink.token(i == 0 ? words[i] : " " + words[i]);
            }
        } catch (IOException ignored) {
            // The interactive loop notices a closed stdout on its own.
        }
    }
}
