package ultron;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Small, bounded persistent memory for the interactive chat.
 *
 * Everything lives under ~/.config/ultron (or $XDG_CONFIG_HOME/ultron):
 *   memory.tsv            durable user facts, at most MAX_FACTS entries
 *   sessions/*.log        append-only conversation transcripts
 *
 * Only the newest session tail is read on startup, facts are capped, and the
 * neural prompt only receives a short summary. This keeps memory use flat even
 * when the session files grow.
 */
public class Memory implements Closeable {
    private static final int MAX_FACTS = 128;
    private static final int MAX_RECENT_TURNS = 24;
    private static final long MAX_SESSION_BYTES = 4L * 1024 * 1024;
    private static final long TAIL_BYTES = 64L * 1024;
    private static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS");

    private static final Set<String> NON_NAME = new HashSet<>(Arrays.asList(
            "tired", "happy", "sad", "hungry", "thirsty", "bored", "fine", "good", "ok",
            "okay", "great", "back", "here", "ready", "sorry", "sure", "not", "a", "an",
            "the", "doing", "going", "learning", "trying", "working", "interested", "looking",
            "new", "confused", "curious", "alive", "dead", "busy", "free", "angry", "excited"));

    private static final Pattern FAVORITE = Pattern.compile(
            "(?i)\\bmy favou?rite ([a-z][a-z ]{0,30}?) is (.+)");
    private static final Pattern MY_ATTRIBUTE = Pattern.compile(
            "(?i)\\bmy ([a-z][a-z ]{0,30}?) is (.+)");
    private static final Pattern ASK_FAVORITE = Pattern.compile(
            "(?i)^(?:what|which)(?:'s| is| are)? my favou?rite ([a-z][a-z ]{0,30}?)[?.!]*$");

    public static class Turn {
        public final String role, text;

        Turn(String role, String text) {
            this.role = role;
            this.text = text;
        }
    }

    public static class Result {
        public boolean changed;
        public String name;
    }

    private final Path root, sessionsDir, factsFile;
    private final LinkedHashMap<String, String> facts = new LinkedHashMap<>();
    private final ArrayDeque<Turn> recent = new ArrayDeque<>();
    private String sessionId;
    private BufferedWriter sessionOut;
    private boolean factsDirty;

    public Memory(Path root) throws IOException {
        this.root = root;
        this.sessionsDir = root.resolve("sessions");
        this.factsFile = root.resolve("memory.tsv");
        Files.createDirectories(sessionsDir);
        loadFacts();
        resumeLatestOrNew();
    }

    public static Memory openDefault() throws IOException {
        String xdg = System.getenv("XDG_CONFIG_HOME");
        Path base = (xdg != null && !xdg.trim().isEmpty())
                ? Paths.get(xdg.trim())
                : Paths.get(System.getProperty("user.home"), ".config");
        return new Memory(base.resolve("ultron"));
    }

    public Path directory() { return root; }

    public String sessionId() { return sessionId; }

    public synchronized String lastTurnRole() {
        Turn t = recent.peekLast();
        return t == null ? null : t.role;
    }

    public synchronized void append(String role, String text) throws IOException {
        if (text == null) return;
        text = text.replaceAll("\\s+", " ").trim();
        if (text.isEmpty()) return;
        if (sessionOut == null) newSession();

        Path file = sessionsDir.resolve(sessionId);
        if (Files.exists(file) && Files.size(file) > MAX_SESSION_BYTES) {
            newSession();
            file = sessionsDir.resolve(sessionId);
        }
        sessionOut.write(LocalDateTime.now().toString());
        sessionOut.write('\t');
        sessionOut.write(role);
        sessionOut.write('\t');
        sessionOut.write(escape(text));
        sessionOut.write('\n');
        sessionOut.flush();

        recent.addLast(new Turn(role, text));
        while (recent.size() > MAX_RECENT_TURNS) recent.removeFirst();
    }

    /** Returns the id of the new session. */
    public synchronized String newSession() throws IOException {
        closeSession();
        String base = LocalDateTime.now().format(STAMP);
        String id = base + ".log";
        int n = 1;
        while (Files.exists(sessionsDir.resolve(id))) id = base + "-" + (n++) + ".log";
        sessionId = id;
        sessionOut = Files.newBufferedWriter(sessionsDir.resolve(id), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        recent.clear();
        return id;
    }

    public synchronized List<String> listSessions(int limit) throws IOException {
        ArrayList<String> ids = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(sessionsDir, "*.log")) {
            for (Path p : ds) ids.add(p.getFileName().toString());
        }
        ids.sort(Comparator.reverseOrder());
        if (ids.size() > limit) return new ArrayList<>(ids.subList(0, limit));
        return ids;
    }

    public synchronized void loadSession(String id) throws IOException {
        String safe = Paths.get(id).getFileName().toString();
        if (!safe.equals(id) || !safe.matches("[A-Za-z0-9._-]+")) {
            throw new IOException("invalid session id");
        }
        Path file = sessionsDir.resolve(safe);
        if (!Files.exists(file)) throw new IOException("session not found: " + safe);
        closeSession();
        sessionId = safe;
        sessionOut = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        recent.clear();
        for (String line : readTail(file, MAX_RECENT_TURNS)) {
            String[] parts = line.split("\t", 3);
            if (parts.length == 3) recent.addLast(new Turn(parts[1], unescape(parts[2])));
        }
    }

    private void closeSession() throws IOException {
        if (sessionOut != null) {
            sessionOut.close();
            sessionOut = null;
        }
    }

    private void resumeLatestOrNew() throws IOException {
        List<String> ids = listSessions(1);
        if (ids.isEmpty()) {
            newSession();
        } else {
            loadSession(ids.get(0));
        }
    }

    private static List<String> readTail(Path file, int maxLines) throws IOException {
        ArrayList<String> lines = new ArrayList<>();
        try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "r")) {
            long len = raf.length();
            long start = Math.max(0, len - TAIL_BYTES);
            raf.seek(start);
            if (start > 0) raf.readLine();
            String ln;
            while ((ln = raf.readLine()) != null) {
                lines.add(new String(ln.getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8));
                if (lines.size() > maxLines * 2) lines.remove(0);
            }
        }
        if (lines.size() > maxLines) return new ArrayList<>(lines.subList(lines.size() - maxLines, lines.size()));
        return lines;
    }

    private void loadFacts() {
        if (!Files.exists(factsFile)) return;
        try (BufferedReader br = Files.newBufferedReader(factsFile, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\t", 2);
                if (parts.length != 2) continue;
                facts.put(unescape(parts[0]), unescape(parts[1]));
                if (facts.size() > MAX_FACTS) {
                    String oldest = facts.keySet().iterator().next();
                    facts.remove(oldest);
                }
            }
        } catch (IOException ignored) {
            // A damaged memory file must never stop the chat from starting.
        }
    }

    private void saveFacts() {
        if (!factsDirty) return;
        try {
            Files.createDirectories(root);
            Path tmp = root.resolve("memory.tsv.tmp");
            try (BufferedWriter w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                for (Map.Entry<String, String> e : facts.entrySet()) {
                    w.write(escape(e.getKey()));
                    w.write('\t');
                    w.write(escape(e.getValue()));
                    w.write('\n');
                }
            }
            try {
                Files.move(tmp, factsFile, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, factsFile, StandardCopyOption.REPLACE_EXISTING);
            }
            factsDirty = false;
        } catch (IOException ignored) {
            // Memory is best-effort; the chat session itself still works.
        }
    }

    private void put(String key, String value) {
        if (key == null || value == null) return;
        key = key.toLowerCase(Locale.ROOT).replaceAll("\\s+", " ").trim();
        value = value.replaceAll("\\s+", " ").trim();
        if (key.isEmpty() || value.isEmpty() || value.length() > 240) return;
        facts.remove(key);
        facts.put(key, value);
        while (facts.size() > MAX_FACTS) {
            String oldest = facts.keySet().iterator().next();
            facts.remove(oldest);
        }
        factsDirty = true;
        saveFacts();
    }

    // ---------------- learning ----------------

    public synchronized Result learn(String text) {
        Result r = new Result();
        if (text == null) return r;
        String t = text.trim();
        if (t.isEmpty()) return r;

        String name = extractName(t);
        if (name != null) {
            put("name", name);
            r.changed = true;
            r.name = name;
        }

        Matcher fav = FAVORITE.matcher(t);
        if (fav.find()) {
            String thing = cleanKey(fav.group(1));
            String value = clause(fav.group(2), false);
            if (!thing.isEmpty() && !value.isEmpty() && !thing.equals("name")) {
                put("favorite " + thing, value);
                r.changed = true;
            }
        }

        String v;
        if ((v = after(t, "i like ")) != null) { put("likes", clause(v, false)); r.changed = true; }
        if ((v = after(t, "i love ")) != null) { put("loves", clause(v, false)); r.changed = true; }
        if ((v = after(t, "i live in ")) != null) { put("location", clause(v, false)); r.changed = true; }
        if ((v = after(t, "i am from ")) != null) { put("from", clause(v, false)); r.changed = true; }
        if ((v = after(t, "i work at ")) != null) { put("work", clause(v, false)); r.changed = true; }
        if ((v = after(t, "i work as ")) != null) { put("work", clause(v, false)); r.changed = true; }
        if ((v = after(t, "remember that ")) != null) { put("note", clause(v, false)); r.changed = true; }
        if ((v = after(t, "remember ")) != null && after(t, "remember that ") == null) {
            put("note", clause(v, false)); r.changed = true;
        }

        Matcher attr = MY_ATTRIBUTE.matcher(t);
        if (attr.find()) {
            String key = cleanKey(attr.group(1));
            String value = clause(attr.group(2), false);
            if (!key.isEmpty() && !key.equals("name") && !value.isEmpty()) {
                put(key, value);
                r.changed = true;
            }
        }
        return r;
    }

    private static String extractName(String t) {
        String v = after(t, "my name is ");
        if (v == null) v = after(t, "my name's ");
        if (v == null) v = after(t, "call me ");
        if (v == null) v = after(t, "i am called ");
        if (v == null) v = after(t, "i'm called ");
        if (v != null) {
            v = clause(v, true);
            return v.isEmpty() ? null : v;
        }

        v = after(t, "i am ");
        if (v == null) v = after(t, "i'm ");
        if (v != null) {
            String candidate = clause(v, true);
            if (looksLikeName(candidate)) return candidate;
        }
        return null;
    }

    private static boolean looksLikeName(String s) {
        if (s == null || s.isEmpty() || s.length() > 60) return false;
        String[] words = s.split("\\s+");
        if (words.length == 0 || words.length > 3) return false;
        for (String w : words) {
            if (!w.matches("[A-Za-z][A-Za-z'\\-]*")) return false;
            if (NON_NAME.contains(w.toLowerCase(Locale.ROOT))) return false;
        }
        if (words.length == 1) return true;
        for (String w : words) if (!Character.isUpperCase(w.charAt(0))) return false;
        return true;
    }

    /** Finds text after a case-insensitive phrase, or null. */
    private static String after(String original, String phrase) {
        int i = original.toLowerCase(Locale.ROOT).indexOf(phrase);
        if (i < 0) return null;
        return original.substring(i + phrase.length()).trim();
    }

    /** Keeps the first sentence/clause. For names also stops at " and "/" but ". */
    private static String clause(String s, boolean name) {
        if (s == null) return "";
        s = s.trim();
        int cut = -1;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == ',' || c == ';' || c == '.' || c == '!' || c == '?') {
                cut = i;
                break;
            }
        }
        if (cut > 0) s = s.substring(0, cut);
        if (name) {
            String low = s.toLowerCase(Locale.ROOT);
            int and = low.indexOf(" and ");
            int but = low.indexOf(" but ");
            if (and > 0) s = s.substring(0, and);
            if (but > 0) s = s.substring(0, but);
        }
        return s.replaceAll("\\s+", " ").trim();
    }

    private static String cleanKey(String s) {
        return s.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9 ]+", " ")
                .replaceAll("\\s+", " ").trim();
    }

    // ---------------- recall ----------------

    public synchronized String recall(String question) {
        if (question == null) return null;
        String q = question.toLowerCase(Locale.ROOT).replaceAll("\\s+", " ").trim();
        if (q.isEmpty()) return null;

        boolean asksName = q.contains("my name") || q.contains("what am i called")
                || q.equals("who am i") || q.equals("who am i?")
                || q.contains("do you know my name") || q.contains("tell me my name");
        if (asksName) {
            String name = facts.get("name");
            return name != null
                    ? "Your name is " + name + "."
                    : "I do not know your name yet. Tell me by saying: My name is ...";
        }

        if (q.contains("what do you know about me") || q.contains("what do you remember")
                || q.contains("what have you remembered") || q.contains("show my memory")
                || q.equals("memory") || q.equals(":memory")) {
            return summary();
        }

        Matcher fav = ASK_FAVORITE.matcher(q);
        if (fav.matches()) {
            String thing = cleanKey(fav.group(1));
            String value = facts.get("favorite " + thing);
            if (value != null) return "Your favorite " + thing + " is " + value + ".";
            return "I do not know your favorite " + thing + " yet.";
        }

        if (q.contains("what do i like") || q.contains("what i like")) {
            String v = facts.get("likes");
            return v != null ? "You like " + v + "." : "You have not told me what you like yet.";
        }
        if (q.contains("what do i love")) {
            String v = facts.get("loves");
            return v != null ? "You love " + v + "." : "You have not told me what you love yet.";
        }
        if (q.contains("where do i live") || q.contains("where am i from")) {
            String v = q.contains("live") ? facts.get("location") : facts.get("from");
            return v != null ? "You are from " + v + "." : "You have not told me where you are from yet.";
        }
        if (q.contains("where do i work")) {
            String v = facts.get("work");
            return v != null ? "You work at or as " + v + "." : "You have not told me where you work yet.";
        }
        if (q.contains("how old am i")) {
            String v = facts.get("age");
            return v != null ? "You are " + v + " years old." : "You have not told me your age yet.";
        }

        if (q.startsWith("what is my ") || q.startsWith("what's my ") || q.startsWith("what are my ")) {
            String key = cleanKey(q.replaceFirst("^(what is|what's|what are) my ", "").replaceAll("[?.!]+$", ""));
            String v = facts.get("favorite " + key);
            if (v == null) v = facts.get(key);
            if (v != null) return "Your " + key + " is " + v + ".";
        }

        if (!facts.isEmpty() && (q.equals("who am i") || q.contains("do you remember me"))) {
            return summary();
        }
        return null;
    }

    public synchronized String summary() {
        if (facts.isEmpty()) {
            return "I do not know anything about you yet. Tell me something like: My name is ...";
        }
        StringBuilder sb = new StringBuilder("Here is what I remember about you:");
        int n = 0;
        for (Map.Entry<String, String> e : facts.entrySet()) {
            if (n++ >= 12) { sb.append(" ..."); break; }
            sb.append(' ').append(e.getKey()).append(" = ").append(shorten(e.getValue(), 80)).append(';');
        }
        return sb.toString();
    }

    public synchronized void forget(String key) {
        if (key == null || key.trim().isEmpty() || key.equalsIgnoreCase("all")) {
            facts.clear();
            factsDirty = true;
            saveFacts();
            return;
        }
        key = cleanKey(key);
        if (facts.remove(key) != null) {
            factsDirty = true;
            saveFacts();
        }
    }

    /** Short factual prefix injected into the neural model prompt. */
    public synchronized String promptSummary() {
        if (facts.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        int n = 0;
        for (Map.Entry<String, String> e : facts.entrySet()) {
            if (n++ >= 8) break;
            if (sb.length() > 0) sb.append("; ");
            sb.append(e.getKey()).append(" is ").append(shorten(e.getValue(), 60));
            if (sb.length() > 240) break;
        }
        return sb.toString();
    }

    /** Recent dialogue for the model context, bounded in both turns and chars. */
    public synchronized String recentDialogue(int maxTurns) {
        ArrayList<Turn> turns = new ArrayList<>(recent);
        int from = Math.max(0, turns.size() - maxTurns);
        StringBuilder sb = new StringBuilder();
        for (int i = from; i < turns.size(); i++) {
            Turn t = turns.get(i);
            sb.append(t.role).append(": ").append(shorten(t.text, 180)).append('\n');
        }
        if (sb.length() > 800) return sb.substring(sb.length() - 800);
        return sb.toString();
    }

    private static String shorten(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, Math.max(0, max - 3)) + "...";
    }

    // ---------------- encoding ----------------

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "").replace("\n", "\\n");
    }

    private static String unescape(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(++i);
                if (n == 't') sb.append('\t');
                else if (n == 'n') sb.append('\n');
                else if (n == '\\') sb.append('\\');
                else sb.append(n);
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    @Override
    public synchronized void close() throws IOException {
        saveFacts();
        closeSession();
    }
}
