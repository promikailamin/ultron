package ultron;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Curated, intent-aware knowledge base for grounded chat answers.
 *
 * Matching is deliberately strict: every distinctive query token must occur in
 * the candidate question, and a domain intent (what_is, recipe, obtain, drops,
 * uses, count) must agree. When no candidate satisfies both rules the caller
 * should prefer the transformer or an explicit "I don't have that" over a
 * loosely related answer.
 */
public final class KnowledgeBase {
    public static final String WHAT_IS = "what_is";
    public static final String RECIPE = "recipe";
    public static final String OBTAIN = "obtain";
    public static final String DROPS = "drops";
    public static final String USES = "uses";
    public static final String COUNT = "count";

    private static final Set<String> DOMAIN_INTENTS = new HashSet<>(Arrays.asList(
            WHAT_IS, RECIPE, OBTAIN, DROPS, USES, COUNT));

    private static final Set<String> KNOWN_INTENTS = new HashSet<>(Arrays.asList(
            WHAT_IS, RECIPE, OBTAIN, DROPS, USES, COUNT,
            "greeting", "identity", "thanks", "goodbye", "smalltalk",
            "capability", "grammar"));

    /**
     * Function words are ignored for matching. Intent words such as make,
     * craft, get, use, find, create, build and obtain are deliberately absent:
     * they carry the meaning of the question and must stay in the query.
     */
    private static final Set<String> STOP = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "of", "to", "in", "on", "at", "for", "with", "by", "from", "about",
            "as", "into", "like", "over", "after", "before", "between", "out",
            "up", "down", "off", "and", "or", "but", "if", "because", "so",
            "than", "then", "that", "this", "these", "those", "it", "its",
            "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
            "their", "his", "her", "them", "us", "do", "does", "did", "can",
            "could", "should", "would", "will", "shall", "may", "might", "must",
            "what", "which", "who", "whom", "whose", "when", "where", "why",
            "how", "tell", "describe", "please", "there", "here", "any", "some",
            "more", "most", "much", "many", "very", "also", "just", "not", "no",
            "yes", "s", "t", "whats", "dont", "doesnt", "isnt", "im", "ive",
            "id", "youre", "youve", "theyre", "weve", "thats", "its"));

    private static final Pattern WHAT = Pattern.compile(
            "^(?:what(?:'s| is| are)|tell me about|describe)\\s+(.+)$");
    private static final Pattern HOW = Pattern.compile(
            "^(?:how (?:do|can|should) i|how to)\\s+"
            + "(?:make|craft|get|obtain|find|create|build|use)\\s+(.+)$");
    private static final Pattern WHERE = Pattern.compile(
            "^(?:where (?:do|can) i (?:find|get)|where to find)\\s+(.+)$");
    private static final Pattern CRAFTABLE = Pattern.compile(
            "^(?:can|is|are)\\s+(.+?)\\s+be\\s+craft(?:ed|able).*$");
    private static final Pattern USED = Pattern.compile(
            "^(?:what is|what are)\\s+(.+?)\\s+used for$");
    private static final Pattern DOES = Pattern.compile(
            "^(?:what does|what do)\\s+(.+?)\\s+do$");

    private static final Pattern FOLLOWUP = Pattern.compile(
            "\\b(it|this|that|them|those)\\b");

    public static final class Entry {
        public final String category, question, answer, intent;
        public final Set<String> tokens;
        public final Set<String> questionTokens;
        public final Set<String> topicTokens;

        Entry(String category, String question, String answer, String intent) {
            this.category = category;
            this.question = question;
            this.answer = answer;
            this.intent = intent;
            this.tokens = contentTokens(question + " " + answer);
            this.questionTokens = contentTokens(question);
            this.topicTokens = contentTokens(extractTopic(question));
        }
    }

    public static final class Query {
        public final String raw, intent, topic;
        public final Set<String> tokens;
        public final Set<String> distinctive;

        Query(String raw) {
            this.raw = raw;
            this.intent = inferIntent(raw);
            this.topic = extractTopic(raw);
            this.tokens = contentTokens(raw);
            this.distinctive = new LinkedHashSet<>();
        }
    }

    private final List<Entry> entries = new ArrayList<>();
    private final Map<String, Entry> exact = new HashMap<>();
    private final Map<String, Integer> documentFrequency = new HashMap<>();
    private int documents;

    private KnowledgeBase() {}

    public static KnowledgeBase load() throws IOException {
        KnowledgeBase kb = new KnowledgeBase();
        try (BufferedReader br = Resources.reader("chat_kb.tsv", "data/chat_kb.tsv")) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\t", 4);
                if (parts.length < 3) continue;
                String category = parts[0].trim();
                String question = parts[1].trim();
                String answer = parts[2].trim();
                String intent = category.toLowerCase(Locale.ROOT);
                if (parts.length == 4) {
                    String tagged = parts[3].trim().toLowerCase(Locale.ROOT);
                    if (KNOWN_INTENTS.contains(tagged)) intent = tagged;
                    else answer = answer + "\t" + parts[3];
                }
                if (question.isEmpty() || answer.isEmpty()) continue;
                if ("minecraft".equalsIgnoreCase(category) && !KNOWN_INTENTS.contains(intent)) {
                    intent = inferIntent(question);
                }
                Entry e = new Entry(category, question, answer, intent);
                kb.entries.add(e);
                String key = normalize(question);
                if (!kb.exact.containsKey(key)) kb.exact.put(key, e);
            }
        }
        kb.index();
        if (kb.entries.isEmpty()) throw new IOException("chat_kb.tsv contains no entries");
        return kb;
    }

    private void index() {
        documents = entries.size();
        for (Entry e : entries) {
            for (String t : e.tokens) {
                documentFrequency.merge(t, 1, Integer::sum);
            }
        }
    }

    public boolean isFollowup(String input) {
        if (input == null) return false;
        String low = normalize(input);
        if (low.startsWith("and ") || low.startsWith("how about ")
                || low.startsWith("what about ")) return true;
        return FOLLOWUP.matcher(low).find();
    }

    /** Returns a strictly grounded answer, or null when nothing is safe. */
    public String answer(String input) {
        if (input == null || input.trim().isEmpty()) return null;
        Query q = new Query(input);

        Entry exactEntry = exact.get(normalize(input));
        if (exactEntry != null) return exactEntry.answer;

        if (q.tokens.isEmpty()) return null;
        for (String t : q.tokens) {
            int df = documentFrequency.getOrDefault(t, 0);
            double idf = Math.log((documents + 1.0) / (df + 1.0)) + 1.0;
            if (idf >= 1.8) q.distinctive.add(t);
        }
        if (q.distinctive.isEmpty()) return null;

        ArrayList<Entry> candidates = new ArrayList<>();
        for (Entry e : entries) {
            if (e.questionTokens.containsAll(q.distinctive)) candidates.add(e);
        }
        if (candidates.isEmpty()) return null;

        if (DOMAIN_INTENTS.contains(q.intent)) {
            ArrayList<Entry> sameIntent = new ArrayList<>();
            for (Entry e : candidates) {
                if (q.intent.equals(e.intent)) sameIntent.add(e);
            }
            if (sameIntent.isEmpty()) return null;
            candidates = sameIntent;
        }

        Entry best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        Set<String> queryTopic = contentTokens(q.topic);
        for (Entry e : candidates) {
            double score = 0;
            for (String t : q.distinctive) {
                int df = documentFrequency.getOrDefault(t, 0);
                score += Math.log((documents + 1.0) / (df + 1.0)) + 1.0;
            }
            for (String t : queryTopic) {
                if (e.topicTokens.contains(t)) score += 3.0;
                else if (e.questionTokens.contains(t)) score += 1.0;
            }
            if (q.intent.equals(e.intent)) score += 4.0;
            if (score > bestScore) {
                bestScore = score;
                best = e;
            }
        }
        return best == null ? null : best.answer;
    }

    /**
     * True when the question's distinctive words are all absent from the
     * knowledge base. Those questions should abstain rather than let a small
     * model invent an answer.
     */
    public boolean isUnknown(String input) {
        if (input == null) return false;
        Query q = new Query(input);
        if (q.tokens.isEmpty()) return false;
        boolean sawToken = false;
        for (String t : q.tokens) {
            int df = documentFrequency.getOrDefault(t, 0);
            double idf = Math.log((documents + 1.0) / (df + 1.0)) + 1.0;
            if (idf < 1.8) continue;
            sawToken = true;
            if (df > 0) return false;
        }
        return sawToken;
    }

    public static String inferIntent(String question) {
        String q = normalize(question);
        q = q.replaceFirst("^(?:and|so|then|ok|okay|also)\\s+", "");
        q = q.replaceFirst("^how about\\s+", "");
        q = q.replaceFirst("^what about\\s+", "");
        if (q.isEmpty()) return WHAT_IS;
        if (q.startsWith("how many ") || q.startsWith("how much ")
                || q.contains("number of ")) return COUNT;
        if (q.contains(" drop") || q.startsWith("what does ") && q.endsWith(" drop")
                || q.startsWith("what do ") && q.endsWith(" drop")) return DROPS;
        if (q.matches("^(?:can|is|are)\\s+.+?\\s+(?:be\\s+)?craft(?:ed|able).*$")
                || q.startsWith("how to make ") || q.startsWith("how do i make ")
                || q.startsWith("how can i make ") || q.startsWith("how to craft ")
                || q.startsWith("how do i craft ") || q.startsWith("how can i craft ")
                || q.contains(" recipe") || q.startsWith("recipe for ")
                || q.startsWith("can we craft ")) return RECIPE;
        if (q.startsWith("how to get ") || q.startsWith("how do i get ")
                || q.startsWith("how can i get ") || q.startsWith("how to obtain ")
                || q.startsWith("how do i obtain ") || q.startsWith("how can i obtain ")
                || q.startsWith("how to find ") || q.startsWith("how do i find ")
                || q.startsWith("where can i find ") || q.startsWith("where do i find ")
                || q.startsWith("where can i get ") || q.startsWith("where do i get ")
                || q.startsWith("where to find ")) return OBTAIN;
        if (q.startsWith("what is ") && q.endsWith(" used for")
                || q.startsWith("what are ") && q.endsWith(" used for")
                || q.startsWith("how do i use ") || q.startsWith("how to use ")
                || q.startsWith("what does ") && q.endsWith(" do")
                || q.startsWith("what do ") && q.endsWith(" do")) return USES;
        // Remaining behavioural/instruction questions ("how do i stop ...",
        // "why did ...", "how does ... work") are not definitions.
        if (q.startsWith("how ") || q.startsWith("why ") || q.startsWith("when ")) return USES;
        return WHAT_IS;
    }

    public static String extractTopic(String input) {
        if (input == null) return "";
        String q = input.toLowerCase(Locale.ROOT).trim();
        q = q.replaceAll("[?!.,;:]+$", "").replaceAll("\\s+", " ").trim();
        if (q.isEmpty()) return "";

        String topic = null;
        Matcher m = CRAFTABLE.matcher(q);
        if (m.matches()) topic = m.group(1);
        if (topic == null) {
            m = USED.matcher(q);
            if (m.matches()) topic = m.group(1);
        }
        if (topic == null) {
            m = DOES.matcher(q);
            if (m.matches()) topic = m.group(1);
        }
        if (topic == null) {
            m = HOW.matcher(q);
            if (m.matches()) topic = m.group(1);
        }
        if (topic == null) {
            m = WHERE.matcher(q);
            if (m.matches()) topic = m.group(1);
        }
        if (topic == null) {
            m = WHAT.matcher(q);
            if (m.matches()) topic = m.group(1);
        }
        if (topic == null) topic = q;

        topic = topic.replaceAll("^(?:a|an|the)\\s+", "");
        topic = topic.replaceAll("\\s+(?:in minecraft|on a crafting table)$", "");
        topic = topic.replaceAll("^(?:it|this|that|them|those)\\s+", "");
        topic = topic.replaceAll("\\s+", " ").trim();
        if (topic.matches("(?:it|this|that|them|those|one)")) return "";
        if (topic.length() < 2 || topic.length() > 80) return "";
        return topic;
    }

    static Set<String> contentTokens(String text) {
        LinkedHashSet<String> out = new LinkedHashSet<>();
        String normalized = normalize(text);
        if (normalized.isEmpty()) return out;
        for (String raw : normalized.split(" ")) {
            if (raw.isEmpty()) continue;
            if (STOP.contains(raw)) continue;
            String token = canonical(raw);
            if (token.isEmpty() || STOP.contains(token) || token.length() < 2) continue;
            out.add(token);
        }
        return out;
    }

    static String normalize(String text) {
        if (text == null) return "";
        String s = text.toLowerCase(Locale.ROOT).replace('\u2019', '\'');
        s = s.replace("'", "");
        s = s.replaceAll("[^a-z0-9]+", " ");
        return s.replaceAll("\\s+", " ").trim();
    }

    static String canonical(String word) {
        if (word.isEmpty()) return word;
        switch (word) {
            case "crafts": case "crafted": case "crafting": case "craftable":
            case "craftability": return "craft";
            case "makes": case "made": case "making": return "make";
            case "gets": case "getting": case "got": case "gotten": return "get";
            case "uses": case "used": case "using": return "use";
            case "finds": case "finding": case "found": return "find";
            case "creates": case "created": case "creating": return "create";
            case "builds": case "building": case "built": return "build";
            case "obtains": case "obtained": case "obtaining": return "obtain";
            case "drops": case "dropped": case "dropping": return "drop";
            case "editions": return "edition";
            case "versions": return "version";
            case "recipes": return "recipe";
            case "exists": case "existed": case "existing": return "exist";
            default:
                break;
        }
        if (word.endsWith("ies") && word.length() > 4) {
            return word.substring(0, word.length() - 3) + "y";
        }
        if (word.endsWith("es") && word.length() > 4
                && (word.endsWith("ches") || word.endsWith("shes")
                    || word.endsWith("sses") || word.endsWith("xes"))) {
            return word.substring(0, word.length() - 2);
        }
        if (word.endsWith("s") && !word.endsWith("ss") && word.length() > 3) {
            return word.substring(0, word.length() - 1);
        }
        return word;
    }
}
