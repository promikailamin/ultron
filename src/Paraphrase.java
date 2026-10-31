package ultron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Deterministic grounded generation fallback.
 *
 * The transformer is always tried first. When its checkpoint is still too
 * small to produce a usable answer, this class turns the retrieved passage
 * into a fresh sentence instead of replaying a stored answer. It changes the
 * wording, sentence order and opening, and never returns the context verbatim.
 */
public final class Paraphrase {
    private Paraphrase() {}

    private static final String[][] REWRITES = {
            {" is a ", " is essentially a "},
            {" are a ", " are essentially a "},
            {" can be ", " may be "},
            {" is used for ", " is helpful for "},
            {" are used for ", " are helpful for "},
            {" is used to ", " helps to "},
            {" are used to ", " help to "},
            {" is made from ", " is built from "},
            {" are made from ", " are built from "},
            {" is made with ", " is built with "},
            {" are made with ", " are built with "},
            {" requires ", " needs "},
            {" require ", " need "},
            {" found in ", " located in "},
            {" found on ", " located on "},
            {" generates in ", " appears in "},
            {" generate in ", " appear in "},
            {" obtained by ", " gathered by "},
            {" obtained from ", " gathered from "},
            {" such as ", " for example "},
            {" because ", " since "},
            {" very ", " quite "},
            {" sometimes ", " occasionally "},
            {" usually ", " normally "},
    };

    private static final Pattern SENTENCE = Pattern.compile("(?<=[.!?])\\s+");
    private static final Pattern PRONOUN_START = Pattern.compile(
            "^(?:it|this|that|they|these|those|there|he|she)\\b",
            Pattern.CASE_INSENSITIVE);

    private static final String[] WHAT_IS = {
            "In short: ", "Put simply: ", "The key idea is this: ",
            "Here is the short answer: ", "Basically: ",
    };
    private static final String[] RECIPE = {
            "To make it: ", "Here is the recipe: ", "You can do it this way: ",
            "For this, follow these steps: ", "This is how it is made: ",
    };
    private static final String[] OBTAIN = {
            "You can get it this way: ", "Here is how to find it: ",
            "This is how you obtain it: ", "Try this method: ",
            "The usual way is this: ",
    };
    private static final String[] DROPS = {
            "Here is what it leaves behind: ", "Here is what it drops: ",
            "Its drops are these: ", "This is what you can collect: ",
            "The loot is as follows: ",
    };
    private static final String[] USES = {
            "It is useful this way: ", "Here is how it is used: ",
            "This is what it does: ", "Its main use is this: ",
            "You can use it like this: ",
    };
    private static final String[] COUNT = {
            "Here is the count: ", "The number is this: ", "In total: ",
            "The game has this many: ", "Counting them gives this: ",
    };

    /** Builds a fresh, grounded answer from retrieved context. */
    public static String compose(String question, String context, int variant) {
        if (context == null || context.trim().isEmpty()) return null;
        String intent = KnowledgeBase.inferIntent(question);
        String topic = KnowledgeBase.extractTopic(question);

        List<String> sentences = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        List<String> raw = new ArrayList<>();
        for (String part : SENTENCE.split(context.trim())) {
            String sentence = part.replaceAll("\\s+", " ").trim();
            if (sentence.isEmpty()) continue;
            raw.add(sentence);
        }
        for (String sentence : raw) {
            if (raw.size() > 1 && sentence.matches("(?i)^(?:no|yes)[.!]$")) continue;
            String key = KnowledgeBase.normalize(sentence);
            if (key.isEmpty() || !seen.add(key)) continue;
            sentences.add(sentence);
            if (sentences.size() >= 6) break;
        }
        if (sentences.isEmpty()) return null;

        boolean definition = KnowledgeBase.WHAT_IS.equals(intent)
                || "grammar".equals(intent) || KnowledgeBase.COUNT.equals(intent);
        if (!definition && variant % 3 == 1 && sentences.size() >= 2
                && !PRONOUN_START.matcher(sentences.get(1)).find()) {
            String first = sentences.remove(0);
            sentences.add(1, first);
        }

        StringBuilder body = new StringBuilder();
        for (int i = 0; i < sentences.size() && i < 6; i++) {
            String sentence = rewrite(sentences.get(i), intent);
            if (body.length() > 0) body.append(' ');
            body.append(sentence);
        }

        String opener = opener(intent, Math.floorMod(variant, 5));
        String out = opener + body;
        if (variant % 5 == 4 && !topic.isEmpty()) {
            out += " That is the main point about " + topic + ".";
        }
        out = out.replaceAll("\\s+", " ").trim();
        return out.isEmpty() ? null : out;
    }

    private static String rewrite(String sentence, String intent) {
        String out = " " + sentence + " ";
        for (String[] pair : REWRITES) {
            out = out.replace(pair[0], pair[1]);
        }
        if (KnowledgeBase.DROPS.equals(intent)) {
            out = out.replace(" drops ", " leaves behind ")
                    .replace(" drop ", " leave behind ")
                    .replace("Drops ", "Leaves behind ")
                    .replace("Drop ", "Leave behind ");
        }
        return out.replaceAll("\\s+", " ").trim();
    }

    private static String opener(String intent, int variant) {
        String[] choices = WHAT_IS;
        switch (intent == null ? "" : intent) {
            case KnowledgeBase.RECIPE: choices = RECIPE; break;
            case KnowledgeBase.OBTAIN: choices = OBTAIN; break;
            case KnowledgeBase.DROPS: choices = DROPS; break;
            case KnowledgeBase.USES: choices = USES; break;
            case KnowledgeBase.COUNT: choices = COUNT; break;
            default: break;
        }
        return choices[variant % choices.length];
    }
}
