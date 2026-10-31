package ultron;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Runs evaluation questions through the same Chat.reply path in one process.
 * Input TSV columns: id, category, question, must_contain, must_not_contain.
 * A question may contain " ||| " to represent multiple turns.
 */
public class ChatBatch {
    public static void main(String[] args) throws Exception {
        Chat chat = new Chat();
        chat.load();
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = br.readLine()) != null) {
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] p = line.split("\t", -1);
                if (p.length < 3) continue;
                String id = p[0];
                String question = p[2];
                chat.resetTopic();
                String answer = "";
                for (String turn : question.split(" \\|\\|\\| ")) {
                    answer = chat.reply(turn.trim(), null);
                }
            System.out.println(id + "\t" + answer.replace("\n", "\\n").replace("\t", " "));
        }
    }
}
