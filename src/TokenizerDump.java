package ultron;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/** Prints token ids for each input line; used by the Python/Java tokenizer parity test. */
public class TokenizerDump {
    public static void main(String[] args) throws Exception {
        Tokenizer tok = Tokenizer.loadDefault();
        if (args.length > 0) {
            for (String s : args) System.out.println(join(tok.encode(s)));
            return;
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = br.readLine()) != null) System.out.println(join(tok.encode(line)));
    }

    private static String join(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(ids[i]);
        }
        return sb.toString();
    }
}
