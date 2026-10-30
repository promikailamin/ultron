package ultron;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;

/**
 * Opens runtime assets from the jar first, then falls back to the repository
 * layout so the same classes work for local development and a packaged jar.
 */
public final class Resources {
    private Resources() {}

    public static InputStream open(String name, String fallback) throws IOException {
        InputStream in = Resources.class.getResourceAsStream("/" + name);
        if (in != null) return in;
        Path p = Paths.get(fallback);
        if (Files.exists(p)) return Files.newInputStream(p);
        throw new FileNotFoundException(name + " (classpath) or " + fallback);
    }

    public static boolean exists(String name, String fallback) {
        return Resources.class.getResource("/" + name) != null || Files.exists(Paths.get(fallback));
    }

    public static String readText(String name, String fallback) throws IOException {
        try (InputStream in = open(name, fallback)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    public static BufferedReader reader(String name, String fallback) throws IOException {
        return new BufferedReader(new InputStreamReader(open(name, fallback), StandardCharsets.UTF_8));
    }
}
