using System.Diagnostics.CodeAnalysis;
using System.Text.RegularExpressions;

namespace PgsToSrtPlus;

/// <summary>
/// File-level post-processing passes applied to the complete SRT entry list
/// after all per-subtitle OCR has finished.
/// </summary>
static class SrtPostProcessor
{
    public delegate SrtEntry[] Step(SrtEntry[] entries);

    private static readonly char[] MusicGlyphs = ['♭', '♬', '♪', '♩', '♫'];

    // Regex character class covering all music glyphs — used in spacing patterns.
    const string GlyphClass = "♭♬♪♩♫";

    // Entry point
    public static SrtEntry[] Process(
        SrtEntry[] entries, string language)
    {
        foreach (var step in OcrLanguageConfigs.ForLanguage(language).PostProcessingSteps)
            entries = step(entries);
        return entries;
    }

    // Rule 0: Merge split italic spans across line breaks

    /// <summary>
    /// When per-line OCR tags each physical line independently, a multi-line
    /// subtitle that is entirely italic is emitted as:
    ///   <i>line 1</i>\n<i>line 2</i>
    /// This collapses adjacent closing/opening italic pairs back into one span:
    ///   <i>line 1\nline 2</i>
    /// Regex. Replace scans the full string in one pass, so chains of three or
    /// more lines are handled without looping.
    /// Punctuation characters that should be absorbed into an italic span when
    /// they sit between two italic runs (e.g. <c>&lt;/i&gt;・&lt;i&gt;</c> → <c>・</c>).
    /// </summary>
    static readonly string CollapseItalicPunct = "・";

    public static SrtEntry[] MergeConsecutiveItalicSpans(
        SrtEntry[] entries)
    {
        // Match </i>{separator}<i> where {separator} is a newline or any run
        // of CollapseItalicPunct characters (optionally surrounded by spaces).
        string punctClass = Regex.Escape(CollapseItalicPunct);
        var mergeRx = new Regex(@"</i>(\n| *[" + punctClass + "]+ *)<i>");

        int count = 0;
        var result = new SrtEntry[entries.Length];

        for (int i = 0; i < entries.Length; i++)
        {
            var e = entries[i];
            if (mergeRx.IsMatch(e.Text))
            {
                result[i] = e with { Text = mergeRx.Replace(e.Text, "$1") };
                count++;
            }
            else
            {
                result[i] = e;
            }
        }

        if (count > 0)
            Console.WriteLine(
                $"[post] italic-span-merge: merged split spans in {count} entr{(count == 1 ? "y" : "ies")}.");

        return result;
    }

    // ── Rule 1: Standardize music-note glyph ──────────────────────────────────

    /// <summary>
    /// If a single music-note glyph from the set {♭ ♬ ♪ ♩ ♫} accounts for
    /// more than 60 % of all music-note glyph occurrences in the file, every
    /// other music-note glyph is replaced with that dominant glyph.
    ///
    /// Example: 120 × ♪, 5 × ♫  →  ♪ is 96 % → replace all ♫ with ♪.
    /// </summary>
    public static SrtEntry[] StandardizeMusicGlyph(
        SrtEntry[] entries)
    {
        var counts = new Dictionary<char, int>(MusicGlyphs.Length);
        foreach (char g in MusicGlyphs) counts[g] = 0;

        foreach (var e in entries)
        foreach (char c in e.Text)
            if (counts.ContainsKey(c))
                counts[c]++;

        int total = counts.Values.Sum();
        if (total == 0) return entries;

        char dominant = '\0';
        foreach (char g in MusicGlyphs)
            if (counts[g] > total * 0.60)
            {
                dominant = g;
                break;
            }

        if (dominant == '\0') return entries;

        Console.WriteLine(
            $"[post] music-glyph: dominant '{dominant}' ({counts[dominant]}/{total} = " +
            $"{counts[dominant] * 100.0 / total:F0}%) — replacing all others.");

        char[] others = MusicGlyphs.Where(g => g != dominant).ToArray();
        var result = new SrtEntry[entries.Length];

        for (int i = 0; i < entries.Length; i++)
        {
            var e = entries[i];
            if (!others.Any(g => e.Text.Contains(g)))
            {
                result[i] = e;
                continue;
            }

            string text = e.Text;
            foreach (char g in others)
                text = text.Replace(g, dominant);
            result[i] = e with { Text = text };
        }

        return result;
    }

    // ── Rule 2: Standardize spacing around music-note glyphs ──────────────────

    /// <summary>
    /// Counts the physical text lines (newline-delimited) that contain at least
    /// one music-note glyph and checks whether each glyph in that line has a
    /// space between it and every adjacent non-glyph, non-whitespace character
    /// (evaluated on the tag-stripped text).
    ///
    /// If more than 50 % of such lines already have spaces, spaces are added to
    /// lines that are missing them.  Otherwise, all surrounding spaces are removed.
    /// </summary>
    public static SrtEntry[] StandardizeMusicGlyphSpacing(
        SrtEntry[] entries)
    {
        int withSpaces = 0, withoutSpaces = 0;

        foreach (var e in entries)
        foreach (string line in e.Text.Split('\n'))
        {
            if (!ContainsMusicGlyph(line)) continue;
            if (LineHasSpacesAroundGlyphs(line)) withSpaces++;
            else withoutSpaces++;
        }

        int total = withSpaces + withoutSpaces;
        if (total == 0) return entries;

        bool addSpaces = withSpaces > withoutSpaces; // strictly > 50 %
        bool alreadyUniform = addSpaces ? withoutSpaces == 0 : withSpaces == 0;

        Console.WriteLine(
            $"[post] music-glyph spacing: {withSpaces}/{total} lines have spaces — " +
            (alreadyUniform
                ? "already uniform, skipping."
                : addSpaces ? "adding spaces around glyphs." : "removing spaces around glyphs."));

        if (alreadyUniform) return entries;

        var result = new SrtEntry[entries.Length];
        for (int i = 0; i < entries.Length; i++)
        {
            var e = entries[i];
            if (!ContainsMusicGlyph(e.Text))
            {
                result[i] = e;
                continue;
            }

            string text = addSpaces
                ? AddSpacesAroundGlyphs(e.Text)
                : RemoveSpacesAroundGlyphs(e.Text);
            result[i] = e with { Text = text };
        }

        return result;
    }

    // ── Rule 3: Standardize music-note position relative to italic tags ────────

    /// <summary>
    /// For lines where all non-glyph text content is italic ("full italic" and
    /// "full italic less music notes"), counts whether music-note glyphs sit
    /// inside or outside the &lt;i&gt; span and standardizes to the majority.
    ///
    ///   Inside  example:  <i>♪ text ♪</i>
    ///   Outside example:  ♪ <i>text</i> ♪
    ///
    /// A line qualifies only when:
    ///   • it contains at least one music-note glyph, AND
    ///   • every non-glyph, non-whitespace character is inside a &lt;i&gt; span
    ///     (checked via a tag-aware character walk), AND
    ///   • there is at least one such text character (lines of pure glyphs are skipped).
    ///
    /// Lines whose glyphs are split across both inside and outside (mixed) are
    /// counted for measurement but not transformed.
    /// </summary>
    public static SrtEntry[] StandardizeMusicNoteItalic(
        SrtEntry[] entries)
    {
        int insideCount = 0, outsideCount = 0;

        foreach (var e in entries)
        foreach (string line in e.Text.Split('\n'))
        {
            if (!IsQualifyingItalicLine(line)) continue;
            switch (ClassifyNoteItalicStyle(line))
            {
                case NoteItalicStyle.Inside: insideCount++; break;
                case NoteItalicStyle.Outside: outsideCount++; break;
            }
        }

        int total = insideCount + outsideCount;
        if (total == 0) return entries;

        bool targetInside = insideCount >= outsideCount;
        bool alreadyUniform = targetInside ? outsideCount == 0 : insideCount == 0;

        Console.WriteLine(
            $"[post] music-note italic: {insideCount} inside / {outsideCount} outside — " +
            (alreadyUniform
                ? "already uniform, skipping."
                : targetInside
                    ? "standardizing: notes inside <i>."
                    : "standardizing: notes outside <i>."));

        if (alreadyUniform) return entries;

        var result = new SrtEntry[entries.Length];
        for (int i = 0; i < entries.Length; i++)
        {
            var e = entries[i];
            if (!ContainsMusicGlyph(e.Text))
            {
                result[i] = e;
                continue;
            }

            string[] lines = e.Text.Split('\n');
            bool changed = false;

            for (int j = 0; j < lines.Length; j++)
            {
                if (!IsQualifyingItalicLine(lines[j])) continue;
                var style = ClassifyNoteItalicStyle(lines[j]);

                if (targetInside && style == NoteItalicStyle.Outside)
                {
                    lines[j] = TransformNotesInside(lines[j]);
                    changed = true;
                }
                else if (!targetInside && style == NoteItalicStyle.Inside)
                {
                    lines[j] = TransformNotesOutside(lines[j]);
                    changed = true;
                }
            }

            result[i] = changed ? e with { Text = string.Join("\n", lines) } : e;
        }

        return result;
    }

    // Note italic-style classification ─────────────────────────────────────────

    enum NoteItalicStyle
    {
        Inside,
        Outside,
        Mixed
    }

    /// <summary>
    /// Returns true when the line contains at least one music-note glyph, has
    /// at least one non-glyph visible character, and every such character is
    /// inside an &lt;i&gt; span (tag-aware walk).
    /// </summary>
    static bool IsQualifyingItalicLine(string line)
    {
        if (!ContainsMusicGlyph(line)) return false;

        bool italic = false;
        bool hasNonNoteText = false;

        for (int i = 0; i < line.Length;)
        {
            if (line[i] == '<')
            {
                int close = line.IndexOf('>', i);
                if (close >= 0)
                {
                    string tag = line.Substring(i, close - i + 1).ToLower();
                    if (tag == "<i>") italic = true;
                    else if (tag == "</i>") italic = false;
                    i = close + 1;
                    continue;
                }
            }

            char c = line[i];
            if (!char.IsWhiteSpace(c) && !IsMusicGlyph(c))
            {
                hasNonNoteText = true;
                if (!italic) return false; // roman text found → does not qualify
            }

            i++;
        }

        return hasNonNoteText;
    }

    /// <summary>
    /// Classifies whether all music-note glyphs in the line are inside &lt;i&gt;
    /// spans, all outside, or a mix.
    /// </summary>
    static NoteItalicStyle ClassifyNoteItalicStyle(string line)
    {
        bool italic = false;
        bool anyInside = false;
        bool anyOutside = false;

        for (int i = 0; i < line.Length;)
        {
            if (line[i] == '<')
            {
                int close = line.IndexOf('>', i);
                if (close >= 0)
                {
                    string tag = line.Substring(i, close - i + 1).ToLower();
                    if (tag == "<i>") italic = true;
                    else if (tag == "</i>") italic = false;
                    i = close + 1;
                    continue;
                }
            }

            if (IsMusicGlyph(line[i]))
            {
                if (italic) anyInside = true;
                else anyOutside = true;
            }

            i++;
        }

        if (anyInside && !anyOutside) return NoteItalicStyle.Inside;
        if (anyOutside && !anyInside) return NoteItalicStyle.Outside;
        return NoteItalicStyle.Mixed;
    }

    // Note italic-style transforms ──────────────────────────────────────────────

    /// <summary>
    /// Moves notes inside the italic span.
    /// Strategy: strip all &lt;i&gt;/&lt;/i&gt; tags and wrap the entire
    /// stripped content in a single &lt;i&gt;...&lt;/i&gt;.
    /// Safe for qualifying lines because they contain only italic text + glyphs.
    ///   ♪ <i>text</i> ♪  →  <i>♪ text ♪</i>
    /// Also works on partial lyric lines (opener/closer of a paired block):
    ///   ♪ text         →  <i>♪ text</i>
    ///   text ♪         →  <i>text ♪</i>
    /// </summary>
    static string TransformNotesInside(string line) =>
        "<i>" + StripTags(line) + "</i>";

    /// <summary>
    /// Moves notes outside the italic span.
    /// Strategy: strip tags, peel off leading and trailing glyph runs (plus any
    /// adjacent spaces), wrap only the text content in &lt;i&gt;...&lt;/i&gt;.
    ///   <i>♪ text ♪</i>  →  ♪ <i>text</i> ♪
    /// Also works on partial lyric lines:
    ///   <i>♪ text</i>    →  ♪ <i>text</i>
    ///   <i>text ♪</i>    →  <i>text</i> ♪
    /// </summary>
    static string TransformNotesOutside(string line)
    {
        string s = StripTags(line);

        // Advance past leading glyphs, spaces, and speaker-change dashes so that
        // a prefix like "-♪ " is kept outside the italic span as a unit.
        int contentStart = 0;
        while (contentStart < s.Length
               && (IsMusicGlyph(s[contentStart]) || s[contentStart] == ' ' || s[contentStart] == '-'))
            contentStart++;

        // Retreat past trailing glyphs and any immediately adjacent spaces.
        int contentEnd = s.Length;
        while (contentEnd > contentStart
               && (IsMusicGlyph(s[contentEnd - 1]) || s[contentEnd - 1] == ' '))
            contentEnd--;

        if (contentStart >= contentEnd) return line; // only glyphs — nothing to wrap

        string leading = s[..contentStart];
        string content = s[contentStart..contentEnd];
        string trailing = s[contentEnd..];

        return leading + "<i>" + content + "</i>" + trailing;
    }

    // ── Rule 4: Standardize italic style of lyric content ─────────────────────

    /// <summary>
    /// Identifies "lyric blocks" — lines whose text content is bracketed by
    /// music-note glyphs — and standardizes whether that content is italic.
    ///
    /// A lyric block is one of:
    ///   • Self-contained: a single newline-delimited line that begins AND ends
    ///     with a music-note glyph (possibly preceded/followed by whitespace/tags).
    ///   • Paired: two consecutive lines within the same SRT entry where line 1
    ///     begins with a glyph but does NOT end with one, and line 2 does NOT
    ///     begin with a glyph but ends with one.
    ///
    /// A block is "italic"     when all its non-glyph, non-whitespace text is
    ///                          inside &lt;i&gt; spans.
    /// A block is "non-italic" when none of that text is inside &lt;i&gt; spans.
    /// Mixed / empty blocks are skipped during both measurement and transformation.
    ///
    /// When making content italic the note-position style (inside/outside &lt;i&gt;)
    /// is taken from the majority of existing italic lyric blocks; defaults to
    /// outside when there are none.
    /// </summary>
    public static SrtEntry[] StandardizeLyricItalic(
        SrtEntry[] entries)
    {
        int italicCount = 0, nonItalicCount = 0;
        int notesInsideCount = 0, notesOutsideCount = 0;

        foreach (var e in entries)
        {
            string[] lines = e.Text.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                var block = TryGetLyricBlock(lines, i);
                if (block == null) continue;
                var (line1, line2, consumed) = block.Value;
                i += consumed - 1;

                bool? isItalic = ClassifyLyricBlockItalic(line1, line2);
                if (isItalic == true)
                {
                    italicCount++;
                    var ns = ClassifyNoteItalicStyle(line1);
                    if (ns == NoteItalicStyle.Inside) notesInsideCount++;
                    else if (ns == NoteItalicStyle.Outside) notesOutsideCount++;
                }
                else if (isItalic == false)
                    nonItalicCount++;
            }
        }

        int total = italicCount + nonItalicCount;
        if (total == 0) return entries;

        bool targetItalic = italicCount >= nonItalicCount;
        // Default to notes-outside when no italic lyric blocks exist yet.
        bool notesInside = notesInsideCount > notesOutsideCount;
        bool alreadyUniform = targetItalic ? nonItalicCount == 0 : italicCount == 0;

        Console.WriteLine(
            $"[post] lyric italic: {italicCount} italic / {nonItalicCount} non-italic" +
            (targetItalic ? $" (notes {(notesInside ? "inside" : "outside")} <i>)" : "") +
            " — " +
            (alreadyUniform
                ? "already uniform, skipping."
                : targetItalic
                    ? "making all lyric content italic."
                    : "making all lyric content non-italic."));

        if (alreadyUniform) return entries;

        var result = new SrtEntry[entries.Length];
        for (int i = 0; i < entries.Length; i++)
        {
            var e = entries[i];
            if (!ContainsMusicGlyph(e.Text))
            {
                result[i] = e;
                continue;
            }

            string[] lines = e.Text.Split('\n');
            bool changed = false;

            for (int j = 0; j < lines.Length; j++)
            {
                var block = TryGetLyricBlock(lines, j);
                if (block == null) continue;
                var (_, _, consumed) = block.Value;

                bool? isItalic = ClassifyLyricBlockItalic(
                    lines[j], consumed == 2 ? lines[j + 1] : null);

                if (targetItalic && isItalic == false)
                {
                    // Add italic to each line of the block independently.
                    lines[j] = notesInside
                        ? TransformNotesInside(lines[j])
                        : TransformNotesOutside(lines[j]);
                    if (consumed == 2)
                        lines[j + 1] = notesInside
                            ? TransformNotesInside(lines[j + 1])
                            : TransformNotesOutside(lines[j + 1]);
                    changed = true;
                }
                else if (!targetItalic && isItalic == true)
                {
                    // Remove all italic tags from each line of the block.
                    lines[j] = StripItalicTags(lines[j]);
                    if (consumed == 2) lines[j + 1] = StripItalicTags(lines[j + 1]);
                    changed = true;
                }

                j += consumed - 1; // skip any line already processed as part of this block
            }

            result[i] = changed ? e with { Text = string.Join("\n", lines) } : e;
        }

        return result;
    }

    // ── Lyric-block helpers ────────────────────────────────────────────────────

    /// <summary>
    /// Tries to form a complete lyric block starting at <paramref name="lines"/>[i].
    /// Returns (line1, line2_or_null, lines_consumed) or null if no block starts here.
    /// </summary>
    static (string Line1, string? Line2, int Consumed)?
        TryGetLyricBlock(string[] lines, int i)
    {
        if (!ContainsMusicGlyph(lines[i])) return null;

        bool hasLead = HasLeadingGlyph(lines[i]);
        bool hasTail = HasTrailingGlyph(lines[i]);

        if (hasLead && hasTail)
            return (lines[i], null, 1);

        if (hasLead && !hasTail
                    && i + 1 < lines.Length
                    && !HasLeadingGlyph(lines[i + 1])
                    && HasTrailingGlyph(lines[i + 1]))
            return (lines[i], lines[i + 1], 2);

        return null;
    }

    /// <summary>
    /// Returns true if the first non-tag, non-whitespace character in the line
    /// is a music-note glyph.
    /// </summary>
    static bool HasLeadingGlyph(string line)
    {
        for (int i = 0; i < line.Length;)
        {
            if (line[i] == '<')
            {
                int close = line.IndexOf('>', i);
                if (close >= 0)
                {
                    i = close + 1;
                    continue;
                }
            }

            if (char.IsWhiteSpace(line[i]))
            {
                i++;
                continue;
            }

            return IsMusicGlyph(line[i]);
        }

        return false;
    }

    /// <summary>
    /// Returns true if the last non-tag, non-whitespace character in the line
    /// is a music-note glyph.
    /// </summary>
    static bool HasTrailingGlyph(string line)
    {
        for (int i = line.Length - 1; i >= 0;)
        {
            if (line[i] == '>')
            {
                int open = line.LastIndexOf('<', i);
                if (open >= 0)
                {
                    i = open - 1;
                    continue;
                }
            }

            if (char.IsWhiteSpace(line[i]))
            {
                i--;
                continue;
            }

            return IsMusicGlyph(line[i]);
        }

        return false;
    }

    /// <summary>
    /// Returns true  if every non-glyph, non-whitespace character in the line
    ///               is inside an &lt;i&gt; span (entirely italic content),
    ///         false if none of them are (entirely non-italic),
    ///         null  if mixed or if there are no such characters.
    /// </summary>
    static bool? IsLyricContentItalic(string line)
    {
        bool italic = false, hasItalic = false, hasNonItalic = false;

        for (int i = 0; i < line.Length;)
        {
            if (line[i] == '<')
            {
                int close = line.IndexOf('>', i);
                if (close >= 0)
                {
                    string tag = line.Substring(i, close - i + 1).ToLower();
                    if (tag == "<i>") italic = true;
                    else if (tag == "</i>") italic = false;
                    i = close + 1;
                    continue;
                }
            }

            char c = line[i];
            if (!char.IsWhiteSpace(c) && !IsMusicGlyph(c))
            {
                if (italic) hasItalic = true;
                else hasNonItalic = true;
            }

            i++;
        }

        if (hasItalic && !hasNonItalic) return true;
        if (hasNonItalic && !hasItalic) return false;
        return null; // mixed or no text content
    }

    /// <summary>
    /// Combines the italic classification of both lines in a lyric block.
    /// Returns true/false only when both lines agree; null otherwise.
    /// </summary>
    static bool? ClassifyLyricBlockItalic(string line1, string? line2)
    {
        bool? r1 = IsLyricContentItalic(line1);
        if (line2 == null) return r1;
        bool? r2 = IsLyricContentItalic(line2);
        if (r1 == true && r2 == true) return true;
        if (r1 == false && r2 == false) return false;
        return null;
    }

    /// <summary>Removes only &lt;i&gt; and &lt;/i&gt; tags, leaving other markup intact.</summary>
    static string StripItalicTags(string text) =>
        Regex.Replace(text, "</?i>", "", RegexOptions.IgnoreCase);

    // ── Shared helpers ─────────────────────────────────────────────────────────

    static bool ContainsMusicGlyph(string text)
    {
        foreach (char c in text)
            if (IsMusicGlyph(c))
                return true;
        return false;
    }

    static bool IsMusicGlyph(char c) => Array.IndexOf(MusicGlyphs, c) >= 0;

    static string StripTags(string text) =>
        Regex.Replace(text, "<[^>]+>", "");

    /// <summary>
    /// Returns true if every music-note glyph in the tag-stripped line has a
    /// space (or a string boundary / another glyph) on each side.
    /// A single missing space → false.
    /// </summary>
    static bool LineHasSpacesAroundGlyphs(string rawLine)
    {
        string s = StripTags(rawLine);
        for (int i = 0; i < s.Length; i++)
        {
            if (!IsMusicGlyph(s[i])) continue;
            if (i > 0
                && !IsMusicGlyph(s[i - 1])
                && !char.IsWhiteSpace(s[i - 1]))
                return false;
            if (i < s.Length - 1
                && !IsMusicGlyph(s[i + 1])
                && !char.IsWhiteSpace(s[i + 1]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Inserts a space between each music-note glyph and any adjacent
    /// non-glyph, non-whitespace content (HTML-tag-aware).
    ///
    /// Three passes handle the common layout patterns:
    ///   pass 1 — glyph immediately before text/tag:  ♪text  →  ♪ text
    ///   pass 2 — text immediately before glyph:      text♪  →  text ♪
    ///   pass 3 — closing tag immediately before glyph: </i>♪ → </i> ♪
    /// </summary>
    [SuppressMessage("ReSharper", "InvalidXmlDocComment")]
    static string AddSpacesAroundGlyphs(string text)
    {
        // After glyph: add space unless already followed by space, newline, glyph, or '<' (HTML tag).
        text = Regex.Replace(text, $@"([{GlyphClass}])(?=[^ \n<{GlyphClass}])", "$1 ");
        // Before glyph: add space unless already preceded by space, newline, glyph, '>', or '-'.
        // '>' is excluded because it is the end of an HTML tag — not a visible character.
        // '-' is excluded so that speaker-change dashes ("-♪") are not split into "- ♪".
        text = Regex.Replace(text, $@"(?<=[^ \n{GlyphClass}>\-])([{GlyphClass}])", " $1");
        // Closing tag immediately before glyph (the '>' exclusion above suppresses step 2
        // for this case, so handle it explicitly):  </i>♪ → </i> ♪
        text = Regex.Replace(text, $@"(</[^>]+>)[^\S\n]*([{GlyphClass}])", "$1 $2");
        return text;
    }

    /// <summary>
    /// Removes all horizontal whitespace (spaces/tabs, but not newlines) that
    /// sits immediately before or after a music-note glyph.
    /// </summary>
    static string RemoveSpacesAroundGlyphs(string text)
    {
        text = Regex.Replace(text, $@"([{GlyphClass}])[^\S\n]+", "$1");
        text = Regex.Replace(text, $@"[^\S\n]+([{GlyphClass}])", "$1");
        return text;
    }
}