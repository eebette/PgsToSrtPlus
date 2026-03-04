using System.Text;
using System.Text.RegularExpressions;

namespace PgsToSrtPlus;

/// <summary>
/// Mid-processing steps applied to raw OCR text before italic detection.
/// Each language config defines which steps to run and in what order.
/// </summary>
static class OcrMidProcessor
{
    public delegate string Step(string text);

    public static string StripItalicTags(string text) =>
        Regex.Replace(text, "</?i>", "", RegexOptions.IgnoreCase).Trim();

    /// <summary>
    /// Ensures that square-bracketed expressions and non-ASCII symbol glyphs
    /// (♪, ♫, …) are each surrounded by a single space, then trims the result.
    ///
    /// Rules:
    ///   • A space is inserted before '[' when the preceding character is not a space.
    ///   • A space is inserted after ']' when the following character is not a space.
    ///   • A space is inserted on both sides of any non-ASCII, non-letter/digit character
    ///     (e.g. ♪) when the adjacent character is not already a space.
    ///   • The whole string is trimmed afterward, so leading/trailing spaces produced
    ///     by the above rules are removed.
    /// </summary>
    public static string NormalizeSpacing(string text)
    {
        var sb = new StringBuilder(text.Length + 8);

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];

            bool spaceBefore = c == '[' || (!char.IsAscii(c) && !char.IsLetterOrDigit(c));
            bool spaceAfter = c == ']' || (!char.IsAscii(c) && !char.IsLetterOrDigit(c));

            // Exception: -[Name] at the start of a line (speaker-change indicator)
            // does not need a space between the dash and the bracket.
            bool prevIsDash = sb.Length > 0 && sb[^1] == '-';
            if (spaceBefore && sb.Length > 0 && sb[^1] != ' ' && !prevIsDash)
                sb.Append(' ');

            sb.Append(c);

            if (spaceAfter && i + 1 < text.Length && text[i + 1] != ' ')
                sb.Append(' ');
        }

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Normalize Unicode ellipsis (…) to three ASCII dots and join tightly to
    /// surrounding text (no spaces), e.g. "… the fog" → "...the fog".
    /// </summary>
    public static string NormalizeUnicodeEllipsis(string text)
    {
        return Regex.Replace(text, @"\s*…\s*", "...");
    }

    /// <summary>
    /// Ensure every '.' is followed by a space when immediately preceding a
    /// non-whitespace, non-dot character.  Excludes decimal separators between
    /// two digits (e.g. "99.9") and trailing periods (no following character).
    /// Dots within an ellipsis run ("...") are left alone — the lookbehind
    /// also excludes dots preceded by another dot, so the last dot of "..."
    /// is never matched even when directly touching the next word.
    ///   "Thank you.I love you." → "Thank you. I love you."
    ///   "99.9 percent"          → unchanged
    ///   "...word"               → unchanged (last dot preceded by '.')
    /// </summary>
    public static string FixDotSpacing(string text)
    {
        return Regex.Replace(
            text,
            @"(?<![.\d])\.(?=[^\s.])|(?<=\d)\.(?=[^\d\s.])",
            ". ");
    }

    /// <summary>
    /// Collapse space(s) after a leading ellipsis: "... word" → "...word".
    /// Also handles ellipsis after a bracketed term: "[Name] ... word" → "[Name] ...word".
    /// Runs after the period rule so the two cannot interfere.
    /// Mid-sentence ellipses ("word ... word") are left unchanged.
    /// </summary>
    public static string CollapseSpaceAfterLeadingEllipsis(string text)
    {
        return Regex.Replace(text, @"(?:^|(?<=\] ))\.\.\.\ +", "...");
    }

    public static string Process(string text, string language)
    {
        foreach (var step in OcrLanguageConfigs.ForLanguage(language).MidProcessingSteps)
            text = step(text);
        return text;
    }
}