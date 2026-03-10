using System.Collections.Concurrent;
using System.Collections.Frozen;
using SkiaSharp;

namespace PgsToSrtPlus;

internal record OcrLanguageConfig(
    string Prompt,
    string ItalicsPrompt,
    string? ReferenceFontPath,
    string ReferenceFontFamily,
    OcrMidProcessor.Step[] MidProcessingSteps,
    SrtPostProcessor.Step[] PostProcessingSteps
);

static class OcrLanguageConfigs
{
    static readonly ConcurrentDictionary<string, SKTypeface> TypefaceCache = new();

    /// <summary>
    /// Resolves and caches the reference typeface for a language config.
    /// Tries the font file path first, then the family name, then SKTypeface.Default.
    /// </summary>
    public static SKTypeface GetReferenceTypeface(OcrLanguageConfig config)
    {
        string key = config.ReferenceFontPath ?? config.ReferenceFontFamily;
        return TypefaceCache.GetOrAdd(key, _ =>
            (config.ReferenceFontPath != null && File.Exists(config.ReferenceFontPath)
                ? SKTypeface.FromFile(config.ReferenceFontPath)
                : SKTypeface.FromFamilyName(
                    config.ReferenceFontFamily,
                    SKFontStyleWeight.Normal,
                    SKFontStyleWidth.Normal,
                    SKFontStyleSlant.Upright))
            ?? SKTypeface.Default);
    }

    static readonly SrtPostProcessor.Step[] DefaultPostProcessingSteps =
    [
        SrtPostProcessor.StandardizeMusicGlyph,
        SrtPostProcessor.StandardizeMusicGlyphSpacing,
        SrtPostProcessor.StandardizeMusicNoteItalic,
        SrtPostProcessor.StandardizeLyricItalic,
        SrtPostProcessor.MergeConsecutiveItalicSpans
    ];

    static readonly OcrLanguageConfig Default = new(
        ReferenceFontPath: "fonts/arial unicode ms.otf",
        ReferenceFontFamily: "Arial Unicode MS",
        Prompt:
        "You are a precision OCR engine for movie subtitle images. Rules: " +
        "1) Transcribe decorative symbols exactly (NO substitutions for symbols with similar meaning but different shape, such as replacing ♪ with ♫). " +
        "2) Preserve spacing. " +
        "3) Output 1 line for each image.",
        ItalicsPrompt:
        "You are a visual italic detector for subtitle images.\n\n" +
        "You will receive two images in separate messages, then a token list.\n" +
        "  Image 1: original PGS subtitle bitmap.\n" +
        "  Image 2: the same text rendered in upright (roman) style for reference.\n\n" +
        "For each token, determine if it is italic (I) or roman (R) in Image 1.\n\n" +
        "Italic text has a consistent rightward lean — vertical strokes (like l, t, h, d, b, f)\n" +
        "tilt clockwise instead of being straight up and down. If the vertical strokes in a word\n" +
        "are straight/vertical in Image 1, it is R, even if the two images look slightly different\n" +
        "in other ways (thickness, smoothness, spacing).\n\n" +
        "Image 2 shows what upright text looks like for this line. Use it to calibrate your eye,\n" +
        "but do NOT mark a token as I simply because it looks \"different\" from Image 2.\n" +
        "Differences in stroke weight, anti-aliasing, or sharpness are normal and do NOT indicate italic.\n" +
        "Only a visible rightward tilt of the letters indicates italic.\n\n" +
        "Rules:\n" +
        "- Classify each token independently based on its visual slant in Image 1.\n" +
        "- Bracketed tags like [Name] are usually upright.\n" +
        "- Base decisions ONLY on visual slant, never on word meaning or grammar.\n\n" +
        "Output STRICT JSON only, no other text:\n" +
        "{\"classification\": [\"I\" or \"R\", ...]}\n" +
        "Array length must equal token list length.",
        MidProcessingSteps:
        [
            OcrMidProcessor.StripItalicTags,
            OcrMidProcessor.NormalizeApostrophes,
            OcrMidProcessor.FixMissingWordBreaks,
            OcrMidProcessor.NormalizeSpacing,
            OcrMidProcessor.NormalizeUnicodeEllipsis,
            OcrMidProcessor.FixDotSpacing,
            OcrMidProcessor.CollapseSpaceAfterLeadingEllipsis
        ],
        PostProcessingSteps: DefaultPostProcessingSteps
    );

    static readonly FrozenDictionary<string, OcrLanguageConfig> ByLanguage =
        new Dictionary<string, OcrLanguageConfig>(StringComparer.OrdinalIgnoreCase)
        {
            ["en"] = Default,
            ["ja"] = new(
                ReferenceFontPath: "fonts/FOT-SeuratCapie Pro M.otf",
                ReferenceFontFamily: "FOT-SeuratCapie Pro M",
                Prompt:
                "You are a precision OCR engine for Japanese movie subtitle images. Rules: " +
                "1) Transcribe all kanji, hiragana, katakana, and symbols exactly. " +
                "2) Preserve spacing (there is usually none between Japanese words). " +
                "3) Output 1 line for each image.",
                ItalicsPrompt:
                "You are a visual italic detector for subtitle images.\n\n" +
                "You will receive two images in separate messages, then a token list.\n" +
                "  Image 1: original PGS subtitle bitmap.\n" +
                "  Image 2: the same text rendered in upright style for reference.\n\n" +
                "For each token, determine if it is italic (I) or roman (R) in Image 1.\n\n" +
                "Italic Japanese text has a consistent rightward lean — vertical strokes of kanji,\n" +
                "hiragana, and katakana tilt clockwise instead of being straight up and down.\n" +
                "If the strokes in Image 1 are straight/vertical, it is R.\n\n" +
                "Image 2 shows what upright text looks like for this line. Use it to calibrate your eye,\n" +
                "but do NOT mark a token as I simply because it looks \"different\" from Image 2.\n" +
                "Differences in stroke weight, anti-aliasing, or sharpness are normal and do NOT indicate italic.\n" +
                "Only a visible rightward tilt indicates italic.\n\n" +
                "Rules:\n" +
                "- Classify each token independently based on its visual slant in Image 1.\n" +
                "- Bracketed tags like [Name] are usually upright.\n" +
                "- Base decisions ONLY on visual slant, never on word meaning or grammar.\n\n" +
                "Output STRICT JSON only, no other text:\n" +
                "{\"classification\": [\"I\" or \"R\", ...]}\n" +
                "Array length must equal token list length.",
                MidProcessingSteps:
                [
                    OcrMidProcessor.StripItalicTags,
                    OcrMidProcessor.NormalizeSpacing,
                    OcrMidProcessor.NormalizeUnicodeEllipsis
                ],
                PostProcessingSteps: DefaultPostProcessingSteps
            ),
        }.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);

    public static OcrLanguageConfig ForLanguage(string language) =>
        ByLanguage.GetValueOrDefault(language, Default);
}
