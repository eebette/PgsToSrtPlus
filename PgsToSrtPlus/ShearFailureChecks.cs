using System.Diagnostics.CodeAnalysis;

namespace PgsToSrtPlus;

/// <summary>
/// Defines conditions under which the normal PaddleOCR + shear italic pipeline
/// should be bypassed in favor of full VLM fallback (both OCR recognition and
/// per-token italic detection).
///
/// To add a new rule, append a <see cref="VlmFallbackRule"/> entry to
/// <see cref="VlmFallbackRules"/>.
/// </summary>
static partial class OcrEngine
{
    /// <summary>State available to VLM fallback rule predicates.</summary>
    [SuppressMessage("ReSharper", "NotAccessedPositionalProperty.Local",
        Justification = "Properties are available for future fallback rules")]
    record VlmFallbackContext(
        string PlainText,
        int TagLen,
        int BodyStart,
        WordBoundary[] Words,
        bool TagVerified);

    record VlmFallbackRule(string Name, Func<VlmFallbackContext, bool> Predicate);

    /// <summary>All registered VLM fallback rules, evaluated in order.</summary>
    static readonly VlmFallbackRule[] VlmFallbackRules =
    [
        // Tag detected in text but iterative segment matching couldn't verify it.
        // Recognition-only and segment-level recognition disagree, so neither
        // can be trusted for tag/body splitting.
        new("tag-segment-mismatch", ctx =>
            ctx.TagLen > 0
            && ctx.BodyStart < ctx.PlainText.Length
            && !ctx.TagVerified),
    ];

    /// <summary>
    /// Returns the name of the first VLM fallback rule that fired, or null if
    /// none did (meaning the normal pipeline can proceed).
    /// </summary>
    static string? CheckVlmFallback(VlmFallbackContext ctx)
    {
        foreach (var rule in VlmFallbackRules)
        {
            if (rule.Predicate(ctx))
                return rule.Name;
        }

        return null;
    }

    /// <summary>
    /// Runs the legacy VLM italic detection fallback: re-OCR via VLM if needed,
    /// then tokenize → render roman reference → two-image chat → per-token I/R
    /// classification → post-processing → styled text.
    /// </summary>
    static string RunVlmItalicFallback(
        string plainText,
        byte[] pngBytes,
        bool paddleFallback,
        string ocrPrompt,
        HttpClient http,
        string ollamaUrl,
        string model,
        string language,
        OcrLanguageConfig langConfig,
        bool? debug,
        string? debugDir,
        string? debugPrefix)
    {
        // Force VLM for text recognition (PaddleOCR segmentation is unreliable here).
        if (!paddleFallback)
        {
            string? vlmText = RunGenerate(ocrPrompt, pngBytes, http, ollamaUrl, model);
            if (!string.IsNullOrWhiteSpace(vlmText))
                plainText = OcrMidProcessor.Process(vlmText, language);
        }

        var vlmItalicResult =
            GetItalicsClassification(plainText, pngBytes, debug, langConfig, ollamaUrl, model, debugDir,
                debugPrefix);
        if (vlmItalicResult == null) return plainText;

        var (italicTokens, detection) = vlmItalicResult.Value;
        string[] classification = PostProcessClassification(detection.Classification!, italicTokens);
        classification = ExpandItalicBoundaries(classification, italicTokens);
        classification = PromoteNearlyAllItalic(classification, italicTokens);
        return BuildStyledText(plainText, italicTokens, classification);
    }
}
