using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using JetBrains.Annotations;
using SkiaSharp;

namespace PgsToSrtPlus;

/// <summary>
/// Sends a preprocessed subtitle line image to PaddleOCR for text recognition
/// with segment bounding boxes, then uses shear-projection-variance analysis
/// to detect italic text.
///
/// Pipeline per image:
///   1. Full PaddleOCR pipeline (det + rec) -> text segments with bounding boxes.
///      If average confidence is below threshold, fall back to Ollama VLM OCR.
///   2. Mid-processing (normalize spacing, strip tags, etc.).
///   3. Identify leading tag segments (speaker tags, sound cues) — always roman.
///   4. Crop bitmap to body region (excluding tags), run shear italic detection.
///   5. Reconstruct final string with &lt;i&gt;…&lt;/i&gt; tags around body if italic.
///
/// Fallback: when PaddleOCR merges a tag+body into a single segment (preventing
/// segment-level tag splitting), the line is routed to the legacy VLM italic
/// detection path: tokenize → render roman reference → two-image chat → per-token
/// I/R classification.
/// </summary>
static partial class OcrEngine
{
    // HTTP / JSON record types

    record OllamaGenerateRequest(
        [UsedImplicitly]
        [property: JsonPropertyName("model")]
        string Model,
        [property: JsonPropertyName("prompt")] string Prompt,
        [property: JsonPropertyName("images")] string[] Images,
        [property: JsonPropertyName("stream")] bool Stream,
        [property: JsonPropertyName("options")]
        OllamaOptions Options
    );

    record OllamaChatRequest(
        [UsedImplicitly]
        [property: JsonPropertyName("model")]
        string Model,
        [property: JsonPropertyName("think")] bool Think,
        [property: JsonPropertyName("messages")]
        OllamaChatMessage[] Messages,
        [property: JsonPropertyName("stream")] bool Stream,
        [property: JsonPropertyName("options")]
        OllamaOptions? Options
    );

    record OllamaChatMessage(
        [UsedImplicitly]
        [property: JsonPropertyName("role")]
        string Role,
        [property: JsonPropertyName("content")]
        string Content,
        [property: JsonPropertyName("images")] string[]? Images
    );

    record OllamaOptions(
        [UsedImplicitly]
        [property: JsonPropertyName("temperature")]
        double Temperature,
        [property: JsonPropertyName("top_k")] int TopK,
        [property: JsonPropertyName("top_p")] double TopP,
        [property: JsonPropertyName("seed")] int Seed,
        [property: JsonPropertyName("num_predict")]
        int NumPredict,
        [property: JsonPropertyName("repeat_penalty")]
        double RepeatPenalty,
        [property: JsonPropertyName("stop")] string[]? Stop
    );

    record OllamaGenerateResponse(
        [property: JsonPropertyName("response")]
        string? Response
    );

    record OllamaChatResponse(
        [property: JsonPropertyName("message")]
        OllamaChatMessage? Message
    );

    record ItalicDetectionResult(
        [property: JsonPropertyName("classification")]
        string[]? Classification
    );

    // Serialize chat requests omitting null fields (e.g. images on system messages).
    static readonly JsonSerializerOptions JsonOmitNull = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    // -- VLM helpers -----------------------------------------------------------

    /// <summary>Single-image /api/generate call -> trimmed response text, or null on failure.</summary>
    static string? RunGenerate(
        string prompt, byte[] pngBytes, HttpClient http, string ollamaUrl, string model)
    {
        var payload = new OllamaGenerateRequest(
            Model: model,
            Prompt: prompt,
            Images: [Convert.ToBase64String(pngBytes)],
            Stream: false,
            Options: new OllamaOptions(Temperature: 0, TopK: 1, TopP: 1, Seed: 42, NumPredict: 1280, RepeatPenalty: 1.0,
                Stop: [])
        );

        try
        {
            string json = JsonSerializer.Serialize(payload);
            using var req = new HttpRequestMessage(HttpMethod.Post, $"{ollamaUrl}/api/generate");
            req.Content = new StringContent(json, Encoding.UTF8, "application/json");
            using var resp = http.Send(req);
            resp.EnsureSuccessStatusCode();
            using var stream = resp.Content.ReadAsStream();
            var result = JsonSerializer.Deserialize<OllamaGenerateResponse>(stream);
            return result?.Response?.Trim();
        }
        catch (HttpRequestException ex) when (ex.Message.Contains("Connection refused"))
        {
            throw new InvalidOperationException(
                $"Cannot reach Ollama at {ollamaUrl}. Is it running?  (ollama serve)", ex);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] Ollama generate request failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Two-image /api/chat call for italic detection.
    /// Sends originalPng as Image 1 and referencePng as Image 2.
    /// Returns (raw model response text or null, request GUID used as cache-busting nonce).
    /// </summary>
    static (string? Raw, string RequestId) RunItalicChat(
        string[] tokenTexts,
        byte[] originalPng,
        byte[] referencePng,
        string ollamaUrl,
        string model,
        string systemPrompt)
    {
        string tokensJson = JsonSerializer.Serialize(tokenTexts);
        string requestId = Guid.NewGuid().ToString();

        var payload = new OllamaChatRequest(
            Model: model,
            Think: false,
            Messages:
            [
                new OllamaChatMessage("system", $"Request-ID: {requestId}\n" + systemPrompt, null),
                new OllamaChatMessage("user", "Image 1 (PGS bitmap to classify):",
                    [Convert.ToBase64String(originalPng)]),
                new OllamaChatMessage("assistant", "Received Image 1.", null),
                new OllamaChatMessage("user", "Image 2 (upright roman reference):",
                    [Convert.ToBase64String(referencePng)]),
                new OllamaChatMessage("assistant", "Received Image 2.", null),
                new OllamaChatMessage("user", $"Tokens: {tokensJson}", null)
            ],
            Stream: false,
            Options: new OllamaOptions(
                Temperature: 0,
                TopK: 1,
                TopP: 1.0,
                Seed: 42,
                NumPredict: 50,
                RepeatPenalty: 1.0,
                Stop: null)
        );

        try
        {
            string json = JsonSerializer.Serialize(payload, JsonOmitNull);

            // Use a fresh HttpClient per italic request to avoid Ollama KV-cache
            // contamination from previous requests on the same persistent connection.
            using var freshHttp = new HttpClient();
            freshHttp.Timeout = TimeSpan.FromSeconds(300);

            using var req = new HttpRequestMessage(HttpMethod.Post, $"{ollamaUrl}/api/chat");
            req.Content = new StringContent(json, Encoding.UTF8, "application/json");
            req.Version = System.Net.HttpVersion.Version11;

            using var resp = freshHttp.Send(req);
            resp.EnsureSuccessStatusCode();

            using var stream = resp.Content.ReadAsStream();

            var result = JsonSerializer.Deserialize<OllamaChatResponse>(stream);
            return (result?.Message?.Content.Trim(), requestId);
        }
        catch (HttpRequestException ex) when (ex.Message.Contains("Connection refused"))
        {
            throw new InvalidOperationException(
                $"Cannot reach Ollama at {ollamaUrl}. Is it running?  (ollama serve)", ex);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] Ollama chat request failed: {ex.Message}");
            return (null, requestId);
        }
    }

    /// <summary>
    /// Extracts a JSON object from the model's response (handles Markdown code fences).
    /// Returns null if no valid JSON object can be found or deserialized.
    /// </summary>
    static ItalicDetectionResult? ParseItalicResponse(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;

        var match = Regex.Match(raw, @"\{[\s\S]*\}", RegexOptions.Singleline);
        if (!match.Success) return null;

        try
        {
            return JsonSerializer.Deserialize<ItalicDetectionResult>(
                match.Value,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        }
        catch
        {
            return null;
        }
    }

    // -- Tokenization ----------------------------------------------------------

    /// <summary>
    /// Returns true for characters in CJK unified ideograph, hiragana, katakana,
    /// CJK extension A, CJK compatibility ideograph, and fullwidth form blocks.
    /// </summary>
    static bool IsCjk(char c) =>
        c is >= '\u3040' and <= '\u30FF' || // hiragana + katakana
        c is >= '\u4E00' and <= '\u9FFF' || // CJK unified ideographs (common kanji)
        c is >= '\u3400' and <= '\u4DBF' || // CJK extension A
        c is >= '\uF900' and <= '\uFAFF' || // CJK compatibility ideographs
        c is >= '\uFF00' and <= '\uFFEF'; // fullwidth forms

    internal record Tokens(
        string Text,
        int Start
    );

    /// <summary>
    /// Tokenizes subtitle text into (Token, StartIndex) pairs for italic detection.
    ///
    /// Rules (in priority order):
    ///   1. [bracketed content] including the brackets → one token.
    ///   2+3. ASCII punctuation/symbols → one token per consecutive run.
    ///   4. Non-ASCII non-letter-digit symbols (♪, ♫, …) → one token per consecutive run.
    ///   4b. CJK characters (kanji, hiragana, katakana, fullwidth) → one token per character.
    ///   5. Everything else: one token per whitespace-delimited word
    ///      (letters and digits, ASCII and non-ASCII).
    /// </summary>
    internal static Tokens[] Tokenize(string text)
    {
        var tokens = new List<Tokens>();
        int i = 0;

        while (i < text.Length)
        {
            char c = text[i];

            // Skip whitespace.
            if (char.IsWhiteSpace(c))
            {
                i++;
                continue;
            }

            // Rule 1: [bracketed content] → single token.
            if (c == '[')
            {
                int close = text.IndexOf(']', i + 1);
                if (close >= 0)
                {
                    tokens.Add(new Tokens(text[i..(close + 1)], i));
                    i = close + 1;
                    continue;
                }
                // No matching ']' — fall through to ASCII punctuation rule.
            }

            // Rule 2: Non-ASCII non-letter/digit symbols (♪, ♫, etc.) → grouped.
            if (!char.IsAscii(c) && !char.IsLetterOrDigit(c))
            {
                int j = i + 1;
                while (j < text.Length && !char.IsAscii(text[j]) && !char.IsLetterOrDigit(text[j]))
                    j++;
                tokens.Add(new Tokens(text[i..j], i));
                i = j;
                continue;
            }

            // Rule 2b: CJK character → one token per character.
            if (IsCjk(c))
            {
                tokens.Add(new Tokens(text[i..(i + 1)], i));
                i++;
                continue;
            }

            // Rules 3+4: ASCII punctuation/symbols → grouped consecutive run.
            // Stop before '[' so a new bracket token can start cleanly.
            if (char.IsAscii(c) && !char.IsLetterOrDigit(c) && !char.IsWhiteSpace(c))
            {
                int j = i + 1;
                while (j < text.Length
                       && char.IsAscii(text[j])
                       && !char.IsLetterOrDigit(text[j])
                       && !char.IsWhiteSpace(text[j])
                       && text[j] != '[')
                    j++;
                tokens.Add(new Tokens(text[i..j], i));
                i = j;
                continue;
            }

            // Rule 5: Word — letters and digits (ASCII and non-ASCII).
            // A mid-word apostrophe (e.g. "it's", "don't") is absorbed into the
            // word token when it is immediately followed by another letter or digit.
            {
                int j = i + 1;
                while (j < text.Length)
                {
                    if (char.IsLetterOrDigit(text[j]))
                    {
                        j++;
                        continue;
                    }

                    if (text[j] == '\'' && j + 1 < text.Length && char.IsLetterOrDigit(text[j + 1]))
                    {
                        j++;
                        continue;
                    }

                    break;
                }

                tokens.Add(new Tokens(text[i..j], i));
                i = j;
            }
        }

        return tokens.ToArray();
    }

    // -- Reference image rendering ---------------------------------------------

    /// <summary>
    /// Cleans text for use in a synthetic reference image:
    ///   • Replaces leading speaker-change dashes with 2 spaces.
    ///   • Replaces non-ASCII symbol glyphs (♪ ♫ … etc.) with 2 spaces each.
    ///   • Collapses runs of 3+ spaces to exactly 2.
    /// Returns null when nothing renderable remains.
    /// </summary>
    static string? CleanForReference(string text)
    {
        text = Regex.Replace(text, @"^-+\s*", "  ");

        var sb = new StringBuilder(text.Length + 8);
        foreach (char c in text)
        {
            if (char.IsAscii(c) || char.IsLetterOrDigit(c))
                sb.Append(c);
            else
                sb.Append("  ");
        }

        string result = Regex.Replace(sb.ToString(), " {3,}", "  ");
        return result.Trim().Length > 0 ? result : null;
    }

    /// <summary>
    /// Renders <paramref name="text"/> as a dark-on-white reference image sized to match
    /// <paramref name="sourcePng"/> and returns it as a PNG byte array. Returns null on failure.
    /// </summary>
    static byte[]? RenderAndPreprocess(string text, SKTypeface typeface, byte[]? sourcePng = null)
    {
        string? cleaned = CleanForReference(text);
        if (cleaned == null) return null;
        text = cleaned;

        int contentH = 80, padPx = 14;
        if (sourcePng != null)
        {
            using var srcBmp = SKBitmap.Decode(sourcePng);
            if (srcBmp != null)
            {
                int c = srcBmp.Height - 28;
                if (c > 0 && Math.Max(14, (int)MathF.Round(0.15f * c)) == 14)
                {
                    contentH = Math.Max(10, c);
                    padPx = 14;
                }
                else
                {
                    contentH = Math.Max(10, (int)MathF.Round(srcBmp.Height / 1.3f));
                    padPx = Math.Max(14, (int)MathF.Round(0.15f * contentH));
                }
            }
        }

        float inkH42;
        using (var mf = new SKFont(typeface, 42f))
        {
            mf.MeasureText(text, out var b);
            inkH42 = b.Height > 0 ? b.Height : 30f;
        }

        float fontSize = 42f * (contentH / inkH42);

        using var font = new SKFont(typeface, fontSize);
        font.MeasureText(text, out var inkBounds);

        int imgW = Math.Max(1, (int)MathF.Ceiling(font.MeasureText(text)) + 2 * padPx + 4);
        int imgH = contentH + 2 * padPx;

        using var bmp = new SKBitmap(imgW, imgH, SKColorType.Bgra8888, SKAlphaType.Opaque);
        using var canvas = new SKCanvas(bmp);
        canvas.Clear(SKColors.White);

        using var paint = new SKPaint();
        paint.Color = SKColors.Black;
        paint.IsAntialias = true;
        paint.Style = SKPaintStyle.StrokeAndFill;
        paint.StrokeWidth = 0.3f;
        paint.StrokeJoin = SKStrokeJoin.Round;

        canvas.DrawText(text, padPx, padPx - inkBounds.Top, font, paint);
        canvas.Flush();

        const float blurSigma = 0.6f;
        using var blurredBmp = new SKBitmap(imgW, imgH, SKColorType.Bgra8888, SKAlphaType.Opaque);
        using var blurCanvas = new SKCanvas(blurredBmp);
        blurCanvas.Clear(SKColors.White);
        using var blurFilter = SKImageFilter.CreateBlur(blurSigma, blurSigma);
        using var blurPaint = new SKPaint();
        blurPaint.ImageFilter = blurFilter;
        blurCanvas.DrawBitmap(bmp, 0, 0, blurPaint);
        blurCanvas.Flush();

        using var gray = new SKBitmap(imgW, imgH, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            byte* sp = (byte*)blurredBmp.GetPixels();
            byte* dp = (byte*)gray.GetPixels();
            for (int i = 0; i < imgW * imgH; i++)
            {
                byte b2 = sp[i * 4 + 0];
                byte g2 = sp[i * 4 + 1];
                byte r2 = sp[i * 4 + 2];
                dp[i] = (byte)(0.299f * r2 + 0.587f * g2 + 0.114f * b2);
            }
        }

        using var enc = gray.Encode(SKEncodedImageFormat.Png, 100);
        return enc.ToArray();
    }

    // -- VLM italic detection --------------------------------------------------

    /// <summary>
    /// Tokenize the text, render a roman reference image, then run two-image
    /// italic detection via Ollama VLM chat.
    /// </summary>
    static (Tokens[] italicTokens, ItalicDetectionResult detection)? GetItalicsClassification(
        string text,
        byte[] pngBytes,
        bool? debug,
        OcrLanguageConfig langConfig,
        string ollamaUrl,
        string model,
        string? debugDir,
        string? debugPrefix
    )
    {
        var italicTokens = Tokenize(text);
        if (italicTokens.Length == 0) return null;

        // Skip when every letter-bearing token is a bracketed stage direction.
        bool allBracketed = Array.TrueForAll(italicTokens, t =>
            !t.Text.Any(char.IsLetterOrDigit)
            || (t.Text.Length > 1 && t.Text[0] == '[' && t.Text[^1] == ']'));
        if (allBracketed)
        {
            if (debug ?? false)
                Console.WriteLine($"  [dbg] {debugPrefix}  vlm-italic  skipped (all-bracketed)");
            return null;
        }

        var typeface = OcrLanguageConfigs.GetReferenceTypeface(langConfig);
        byte[]? italicRef = RenderAndPreprocess(text, typeface, pngBytes);
        if (italicRef == null)
        {
            Console.WriteLine("  [warn] italic reference render failed; using plain text.");
            return null;
        }

        if ((debug ?? false) && debugDir != null)
            File.WriteAllBytes(Path.Combine(debugDir, $"{debugPrefix}_italic_ref.png"), italicRef);

        string[] italicTexts = Array.ConvertAll(italicTokens, t => t.Text);

        var (italicRaw, italicReqId) =
            RunItalicChat(italicTexts, pngBytes, italicRef, ollamaUrl, model, langConfig.ItalicsPrompt);

        if (debug ?? false)
            Console.WriteLine($"  [dbg] {debugPrefix}  vlm-italic  req={italicReqId}");

        var detection = ParseItalicResponse(italicRaw);

        if (detection?.Classification == null
            || detection.Classification.Length != italicTokens.Length)
        {
            Console.WriteLine(
                "  [warn] italic detection returned no usable classification; using plain text.");
            return null;
        }

        return (italicTokens, detection);
    }

    // -- Classification post-processing ----------------------------------------

    /// <summary>
    /// If an ASCII punctuation token sits between two italic tokens and is not
    /// sentence-ending, promote it to italic so tags don't split mid-phrase.
    /// </summary>
    static string[] PostProcessClassification(
        string[] classification, Tokens[] tokens)
    {
        var cls = (string[])classification.Clone();

        for (int i = 1; i < tokens.Length - 1; i++)
        {
            if (cls[i].Equals("I", StringComparison.OrdinalIgnoreCase)) continue;

            string tok = tokens[i].Text;

            bool allAsciiPunct = true;
            foreach (char c in tok)
            {
                if (!char.IsAscii(c) || char.IsLetterOrDigit(c))
                {
                    allAsciiPunct = false;
                    break;
                }
            }

            if (!allAsciiPunct) continue;

            bool sentenceEnding = true;
            foreach (char c in tok)
            {
                if (c != '.' && c != '!' && c != '?')
                {
                    sentenceEnding = false;
                    break;
                }
            }

            if (sentenceEnding) continue;

            if (cls[i - 1].Equals("I", StringComparison.OrdinalIgnoreCase) &&
                cls[i + 1].Equals("I", StringComparison.OrdinalIgnoreCase))
                cls[i] = "I";
        }

        return cls;
    }

    /// <summary>
    /// Expands each italic span outward through adjacent ASCII punctuation tokens,
    /// provided the token on the far side is not a roman word token.
    /// </summary>
    static string[] ExpandItalicBoundaries(
        string[] classification, Tokens[] tokens)
    {
        var cls = (string[])classification.Clone();
        int n = tokens.Length;

        static bool IsAsciiPunct(string tok)
        {
            if (tok.Length == 0) return false;
            foreach (char c in tok)
                if (!char.IsAscii(c) || char.IsLetterOrDigit(c))
                    return false;
            return true;
        }

        bool IsRomanWord(int idx) =>
            !cls[idx].Equals("I", StringComparison.OrdinalIgnoreCase)
            && tokens[idx].Text.Any(char.IsLetterOrDigit);

        int i = 0;
        while (i < n)
        {
            if (!cls[i].Equals("I", StringComparison.OrdinalIgnoreCase))
            {
                i++;
                continue;
            }

            int spanEnd = i;
            while (spanEnd + 1 < n
                   && cls[spanEnd + 1].Equals("I", StringComparison.OrdinalIgnoreCase))
                spanEnd++;

            int left = i - 1;
            while (left >= 0 && IsAsciiPunct(tokens[left].Text))
            {
                if (left - 1 >= 0 && IsRomanWord(left - 1)) break;
                cls[left] = "I";
                left--;
            }

            int right = spanEnd + 1;
            while (right < n && IsAsciiPunct(tokens[right].Text))
            {
                if (right + 1 < n && IsRomanWord(right + 1)) break;
                cls[right] = "I";
                right++;
            }

            i = spanEnd + 1;
        }

        return cls;
    }

    /// <summary>
    /// If every non-bracketed word token is italic except exactly one, promotes
    /// all non-bracketed tokens to italic (catches stray misclassifications).
    /// </summary>
    static string[] PromoteNearlyAllItalic(
        string[] classification, Tokens[] tokens)
    {
        int wordCount = 0;
        int romanWords = 0;

        for (int i = 0; i < tokens.Length; i++)
        {
            string tok = tokens[i].Text;
            if (!tok.Any(char.IsLetterOrDigit)) continue;
            if (tok.Length > 1 && tok[0] == '[' && tok[^1] == ']') continue;
            wordCount++;
            if (!classification[i].Equals("I", StringComparison.OrdinalIgnoreCase))
                romanWords++;
        }

        if (romanWords != 1 || wordCount < 2) return classification;

        var cls = (string[])classification.Clone();
        for (int i = 0; i < cls.Length; i++)
        {
            string tok = tokens[i].Text;
            if (tok.Length > 1 && tok[0] == '[' && tok[^1] == ']') continue;
            cls[i] = "I";
        }

        return cls;
    }

    // -- Text reconstruction ---------------------------------------------------

    /// <summary>
    /// Reconstructs the original text with &lt;i&gt;…&lt;/i&gt; wrapping italic token runs.
    /// Whitespace and characters between tokens are preserved verbatim.
    /// </summary>
    static string BuildStyledText(
        string text,
        Tokens[] tokens,
        string[] classification)
    {
        if (tokens.Length == 0) return text;

        var sb = new StringBuilder(text.Length + 32);
        int pos = 0;
        bool inItalic = false;

        for (int i = 0; i < tokens.Length; i++)
        {
            int start = tokens[i].Start;
            int len = tokens[i].Text.Length;
            bool italic = i < classification.Length
                          && classification[i].Equals("I", StringComparison.OrdinalIgnoreCase);

            if (start > pos)
            {
                if (!italic && inItalic)
                {
                    sb.Append("</i>");
                    inItalic = false;
                }

                sb.Append(text, pos, start - pos);
            }

            if (italic && !inItalic)
            {
                sb.Append("<i>");
                inItalic = true;
            }

            if (!italic && inItalic)
            {
                sb.Append("</i>");
                inItalic = false;
            }

            sb.Append(tokens[i].Text);
            pos = start + len;
        }

        if (inItalic) sb.Append("</i>");

        if (pos < text.Length)
            sb.Append(text, pos, text.Length - pos);

        return sb.ToString();
    }

    // -- Tag identification (segment-level) ------------------------------------

    // -- Text-based tag identification ------------------------------------------

    /// <summary>
    /// Regex pattern matching leading tags in OCR text.
    /// </summary>
    static readonly Regex LeadingTagPattern = new(
        @"^(?:-+\s*)?(?:(?:\[[^\]]*\]\s*)+(?:[A-Z][A-Z0-9]*:\s*)?(?:\[[^\]]*\]\s*)*|(?:[A-Z][A-Z0-9]*:\s*)(?:\[[^\]]*\]\s*)*)",
        RegexOptions.Compiled);

    /// <summary>
    /// Finds the tag/body boundary in the final OCR text string.
    /// Returns (tagLength, bodyStart) where fullText[..tagLength] is the tag prefix.
    /// Returns (0, 0) if no tag is found.
    /// </summary>
    static (int TagLength, int BodyStart) FindTagBoundaryInText(string text)
    {
        var m = LeadingTagPattern.Match(text);
        if (!m.Success || m.Length == 0) return (0, 0);

        string remainder = text[m.Length..];
        if (string.IsNullOrWhiteSpace(remainder)) return (0, 0);

        int tagEnd = m.Length;
        while (tagEnd > 0 && text[tagEnd - 1] == ' ') tagEnd--;

        return (tagEnd, m.Length);
    }

    // -- Helpers ----------------------------------------------------------------

    /// <summary>
    /// Crops the image to the horizontal pixel range [xStart, xEnd), full height.
    /// </summary>
    static byte[]? CropXRange(byte[] pngBytes, int xStart, int xEnd)
    {
        using var srcBmp = SKBitmap.Decode(pngBytes);
        if (srcBmp == null) return null;

        xStart = Math.Max(0, xStart);
        xEnd = Math.Min(xEnd, srcBmp.Width);
        if (xStart >= xEnd) return null;

        int w = xEnd - xStart;
        int h = srcBmp.Height;

        using var cropped = new SKBitmap(w, h, srcBmp.ColorType, srcBmp.AlphaType);
        using var canvas = new SKCanvas(cropped);
        canvas.DrawBitmap(srcBmp, new SKRect(xStart, 0, xEnd, h), new SKRect(0, 0, w, h));
        canvas.Flush();

        using var enc = cropped.Encode(SKEncodedImageFormat.Png, 100);
        return enc.ToArray();
    }

    /// <summary>
    /// Strips whitespace and lowercases for fuzzy tag verification.
    /// </summary>
    static string NormalizeForCompare(string s) =>
        new string(s.Where(c => !char.IsWhiteSpace(c)).ToArray()).ToLowerInvariant();

    // -- Public API -------------------------------------------------------------

    /// <summary>
    /// Full pipeline for a single preprocessed subtitle line image.
    ///
    ///   1. Recognition-only PaddleOCR on the full line image.
    ///   2. Mid-processing (normalize spacing, strip tags, etc.).
    ///   3. If a leading tag is detected in the text:
    ///      a. Projection-profile word segmentation to find pixel boundaries.
    ///      b. Verify tag crop via recognition-only OCR.
    ///      c. Shear italic detection on tag crop and body crop separately.
    ///   4. Otherwise, shear italic detection on the full line.
    /// </summary>
    public static string? OcrImage(
        byte[] pngBytes,
        HttpClient http,
        string ollamaUrl,
        string model,
        PaddleOcrWorker paddle,
        bool? debug = false,
        string? debugDir = null,
        string? debugPrefix = null,
        double paddleVerifyThreshold = 1.0,
        string language = "en"
    )
    {
        var langConfig = OcrLanguageConfigs.ForLanguage(language);
        string ocrPrompt = langConfig.Prompt;

        // Step 1: Recognition-only OCR on the full line image.
        var (plainText, paddleScore) = paddle.RecognizeOnly(pngBytes);
        bool paddleFallback = false;

        if (paddleScore < paddleVerifyThreshold || string.IsNullOrWhiteSpace(plainText))
        {
            paddleFallback = true;
            plainText = RunGenerate(ocrPrompt, pngBytes, http, ollamaUrl, model);
        }

        if (debug ?? false)
        {
            using var bmp = SKBitmap.Decode(pngBytes);
            string scoreInfo = paddleFallback
                ? $"  confidence={paddleScore:F3} -> vlm-fallback"
                : $"  confidence={paddleScore:F3}";
            Console.WriteLine(
                $"  [dbg] {debugPrefix}  step1-ocr  {scoreInfo}  image={bmp?.Width}x{bmp?.Height}");
        }

        if (string.IsNullOrWhiteSpace(plainText)) return null;

        // Step 2: Mid-processing.
        plainText = OcrMidProcessor.Process(plainText, language);

        // Step 3: Tag detection + projection-profile word segmentation.
        var (tagLen, bodyStart) = FindTagBoundaryInText(plainText);
        WordBoundary[] words = [];
        bool tagVerified = false;
        int tagSegCount = 0;
        byte[]? tagCrop = null;

        if (tagLen > 0 && bodyStart < plainText.Length)
        {
            string tagText = plainText[..tagLen];
            string bodyText = plainText[bodyStart..];
            words = paddle.WordSegment(pngBytes);

            if (debug ?? false)
            {
                int tagWordCount = tagText.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
                Console.WriteLine(
                    $"  [dbg] {debugPrefix}  tag=\"{tagText}\"  body=\"{bodyText}\"  words={words.Length}  tagWords={tagWordCount}");
            }

            // Progressively accumulate word segments from left to right
            // until the OCR of the crop matches the full tag text.
            // Brackets and punctuation may be split into their own segments,
            // so a fixed word-count estimate is insufficient.
            string normTag = NormalizeForCompare(tagText);

            for (int n = 1; n < words.Length; n++)
            {
                byte[]? crop = CropXRange(pngBytes, words[0].Start, words[n - 1].End);
                if (crop == null) continue;

                var (verifyText, _) = paddle.RecognizeOnly(crop);
                string normVerify = NormalizeForCompare(verifyText ?? "");

                if (debug ?? false)
                    Console.WriteLine(
                        $"  [dbg] {debugPrefix}  tag-verify  n={n}  expected=\"{tagText}\"  got=\"{verifyText}\"  match={normTag == normVerify}");

                if (normTag == normVerify)
                {
                    tagCrop = crop;
                    tagSegCount = n;
                    tagVerified = true;
                    break;
                }
            }
        }

        // ── VLM fallback gate ─────────────────────────────────────────────────
        var fallbackCtx = new VlmFallbackContext(plainText, tagLen, bodyStart, words, tagVerified);
        if (CheckVlmFallback(fallbackCtx) is { } vlmReason)
        {
            if (debug ?? false)
                Console.WriteLine($"  [dbg] {debugPrefix}  vlm-fallback={vlmReason}");

            return RunVlmItalicFallback(
                plainText, pngBytes, paddleFallback, ocrPrompt,
                http, ollamaUrl, model, language, langConfig,
                debug, debugDir, debugPrefix);
        }

        // ── Normal path: shear italic detection ──────────────────────────────

        // Tag verified — italic detect tag and body separately.
        if (tagVerified && tagCrop != null && tagSegCount < words.Length)
        {
            int bodyPixelStart = words[tagSegCount].Start;
            byte[]? bodyCrop = CropXRange(pngBytes, bodyPixelStart, int.MaxValue);

            if ((debug ?? false) && debugDir != null)
            {
                File.WriteAllBytes(
                    Path.Combine(debugDir, $"{debugPrefix}_tag_crop.png"), tagCrop);
                if (bodyCrop != null)
                    File.WriteAllBytes(
                        Path.Combine(debugDir, $"{debugPrefix}_body_crop.png"), bodyCrop);
            }

            var tagItalicResult = paddle.DetectItalicAngle(tagCrop);

            if (bodyCrop != null)
            {
                var bodyItalicResult = paddle.DetectItalicAngle(bodyCrop);

                if (debug ?? false)
                {
                    Console.WriteLine(
                        $"  [dbg] {debugPrefix}  italic-tag  angle={tagItalicResult.Angle:F1}  italic={tagItalicResult.IsItalic}");
                    Console.WriteLine(
                        $"  [dbg] {debugPrefix}  italic-body  angle={bodyItalicResult.Angle:F1}  italic={bodyItalicResult.IsItalic}");
                }

                if (tagItalicResult.IsItalic && bodyItalicResult.IsItalic)
                    return $"<i>{plainText.TrimEnd()}</i>";
                if (bodyItalicResult.IsItalic)
                    return $"{plainText[..bodyStart]}<i>{plainText[bodyStart..].TrimEnd()}</i>";
                if (tagItalicResult.IsItalic)
                    return $"<i>{plainText[..tagLen]}</i>{plainText[tagLen..].TrimEnd()}";
                return plainText;
            }
        }

        // No tag split: italic detect on full line.
        var italicResult = paddle.DetectItalicAngle(pngBytes);

        if (debug ?? false)
            Console.WriteLine(
                $"  [dbg] {debugPrefix}  italic-detect  angle={italicResult.Angle:F1}  italic={italicResult.IsItalic}");

        return italicResult.IsItalic ? $"<i>{plainText.TrimEnd()}</i>" : plainText;
    }
}
