using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using JetBrains.Annotations;
using SkiaSharp;

namespace PgsToSrtPlus;

/// <summary>
/// Sends a preprocessed subtitle line image to a local Ollama vision model
/// and returns the OCR text with italic markup preserved.
///
/// Pipeline per image:
///   1. Plain OCR via PaddleOcr or OcrPrompt → plain text.
///   2. Render a non-italic roman reference image from that text.
///   3. Tokenize the text; send both images + token array to the italic-detection
///      model (two-image chat call) → per-token I/R/U classification JSON.
///   4. Reconstruct the final string with &lt;i&gt;…&lt;/i&gt; tags.
/// </summary>
static class OcrEngine
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

    // Tokenization

    /// <summary>
    /// Returns true for characters in CJK unified ideograph, hiragana, katakana,
    /// CJK extension A, CJK compatibility ideograph, and fullwidth form blocks.
    /// These characters are emitted one per token (Rule 4b) instead of being
    /// grouped into a space-delimited word.
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

    // Reference image rendering

    /// <summary>
    /// Cleans text for use in a synthetic reference image:
    ///   • Replaces leading speaker-change dashes with 2 spaces (e.g. "- word" → "  word").
    ///   • Replaces non-ASCII symbol glyphs (♪ ♫ … etc.) with 2 spaces each so the
    ///     surrounding words keep approximately the same horizontal position as they have
    ///     in the source PGS image.  Non-ASCII letters and digits (CJK, accented Latin,
    ///     kana, …) are preserved.
    ///   • Collapses runs of 3+ spaces to exactly 2 so adjacent replacements don't
    ///     create a large gap.
    /// Returns null when nothing renderable remains (e.g. the text was only music notes).
    /// </summary>
    static string? CleanForReference(string text)
    {
        // Replace leading dashes (and any whitespace directly after them) with 2 spaces.
        text = Regex.Replace(text, @"^-+\s*", "  ");

        // Replace non-ASCII non-letter/digit glyphs with 2 spaces each.
        // char.IsAscii  → keep all ASCII (letters, digits, punctuation, spaces).
        // char.IsLetterOrDigit → keep non-ASCII letters (CJK, accented Latin, kana) and digits.
        var sb = new StringBuilder(text.Length + 8);
        foreach (char c in text)
        {
            if (char.IsAscii(c) || char.IsLetterOrDigit(c))
                sb.Append(c);
            else
                sb.Append("  "); // 2-space placeholder for removed glyph
        }

        // Collapse runs of 3+ spaces to 2 (adjacent replacements, or replacement next to
        // an existing space, produce at most a 2-space gap rather than an open chasm).
        string result = Regex.Replace(sb.ToString(), " {3,}", "  ");

        // Return null only if there is no non-whitespace content left.
        return result.Trim().Length > 0 ? result : null;
    }

    // HTTP helpers

    /// <summary>Single-image /api/generate call → trimmed response text, or null on failure.</summary>
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
            // Serialize with JsonOmitNull to suppress null "images" on the system message.
            string json = JsonSerializer.Serialize(payload, JsonOmitNull);

            // Use a fresh HttpClient per italic request to avoid Ollama KV-cache
            // contamination from previous requests on the same persistent connection.
            // curl opens a fresh TCP connection per invocation — this replicates that.
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


    // Response parsing

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

    /// <summary>
    /// Renders <paramref name="text"/> as a dark-on-white reference image sized to match
    /// <paramref name="sourcePng"/> and returns it as a PNG byte array.  Returns null on failure.
    ///
    /// The source was produced by <see cref="Preprocessor.Preprocess"/> +
    /// <see cref="Preprocessor.SplitLines"/>, which pads with:
    ///   padPx = max(14, round(0.15 × contentH));  lineH = contentH + 2 × padPx.
    /// We invert that formula to get <c>contentH</c> from <c>srcBmp.Height</c>, then scale the
    /// font so that the rendered ink fills exactly <c>contentH</c> pixels.  Rendering directly
    /// (without running through <see cref="Preprocessor.Preprocess"/> again) avoids the bucket
    /// quantization that would otherwise pin the output to 108 / 124 / 208 px regardless of
    /// font size, making it impossible to match small or sparse source images like 61 px lines.
    /// </summary>
    static byte[]? RenderAndPreprocess(string text, SKTypeface typeface, byte[]? sourcePng = null)
    {
        // Strip leading dashes and non-ASCII symbol glyphs before rendering.
        string? cleaned = CleanForReference(text);
        if (cleaned == null) return null;
        text = cleaned;

        // Invert SplitLines padding to get target (contentH, padPx)
        // For small lines (padPx = 14):  lineH = contentH + 28  →  contentH = lineH - 28.
        // For large lines (padPx > 14):  lineH ≈ 1.3 × contentH.
        int contentH = 80, padPx = 14; // defaults match BucketSmall
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

        // Scale font so ink height ≈ contentH
        float inkH42;
        using (var mf = new SKFont(typeface, 42f))
        {
            mf.MeasureText(text, out var b);
            inkH42 = b.Height > 0 ? b.Height : 30f;
        }

        float fontSize = 42f * (contentH / inkH42);

        // Render dark text on white directly
        // Bypassing Preprocessor.Preprocess keeps the output at exactly
        // (contentH + 2×padPx) pixels tall instead of a quantized bucket height.
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

        // Position baseline so ink top lands at padPx from the top of the canvas.
        // inkBounds.Top is negative (distance above baseline to the tallest glyph top).
        canvas.DrawText(text, padPx, padPx - inkBounds.Top, font, paint);
        canvas.Flush();

        // Gaussian blur to match PGS preprocessing
        // Matches Preprocessor.Options.BlurSigma so both images look similarly soft.
        const float blurSigma = 0.6f;
        using var blurredBmp = new SKBitmap(imgW, imgH, SKColorType.Bgra8888, SKAlphaType.Opaque);
        using var blurCanvas = new SKCanvas(blurredBmp);
        blurCanvas.Clear(SKColors.White);
        using var blurFilter = SKImageFilter.CreateBlur(blurSigma, blurSigma);
        using var blurPaint = new SKPaint();
        blurPaint.ImageFilter = blurFilter;
        blurCanvas.DrawBitmap(bmp, 0, 0, blurPaint);
        blurCanvas.Flush();

        // Convert Bgra8888 → Gray8
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


    /// <summary>
    /// Re-tokenize the working text, render a fresh
    /// reference image, then run two-image italic detection.
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

        // Skip italic detection when every letter-bearing token is a bracketed
        // stage direction (e.g. "[chuckles]", "[crowd gasps]").  These are never
        // rendered italic in practice, and skipping saves ~1.5 s per line.
        bool allBracketed = Array.TrueForAll(italicTokens, t =>
            !t.Text.Any(char.IsLetterOrDigit)
            || (t.Text.Length > 1 && t.Text[0] == '[' && t.Text[^1] == ']'));
        if (allBracketed)
        {
            if (debug ?? false)
                Console.WriteLine($"  [dbg] {debugPrefix}  step2-italic  skipped (all-bracketed)");
            return null;
        }

        // Generate reference image from OCR text to use in italic chat
        var typeface = OcrLanguageConfigs.GetReferenceTypeface(langConfig);
        byte[]? italicRef = RenderAndPreprocess(text, typeface, pngBytes);
        if (italicRef == null)
        {
            Console.WriteLine("  [warn] italic reference render failed; using plain text.");
            return null;
        }

        if (debug ?? false)
            File.WriteAllBytes(Path.Combine(debugDir!, $"{debugPrefix}_italic_ref.png"), italicRef);

        string[] italicTexts = Array.ConvertAll(italicTokens, t => t.Text);

        string italicSystemPrompt = langConfig.ItalicsPrompt;

        var (italicRaw, italicReqId) =
            RunItalicChat(italicTexts, pngBytes, italicRef, ollamaUrl, model, italicSystemPrompt);

        if (debug ?? false)
        {
            Console.WriteLine(
                $"  [dbg] {debugPrefix}  step2-italic  req={italicReqId}");
        }

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

    // Classification post-processing 
    /// <summary>
    /// If an ASCII punctuation token sits between two italic tokens and is not
    /// sentence-ending (i.e. does not consist solely of '.', '!', '?'), promote
    /// it to italic so tags don't split across e.g. "word, word" runs.
    /// Non-ASCII symbols (♪ etc.) are intentionally excluded.
    /// </summary>
    static string[] PostProcessClassification(
        string[] classification, Tokens[] tokens)
    {
        var cls = (string[])classification.Clone();

        for (int i = 1; i < tokens.Length - 1; i++)
        {
            if (cls[i].Equals("I", StringComparison.OrdinalIgnoreCase)) continue;

            string tok = tokens[i].Text;

            // Only ASCII punctuation (excludes words, brackets, ♪ etc.).
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

            // Skip sentence-ending tokens (., !, ?, or combinations thereof).
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

            // Promote if both neighbors are italic.
            if (cls[i - 1].Equals("I", StringComparison.OrdinalIgnoreCase) &&
                cls[i + 1].Equals("I", StringComparison.OrdinalIgnoreCase))
                cls[i] = "I";
        }

        return cls;
    }

    /// <summary>
    /// Expands each italic span outward through adjacent ASCII punctuation tokens,
    /// provided the token on the far side of the punctuation is not a roman word token.
    ///
    /// This promotes leading/trailing punctuation (including sentence-ending tokens
    /// such as <c>...</c>) when the whole sentence content is italic.
    ///
    /// Example:  [Name] ♪  <b>...<i>hello world</i></b>  ♪
    ///                      ↑ becomes italic because ♪ (non-word) is the outer neighbor.
    ///
    /// A "roman word token" — one that blocks expansion — is any non-italic token
    /// that contains at least one letter or digit (ordinary words, [brackets], etc.).
    /// Non-ASCII symbols (♪) and already-italic tokens never block expansion.
    ///
    /// Applied after <see cref="PostProcessClassification"/> (mid-sentence rule).
    /// </summary>
    static string[] ExpandItalicBoundaries(
        string[] classification, Tokens[] tokens)
    {
        var cls = (string[])classification.Clone();
        int n = tokens.Length;

        // Local helpers -------------------------------------------------------
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
        // ---------------------------------------------------------------------

        int i = 0;
        while (i < n)
        {
            if (!cls[i].Equals("I", StringComparison.OrdinalIgnoreCase))
            {
                i++;
                continue;
            }

            // Locate the end of this italic span.
            int spanEnd = i;
            while (spanEnd + 1 < n
                   && cls[spanEnd + 1].Equals("I", StringComparison.OrdinalIgnoreCase))
                spanEnd++;

            // Expand leftward past ASCII punctuation.
            int left = i - 1;
            while (left >= 0 && IsAsciiPunct(tokens[left].Text))
            {
                // Stop if the token further left is a roman word.
                if (left - 1 >= 0 && IsRomanWord(left - 1)) break;
                cls[left] = "I";
                left--;
            }

            // Expand rightward past ASCII punctuation.
            int right = spanEnd + 1;
            while (right < n && IsAsciiPunct(tokens[right].Text))
            {
                // Stop if the token further right is a roman word.
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
    /// all non-bracketed tokens (words, punctuation, symbols) to italic.
    ///
    /// This catches single stray roman words in an otherwise fully-italic line —
    /// e.g. a misclassified article ("a", "the") in dialogue that is entirely italic.
    ///
    /// Bracketed tags like [NAME] are excluded from the word count and never promoted.
    /// Applied after <see cref="ExpandItalicBoundaries"/>.
    /// </summary>
    static string[] PromoteNearlyAllItalic(
        string[] classification, Tokens[] tokens)
    {
        int wordCount = 0;
        int romanWords = 0;

        for (int i = 0; i < tokens.Length; i++)
        {
            string tok = tokens[i].Text;
            if (!tok.Any(char.IsLetterOrDigit)) continue; // punctuation / symbols — skip
            if (tok.Length > 1 && tok[0] == '[' && tok[^1] == ']') continue; // bracketed tag — skip
            wordCount++;
            if (!classification[i].Equals("I", StringComparison.OrdinalIgnoreCase))
                romanWords++;
        }

        // Need at least 2 word tokens, and exactly 1 must be non-italic.
        if (romanWords != 1 || wordCount < 2) return classification;

        var cls = (string[])classification.Clone();
        for (int i = 0; i < cls.Length; i++)
        {
            string tok = tokens[i].Text;
            if (tok.Length > 1 && tok[0] == '[' && tok[^1] == ']') continue; // leave bracketed tags alone
            cls[i] = "I";
        }

        return cls;
    }

    // Text reconstruction

    /// <summary>
    /// Reconstructs the original text with &lt;i&gt;…&lt;/i&gt; wrapping italic token runs.
    /// Whitespace and characters between tokens are preserved verbatim.
    /// Tokens classified "U" (uncertain) are treated as roman.
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

            // Preserve any whitespace/characters between the previous token and this one.
            // When closing an italic span, emit </i> BEFORE the inter-token gap so that
            // trailing whitespace lands outside the tag (e.g. "<i>word</i> next").
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

        // Append any trailing characters after the last token.
        if (pos < text.Length)
            sb.Append(text, pos, text.Length - pos);

        return sb.ToString();
    }

    // Public API

    /// <summary>
    /// Full pipeline for a single preprocessed subtitle line image:
    ///   1. Plain OCR → plain text, strip tags, normalize spacing.
    ///   2. Tokenize + render reference → OCR verification pass (corrects wrong tokens).
    ///   3. Reconstruct corrected text, re-render reference → italic detection pass.
    ///   4. Post-process classification + reconstruct with &lt;i&gt; tags.
    /// Returns null on unrecoverable failure.
    /// </summary>
    public static string? OcrImage(
        byte[] pngBytes,
        HttpClient http,
        string ollamaUrl,
        string model,
        Func<byte[], (string? Text, double Score)> paddleOcr,
        bool? debug = false,
        string? debugDir = null,
        string? debugPrefix = null,
        double paddleVerifyThreshold = 1.0,
        string language = "en"
    )
    {
        // Step 1: OCR
        // When PaddleOCR is configured:
        //   confidence >= threshold → use paddle result, skip verify.
        //   confidence <  threshold → discard paddle result, fall back to VLM
        //                             OCR (same as when paddle is not configured),
        //                             and also skip verify.
        bool paddleFallback = false;

        var langConfig = OcrLanguageConfigs.ForLanguage(language);
        string ocrPrompt = langConfig.Prompt;

        var (plainText, paddleScore) = paddleOcr(pngBytes);
        if (paddleScore < paddleVerifyThreshold)
        {
            paddleFallback = true;
            plainText = RunGenerate(ocrPrompt, pngBytes, http, ollamaUrl, model);
        }

        if (debug ?? false)
        {
            string scoreInfo = paddleFallback
                ? $"  confidence={paddleScore:F3} → vlm-fallback"
                : $"  confidence={paddleScore:F3}";

            Console.WriteLine($"  [dbg] {debugPrefix}  step1-ocr  {scoreInfo}");
        }

        if (string.IsNullOrWhiteSpace(plainText)) return null;


        // Step 2: Normalize / OCR Fixes
        plainText = OcrMidProcessor.Process(plainText, language);


        // Step 3: Italic detection
        // Re-tokenize the working text, render a fresh
        // reference image, then run two-image italic detection.
        var italicResult =
            GetItalicsClassification(plainText, pngBytes, debug, langConfig, ollamaUrl, model, debugDir, debugPrefix);

        if (italicResult == null) return plainText;

        var (italicTokens, detection) = italicResult.Value;

        // Step 4: Post-process + reconstruct
        string[] classification = PostProcessClassification(detection.Classification!, italicTokens);
        classification = ExpandItalicBoundaries(classification, italicTokens);
        classification = PromoteNearlyAllItalic(classification, italicTokens);
        return BuildStyledText(plainText, italicTokens, classification);
    }
}