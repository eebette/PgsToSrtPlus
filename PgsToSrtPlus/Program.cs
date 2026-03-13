using System.CommandLine;
using System.Text;
using Nikse.SubtitleEdit.Core.BluRaySup;
using Nikse.SubtitleEdit.Core.ContainerFormats.Matroska;
using SkiaSharp;

namespace PgsToSrtPlus;

static class Program
{
    static int Main(string[] args)
    {
        var (
                inputPathArgument,
                ollamaUrlOption,
                languageOption,
                trackIndexOption,
                outDirOption,
                ollamaModelOption,
                paddlePythonOption,
                paddleModelOption,
                deviceOption,
                paddleAcceptanceThresholdOption,
                italicThresholdOption,
                debugOption,
                debugDirOption
                ) =
            CliOptionFactory.DefineCliOptions();

        var rootCommand = new RootCommand("PGS subtitle extractor and OCR tool")
        {
            inputPathArgument,
            ollamaUrlOption,
            languageOption,
            trackIndexOption,
            outDirOption,
            ollamaModelOption,
            paddlePythonOption,
            paddleModelOption,
            deviceOption,
            paddleAcceptanceThresholdOption,
            italicThresholdOption,
            debugOption,
            debugDirOption
        };

        rootCommand.Validators.Add(result =>
        {
            if (!Path.Exists(result.GetValue(inputPathArgument)))
                result.AddError("Invalid input path");
        });

        rootCommand.SetAction(parseResult =>
        {
            var inputPath = parseResult.GetValue(inputPathArgument)!;
            var trackIndex = parseResult.GetValue(trackIndexOption);
            var outDir = parseResult.GetValue(outDirOption) ?? Path.GetDirectoryName(inputPath)!;
            var debug = parseResult.GetValue(debugOption);
            string? debugDir = debug ? parseResult.GetValue(debugDirOption)! : null;
            var ollamaUrl = parseResult.GetValue(ollamaUrlOption)!;
            var ollamaModel = parseResult.GetValue(ollamaModelOption)!;
            var paddlePython = parseResult.GetValue(paddlePythonOption)!;
            var paddleModel = parseResult.GetValue(paddleModelOption)!;
            var device = parseResult.GetValue(deviceOption)!;
            var paddleAcceptanceThreshold = parseResult.GetValue(paddleAcceptanceThresholdOption);
            var italicThreshold = parseResult.GetValue(italicThresholdOption);
            var language = parseResult.GetValue(languageOption)!;

            Directory.CreateDirectory(outDir);

            if (debug && debugDir != null)
            {
                Directory.CreateDirectory(debugDir);
                Console.WriteLine($"Debug mode: saving intermediates to {debugDir}");
            }

            // PGS extraction / OCR
            List<BluRaySupParser.PcsData> sets = LoadFromMkv(inputPath, trackIndex, language);
            DisplaySet[] displaySets = CollectDisplaySets(sets);
            Console.WriteLine($"{displaySets.Length} subtitle entries.");

            return Ocr(
                displaySets,
                outDir,
                debugDir,
                ollamaUrl,
                ollamaModel,
                inputPath,
                paddlePython,
                paddleModel,
                device,
                paddleAcceptanceThreshold,
                italicThreshold,
                language
            );
        });

        return rootCommand.Parse(args).Invoke();
    }

    // Display set collection

    /// <summary>
    /// Groups PCS display sets into subtitle entries with start/end timestamps.
    /// Each display set's end time is inferred from the next set's start time,
    /// or a 5-second fallback for the final entry.
    /// </summary>
    static DisplaySet[] CollectDisplaySets(
        List<BluRaySupParser.PcsData> displaySets)
    {
        var result = new List<DisplaySet>();
        long? previousStart = null;
        BluRaySupParser.PcsData? previousDisplaySet = null;
        int visible = 0, clear = 0, mergedRle = 0, mergedPixel = 0;

        foreach (var currentDisplaySet in displaySets)
        {
            bool currentIsDisplay = currentDisplaySet.PcsObjects.Count > 0 && currentDisplaySet.BitmapObjects.Count > 0;
            long currentStartTimeInMs = currentDisplaySet.StartTime / 90;

            if (currentIsDisplay)
            {
                visible++;
                if (previousDisplaySet != null)
                {
                    var matchKind = BitmapMatchKind(previousDisplaySet, currentDisplaySet);
                    if (matchKind != MatchKind.None)
                    {
                        if (matchKind == MatchKind.ExactRle) mergedRle++;
                        else mergedPixel++;
                        continue; // Same bitmap — extend previous entry's duration
                    }

                    long previousEndTimeInMs = previousDisplaySet.EndTime / 90;
                    result.Add(new DisplaySet(previousDisplaySet, previousStart!.Value, previousEndTimeInMs));
                }
                previousStart = currentStartTimeInMs;
                previousDisplaySet = currentDisplaySet;
            }
            else
            {
                clear++;
                if (previousDisplaySet != null)
                {
                    result.Add(new DisplaySet(previousDisplaySet, previousStart!.Value, currentStartTimeInMs));
                    previousDisplaySet = null;
                    previousStart = null;
                }
            }
        }

        // Final display set rendering — use PGS EndTime, fall back to 5000 ms
        if (previousDisplaySet != null)
        {
            long endMs = previousDisplaySet.EndTime > 0
                ? previousDisplaySet.EndTime / 90
                : previousStart!.Value + 5000;
            result.Add(new DisplaySet(previousDisplaySet, previousStart!.Value, endMs));
        }

        // Show last PCS timestamp so we can verify the parser read the full file.
        if (displaySets.Count > 0)
        {
            long lastMs = displaySets[^1].StartTime / 90;
            var lastTs = TimeSpan.FromMilliseconds(lastMs);
            Console.WriteLine(
                $"  PCS stats: {visible} visible, {clear} clear  |  " +
                $"merged: {mergedRle} exact-RLE + {mergedPixel} pixel  |  " +
                $"last PCS @ {lastTs:hh\\:mm\\:ss}");
        }

        return result.ToArray();
    }

    enum MatchKind { None, ExactRle, Pixel }

    /// <summary>
    /// Returns the kind of bitmap match between two PCS display sets, or None.
    /// Fast path: exact RLE match (same dimensions and byte-identical data).
    /// Slow path: for bitmaps within ±2px dimensions, decodes both and compares
    /// alpha channels pixel-by-pixel at the common resolution, requiring ≥99.5%
    /// of pixels to match within tolerance.
    /// </summary>
    static MatchKind BitmapMatchKind(BluRaySupParser.PcsData a, BluRaySupParser.PcsData b)
    {
        if (a.PcsObjects.Count != b.PcsObjects.Count || a.PcsObjects.Count == 0)
            return MatchKind.None;

        var aObj = a.PcsObjects[0];
        var bObj = b.PcsObjects[0];

        if (aObj.Origin.X != bObj.Origin.X || aObj.Origin.Y != bObj.Origin.Y)
            return MatchKind.None;

        if (aObj.ObjectId >= a.BitmapObjects.Count || bObj.ObjectId >= b.BitmapObjects.Count)
            return MatchKind.None;

        var aBmps = a.BitmapObjects[aObj.ObjectId];
        var bBmps = b.BitmapObjects[bObj.ObjectId];

        if (aBmps.Count == 0 || bBmps.Count == 0)
            return MatchKind.None;

        var aOds = aBmps[0];
        var bOds = bBmps[0];

        // Fast path: exact RLE match
        if (aOds.Size.Width == bOds.Size.Width && aOds.Size.Height == bOds.Size.Height &&
            aOds.Fragment.ImageBuffer.AsSpan().SequenceEqual(bOds.Fragment.ImageBuffer.AsSpan()))
            return MatchKind.ExactRle;

        // Dimensions must be within ±2px for fuzzy comparison
        if (Math.Abs(aOds.Size.Width - bOds.Size.Width) > 2 ||
            Math.Abs(aOds.Size.Height - bOds.Size.Height) > 2)
            return MatchKind.None;

        // Slow path: decode both and compare alpha channels at full resolution
        using var aBmp = PgsDecoder.DecodePgsImage(a);
        using var bBmp = PgsDecoder.DecodePgsImage(b);
        if (aBmp == null || bBmp == null)
            return MatchKind.None;

        return AlphaMatch(aBmp, bBmp) ? MatchKind.Pixel : MatchKind.None;
    }

    /// <summary>
    /// Compares two decoded BGRA bitmaps by their alpha channels at the common
    /// (minimum) resolution. Returns true when ≥99.5% of pixels match within
    /// a tolerance of ±16. This catches re-rasterized duplicates that differ
    /// by ±1–2px in dimensions while rejecting genuinely different subtitles.
    /// </summary>
    static unsafe bool AlphaMatch(SKBitmap a, SKBitmap b)
    {
        int w = Math.Min(a.Width, b.Width);
        int h = Math.Min(a.Height, b.Height);
        if (w == 0 || h == 0) return false;

        int bppA = a.BytesPerPixel, bppB = b.BytesPerPixel;
        int strideA = a.RowBytes, strideB = b.RowBytes;
        byte* pA = (byte*)a.GetPixels(), pB = (byte*)b.GetPixels();

        // Alpha is byte 3 in both BGRA and RGBA
        const int alphaOff = 3;
        const int tolerance = 16;

        int mismatches = 0;
        int total = w * h;

        for (int y = 0; y < h; y++)
        {
            byte* rowA = pA + y * strideA;
            byte* rowB = pB + y * strideB;
            for (int x = 0; x < w; x++)
            {
                int diff = rowA[x * bppA + alphaOff] - rowB[x * bppB + alphaOff];
                if (diff > tolerance || diff < -tolerance)
                    mismatches++;
            }
        }

        return mismatches <= total * 0.005; // 99.5% match
    }

    // Decode + preprocess

    /// <summary>
    /// Decodes one PCS display set and runs the preprocessing pipeline.
    /// Returns the Gray8 bitmap (caller must dispose) plus raw-image metrics,
    /// or (null, ...) if the display set contains no renderable content.
    /// Line splitting is intentionally NOT done here — callers do it explicitly.
    /// </summary>
    static (SKBitmap? processed, int originX, int originY, int rawW, int rawH)
        DecodeAndPreprocess(BluRaySupParser.PcsData ds, int index, string? debugDir)
    {
        using var rawBmp = PgsDecoder.DecodePgsImage(ds);
        if (rawBmp == null || rawBmp.Width == 0)
            return (null, 0, 0, 0, 0);

        int originX = ds.PcsObjects.Count > 0 ? ds.PcsObjects[0].Origin.X : 0;
        int originY = ds.PcsObjects.Count > 0 ? ds.PcsObjects[0].Origin.Y : 0;
        int rawW = rawBmp.Width;
        int rawH = rawBmp.Height;

        var processed = Preprocessor.Preprocess(rawBmp, new Preprocessor.Options
        {
            DebugDir = debugDir,
            DebugPrefix = $"{index:D5}"
        });

        return (processed, originX, originY, rawW, rawH);
    }

    static List<byte[]> EncodeBitmapsToPngs(List<(SKBitmap Bmp, int Y0, int Y1)> bitmaps, int index, string? debugDir)
    {
        var linePngs = new List<byte[]>(bitmaps.Count);

        for (int li = 0; li < bitmaps.Count; li++)
        {
            var (lineBmp, _, _) = bitmaps[li];
            using var encoded = lineBmp.Encode(SKEncodedImageFormat.Png, 100);
            var pngBytes = encoded.ToArray();
            linePngs.Add(pngBytes);

            if (debugDir != null)
                File.WriteAllBytes(
                    Path.Combine(debugDir, $"{index:D5}_L{li}_ocr_input.png"),
                    pngBytes);

            lineBmp.Dispose();
        }

        return linePngs;
    }

    static List<SubtitleData> PrepareSubtitleData(
        DisplaySet[] displaySets, string? debugDir)

    {
        Console.WriteLine("Preprocessing and splitting lines…");

        var subtitleData = new List<SubtitleData>(displaySets.Length);

        for (int i = 0; i < displaySets.Length; i++)
        {
            var (ds, startMs, endMs) = displaySets[i];
            int index = i + 1;

            var (processed, _, _, _, _) = DecodeAndPreprocess(ds, index, debugDir);
            if (processed == null) continue;

            using (processed)
            {
                // Split the preprocessed bitmap into individual line images.
                var splitLines = Preprocessor.SplitLines(processed);
                var linePngs = EncodeBitmapsToPngs(splitLines, index, debugDir);

                if (linePngs.Count > 0)
                    subtitleData.Add(new SubtitleData(index, startMs, endMs, linePngs));
            }
        }

        return subtitleData;
    }

    static SrtEntry[] DataToSrtEntries(
        List<SubtitleData> subtitleData,
        string ollamaUrl,
        string model,
        PaddleOcrWorker paddle,
        double paddleVerifyThreshold,
        string language,
        string? debugDir
    )
    {
        var srtEntries = new SrtEntry[subtitleData.Count];

        using var http = new HttpClient();
        http.Timeout = TimeSpan.FromSeconds(20);

        for (int i = 0; i < subtitleData.Count; i++)
        {
            var (idx, startMs, endMs, linePngs) = subtitleData[i];
            var lineTexts = new List<string>(linePngs.Count);
            var vlmLines = new bool[linePngs.Count];

            for (int li = 0; li < linePngs.Count; li++)
            {
                var (text, vlm) =
                    OcrEngine.OcrImage(
                        linePngs[li],
                        http,
                        ollamaUrl,
                        model,
                        paddle,
                        debugDir != null,
                        debugDir,
                        debugDir != null ? $"{idx:D5}_L{li}" : null,
                        paddleVerifyThreshold, language
                    );
                lineTexts.Add(text ?? "");
                vlmLines[li] = vlm;
            }

            string joined = string.Join("\n", lineTexts);
            srtEntries[i] = new SrtEntry(idx, startMs, endMs, joined, vlmLines);
            Console.WriteLine($"  [{i + 1}/{subtitleData.Count}] #{idx}  {joined}");
        }

        return srtEntries;
    }


    // OCR (preprocess → split lines → OCR → SRT)
    static int Ocr(
        DisplaySet[] displaySets,
        string outDir,
        string? debugDir,
        string ollamaUrl,
        string model,
        string inputPath,
        string paddlePython,
        string paddleModel = "PP-OCRv5_server_rec",
        string paddleDevice = "gpu",
        double paddleVerifyThreshold = 1.0,
        double italicThreshold = 3.0,
        string language = "en")
    {
        // Step 1: decode, preprocess, and split every subtitle into line images.
        var subtitleData = PrepareSubtitleData(displaySets, debugDir);

        // Step 2: Start PaddleOCR worker
        string script = PaddleOcrWorker.FindScript();

        PaddleOcrWorker paddle = PaddleOcrWorker.Start(
            paddlePython, script, paddleModel, paddleDevice, italicThreshold, debugDir);

        using var _ = paddle; // disposed at end of method

        Console.WriteLine(
            $"OCR-ing {subtitleData.Count} subtitles ({subtitleData.Sum(s => s.LinePngs.Count)} line images) " +
            $"via PaddleOCR  model={paddleModel}  device={paddleDevice}");

        // Step 3: Run OCR engine on all subtitle data
        var srtEntries =
            DataToSrtEntries(
                subtitleData,
                ollamaUrl,
                model,
                paddle,
                paddleVerifyThreshold,
                language,
                debugDir
            );

        // Step 4: file-level post-processing.
        srtEntries = SrtPostProcessor.Process(srtEntries, language);

        // Step 5: write SRT ordered by original subtitle index.
        string srtName = Path.GetFileNameWithoutExtension(inputPath) + $".{language}.srt";
        string srtPath = Path.Combine(outDir, srtName);
        WriteSrt(srtPath, srtEntries);
        Console.WriteLine($"SRT written → {srtPath}");
        return 0;
    }

    // ── SRT writer ─────────────────────────────────────────────────────────────

    static void WriteSrt(
        string path,
        SrtEntry[] entries)
    {
        int srtIndex = 0;
        using var w = new StreamWriter(
            path,
            append: false,
            encoding: new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)
        );

        foreach (var e in entries.OrderBy(e => e.Index))
        {
            if (string.IsNullOrWhiteSpace(e.Text))
                continue;

            srtIndex++;
            w.WriteLine(srtIndex);
            w.WriteLine($"{SrtTime(e.StartMs)} --> {SrtTime(e.EndMs)}");
            w.WriteLine(e.Text);
            w.WriteLine();
        }
    }

    static string SrtTime(long ms)
    {
        var t = TimeSpan.FromMilliseconds(ms);
        return $"{(int)t.TotalHours:D2}:{t.Minutes:D2}:{t.Seconds:D2},{t.Milliseconds:D3}";
    }

    /// <summary>
    /// Map a language code (ISO 639-1 or 639-2) to a set of aliases so we can
    /// match either form against MKV track metadata.
    /// </summary>
    static HashSet<string> LangAliases(string code)
    {
        // Small lookup covering the languages most likely to appear in MKV PGS tracks.
        var map = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["en"] = ["en", "eng"],
            ["eng"] = ["en", "eng"],
            ["ja"] = ["ja", "jpn"],
            ["jpn"] = ["ja", "jpn"]
            // ["zh"] = ["zh", "zho", "chi"],
            // ["zho"] = ["zh", "zho", "chi"],
            // ["chi"] = ["zh", "zho", "chi"],
            // ["ko"] = ["ko", "kor"],
            // ["kor"] = ["ko", "kor"],
            // ["fr"] = ["fr", "fre", "fra"],
            // ["fre"] = ["fr", "fre", "fra"],
            // ["fra"] = ["fr", "fre", "fra"],
            // ["de"] = ["de", "deu", "ger"],
            // ["deu"] = ["de", "deu", "ger"],
            // ["ger"] = ["de", "deu", "ger"],
            // ["es"] = ["es", "spa"],
            // ["spa"] = ["es", "spa"],
            // ["pt"] = ["pt", "por"],
            // ["por"] = ["pt", "por"],
            // ["it"] = ["it", "ita"],
            // ["ita"] = ["it", "ita"],
            // ["ru"] = ["ru", "rus"],
            // ["rus"] = ["ru", "rus"],
            // ["ar"] = ["ar", "ara"],
            // ["ara"] = ["ar", "ara"],
        };

        if (map.TryGetValue(code, out var aliases))
            return new HashSet<string>(aliases, StringComparer.OrdinalIgnoreCase);

        // Unknown code — just match literally.
        return new HashSet<string>(StringComparer.OrdinalIgnoreCase) { code };
    }

    // MKV loaders
    static List<BluRaySupParser.PcsData> LoadFromMkv(string mkvPath, int trackIndex = -1, string language = "en")
    {
        using var mkv = new MatroskaFile(mkvPath);
        if (!mkv.IsValid)
            throw new Exception($"Not a valid Matroska file: {mkvPath}");

        var tracks = mkv.GetTracks(true);
        var pgsTracks = tracks.Where(t => t.CodecId == "S_HDMV/PGS").ToList();

        if (pgsTracks.Count == 0)
            throw new Exception("No PGS subtitle tracks found in MKV.");

        if (trackIndex >= 0)
        {
            // Explicit --track N
            if (trackIndex >= pgsTracks.Count)
                throw new Exception(
                    $"Track index {trackIndex} out of range (found {pgsTracks.Count} PGS track(s)).");
        }
        else
        {
            // Auto-select first PGS track matching the target language.
            // MKV tracks typically use ISO 639-2 (3-letter) codes like "eng", "jpn",
            // but --language uses short codes like "en", "ja".  Match either form.
            var langAliases = LangAliases(language);
            trackIndex = pgsTracks.FindIndex(t =>
                t.Language != null && langAliases.Contains(t.Language));
            if (trackIndex < 0)
            {
                Console.WriteLine(
                    $"  [warn] no PGS track with language \"{language}\"; falling back to first PGS track.");
                trackIndex = 0;
            }
        }

        var track = pgsTracks[trackIndex];
        Console.WriteLine(
            $"Loading PGS track {track.TrackNumber} ({track.Language ?? "unknown"}) " +
            $"from {Path.GetFileName(mkvPath)}…");

        var result = BluRaySupParser.ParseBluRaySupFromMatroska(track, mkv);
        Console.WriteLine($"Parsed {result.Count} display sets.");
        return result;
    }
}

// ── Record types ───────────────────────────────────────────────────────────────

record DisplaySet(BluRaySupParser.PcsData Ds, long StartMs, long EndMs);

record SrtEntry(int Index, long StartMs, long EndMs, string Text, bool[]? VlmLines = null);

record SubtitleData(int Index, long StartMs, long EndMs, List<byte[]> LinePngs);