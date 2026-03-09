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

        foreach (var currentDisplaySet in displaySets)
        {
            bool currentIsDisplay = currentDisplaySet.PcsObjects.Count > 0 && currentDisplaySet.BitmapObjects.Count > 0;
            long currentStartTimeInMs = currentDisplaySet.StartTime / 90;

            // If this display set is visually rendered, add previous display set to return collection with this 
            // display set's start time as its end time if previous display set is not null, then set this display set
            // as the previous display set for next iteration 
            if (currentIsDisplay)
            {
                if (previousDisplaySet != null)
                    result.Add(new DisplaySet(previousDisplaySet, previousStart!.Value, currentStartTimeInMs));
                previousStart = currentStartTimeInMs;
                previousDisplaySet = currentDisplaySet;
            }
            // If this display set is not visually rendered, add previous display set to return collection with this 
            // display set's start time as its end time if previous display set is not null, then set the previous
            // display set as null for next iteration
            else if (previousDisplaySet != null)
            {
                result.Add(new DisplaySet(previousDisplaySet, previousStart!.Value, currentStartTimeInMs));
                previousDisplaySet = null;
                previousStart = null;
            }
        }

        // Final display set rendering (5000 ms display time)
        if (previousDisplaySet != null)
            result.Add(new DisplaySet(previousDisplaySet, previousStart!.Value, previousStart.Value + 5000));

        return result.ToArray();
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

            for (int li = 0; li < linePngs.Count; li++)
            {
                var text =
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
            }

            string joined = string.Join("\n", lineTexts);
            srtEntries[i] = new SrtEntry(idx, startMs, endMs, joined);
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

record SrtEntry(int Index, long StartMs, long EndMs, string Text);

record SubtitleData(int Index, long StartMs, long EndMs, List<byte[]> LinePngs);