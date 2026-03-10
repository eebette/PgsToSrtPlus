using System.Diagnostics;
using System.Text.Json;

namespace PgsToSrtPlus;

record ItalicAngleResult(double Angle, double PeakRatio, bool IsItalic);

/// <summary>A horizontal pixel span [Start, End) from projection-profile word segmentation.</summary>
record WordBoundary(int Start, int End);

/// <summary>
/// Manages a persistent PaddleOCR Python subprocess for text recognition,
/// segment detection, and shear-based italic detection.
///
/// The subprocess (paddle_ocr_bridge.py) loads the OCR models once at startup,
/// then handles per-image requests over a newline-delimited JSON stdin/stdout
/// protocol — avoiding per-image process start overhead.
/// </summary>
sealed class PaddleOcrWorker : IDisposable
{
    readonly Process _proc;
    readonly Lock _lock = new();

    PaddleOcrWorker(Process proc) => _proc = proc;

    // -- Factory ---------------------------------------------------------------

    /// <summary>
    /// Starts the Python bridge subprocess and waits for it to signal ready.
    /// </summary>
    public static PaddleOcrWorker Start(
        string pythonPath,
        string scriptPath,
        string modelName = "PP-OCRv5_server_rec",
        string device = "gpu",
        double italicThreshold = 3.0,
        string? debugDir = null)
    {
        Console.WriteLine(
            $"Starting PaddleOCR worker  model={modelName}  device={device}…");

        if (!File.Exists(scriptPath))
        {
            throw new FileNotFoundException("PaddleOCR script not found.", scriptPath);
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"\"{scriptPath}\"",
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            Environment =
            {
                ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            }
        };

        ConfigureGpu(psi.Environment, device);

        Process? proc = Process.Start(psi);

        if (proc == null)
        {
            throw new InvalidOperationException("Failed to start PaddleOCR process.");
        }

        proc.StandardInput.AutoFlush = true;

        try
        {
            proc.StandardInput.WriteLine(
                JsonSerializer.Serialize(new
                {
                    model_name = modelName,
                    device,
                    italic_threshold = italicThreshold,
                }));

            string? ready = proc.StandardOutput.ReadLine();
            if (ready == null)
            {
                proc.Kill();
                proc.Dispose();
                throw new InvalidOperationException("PaddleOCR process exited before signalling ready.");
            }

            using var doc = JsonDocument.Parse(ready);
            if (doc.RootElement.TryGetProperty("error", out var err))
            {
                proc.Kill();
                proc.Dispose();
                throw new InvalidOperationException($"PaddleOCR init error: {err.GetString()}");
            }

            Console.WriteLine("PaddleOCR worker ready.");

            return new PaddleOcrWorker(proc);
        }
        catch (Exception ex)
        {
            try
            {
                proc.Kill();
            }
            catch
            {
                // ignored
            }

            proc.Dispose();
            throw new InvalidOperationException($"PaddleOCR worker startup failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Sends an image to the recognition-only model (no detection).
    /// Used for VLM fallback path where detection is not needed.
    /// </summary>
    public (string? Text, double Score) RecognizeOnly(byte[] pngBytes)
    {
        string tmp = Path.GetTempFileName() + ".png";
        try
        {
            File.WriteAllBytes(tmp, pngBytes);

            lock (_lock)
            {
                _proc.StandardInput.WriteLine(
                    JsonSerializer.Serialize(new { image = tmp, recognize_only = true }));

                string? response = _proc.StandardOutput.ReadLine();
                if (response == null)
                {
                    Console.WriteLine("  [warn] PaddleOCR worker closed unexpectedly.");
                    return (null, 0);
                }

                using var doc = JsonDocument.Parse(response);
                if (doc.RootElement.TryGetProperty("error", out var err))
                {
                    Console.WriteLine($"  [warn] PaddleOCR error: {err.GetString()}");
                    return (null, 0);
                }

                string? text = doc.RootElement.TryGetProperty("text", out var textEl)
                    ? textEl.GetString()?.Trim()
                    : null;
                double score = doc.RootElement.TryGetProperty("score", out var scoreEl)
                    ? scoreEl.GetDouble()
                    : 0.0;
                return (text, score);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] PaddleOCR recognize-only failed: {ex.Message}");
            return (null, 0);
        }
        finally
        {
            try { File.Delete(tmp); } catch { /* ignored */ }
        }
    }

    // -- Italic angle detection ------------------------------------------------

    /// <summary>
    /// Sends a cropped image to the Python bridge for shear-projection-variance
    /// italic angle detection.
    /// </summary>
    public ItalicAngleResult DetectItalicAngle(byte[] pngBytes)
    {
        string tmp = Path.GetTempFileName() + ".png";
        try
        {
            File.WriteAllBytes(tmp, pngBytes);

            lock (_lock)
            {
                _proc.StandardInput.WriteLine(
                    JsonSerializer.Serialize(new { italic_detect = tmp }));

                string? response = _proc.StandardOutput.ReadLine();
                if (response == null)
                {
                    Console.WriteLine("  [warn] PaddleOCR worker closed unexpectedly.");
                    return new ItalicAngleResult(0, 0, false);
                }

                using var doc = JsonDocument.Parse(response);
                if (doc.RootElement.TryGetProperty("error", out var err))
                {
                    Console.WriteLine($"  [warn] Italic detection error: {err.GetString()}");
                    return new ItalicAngleResult(0, 0, false);
                }

                double angle = doc.RootElement.TryGetProperty("angle", out var angleEl)
                    ? angleEl.GetDouble()
                    : 0.0;
                double peakRatio = doc.RootElement.TryGetProperty("peak_ratio", out var ratioEl)
                    ? ratioEl.GetDouble()
                    : 0.0;
                bool isItalic = doc.RootElement.TryGetProperty("is_italic", out var italicEl)
                    && italicEl.GetBoolean();
                return new ItalicAngleResult(angle, peakRatio, isItalic);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] Italic detection failed: {ex.Message}");
            return new ItalicAngleResult(0, 0, false);
        }
        finally
        {
            try { File.Delete(tmp); } catch { /* ignored */ }
        }
    }

    // -- Projection-profile word segmentation -----------------------------------

    /// <summary>
    /// Sends an image to the Python bridge for projection-profile word
    /// segmentation.  Returns horizontal pixel spans [Start, End) for each
    /// contiguous run of ink columns.
    /// </summary>
    public WordBoundary[] WordSegment(byte[] pngBytes)
    {
        string tmp = Path.GetTempFileName() + ".png";
        try
        {
            File.WriteAllBytes(tmp, pngBytes);

            lock (_lock)
            {
                _proc.StandardInput.WriteLine(
                    JsonSerializer.Serialize(new { word_segment = tmp }));

                string? response = _proc.StandardOutput.ReadLine();
                if (response == null)
                {
                    Console.WriteLine("  [warn] PaddleOCR worker closed unexpectedly.");
                    return [];
                }

                using var doc = JsonDocument.Parse(response);
                if (doc.RootElement.TryGetProperty("error", out var err))
                {
                    Console.WriteLine($"  [warn] Word segment error: {err.GetString()}");
                    return [];
                }

                if (!doc.RootElement.TryGetProperty("words", out var wordsEl)
                    || wordsEl.ValueKind != JsonValueKind.Array)
                    return [];

                var words = new List<WordBoundary>();
                foreach (var w in wordsEl.EnumerateArray())
                {
                    var items = w.EnumerateArray().ToArray();
                    if (items.Length >= 2)
                        words.Add(new WordBoundary(items[0].GetInt32(), items[1].GetInt32()));
                }

                return words.ToArray();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] Word segment failed: {ex.Message}");
            return [];
        }
        finally
        {
            try { File.Delete(tmp); } catch { /* ignored */ }
        }
    }

    // Script discovery

    /// <summary>
    /// Looks for paddle_ocr_bridge.py alongside the binary, in ancestor
    /// directories of the binary (to find the repo root), and in the current
    /// working directory.
    /// </summary>
    public static string FindScript(string name = "paddle_ocr_bridge.py")
    {
        var candidates = new List<string>
        {
            Path.Combine(AppContext.BaseDirectory, name),
            Path.Combine(Directory.GetCurrentDirectory(), name)
        };

        var dir = Directory.GetParent(AppContext.BaseDirectory);
        while (dir != null)
        {
            candidates.Add(Path.Combine(dir.FullName, name));
            dir = dir.Parent;
        }

        var scriptFile = candidates.Find(File.Exists);

        if (scriptFile == null)
        {
            throw new InvalidOperationException(
                $"PaddleOCR script file not found. Searched:\n  {string.Join("\n  ", candidates)}");
        }

        return scriptFile;
    }

    // Cleanup

    public void Dispose()
    {
        try
        {
            _proc.StandardInput.WriteLine(JsonSerializer.Serialize(new { quit = true }));
            _proc.WaitForExit(3000);
        }
        catch
        {
            // ignored
        }
        finally
        {
            try
            {
                _proc.Kill(entireProcessTree: true);
            }
            catch
            {
                // ignored
            }

            _proc.Dispose();
        }
    }

    // Helpers

    static void ConfigureGpu(IDictionary<string, string?> env, string device)
    {
        if (device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase))
        {
            env["FLAGS_use_cuda"] = "1";
            env["CUDA_VISIBLE_DEVICES"] = device.Contains(':') ? device.Split(':')[1] : "0";
        }
        else
        {
            env["FLAGS_use_cuda"] = "0";
            env["CUDA_VISIBLE_DEVICES"] = "";
        }
    }
}
