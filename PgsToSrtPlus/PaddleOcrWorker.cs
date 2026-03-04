using System.Diagnostics;
using System.Text.Json;

namespace PgsToSrtPlus;

/// <summary>
/// Manages a persistent PaddleOCR Python subprocess for text recognition.
///
/// The subprocess (paddle_ocr_bridge.py) loads the OCR model once at startup,
/// then handles per-image recognition requests over a newline-delimited JSON
/// stdin/stdout protocol — avoiding per-image process start overhead.
/// </summary>
sealed class PaddleOcrWorker : IDisposable
{
    readonly Process _proc;
    readonly Lock _lock = new();

    PaddleOcrWorker(Process proc) => _proc = proc;

    // ── Factory ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Starts the Python bridge subprocess and waits for it to signal ready.
    /// Returns null (with a logged error) if the process cannot start or the
    /// model fails to load.
    /// </summary>
    public static PaddleOcrWorker Start(
        string pythonPath,
        string scriptPath,
        string modelName = "PP-OCRv5_server_rec",
        string device = "gpu")
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
            // Send startup config as the first line.
            proc.StandardInput.WriteLine(
                JsonSerializer.Serialize(new { model_name = modelName, device }));

            // Wait for {"ready":true} or {"error":"..."}.
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

    // Recognition

    /// <summary>
    /// Writes <paramref name="pngBytes"/> to a temporary file, sends the path
    /// to the worker process, and returns the recognised text and confidence score.
    /// Thread-safe. Returns (null, 0) on any failure.
    /// </summary>
    public (string? Text, double Score) Recognize(byte[] pngBytes)
    {
        string tmp = Path.GetTempFileName() + ".png";
        try
        {
            File.WriteAllBytes(tmp, pngBytes);

            lock (_lock)
            {
                _proc.StandardInput.WriteLine(
                    JsonSerializer.Serialize(new { image = tmp }));

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
            Console.WriteLine($"  [warn] PaddleOCR recognize failed: {ex.Message}");
            return (null, 0);
        }
        finally
        {
            try
            {
                File.Delete(tmp);
            }
            catch
            {
                // ignored
            }
        }
    }

    // Script discovery

    /// <summary>
    /// Looks for paddle_ocr_bridge.py alongside the binary, in ancestor
    /// directories of the binary (to find the repo root), and in the current
    /// working directory. Returns null if not found.
    /// </summary>
    public static string FindScript(string name = "paddle_ocr_bridge.py")
    {
        // Check alongside the binary and in the working directory first.
        var candidates = new List<string>
        {
            Path.Combine(AppContext.BaseDirectory, name),
            Path.Combine(Directory.GetCurrentDirectory(), name)
        };

        // Walk up from the binary directory (e.g. bin/Debug/net9.0 → repo root).
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