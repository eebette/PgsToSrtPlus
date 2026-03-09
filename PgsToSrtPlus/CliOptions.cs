using System.CommandLine;

namespace PgsToSrtPlus;

record CliOptions(
    Argument<string> InputPath,
    Option<string> OllamaUrlOption,
    Option<string> LanguageOption,
    Option<int> TrackIndex,
    Option<string?> OutDir,
    Option<string> OllamaModelOption,
    Option<string> PaddlePythonOption,
    Option<string> PaddleModelOption,
    Option<string> DeviceOption,
    Option<double> PaddleAcceptanceThresholdOption,
    Option<double> ItalicThresholdOption,
    Option<bool> Debug,
    Option<string> DebugDir
);

static class CliOptionFactory
{
    public static CliOptions DefineCliOptions()
    {
        var inputPathArgument = new Argument<string>("input")
        {
            Description = "Path to .sup or .mkv file"
        };

        var ollamaUrlOption = new Option<string>("--ollama")
        {
            Description = "Ollama Endpoint URL",
            Required = false,
            DefaultValueFactory = _ => "http://127.0.0.1:11434"
        };

        var trackIndexOption = new Option<int>("--track")
        {
            Description = "Index of track to convert",
            Required = false,
            DefaultValueFactory = _ => -1
        };

        var outDirOption = new Option<string?>("--output", "-o")
        {
            Description = "Output directory or filename",
            Required = false
        };

        var debugOption = new Option<bool>("--debug")
        {
            Description = "Debug mode: saves extracted bitmaps and writes execution details to console",
            Required = false,
            DefaultValueFactory = _ => false
        };

        var debugDirOption = new Option<string>("--debug-dir")
        {
            Description = "Directory to write bitmaps when in debug mode",
            Required = false,
            DefaultValueFactory = _ => "debug"
        };

        var ollamaModelOption = new Option<string>("--model")
        {
            Description = "Ollama model for VLM fallback OCR on low-confidence lines",
            Required = false,
            DefaultValueFactory = _ => "qwen3-vl:32b-instruct"
        };

        var paddlePythonOption = new Option<string>("--paddle-python")
        {
            Description = "Python executable path with Paddle requirements installed",
            Required = false,
            DefaultValueFactory = _ => "python3"
        };

        var paddleModelOption = new Option<string>("--paddle-model")
        {
            Description = "Paddle model for first OCR pass on all lines",
            Required = false,
            DefaultValueFactory = _ => "PP-OCRv5_server_rec"
        };

        var deviceOption = new Option<string>("--device")
        {
            Description = "Device on which to run PaddleOCR",
            Required = false,
            DefaultValueFactory = _ => "cpu"
        };
        deviceOption.AcceptOnlyFromAmong("cpu", "gpu");

        var paddleAcceptanceThresholdOption = new Option<double>("--verify-threshold")
        {
            Description = "Paddle confidence lower limit at which to run OCR using Ollama VLM instead",
            Required = false,
            DefaultValueFactory = _ => 0.97
        };
        paddleAcceptanceThresholdOption.Validators.Add(result =>
        {
            var val = result.GetValueOrDefault<double>();
            if (val < 0.0 || val > 1.0)
                result.AddError("Value must be between 0.0 and 1.0");
        });

        var italicThresholdOption = new Option<double>("--italic-threshold")
        {
            Description = "Shear angle (degrees) above which text is classified as italic",
            Required = false,
            DefaultValueFactory = _ => 3.0
        };
        italicThresholdOption.Validators.Add(result =>
        {
            var val = result.GetValueOrDefault<double>();
            if (val < 0.0 || val > 45.0)
                result.AddError("Value must be between 0.0 and 45.0");
        });

        var languageOption = new Option<string>("--language", "-l")
        {
            Description = "Language of the subtitle track to select",
            Required = false,
            DefaultValueFactory = _ => "en"
        };
        languageOption.AcceptOnlyFromAmong("en", "eng", "ja", "jpn");

        return new CliOptions(
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
        );
    }
}