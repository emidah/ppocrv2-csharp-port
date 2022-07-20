using System.CommandLine;
using System.Diagnostics;
using System.Text;

namespace PPOCRv2DemoApp;

public class Program {
    private static readonly HashSet<string> SupportedExtensions = new() {
        ".png", ".jpeg", ".jpg", ".jpg", ".jpe", ".bmp"
    };

    private const string LogLevel = "3";
    private const string TfCppMinLogLevel = "TF_CPP_MIN_LOG_LEVEL";

    public static int Main(string[] args) {
        Console.OutputEncoding = Encoding.UTF8;

        if (Environment.GetEnvironmentVariable(TfCppMinLogLevel) != LogLevel) {
            Environment.SetEnvironmentVariable(TfCppMinLogLevel, LogLevel, EnvironmentVariableTarget.Process);
            // This is a hack to set the environment variable before execution - otherwise it is not loaded correctly.
            var thisProcess = Process.GetCurrentProcess();
            var p = Process.Start(new ProcessStartInfo(thisProcess.MainModule!.FileName!));
            p!.WaitForExit();
            return p.ExitCode;
        }

        var fileOption = new Option<FileSystemInfo>(
            "--image",
            description: "Image path (jpg, png, bmp",
            getDefaultValue: () => new FileInfo("./images/lite_demo.png"));

        var maxSideLength = new Option<int>(
            "--max-side-len",
            description: "Maximum image side length used by the detector",
            getDefaultValue: () => 1920);

        var useAngleClassifier = new Option<bool>(
            "--use-angle-cls",
            description: "Use angle classifier between detection and recognition steps",
            getDefaultValue: () => true);

        var useSpaces = new Option<bool>(
            "--use-space-char",
            description: "Recognize space characters",
            getDefaultValue: () => true);

        var threshold = new Option<float>(
            "--rec-thresh",
            description: "Set recognition threshold",
            getDefaultValue: () => 0.5f);

        var rootCommand = new RootCommand("Sample app for System.CommandLine");
        rootCommand.AddOption(fileOption);
        rootCommand.AddOption(maxSideLength);
        rootCommand.AddOption(useAngleClassifier);
        rootCommand.AddOption(useSpaces);
        rootCommand.AddOption(threshold);

        rootCommand.SetHandler((file, side, cls, thres, useSpace) => {
                var files = new List<FileInfo>();
                if (file is FileInfo fi) {
                    files.Add(fi);
                } else if (file is DirectoryInfo d) {
                    files = d.EnumerateFiles()
                        .Where(f => SupportedExtensions.Contains(Path.GetExtension(f.Name).ToLowerInvariant())).ToList();
                    if (files.Count == 0) {
                        throw new ArgumentException("No image files were found in this directory");
                    }
                } else {
                    throw new FileNotFoundException("No such file or folder exists");
                }

                foreach (var fileInfo in files) {
                    var ppocr = new PPOCRv2.PPOCRv2(side, cls, thres, useSpace);
                    var res = ppocr.Ocr(fileInfo.FullName);
                    Console.WriteLine($"OCR for {Path.GetFileName(fileInfo.FullName)}");
                    foreach (var ocrResult in res) {
                        Console.WriteLine($"\"{ocrResult.RecognitionText}\" - {ocrResult.RecognitionScore:F2}");
                    }
                }
            },
            fileOption, maxSideLength, useAngleClassifier, threshold, useSpaces);

        return rootCommand.Invoke(args);
    }
}