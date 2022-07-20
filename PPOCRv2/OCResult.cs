namespace PPOCRv2;

public class OCRResult {
    public OCRResult(int[,] detectionBox, string recognitionText, float recognitionScore, string sourceImagePath, int sourceImageHeight,
        int sourceImageWidth) {
        this.DetectionBox = detectionBox;
        this.RecognitionText = recognitionText;
        this.RecognitionScore = recognitionScore;
        this.SourceImagePath = sourceImagePath;
        this.SourceImageHeight = sourceImageHeight;
        this.SourceImageWidth = sourceImageWidth;
    }

    public int[,] DetectionBox { get; }

    public string RecognitionText { get; }

    public float RecognitionScore { get; }
    public string SourceImagePath { get; }
    public int SourceImageHeight { get; }
    public int SourceImageWidth { get; }
}