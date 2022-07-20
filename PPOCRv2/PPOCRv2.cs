using System.Globalization;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using PPOCRv2.AngleClassifier;
using PPOCRv2.Helpers;
using PPOCRv2.TextDetector;
using Tensorflow;
using Binding = SharpCV.Binding;

namespace PPOCRv2;

using static Binding;

public class PPOCRv2 {
    private readonly Args flags;

    public PPOCRv2(Args flags) {
        this.flags = flags;
    }

    public PPOCRv2(int maxSideLength = 1920, bool useAngleCls = true, float recognitionThreshold = 0.5f,
        bool useSpaceChar = true) {
        this.flags = new Args {
            ClsModelDir = "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            RecModelDir = "./models/ch_PP-OCRv2_rec_infer.onnx",
            DetModelDir = "./models/ch_PP-OCRv2_det_infer.onnx",
            DetLimitSideLen = maxSideLength,
            UseAngleCls = useAngleCls,
            DropScore = recognitionThreshold
        };
    }

    public List<OCRResult> Ocr(string imagePath) {
        this.flags.ImagePath = imagePath;
        var img = cv2.imread(this.flags.ImagePath).data;
        if (img.shape.Length == 0) {
            throw new Exception("No such image found");
        }

        var oriIm = img.Copy();

        // text detect
        var textDetector = new TextDetector.TextDetector(this.flags);
        var dtBoxes = textDetector.Detect(img);
        (dtBoxes, var imgCropList) = PreProcessor.PreprocessBoxes(dtBoxes, oriIm);

        // text classifier
        if (this.flags.UseAngleCls) {
            var textClassifier = new TextClassifier(this.flags);
            (imgCropList, _) = textClassifier.Classify(imgCropList);
        }

        // text recognize
        var textRecognizer = new TextRecognizer.TextRecognizer(this.flags);
        var recRes = textRecognizer.Recognize(imgCropList.ToList());

        (dtBoxes, var filterRecRes) = PostProcessor.PostProcess(this.flags, dtBoxes, recRes);

        var detections = dtBoxes.Select(box => (int[,]) box.astype(TF_DataType.TF_INT32).ToMultiDimArray<int>()).ToList();
        return detections.Select((det, i) => new OCRResult(
            det, 
            filterRecRes[i].Item1, 
            filterRecRes[i].Item2,
            this.flags.ImagePath,
            (int)img.shape[0], 
            (int)img.shape[1])
        ).ToList();
    }
}