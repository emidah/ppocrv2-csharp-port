using System.Globalization;
using System.Text;
using PPOCRv2.CLS;
using PPOCRv2.Helpers;
using SharpCV;

namespace PPOCRv2;

using static Binding;
using static Tensorflow.Binding;

public class PPOCRv2 {
    private static Args flags;

    public static void Main(string[] args) {
        tf.enable_eager_execution();
        flags = new Args {
            cls_model_dir = "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            rec_model_dir = "./models/ch_PP-OCRv2_rec_infer.onnx",
            det_model_dir = "./models/ch_PP-OCRv2_det_infer.onnx",
            image_path = "./images/lite_demo.png"
        };
        var img = cv2.imread(flags.image_path).data;
        var oriIm = img.Copy();

        // text detect
        var textDetector = new TextDetector.TextDetector(flags);
        var dtBoxes = textDetector.Detect(img);
        (dtBoxes, var imgCropList) = PreProcessor.PreprocessBoxes(dtBoxes, oriIm);

        // text classifier
        if (flags.use_angle_cls) {
            var textClassifier = new TextClassifier(flags);
            (imgCropList, _) = textClassifier.Classify(imgCropList);
        }

        // text recognize
        var textRecognizer = new TextRecognizer.TextRecognizer(flags);
        var recRes = textRecognizer.Recognize(imgCropList.ToList());

        var (_, filterRecRes) = PostProcessor.PostProcess(flags, dtBoxes, recRes);
        Console.OutputEncoding = Encoding.UTF8;

        foreach (var (text, score) in filterRecRes) {
            Console.WriteLine("{0}, {1:.3f}", new object[] { text, score.ToString(CultureInfo.InvariantCulture) });
        }

        Console.WriteLine("Finish!");
    }
}