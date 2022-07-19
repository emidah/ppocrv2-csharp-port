using System.Globalization;
using System.Text;
using SharpCV;
using Tensorflow.NumPy;

namespace PaddleOCR;

using static Binding;
using static Tensorflow.Binding;

public class PaddleOcr {
    private static Args flags;

    public static void Main(string[] args) {
        tf.enable_eager_execution();
        flags = new Args {
            cls_model_dir = "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            rec_model_dir = "./models/ch_PP-OCRv2_rec_infer.onnx",
            det_model_dir = "./models/ch_PP-OCRv2_det_infer.onnx",
            image_path = "./images/lite_demo.png",
            use_paddle_predict = false
        };
        var img = cv2.imread(flags.image_path).data;
        var oriIm = img.Copy();

// text detect
        var textDetector = new TextDetector(flags);
        var dtBoxes = textDetector.Detect(img);
        var a = 0;
        (dtBoxes, var imgCropList) = PreProcessor.PreprocessBoxes(dtBoxes, oriIm);

        //// text classifier
        if (flags.use_angle_cls) {
            var textClassifier = new TextClassifier(flags);
            (imgCropList, _) = textClassifier.Classify(imgCropList);
        }

        //// text recognize
        var textRecognizer = new TextRecognizer(flags);
        var recRes = textRecognizer.Recognize(imgCropList.ToList());

        var (_, filterRecRes) = PostProcess(dtBoxes, recRes);
        Console.OutputEncoding = Encoding.UTF8;

        foreach (var (text, score) in filterRecRes) {
            Console.WriteLine("{0}, {1:.3f}", new object[] { text, score.ToString(CultureInfo.InvariantCulture) });
        }

        Console.WriteLine("Finish!");
    }

    private static (NDArray dt_boxes, IList<(string, float)> rec_res) PostProcess(NDArray dtBoxes, IList<(string, float)> recRes) {
        var (filterBoxes, filterRecRes) = (new List<NDArray>(), new List<(string, float)>());
        foreach (var (box, recResult) in zip(dtBoxes, recRes)) {
            var (text, score) = recResult;
            if (score >= flags.drop_score) {
                filterBoxes.append(box);
                filterRecRes.append(recResult);
            }
        }

        return (NdArrayExtensions.FromArray(filterBoxes.ToArray()), filterRecRes);
    }
}