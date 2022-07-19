using System.Globalization;
using System.Runtime.CompilerServices;
using SharpCV;
using Tensorflow.NumPy;

namespace PaddleOCR;

using static Binding;
using static Tensorflow.Binding;

public class PaddleOCR {
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
        var ori_im = img.Copy();

// text detect
        var text_detector = new TextDetector(flags);
        var dt_boxes = text_detector.Detect(img);
        var a = 0;
        (dt_boxes, IList<NDArray> img_crop_list) = PreProcessor.preprocess_boxes(dt_boxes, ori_im);

        //// text classifier
        if (flags.use_angle_cls) {
            var text_classifier = new TextClassifier(flags);
            (img_crop_list, _) = text_classifier.Classify(img_crop_list);
        }

        //// text recognize
        var text_recognizer = new TextRecognizer(flags);
        var rec_res = text_recognizer.Recognize(img_crop_list.ToList());

        var (_, filter_rec_res) = PostProcess(dt_boxes, rec_res);
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        foreach (var (text, score) in filter_rec_res) {
            Console.WriteLine("{0}, {1:.3f}", new object[] { text, score.ToString(CultureInfo.InvariantCulture) });
        }
        Console.WriteLine("Finish!");
    }

    private static (NDArray dt_boxes, IList<(string, float)> rec_res) PostProcess(NDArray dt_boxes, IList<(string, float)> rec_res) {
        var (filter_boxes, filter_rec_res) = (new List<NDArray>(), new List<(string, float)>());
        foreach (var (box, rec_result) in zip(dt_boxes, rec_res)) {
            var (text, score) = rec_result;
            if (score >= flags.drop_score) {
                filter_boxes.append(box);
                filter_rec_res.append(rec_result);
            }
        }

        return (NDArrayExtensions.FromArray(filter_boxes.ToArray()), filter_rec_res);
    }
}