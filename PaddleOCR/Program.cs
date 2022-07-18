namespace PaddleOCR;

using static SharpCV.Binding;
using static Tensorflow.Binding;

public class PaddleOCR {
    public static void Main(string[] args) {
        tf.enable_eager_execution();
        var flags = new Args() {
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
//        (dt_boxes, img_crop_list) = preprocess_boxes(dt_boxes);

//// text classifier
//        var angle_list;
//        if (flags.use_angle_cls) {
//            var text_classifier = new TextClassifier(flags);
//            (img_crop_list, angle_list) = text_classifier.Detect(img_crop_list);
//        }

//// text recognize
//        var text_recognizer = new TextRecognizer(flags);
//        var rec_res = text_recognizer.Recognize(img_crop_list);

//        var (_, filter_rec_res) = PostProcess(dt_boxes, rec_res);

//        foreach (var (text, score) in filter_rec_res) {
//            Console.WriteLine("{0}, {1:.3f}", new object[] { text, score });
//            Console.WriteLine("Finish!");
//        }
    }

    private static (object, IEnumerable<(object, object)>) PostProcess(object dtBoxes, object recRes) {
        throw new NotImplementedException();
    }

    private static object preprocess_boxes(object dtBoxes) {
        throw new NotImplementedException();
    }
}