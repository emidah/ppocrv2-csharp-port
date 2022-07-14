using OpenCvSharp;

namespace PaddleOCR;

public class PaddleOCR {
    public static void Main(string[] args) {
        var flags = new Args();
        var img = Cv2.ImRead(flags.image_path).WrapWithNDArray();
        var ori_im = img.Copy();

// text detect
        var text_detector = new TextDetector(flags);
        var dt_boxes = text_detector.Detect(img);
        var (dt_boxes, img_crop_list) = preprocess_boxes(dt_boxes);

// text classifier
        var angle_list;
        if (flags.use_angle_cls) {
            var text_classifier = new TextClassifier(flags);
            (img_crop_list, angle_list) = text_classifier.Detect(img_crop_list);
        }

// text recognize
        var text_recognizer = new TextRecognizer(flags);
        var rec_res = text_recognizer.Recognize(img_crop_list);

        var (_, filter_rec_res) = PostProcess(dt_boxes, rec_res);

        foreach (var (text, score) in filter_rec_res) {
            Console.WriteLine("{0}, {1:.3f}", new object[] { text, score });
            Console.WriteLine("Finish!");
        }
    }

    private static (object, IEnumerable<(object, object)>) PostProcess(object dtBoxes, object recRes) {
        throw new NotImplementedException();
    }

    private static object preprocess_boxes(object dtBoxes) {
        throw new NotImplementedException();
    }
}