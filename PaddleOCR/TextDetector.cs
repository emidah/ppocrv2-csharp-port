
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tensorflow;
using Tensorflow.NumPy;
using Tensor = Tensorflow.Tensor;

namespace PaddleOCR; 

public class TextDetector {
    private readonly Args args;
    private readonly bool use_paddle_predict;
    private readonly DBPreProcess preprocess_op;
    private readonly DBPostProcess postprocess_op;
    private readonly InferenceSession predictor;

    public TextDetector(Args args) {
        this.args = args;
        this.use_paddle_predict = args.use_paddle_predict;

        this.preprocess_op = new DBPreProcess(args);
        this.postprocess_op = new DBPostProcess(
            thresh: args.det_db_thresh,
            box_thresh: args.det_db_box_thresh,
            max_candidates: 1000,
            unclip_ratio: args.det_db_unclip_ratio,
            use_dilation: args.use_dilation,
            score_mode: args.det_db_score_mode);

        var model_dir = args.det_model_dir;
        //if (args.use_paddle_predict:
        //model = paddle.jit.load(model_dir + "/inference")
        //model.eval()
        //this.predictor = model
        //else:
         
        //import onnxruntime as ort
        var model_file_path = model_dir;
        var sess = new InferenceSession(model_file_path);
        this.predictor = sess;
    }

    public NDArray Detect(NDArray img) {
        var ori_im = img.Copy();
        var data1 = new Dictionary<string, NDArray>() { { "image", img } };

        var data = this.preprocess_op.PreProcess(data1);
        (img, var shape_list) = (data[0], data[1]);
        if (img == null) {
            return null; //, 0;
        }

        img = np.expand_dims(img, axis: 0);
        shape_list = np.expand_dims(shape_list, axis: 0);
        img = img.Copy();

        //if (this.use_paddle_predict)
        //output = self.predictor(img).numpy()
        //outputs = []
        //outputs.append(output)
        //else:
        var mem = new Memory<float>(img.ToArray<float>());
        var inputTensor = new DenseTensor<float>(mem, img.shape.as_int_list());
        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(this.predictor.InputMetadata.Keys.First(), inputTensor) };
        var outputs = this.predictor.Run(input).ToList();

        var preds = new Dictionary<string, NDArray>();
        var tensor = outputs[0].AsTensor<float>();
        preds["maps"] = new NDArray(tensor.ToArray(), new Shape(tensor.Dimensions.ToArray()));
        var post_result = this.postprocess_op.PostProcess(preds, shape_list);
        var dt_boxes = post_result[0].points;
        dt_boxes = this.filter_tag_det_res(dt_boxes, ori_im.shape);
        return dt_boxes;
    }

    private NDArray filter_tag_det_res(NDArray dt_boxes, Shape shape) {
        var (img_height, img_width) = (shape[0], shape[1]);
        var dt_boxes_new = new List<NDArray>();
        foreach (var dBox in dt_boxes) {
            var box = dBox;
            box = this.order_points_clockwise(box);
            box = this.clip_det_res(box, img_height, img_width);
            var rect_width = (int)np.linalg.norm(box[0] - box[1]);
            var rect_height = (int)np.linalg.norm(box[0] - box[3]);
            if (rect_width <= 3 || rect_height <= 3) {
                continue;
            }

            dt_boxes_new.Add(box);
        }

        dt_boxes = NDArrayExtensions.FromArray(dt_boxes_new.ToArray());
        return dt_boxes;
    }

    private NDArray clip_det_res(NDArray points, long img_height, long img_width) {
        for (int pno = 0; pno < (int)points.shape[0]; pno++) {
            points[pno, 0] = Math.Min(Math.Max((float)points[pno, 0], 0), img_width - 1);
            points[pno, 1] = Math.Min(Math.Max((float)points[pno, 1], 0), img_height - 1);
        }
        return points;
    }

    private NDArray order_points_clockwise(NDArray pts) {
        //"""
        //reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        //# sort the points based on their x-coordinates
        //"""
        var xSorted = pts[np.argsort(pts[new Slice(":"), 0])];

        //# grab the left-most and right-most points from the sorted
        //# x-roodinate points
        var leftMost = xSorted[Slice.ParseSlices(":2, :")];
        var rightMost = xSorted[Slice.ParseSlices("2:, :")];

        //# now, sort the left-most coordinates according to their
        //# y-coordinates so we can grab the top-left and bottom-left
        //# points, respectively
        leftMost = leftMost[np.argsort(leftMost[Slice.ParseSlices(":, 1")])];
        var (tl, bl) = (leftMost[0], leftMost[1]);

        rightMost = rightMost[np.argsort(rightMost[Slice.ParseSlices(":, 1")])];
        var (tr, br) = (rightMost[0], rightMost[1]);

        var rect = NDArrayExtensions.FromArray(new[] { tl, tr, br, bl }).astype(TF_DataType.TF_FLOAT);
        return rect;
    }
}