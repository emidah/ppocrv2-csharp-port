using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PPOCRv2.Helpers;
using Tensorflow;
using Tensorflow.NumPy;

namespace PPOCRv2.TextDetector;

public class TextDetector {
    private readonly DbPostProcess postprocessOp;
    private readonly InferenceSession predictor;
    private readonly DbPreProcess preprocessOp;

    public TextDetector(Args args) {
        preprocessOp = new DbPreProcess(args);
        postprocessOp = new DbPostProcess(
            args.det_db_thresh,
            args.det_db_box_thresh,
            1000,
            args.det_db_unclip_ratio,
            args.use_dilation,
            args.det_db_score_mode);

        var modelDir = args.det_model_dir;
        //if (args.use_paddle_predict:
        //model = paddle.jit.load(model_dir + "/inference")
        //model.eval()
        //this.predictor = model
        //else:

        //import onnxruntime as ort
        var modelFilePath = modelDir;
        var sess = new InferenceSession(modelFilePath);
        predictor = sess;
    }

    public NDArray Detect(NDArray img) {
        var oriIm = img.Copy();
        var data1 = new Dictionary<string, NDArray> { { "image", img } };

        var data = preprocessOp.PreProcess(data1);
        (img, var shapeList) = (data[0], data[1]);
        if (img == null) {
            return null; //, 0;
        }

        img = np.expand_dims(img, 0);
        shapeList = np.expand_dims(shapeList, 0);
        img = img.Copy();

        var mem = new Memory<float>(img.ToArray<float>());
        var inputTensor = new DenseTensor<float>(mem, img.shape.as_int_list());
        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(predictor.InputMetadata.Keys.First(), inputTensor) };
        var outputs = predictor.Run(input).ToList();

        var preds = new Dictionary<string, NDArray>();
        var tensor = outputs[0].AsTensor<float>();
        preds["maps"] = new NDArray(tensor.ToArray(), new Shape(tensor.Dimensions.ToArray()));
        var postResult = postprocessOp.PostProcess(preds, shapeList);
        var dtBoxes = postResult[0].Points;
        dtBoxes = FilterTagDetRes(dtBoxes, oriIm.shape);
        return dtBoxes;
    }

    private NDArray FilterTagDetRes(NDArray dtBoxes, Shape shape) {
        var (imgHeight, imgWidth) = (shape[0], shape[1]);
        var dtBoxesNew = new List<NDArray>();
        foreach (var dBox in dtBoxes) {
            var box = dBox;
            box = order_points_clockwise(box);
            box = clip_det_res(box, imgHeight, imgWidth);
            var rectWidth = (int)np.linalg.norm(box[0] - box[1]);
            var rectHeight = (int)np.linalg.norm(box[0] - box[3]);
            if (rectWidth <= 3 || rectHeight <= 3) {
                continue;
            }

            dtBoxesNew.Add(box);
        }

        dtBoxes = NdArrayExtensions.FromArray(dtBoxesNew.ToArray());
        return dtBoxes;
    }

    private NDArray clip_det_res(NDArray points, long imgHeight, long imgWidth) {
        for (var pno = 0; pno < (int)points.shape[0]; pno++) {
            points[pno, 0] = Math.Min(Math.Max((float)points[pno, 0], 0), imgWidth - 1);
            points[pno, 1] = Math.Min(Math.Max((float)points[pno, 1], 0), imgHeight - 1);
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

        var rect = NdArrayExtensions.FromArray(new[] { tl, tr, br, bl }).astype(TF_DataType.TF_FLOAT);
        return rect;
    }
}