using System.Security.Cryptography.X509Certificates;
using OpenCvSharp;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;
using ContourApproximationModes = SharpCV.ContourApproximationModes;
using Mat = OpenCvSharp.Mat;
using MatType = OpenCvSharp.MatType;
using RetrievalModes = SharpCV.RetrievalModes;
using Scalar = OpenCvSharp.Scalar;
using ClipperLib;

namespace PaddleOCR;

public class DBPostProcess {
    private readonly NDArray dilation_kernel = null;
    private readonly string score_mode;
    private readonly float thresh;
    private readonly int max_candidates;
    private readonly float box_thresh;
    private readonly float unclip_ratio;
    private readonly int min_size;

    public DBPostProcess() {
        throw new NotImplementedException();
    }

    public DBPostProcess(float thresh, float box_thresh, int max_candidates, float unclip_ratio, bool use_dilation, string score_mode) {
        this.thresh = thresh;
        this.box_thresh = box_thresh;
        this.max_candidates = max_candidates;
        this.unclip_ratio = unclip_ratio;
        this.min_size = 3;
        this.score_mode = score_mode;
        //assert score_mode in [
        //"slow", "fast"
        //    ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)
        use_dilation = false;
        this.dilation_kernel = use_dilation ? np.array(new[] { new[] { 1, 1 }, new[] { 1, 1 } }) : null;
    }

    public (NDArray, float) get_mini_boxes(NDArray contour) {
        var bounding_box = cv2.minAreaRect(contour);

        var points = NDArrayExtensions.FromArray(cv2.BoxPoints(bounding_box).OrderBy(x => (int) x[0]).ToArray());

        var (index_1, index_2, index_3, index_4) = (0, 1, 2, 3);
        if (points[1][1] > points[0][1]) {
            index_1 = 0;
            index_4 = 1;
        } else {
            index_1 = 1;
            index_4 = 0;
        }

        if (points[3][1] > points[2][1]) {
            index_2 = 2;
            index_3 = 3;
        } else {
            index_2 = 3;
            index_3 = 2;
        }

        var box = NDArrayExtensions.FromArray(new[] {
            points[index_1], points[index_2], points[index_3], points[index_4]
        });
        var (_, i1, _) = bounding_box;
        return (box, Math.Min(i1.Width, i1.Height));
    }

    public (NDArray, List<float>) boxes_from_bitmap(NDArray pred, NDArray bitmap, NDArray dest_width, NDArray dest_height) {
        //'''
        //_bitmap:
        //single map with shape(1, H, W),
        //whose values are binarized as {
        //    0, 1
        //}
        //'''

        var (height, width) = bitmap.shape;
        bitmap = bitmap.astype(TF_DataType.TF_INT8) * 255;
        //bitmap = new NDArray(bitmap..Select(b => (bool)b ? 255 : 0).ToArray(), bitmap.shape);
        using var bitMat = new Mat(bitmap.shape.as_int_list(), MatType.CV_8U, bitmap.ToByteArray());
        Cv2.FindContours(InputArray.Create(bitMat), out var ct, out _, OpenCvSharp.RetrievalModes.List,
            OpenCvSharp.ContourApproximationModes.ApproxSimple);
        var contours = ct.Select(a => NDArrayExtensions.FromArray(
            a.Select(c => new NDArray(new[] { c.X, c.Y })).ToArray())
        ).ToArray();

        var num_contours = Math.Min(contours.Length, this.max_candidates);

        var boxes = new List<NDArray>();
        var scores = new List<float>();
        foreach (var index in Enumerable.Range(0, num_contours)) {
            var contour = contours[index];
            var (points, sside) = this.get_mini_boxes(contour);
            if (sside < this.min_size) {
                continue;
            }

            points = points.Copy();
            var score = this.box_score_fast(pred, points.reshape(new Shape(-1, 2)));
            if (this.box_thresh > score) {
                continue;
            }

            var box = this.unclip(points).reshape(new Shape(-1, 1, 2));
            (box, sside) = this.get_mini_boxes(box);
            if (sside < this.min_size + 2) {
                continue;
            }

            box = box.Copy().astype(TF_DataType.TF_FLOAT);
            var boxLen = box.Count();
            for (var i = 0; i < boxLen; i++) {
                box[i,0] = (float)Math.Round(Math.Clamp((float)box[i,0] / width * (float)dest_width, 0, (float)dest_width));
                box[i,1] = (float)Math.Round(Math.Clamp((float)box[i,1] / height * (float)dest_height, 0, (float)dest_height));
            }
            //box[S(":, 0")] = new NDArray();
            //box[S(":, 1")] = new NDArray(tf.clip_by_value(
            //    tf.round(box[S(":, 1")] / height * dest_height), 0, dest_height));
            var box16 = box.astype(np.int32);
            boxes.append(box16);
            scores.append(score);
        }

        return (NDArrayExtensions.FromArray(boxes.ToArray()), scores);
    }

    public float box_score_fast(NDArray bitmap, NDArray _box) {
        //'''
        //box_score_fast:
        //use bbox mean score as the mean score
        //'''
        var (h, w) = ((int)bitmap.shape[0], (int)bitmap.shape[1]);
        var box = _box.Copy();
        var xmin = new NDArray(tf.clip_by_value(np.floor(box[S(":, 0")].NDMin()).astype(np.int32), 0, w - 1));
        var xmax = new NDArray(tf.clip_by_value(tf.ceil(box[S(":, 0")].NDMax()).AsNDArrayOfType(tf.int32), 0, w - 1));
        var ymin = new NDArray(tf.clip_by_value(np.floor(box[S(":, 1")].NDMin()).astype(np.int32), 0, h - 1));
        var ymax = new NDArray(tf.clip_by_value(tf.ceil(box[S(":, 1")].NDMax()).AsNDArrayOfType(tf.int32), 0, h - 1));

        var mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8);
        var i = 0;
        foreach (var ndArray in box) {
            ndArray[0] -= xmin;
            ndArray[1] -= ymin;
            box[i,0] = ndArray[0];
            box[i,1] = ndArray[1];
            i++;
        }
        //var maskMat = new Mat(mask.shape.as_int_list(), MatType.CV_8U, mask.ToByteArray());
        mask = cv2.FillPoly(mask, box.reshape(new Shape(1, -1, 2)).astype(TF_DataType.TF_INT32), new Scalar(1));
        using var croppedBitmap = (bitmap[S($"{ymin}: {ymax + 1}, {xmin}: {xmax + 1}")] * 255).astype(TF_DataType.TF_UINT8);
        using var bitMat = new Mat(croppedBitmap.shape.as_int_list(), MatType.CV_8U, croppedBitmap.ToByteArray());
        using var maskMat = new Mat(mask.shape.as_int_list(), MatType.CV_8U, mask.ToByteArray());
        var mean = Cv2.Mean(InputArray.Create(bitMat), InputArray.Create(maskMat));

        return (float)mean.Val0 /255;
    }

    private static Slice[] S(string s) {
        return Slice.ParseSlices(s);
    }

    private static double PolygonArea(NDArray polygon) {
        int i, j;
        double area = 0;

        var length = (int)polygon.shape[0];

        for (i = 0; i < length; i++) {
            j = (i + 1) % length;

            area += polygon[i][0] * polygon[j][1];
            area -= polygon[i][1] * polygon[j][0];
        }

        area /= 2;
        return area < 0 ? -area : area;
    }

    public NDArray unclip(NDArray box) {
        var distance = PolygonArea(box) * this.unclip_ratio / PolygonLength(box);
        var offset = new ClipperLib.ClipperOffset();
        var boxPoints = box.Select(nd => new IntPoint((int)nd[0], (int)nd[1])).ToList();
        offset.AddPath(boxPoints, JoinType.jtRound, EndType.etClosedPolygon);
        var output = new List<List<IntPoint>>();
        offset.Execute(ref output, distance);
        var expanded = NDArrayExtensions.FromArray(
            output.Select(a => NDArrayExtensions.FromArray(
                a.Select(b => new NDArray(
                    new[] { (int)b.X, (int)b.Y })
                ).ToArray())
            ).ToArray());
        return expanded;
    }

    private double PolygonLength(NDArray box) {
        NDArray prevPoint = null;
        double sum = 0;
        for (int i = 0; i <= (int)box.shape[0]; i++) {
            var point = box[i % box.shape[0]][0];
            if (prevPoint is not null) {
                var offset = point-prevPoint;
                var x = (float)offset[0];
                var y = (float)offset[1];
                sum += Math.Sqrt(x*x + y*y);
            }
            prevPoint = point;
        }

        return sum;
    }

    public List<Box> PostProcess(Dictionary<string, NDArray> outs_dict, NDArray shape_list) {
        var pred = outs_dict["maps"];
        //if isinstance(pred, paddle.Tensor):
        //pred = pred.numpy()
        pred = pred[Slice.ParseSlices(":, 0, :, :")]; //????
        var segmentation = pred > this.thresh;

        var boxes_batch = new List<Box>();
        foreach (var batch_index in Enumerable.Range(0, (int)pred.shape[0])) {
            var (src_h, src_w, ratio_h, ratio_w) = (shape_list[batch_index][0], shape_list[batch_index][1], shape_list[batch_index][2],
                shape_list[batch_index][3]);
            NDArray mask;
            if (false && this.dilation_kernel != null) {
                //mask = cv2.dilate(
                //    np.array(segmentation[batch_index]).astype(np.uint8),
                //    this.dilation_kernel);
            } else {
                mask = segmentation[batch_index];
                var (boxes, scores) = this.boxes_from_bitmap(pred[batch_index], mask,
                    src_w, src_h);

                boxes_batch.Add(new Box {
                    points = boxes
                });
            }
        }

        return boxes_batch;
    }
}

public class Box {
    public NDArray points;
}