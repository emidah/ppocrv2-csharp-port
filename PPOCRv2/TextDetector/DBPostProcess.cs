using ClipperLib;
using OpenCvSharp;
using PPOCRv2.Helpers;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace PPOCRv2.TextDetector;

public class DbPostProcess {
    private readonly float boxThresh;

    //private readonly NDArray dilationKernel;
    private readonly int maxCandidates;
    private readonly int minSize;
    private readonly string scoreMode;
    private readonly float thresh;
    private readonly float unclipRatio;

    public DbPostProcess(float thresh, float boxThresh, int maxCandidates, float unclipRatio, bool useDilation, string scoreMode) {
        this.thresh = thresh;
        this.boxThresh = boxThresh;
        this.maxCandidates = maxCandidates;
        this.unclipRatio = unclipRatio;
        minSize = 3;
        this.scoreMode = scoreMode;
        useDilation = false;
        //this.dilationKernel = useDilation ? np.array(new[] { new[] { 1, 1 }, new[] { 1, 1 } }) : null;
    }

    public static (NDArray, float) GetMiniBoxes(NDArray contour) {
        var boundingBox = cv2.minAreaRect(contour);

        var points = NdArrayExtensions.FromArray(cv2.BoxPoints(boundingBox).OrderBy(x => (int)x[0]).ToArray());

        int index1, index2, index3, index4;
        if (points[1][1] > points[0][1]) {
            index1 = 0;
            index4 = 1;
        } else {
            index1 = 1;
            index4 = 0;
        }

        if (points[3][1] > points[2][1]) {
            index2 = 2;
            index3 = 3;
        } else {
            index2 = 3;
            index3 = 2;
        }

        var box = NdArrayExtensions.FromArray(new[] {
            points[index1], points[index2], points[index3], points[index4]
        });
        var (_, i1, _) = boundingBox;
        return (box, Math.Min(i1.Width, i1.Height));
    }

    public (NDArray, List<float>) BoxesFromBitmap(NDArray pred, NDArray bitmap, NDArray destWidth, NDArray destHeight) {
        var (height, width) = bitmap.shape;
        bitmap = bitmap.astype(TF_DataType.TF_INT8) * 255;
        using var bitMat = new Mat(bitmap.shape.as_int_list(), MatType.CV_8U, bitmap.ToByteArray());
        Cv2.FindContours(InputArray.Create(bitMat), out var ct, out _, RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);
        var contours = ct.Select(a => NdArrayExtensions.FromArray(
            a.Select(c => new NDArray(new[] { c.X, c.Y })).ToArray())
        ).ToArray();

        var numContours = Math.Min(contours.Length, maxCandidates);

        var boxes = new List<NDArray>();
        var scores = new List<float>();
        foreach (var index in Enumerable.Range(0, numContours)) {
            var contour = contours[index];
            var (points, sside) = GetMiniBoxes(contour);
            if (sside < minSize) {
                continue;
            }

            points = points.Copy();
            var score = BoxScoreFast(pred, points.reshape(new Shape(-1, 2)));
            if (boxThresh > score) {
                continue;
            }

            var box = Unclip(points).reshape(new Shape(-1, 1, 2));
            (box, sside) = GetMiniBoxes(box);
            if (sside < minSize + 2) {
                continue;
            }

            box = box.Copy().astype(TF_DataType.TF_FLOAT);
            var boxLen = box.Count();
            for (var i = 0; i < boxLen; i++) {
                box[i, 0] = (float)Math.Round(Math.Clamp((float)box[i, 0] / width * (float)destWidth, 0, (float)destWidth));
                box[i, 1] = (float)Math.Round(Math.Clamp((float)box[i, 1] / height * (float)destHeight, 0, (float)destHeight));
            }

            //box[S(":, 0")] = new NDArray();
            //box[S(":, 1")] = new NDArray(tf.clip_by_value(
            //    tf.round(box[S(":, 1")] / height * dest_height), 0, dest_height));
            var box16 = box.astype(np.int32);
            boxes.append(box16);
            scores.append(score);
        }

        return (NdArrayExtensions.FromArray(boxes.ToArray()), scores);
    }

    public static float BoxScoreFast(NDArray bitmap, NDArray boxOg) {
        var (h, w) = ((int)bitmap.shape[0], (int)bitmap.shape[1]);
        var box = boxOg.Copy();
        var xmin = new NDArray(tf.clip_by_value(np.floor(box[S(":, 0")].NdMin()).astype(np.int32), 0, w - 1));
        var xmax = new NDArray(tf.clip_by_value(tf.ceil(box[S(":, 0")].NdMax()).AsNdArrayOfType(tf.int32), 0, w - 1));
        var ymin = new NDArray(tf.clip_by_value(np.floor(box[S(":, 1")].NdMin()).astype(np.int32), 0, h - 1));
        var ymax = new NDArray(tf.clip_by_value(tf.ceil(box[S(":, 1")].NdMax()).AsNdArrayOfType(tf.int32), 0, h - 1));

        var mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8);
        var i = 0;
        foreach (var ndArray in box) {
            ndArray[0] -= xmin;
            ndArray[1] -= ymin;
            box[i, 0] = ndArray[0];
            box[i, 1] = ndArray[1];
            i++;
        }
        mask = cv2.FillPoly(mask, box.reshape(new Shape(1, -1, 2)).astype(TF_DataType.TF_INT32), new Scalar(1));
        using var croppedBitmap = (bitmap[S($"{ymin}: {ymax + 1}, {xmin}: {xmax + 1}")] * 255).astype(TF_DataType.TF_UINT8);
        using var bitMat = new Mat(croppedBitmap.shape.as_int_list(), MatType.CV_8U, croppedBitmap.ToByteArray());
        using var maskMat = new Mat(mask.shape.as_int_list(), MatType.CV_8U, mask.ToByteArray());
        var mean = Cv2.Mean(InputArray.Create(bitMat), InputArray.Create(maskMat));

        return (float)mean.Val0 / 255;
    }

    private static Slice[] S(string s) {
        return Slice.ParseSlices(s);
    }

    private static double PolygonArea(NDArray polygon) {
        var area = 0.0d;
        var length = (int)polygon.shape[0];
        for (var i = 0; i < length; i++) {
            var j = (i + 1) % length;
            area += polygon[i][0] * polygon[j][1];
            area -= polygon[i][1] * polygon[j][0];
        }
        area /= 2;
        return Math.Abs(area);
    }

    public NDArray Unclip(NDArray box) {
        var distance = PolygonArea(box) * unclipRatio / PolygonLength(box);
        var offset = new ClipperOffset();
        var boxPoints = box.Select(nd => new IntPoint((int)nd[0], (int)nd[1])).ToList();
        offset.AddPath(boxPoints, JoinType.jtRound, EndType.etClosedPolygon);
        var output = new List<List<IntPoint>>();
        offset.Execute(ref output, distance);
        var expanded = NdArrayExtensions.FromArray(
            output.Select(a => NdArrayExtensions.FromArray(
                a.Select(b => new NDArray(
                    new[] { (int)b.X, (int)b.Y })
                ).ToArray())
            ).ToArray());
        return expanded;
    }

    private double PolygonLength(NDArray box) {
        NDArray prevPoint = null;
        double sum = 0;
        for (var i = 0; i <= (int)box.shape[0]; i++) {
            var point = box[i % box.shape[0]][0];
            if (prevPoint is not null) {
                var offset = point - prevPoint;
                var x = (float)offset[0];
                var y = (float)offset[1];
                sum += Math.Sqrt(x * x + y * y);
            }

            prevPoint = point;
        }

        return sum;
    }

    public List<Box> PostProcess(Dictionary<string, NDArray> outsDict, NDArray shapeList) {
        var pred = outsDict["maps"];
        pred = pred[Slice.ParseSlices(":, 0, :, :")]; //????
        var segmentation = pred > thresh;

        var boxesBatch = new List<Box>();
        foreach (var batchIndex in Enumerable.Range(0, (int)pred.shape[0])) {
            var (srcH, srcW, ratioH, ratioW) = (shapeList[batchIndex][0], shapeList[batchIndex][1], shapeList[batchIndex][2],
                shapeList[batchIndex][3]);
            NDArray mask;

            mask = segmentation[batchIndex];
            var (boxes, scores) = BoxesFromBitmap(pred[batchIndex], mask,
                srcW, srcH);

            boxesBatch.Add(new Box {
                Points = boxes
            });
        }

        return boxesBatch;
    }
}

public class Box {
    public NDArray Points;
}