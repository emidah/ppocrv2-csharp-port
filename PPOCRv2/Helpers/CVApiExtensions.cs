using OpenCvSharp;
using SharpCV;
using Tensorflow;
using Tensorflow.NumPy;
using BorderTypes = OpenCvSharp.BorderTypes;
using InterpolationFlags = OpenCvSharp.InterpolationFlags;
using Mat = OpenCvSharp.Mat;
using MatType = OpenCvSharp.MatType;
using Point = OpenCvSharp.Point;
using Point2f = OpenCvSharp.Point2f;
using RotatedRect = SharpCV.RotatedRect;
using Scalar = OpenCvSharp.Scalar;
using Size = OpenCvSharp.Size;
using Size2f = OpenCvSharp.Size2f;

namespace PPOCRv2.Helpers;

internal static class CvApiExtensions {
    public static NDArray FillPoly(this cv_api cv, NDArray img, NDArray box, Scalar color) {
        using var maskMat = new Mat(img.shape.as_int_list(), MatType.CV_8U, img.ToByteArray());
        var points = box.reshape(new Shape(1, -1, 2)).astype(np.int32).Select(a =>
            a.Select(t => {
                var ints = t.ToArray<int>();
                return new Point(ints[0], ints[1]);
            }).ToArray()).ToArray();
        Cv2.FillPoly(maskMat, points, color);
        maskMat.GetArray(out byte[] data);
        return new NDArray(data, img.shape, img.dtype);
    }

    public static NDArray BoxPoints(this cv_api cv, RotatedRect rect) {
        var ocsRect = new OpenCvSharp.RotatedRect(
            new Point2f(rect.Center.X, rect.Center.Y),
            new Size2f(rect.Size.Width, rect.Size.Height),
            rect.Angle);
        var arr = Cv2.BoxPoints(ocsRect);
        var arr2 = NdArrayExtensions.FromArray(arr
            .Select(p => new NDArray(new[] { p.X, p.Y })).ToArray());
        return arr2;
    }

    public static NDArray GetPerspectiveTransform(this cv_api cv, NDArray points, NDArray dst) {
        var pointList = points.Select(p => new Point2f((float)p[0], (float)p[1])).ToList();
        var dstList = dst.Select(p => new Point2f((float)p[0], (float)p[1])).ToList();
        using var mat = Cv2.GetPerspectiveTransform(pointList, dstList);
        mat.GetArray(out double[] matData);
        return new NDArray(matData, new Shape(mat.Cols, mat.Rows));
    }

    public static NDArray WarpPerspective(this cv_api cv, NDArray src, NDArray m, (int, int) size, InterpolationFlags flags,
        BorderTypes border) {
        var mArray = m.ToArray<double>().Select(d => (float)d).ToArray();
        var m2 = new[,] {
            { mArray[0], mArray[1], mArray[2] },
            { mArray[3], mArray[4], mArray[5] },
            { mArray[6], mArray[7], mArray[8] }
        };
        using var srcMat = new Mat(new[] { (int)src.shape[0], (int)src.shape[1] }, MatType.CV_32FC3,
            (src.astype(TF_DataType.TF_FLOAT) / 255).ToByteArray());
        using var output = new Mat(srcMat.Size(), srcMat.Type());
        var dsize = new Size(size.Item1, size.Item2);
        Cv2.WarpPerspective(InputArray.Create(srcMat), OutputArray.Create(output), m2, dsize, flags, border);
        output.GetArray<Vec3f>(out var outFloats);
        var outs = outFloats.SelectMany(v => new[] { v.Item0, v.Item1, v.Item2 }).ToArray();
        var toReturn = new NDArray(outs, new Shape(output.Height, output.Width, 3));
        toReturn = (toReturn * 255).astype(TF_DataType.TF_UINT8);
        return toReturn;
    }
}