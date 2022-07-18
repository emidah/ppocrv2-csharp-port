using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using SharpCV;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static SharpCV.Binding;
using Mat = OpenCvSharp.Mat;
using MatType = OpenCvSharp.MatType;
using Point = OpenCvSharp.Point;
using RotatedRect = SharpCV.RotatedRect;
using Scalar = OpenCvSharp.Scalar;

namespace PaddleOCR
{
    internal static class CVApiExtensions {
        public static NDArray FillPoly(this cv_api cv, NDArray img, NDArray box, Scalar color) {
            using var maskMat = new Mat(img.shape.as_int_list(), MatType.CV_8U, img.ToByteArray());
            var points = box.reshape(new Shape(1, -1, 2)).astype(np.int32).Select(a =>
                a.Select(t => {
                    var ints = t.ToArray<int>();
                    return new Point(ints[0], ints[1]);
                }).ToArray()).ToArray();
            Cv2.FillPoly(img: maskMat, pts: points, color: color);
            maskMat.GetArray(out byte[] data);
            return new NDArray(data, img.shape, img.dtype);
        }

        public static NDArray BoxPoints(this cv_api cv, RotatedRect rect) {
            var ocsRect = new OpenCvSharp.RotatedRect(
                new OpenCvSharp.Point2f(rect.Center.X, rect.Center.Y),
                new OpenCvSharp.Size2f(rect.Size.Width, rect.Size.Height),
                rect.Angle);
            var arr = Cv2.BoxPoints(ocsRect);
            var arr2 = NDArrayExtensions.FromArray(arr
                .Select(p => new NDArray(new [] { p.X, p.Y })).ToArray());
            return arr2;
        }
    }
}
