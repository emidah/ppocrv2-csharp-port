using OpenCvSharp;
using Tensorflow;
using Tensorflow.NumPy;
using Binding = SharpCV.Binding;

namespace PaddleOCR;

using static Binding;
using static Tensorflow.Binding;

public class PreProcessor {
    public static (NDArray, IList<NDArray>) PreprocessBoxes(NDArray dtBoxes, NDArray oriIm) {
        var imgCropList = new List<NDArray>();
        dtBoxes = SortedBoxes(dtBoxes);
        foreach (var bno in Enumerable.Range(0, dtBoxes.Count())) {
            var tmpBox = dtBoxes[bno].Copy();
            var imgCrop = RotateCropImage(oriIm, tmpBox);
            imgCropList.Add(imgCrop);
        }

        return (dtBoxes, imgCropList);
    }

    public static NDArray RotateCropImage(NDArray img, NDArray points) {
        //assert len(points) == 4, "shape of points must be 4*2"
        var imgCropWidth = Math.Max((int)
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]));
        var imgCropHeight = Math.Max((int)
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]));
        var ptsStd = new NDArray(
            new[] {
                0, 0, imgCropWidth,
                0, imgCropWidth, imgCropHeight, 0, imgCropHeight
            }, new Shape(4, 2));
        var m = cv2.GetPerspectiveTransform(points, ptsStd);
        var dstImg = cv2.WarpPerspective(
            img,
            m,
            (imgCropWidth, imgCropHeight),
            InterpolationFlags.Cubic,
            BorderTypes.Replicate);

        var (dstImgHeight, dstImgWidth) = (dstImg.shape[0], dstImg.shape[1]);
        if (dstImgHeight * 1.0 / dstImgWidth >= 1.5) {
            var rotated = new NDArray(tf.transpose(dstImg, new Axis(1, 0, 2)));
            dstImg = rotated[Slice.ParseSlices("::-1,:,:")];
        }

        return dstImg;
    }

    public static NDArray SortedBoxes(NDArray dtBoxes) {
        var numBoxes = dtBoxes.shape[0];
        var sortedBoxes = dtBoxes.OrderBy(x => (int)x[0][1] /*, x[0][0]*/);
        var boxes = sortedBoxes.ToArray();

        for (var i = 0; i < numBoxes - 1; i++) {
            if (Math.Abs((int)(boxes[i + 1][0][1] - boxes[i][0][1])) < 10 && boxes[i + 1][0][0] < boxes[i][0][0]) {
                (boxes[i], boxes[i + 1]) = (boxes[i + 1], boxes[i]);
            }
        }

        return NdArrayExtensions.FromArray(boxes);
    }
}