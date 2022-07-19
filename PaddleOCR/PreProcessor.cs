using OpenCvSharp;
using Tensorflow;
using Tensorflow.NumPy;
using Binding = SharpCV.Binding;

namespace PaddleOCR;

using static Binding;
using static Tensorflow.Binding;

public class PreProcessor {
    public static (NDArray, IList<NDArray>) preprocess_boxes(NDArray dt_boxes, NDArray ori_im) {
        var img_crop_list = new List<NDArray>();
        dt_boxes = sorted_boxes(dt_boxes);
        foreach (var bno in Enumerable.Range(0, dt_boxes.Count())) {
            var tmp_box = dt_boxes[bno].Copy();
            var img_crop = get_rotate_crop_image(ori_im, tmp_box);
            img_crop_list.Add(img_crop);
        }

        return (dt_boxes, img_crop_list);
    }

    public static NDArray get_rotate_crop_image(NDArray img, NDArray points) {
        //assert len(points) == 4, "shape of points must be 4*2"
        var img_crop_width = Math.Max((int)
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]));
        var img_crop_height = Math.Max((int)
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]));
        var pts_std = new NDArray(
            new[] {
                0, 0, img_crop_width,
                0, img_crop_width, img_crop_height, 0, img_crop_height
            }, new Shape(4, 2));
        var m = cv2.GetPerspectiveTransform(points, pts_std);
        var dst_img = cv2.WarpPerspective(
            img,
            m,
            (img_crop_width, img_crop_height),
            InterpolationFlags.Cubic,
            BorderTypes.Replicate);

        var (dst_img_height, dst_img_width) = (dst_img.shape[0], dst_img.shape[1]);
        if (dst_img_height * 1.0 / dst_img_width >= 1.5) {
            var rotated = new NDArray(tf.transpose(dst_img, new Axis(1, 0, 2)));
            dst_img = rotated[Slice.ParseSlices("::-1,:,:")];
        }

        return dst_img;
    }

    public static NDArray sorted_boxes(NDArray dt_boxes) {
        var num_boxes = dt_boxes.shape[0];
        var sorted_boxes = dt_boxes.OrderBy(x => (int)x[0][1] /*, x[0][0]*/);
        var _boxes = sorted_boxes.ToArray();

        for (var i = 0; i < num_boxes - 1; i++) {
            if (Math.Abs((int)(_boxes[i + 1][0][1] - _boxes[i][0][1])) < 10 && _boxes[i + 1][0][0] < _boxes[i][0][0]) {
                (_boxes[i], _boxes[i + 1]) = (_boxes[i + 1], _boxes[i]);
            }
        }

        return NDArrayExtensions.FromArray(_boxes);
    }
}