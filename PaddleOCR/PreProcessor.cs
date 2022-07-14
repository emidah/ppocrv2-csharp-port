using NumSharp;

namespace PaddleOCR;

public class PreProcessor {
    public static (IList<object>, IList<object>) preprocess_boxes(IList<object> dt_boxes, NDArray ori_im) {
        var img_crop_list = new List<object>();
        dt_boxes = sorted_boxes(dt_boxes);
        foreach(var bno in Enumerable.Range(0, dt_boxes.Count)) {
            var tmp_box = copy.deepcopy(dt_boxes[bno]);
            var img_crop = get_rotate_crop_image(ori_im, tmp_box);
            img_crop_list.Add(img_crop);
        }
        return (dt_boxes, img_crop_list);
    }

    public static object get_rotate_crop_image(object img, float[] points) {
        //assert len(points) == 4, "shape of points must be 4*2"
        var img_crop_width = (int)Math.Max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]));
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width,
        0],
        [img_crop_width, img_crop_height],
        [
        0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode = cv2.BORDER_REPLICATE,
            flags = cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
        return dst_img
    }

    public static object sorted_boxes(object dt_boxes) {
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key = lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in
        range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
        (_boxes[i + 1][0][0] < _boxes[i][0][0]):
        tmp = _boxes[i]
        _boxes[i] = _boxes[i + 1]
        _boxes[i + 1] = tmp
        return _boxes
    }

    
}