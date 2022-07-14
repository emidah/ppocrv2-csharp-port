﻿using System.Runtime.InteropServices;
using NumSharp;
using OpenCvSharp;

namespace PaddleOCR;

public class DBPostProcess {
    private readonly NDArray dilation_kernel;
    private readonly string score_mode;
    private readonly string thresh;
    private readonly int max_candidates;
    private readonly string box_thresh;
    private readonly string unclip_ratio;
    private readonly int min_size;

    public DBPostProcess() {
        throw new NotImplementedException();
    }

    public DBPostProcess(string thresh, string box_thresh, int max_candidates, string unclip_ratio, bool use_dilation, string score_mode)
    {
        this.thresh = thresh;
        this.box_thresh = box_thresh;
        this.max_candidates = max_candidates;
        this.unclip_ratio = unclip_ratio;
        this.min_size = 3;
        this.score_mode = score_mode;
        //assert score_mode in [
        //"slow", "fast"
        //    ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)
        this.dilation_kernel = use_dilation ? np.array(new[] { new[] {1, 1}, new[] {1, 1}} ) : null;
    }

    public (NDArray, List<float>) boxes_from_bitmap(pred, _bitmap, dest_width, dest_height) {
        //'''
        //_bitmap:
        //single map with shape(1, H, W),
        //whose values are binarized as {
        //    0, 1
        //}
        //'''

        var bitmap = _bitmap;
        var (height, width) = bitmap.shape;

        var outMats = Cv2.FindContoursAsMat((bitmap * 255).astype(np.uint8), RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);
        new NDArray()
        Point[]? img;
        Point[]? contours;
        if (outs.Length == 3) {
            (img, contours, _) = (outs[0], outs[1], outs[2]);
        } else if (outs.Length == 2) {
            (contours, _) = (outs[0], outs[1]);
        }

        var num_contours = Math.Min(contours., this.max_candidates)

        var boxes = []
        var scores = []
        foreach (var index in Enumerable.Range(0, num_contours)) {
            var contour = contours[index]
            var (firstPoints, sside) = this.get_mini_boxes(contour);
            if(sside < this.min_size) {
                continue;
            }

            var points = np.array(firstPoints);
            var score = this.box_score_fast(pred, points.reshape(-1, 2));
            if(this.box_thresh > score) {
                continue;
            }

            var abox = this.unclip(points).reshape(-1, 1, 2);
            var (firstBox, sside2) = this.get_mini_boxes(abox);
            if (sside2 < this.min_size + 2)
            continue;
            var box = np.array(firstBox);

            box[:, 0] = np.clip(
                np.round(box[):, 0] / width* dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height* dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16));
            scores.append(score);
        }

        return (np.array(boxes, np.int16), scores);
    }

    public NDArray unclip(NDArray box) {
        var poly = new Polygon(box);
        var distance = poly.area * this.unclip_ratio / poly.length;
        var offset = pyclipper.PyclipperOffset();
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON);
        var expanded = np.array(offset.Execute(distance));
        return expanded
    }
}