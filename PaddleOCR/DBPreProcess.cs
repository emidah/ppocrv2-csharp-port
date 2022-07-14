
using OpenCvSharp;
using OpenCvSharp.Internal;
using OpenCvSharp.Internal.Vectors;
using SharpCV;

namespace PaddleOCR; 

public class DBPreProcess {

    private readonly Args args;
    public DBPreProcess(Args args) {
        this.args = args;
    }

    public List<Mat> PreProcess(Dictionary<string, Mat>? data) {
        data = this.DetResizeForTest(this.args.det_limit_side_len,
            this.args.det_limit_type, data);
        data = this.NormalizeImage(data);
        data = this.ToCHWImage(data);
        var data2 = this.KeepKeys(new string[]{ "image", "shape"}, data);
        return data2;
    }

    public Dictionary<string, Mat>? NormalizeImage(Dictionary<string, Mat>? data) {
        var scale = 1.0f / 255;
        var means = new float[]{ 0.485f, 0.456f, 0.406f };

        var stds = new float[] {
            0.229f, 0.224f, 0.225f
        };
        var order = "hwc";
        var shape = (1, 1, 3);
        var mean = Mat.FromArray(means).reshape(shape).astype("float32");
        var std = Mat.FromArray(stds).reshape(shape).astype("float32");
        var img = data["image"];
        Mat m2 = new Mat();
        img.ConvertTo(m2, MatType.CV_32F);
        data["image"] = m2;
        unsafe {
            m2.ForEachAsVec3f(((value, position) => {
                    Vec3f vector = *value;
                    vector.Item0 = (vector.Item0 * scale - mean[0]) / std[0];
                    vector.Item1 = (vector.Item0 * scale - mean[1]) / std[1];
                    vector.Item2 = (vector.Item0 * scale - mean[2]) / std[2];
                })
            );
        }

        //(m2 * scale - mean) / std);
            return data;
    }

    public Dictionary<string, Mat> ToCHWImage(Dictionary<string, Mat> data) {
        var img = data["image"];
        //from PIL import Image
        //if isinstance(img, Image.Image):
        //img = np.array(img)
        data["image"] = SwapAxes(img);
        return data;
    }

    private static Mat SwapAxes(Mat img) {
        //todo: fix
        int[] size = new []{ 3, img.Rows, img.Cols};
        var chw = new Mat(size, MatType.CV_32F);
        Mat[] planes = {
            new Mat(img.Rows, img.Cols, MatType.CV_32F, img.Ptr(0)),
            new Mat(img.Rows, img.Cols, MatType.CV_32F, img.Ptr(1)),
            new Mat(img.Rows, img.Cols, MatType.CV_32F, img.Ptr(2))
        };
        Split(img, ref planes);
        return chw;
    }

    private static void Split(Mat src, ref Mat[] mv) {
        if (src == null)
            throw new ArgumentNullException(nameof(src));
        src.ThrowIfDisposed();

        using var vec = new VectorOfMat();
        NativeMethods.HandleException(
            NativeMethods.core_split(src.CvPtr, vec.CvPtr));
        mv = vec.ToArray();

        GC.KeepAlive(src);
    }

    public List<Mat> KeepKeys(IEnumerable<string> keepKeys, Dictionary<string, Mat> data) {
        var data_list = new List<Mat>();
        foreach (var key in keepKeys) {
            data_list.Add(data[key]);
        }
        return data_list;
    }

    public Dictionary<string, Mat> DetResizeForTest( int limitSideLen, string limitType, Dictionary<string, Mat> data) {
        var img = data["image"];
        var (srcH, srcW) = (img.Size(0), img.Size(1));

        var (h, w, c) = (img.Size(0), img.Size(1), img.Size(2));

        float ratio;
        // limit the max side
        if (limitType == "max") {
            if (Math.Max(h, w) > limitSideLen) {
                if (h > w) {
                    ratio = (float) limitSideLen / h;
                } else {
                    ratio = (float) limitSideLen / w;
                }
            } else {
                ratio = 1.0f;
            }
        } else {
            throw new Exception("not support limit type, image ");
        }

        var resizeH = h * ratio;
        var resizeW = w * ratio;

        resizeH = Math.Max((int)Math.Round(resizeH / 32) * 32, 32);
        resizeW = Math.Max((int)Math.Round(resizeW / 32) * 32, 32);

        try {
            if ((int)resizeW <= 0 || (int) resizeH <= 0) {
                return new Dictionary<string, Mat>();
            }

            var output = new Mat();
            Cv2.Resize(img, output, new Size((int) resizeW, (int)resizeH));
        } catch {
            Console.WriteLine($"{img.Dims}, {resizeW}, {resizeH}");
            Environment.Exit(0);
        }

        var ratioH = resizeH / (float)h;
        var ratioW = resizeW / (float)w;

        data["image"] = img;
        var shape = new float[] { srcH, srcW, ratioH, ratioW };
        data["shape"] = Mat.FromArray(shape);
        return data;
    }
} 