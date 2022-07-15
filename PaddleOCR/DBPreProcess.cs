

using SharpCV;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace PaddleOCR; 

public class DBPreProcess {

    private readonly Args args;
    public DBPreProcess(Args args) {
        this.args = args;
    }

    public List<NDArray> PreProcess(Dictionary<string, NDArray> data) {
        data = this.DetResizeForTest(this.args.det_limit_side_len,
            this.args.det_limit_type, data);
        data = this.NormalizeImage(data);
        data = this.ToCHWImage(data);
        var data2 = this.KeepKeys(new string[]{ "image", "shape"}, data);
        return data2;
    }

    public Dictionary<string, NDArray>? NormalizeImage(Dictionary<string, NDArray> data) {
        var scale = 1.0f / 255;
        var mean = np.array( 0.485f, 0.456f, 0.406f );
        var std = np.array(0.229f, 0.224f, 0.225f);
        var shape = (1, 1, 3);
        mean = mean.reshape(shape).astype(TF_DataType.TF_FLOAT);
        std = std.reshape(shape).astype(TF_DataType.TF_FLOAT);
        var img = data["image"];
        data["image"] = (img.astype(TF_DataType.TF_FLOAT) * scale - mean) / std;
        return data;
    }

    public Dictionary<string, NDArray> ToCHWImage(Dictionary<string, NDArray> data) {
        var img = data["image"];
        //from PIL import Image
        //if isinstance(img, Image.Image):
        //img = np.array(img)
        data["image"] = new NDArray( tf.transpose(img, new Axis(2, 0, 1)));
        return data;
    }

    //private static Mat SwapAxes(Mat img) {
    //    //todo: fix
    //    int[] size = new []{ 3, img.Rows, img.Cols};
    //    var chw = new Mat(size, MatType.CV_32F);
    //    Mat[] planes = {
    //        new Mat(img.Rows, img.Cols, MatType.CV_32F, img.Ptr(0)),
    //        new Mat(img.Rows, img.Cols, MatType.CV_32F, img.Ptr(1)),
    //        new Mat(img.Rows, img.Cols, MatType.CV_32F, img.Ptr(2))
    //    };
    //    Split(img, ref planes);
    //    return chw;
    //}

    //private static void Split(Mat src, ref Mat[] mv) {
    //    if (src == null)
    //        throw new ArgumentNullException(nameof(src));
    //    src.ThrowIfDisposed();

    //    using var vec = new VectorOfMat();
    //    NativeMethods.HandleException(
    //        NativeMethods.core_split(src.CvPtr, vec.CvPtr));
    //    mv = vec.ToArray();

    //    GC.KeepAlive(src);
    //}

    //private static Mat Reshape113(Mat mat) {
    //    return mat; //todo: fix
    //}

    public List<NDArray> KeepKeys(IEnumerable<string> keepKeys, Dictionary<string, NDArray> data) {
        var data_list = new List<NDArray>();
        foreach (var key in keepKeys) {
            data_list.Add(data[key]);
        }
        return data_list;
    }

    public Dictionary<string, NDArray> DetResizeForTest( float limitSideLen, string limitType, Dictionary<string, NDArray> data) {
        var img = data["image"];
        var (srcH, srcW) = img.shape;

        var (h, w, c) = (img.shape[0], img.shape[1], img.shape[2]);

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
                return new Dictionary<string, NDArray>();
            }

            var output = new Mat();
            var input = new Mat(img);
            img = cv2.resize(new Mat(img), ((int) resizeW, (int)resizeH));
        } catch {
            Console.WriteLine($"{img.shape}, {resizeW}, {resizeH}");
            Environment.Exit(0);
        }

        var ratioH = resizeH / (float)h;
        var ratioW = resizeW / (float)w;

        data["image"] = img;
        var shape = new float[] { srcH, srcW, ratioH, ratioW };
        data["shape"] = np.array(shape);
        return data;
    }
} 