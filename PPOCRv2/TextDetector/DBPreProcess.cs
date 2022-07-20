using SharpCV;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace PPOCRv2.TextDetector;

public class DbPreProcess {
    private readonly Args args;

    public DbPreProcess(Args args) {
        this.args = args;
    }

    public List<NDArray> PreProcess(Dictionary<string, NDArray> data) {
        data = DetResizeForTest(args.DetLimitSideLen,
            args.DetLimitType, data);
        data = NormalizeImage(data);
        data = ToChwImage(data);
        var data2 = KeepKeys(new[] { "image", "shape" }, data);
        return data2;
    }

    public static Dictionary<string, NDArray> NormalizeImage(Dictionary<string, NDArray> data) {
        var scale = 1.0f / 255;
        var mean = np.array(0.485f, 0.456f, 0.406f);
        var std = np.array(0.229f, 0.224f, 0.225f);
        var shape = (1, 1, 3);
        mean = mean.reshape(shape).astype(TF_DataType.TF_FLOAT);
        std = std.reshape(shape).astype(TF_DataType.TF_FLOAT);
        var img = data["image"];
        data["image"] = (img.astype(TF_DataType.TF_FLOAT) * scale - mean) / std;
        return data;
    }

    public Dictionary<string, NDArray> ToChwImage(Dictionary<string, NDArray> data) {
        var img = data["image"];
        data["image"] = new NDArray(tf.transpose(img, new Axis(2, 0, 1)));
        return data;
    }

    public List<NDArray> KeepKeys(IEnumerable<string> keepKeys, Dictionary<string, NDArray> data) {
        return keepKeys.Select(key => data[key]).ToList();
    }

    public Dictionary<string, NDArray> DetResizeForTest(float limitSideLen, string limitType, Dictionary<string, NDArray> data) {
        var img = data["image"];
        var (srcH, srcW) = img.shape;

        var (h, w) = (img.shape[0], img.shape[1]);

        float ratio;
        // limit the max side
        if (limitType == "max") {
            if (Math.Max(h, w) > limitSideLen) {
                if (h > w) {
                    ratio = limitSideLen / h;
                } else {
                    ratio = limitSideLen / w;
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
            if ((int)resizeW <= 0 || (int)resizeH <= 0) {
                return new Dictionary<string, NDArray>();
            }

            img = cv2.resize(new Mat(img), ((int)resizeW, (int)resizeH));
        } catch {
            Console.WriteLine($"{img.shape}, {resizeW}, {resizeH}");
            Environment.Exit(0);
        }

        var ratioH = resizeH / h;
        var ratioW = resizeW / w;

        data["image"] = img;
        var shape = new[] { srcH, srcW, ratioH, ratioW };
        data["shape"] = np.array(shape);
        return data;
    }
}