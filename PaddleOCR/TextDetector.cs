
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tensorflow;
using Tensorflow.NumPy;
using Tensor = Tensorflow.Tensor;

namespace PaddleOCR; 

public class TextDetector {
    private readonly Args args;
    private readonly bool use_paddle_predict;
    private readonly DBPreProcess preprocess_op;
    private readonly DBPostProcess postprocess_op;
    private readonly InferenceSession predictor;

    public TextDetector(Args args) {
        this.args = args;
        this.use_paddle_predict = args.use_paddle_predict;

        this.preprocess_op = new DBPreProcess(args);
        //this.postprocess_op = new DBPostProcess(
        //    thresh: args.det_db_thresh,
        //    box_thresh: args.det_db_box_thresh,
        //    max_candidates: 1000,
        //    unclip_ratio: args.det_db_unclip_ratio,
        //    use_dilation: args.use_dilation,
        //    score_mode: args.det_db_score_mode);

        var model_dir = args.det_model_dir;
        //if (args.use_paddle_predict:
        //model = paddle.jit.load(model_dir + "/inference")
        //model.eval()
        //this.predictor = model
        //else:
         
        //import onnxruntime as ort
        var model_file_path = model_dir;
        var sess = new InferenceSession(model_file_path);
        this.predictor = sess;
    }

    public IList<object> Detect(NDArray img) {
        var ori_im = img.Copy();
        var data1 = new Dictionary<string, NDArray>() { { "image", img } };

        var data = this.preprocess_op.PreProcess(data1);
        (img, var shape_list) = (data[0], data[1]);
        if (img == null) {
            return null; //, 0;
        }

        img = np.expand_dims(img, axis: 0);
        shape_list = np.expand_dims(shape_list, axis: 0);
        img = img.Copy();

        //if (this.use_paddle_predict)
        //output = self.predictor(img).numpy()
        //outputs = []
        //outputs.append(output)
        //else:
        var mem = new Memory<float>(img.ToArray<float>());
        var inputTensor = new DenseTensor<float>(mem, img.shape.as_int_list());
        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
        var outputs = this.predictor.Run(input).ToList();

        //var preds = new Dictionary<string, NDArray>();
        //preds["maps"] = outputs[0].;
        //var post_result = this.postprocess_op.PostProcess(preds, shape_list);
        //var dt_boxes = post_result[0]["points"];
        //dt_boxes = this.filter_tag_det_res(dt_boxes, ori_im.shape);
        //return dt_boxes;
        return null;
    }

    private object filter_tag_det_res(object dtBoxes, Shape shape) {
        throw new NotImplementedException();
    }
}