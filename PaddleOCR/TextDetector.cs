
using Microsoft.ML.OnnxRuntime;
using NumSharp;

namespace PaddleOCR; 

public class TextDetector {
    private readonly Args args;
    private readonly bool use_paddle_predict;
    private readonly object preprocess_op;
    private readonly DBPostProcess postprocess_op;
    private readonly InferenceSession predictor;

    public TextDetector(Args args) {
        this.args = args;
        this.use_paddle_predict = args.use_paddle_predict;

        this.preprocess_op = new DBPreProcess(args);
        this.postprocess_op = new DBPostProcess(
            thresh: args.det_db_thresh,
            box_thresh: args.det_db_box_thresh,
            max_candidates: 1000,
            unclip_ratio: args.det_db_unclip_ratio,
            use_dilation: args.use_dilation,
            score_mode: args.det_db_score_mode);

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
        throw new NotImplementedException();
    }
}