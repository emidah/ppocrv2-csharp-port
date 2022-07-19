using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;
namespace PaddleOCR;

public class TextRecognizer {
    private readonly List<int> rec_image_shape;
    private readonly int rec_batch_num;
    private readonly CTCLabelDecode postprocess_op;
    private readonly Args args;
    private readonly InferenceSession predictor;

    public TextRecognizer(Args args) {
        this.rec_image_shape = args.rec_image_shape.Split(',').Select(s => int.Parse(s.Trim())).ToList();
        this.rec_batch_num = args.rec_batch_num;
        this.postprocess_op = new CTCLabelDecode(args.rec_char_dict_path,
            args.use_space_char);
        this.args = args;

        var model_dir = args.rec_model_dir;
        var sess = new InferenceSession(model_dir);
        this.predictor = sess;
    }

    public List<(string, float)> Recognize(List<NDArray> img_list) {
        var img_num = img_list.Count;
        //# Calculate the aspect ratio of all text bars
        var width_list = img_list.Select(img => (float)img.shape[1] / img.shape[0]).ToList();

        //# Sorting can speed up the recognition process
        var indices = np.argsort(np.array(width_list.ToArray()));
        for (int i = 0; i < img_num; i++) {
            
        }

        var rec_res = new List<(string, float)>(img_num);
        for(int i = 0; i<img_num; i++) {
            rec_res.Add(("", 0.0f));
        }
                      //[['', 0.0]] *img_num
        var batch_num = this.rec_batch_num;

        for (var beg_img_no = 0; beg_img_no < img_num; beg_img_no += batch_num) {
            var end_img_no = Math.Min(img_num, beg_img_no + batch_num);
            var norm_img_batch = new List<NDArray>();
            var max_wh_ratio = 0.0f;
            for (var ino = beg_img_no; ino < end_img_no; ino++) {
                var (h, w) = (img_list[indices[ino]].shape[0], img_list[indices[ino]].shape[1]);
                var wh_ratio = w * 1.0f / h;
                max_wh_ratio = Math.Max(max_wh_ratio, wh_ratio);
            }

            for (var ino = beg_img_no; ino < end_img_no; ino++) {
                var norm_img = this.resize_norm_img(img_list[indices[ino]],
                    max_wh_ratio);
                norm_img = norm_img[np.newaxis, new Slice(":")];
                norm_img_batch.Add(norm_img);
            }

            var arr_norm_img_batch = np.concatenate(norm_img_batch.ToArray());
            arr_norm_img_batch = arr_norm_img_batch.Copy();

            //var input_dict = new Dictionary<int, NDArray>();
            //input_dict[this.predictor.get_inputs()[0].name] = arr_norm_img_batch;
            //outputs = this.predictor.run(None, input_dict)
            //preds = outputs[0]

            var mem = new Memory<float>(arr_norm_img_batch.ToArray<float>());
            var inputTensor = new DenseTensor<float>(mem, arr_norm_img_batch.shape.as_int_list());
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(this.predictor.InputMetadata.Keys.First(), inputTensor) };
            var outputs = this.predictor.Run(input).ToList();
            var preds = outputs[0];

            var tensor = outputs[0].AsTensor<float>();
            var predsArray = new NDArray(tensor.ToArray(), new Shape(tensor.Dimensions.ToArray()));

            var rec_result = this.postprocess_op.DoDecode(predsArray);
            for (var rno = 0; rno < rec_result.Count; rno++) {
                rec_res[indices[beg_img_no + rno]] = rec_result[rno];
            }
        }

        return rec_res;
    }

    private NDArray resize_norm_img(NDArray img, float max_wh_ratio) {
        var (imgC, imgH, imgW) = (this.rec_image_shape[0], this.rec_image_shape[1], this.rec_image_shape[2]);
        //assert imgC == img.shape[2]
        imgW = (int)(32 * max_wh_ratio);
        var w = this.predictor.InputMetadata.First().Value.Dimensions[3]; //TODO
        if(w > 0 ) {
            imgW = w;
        }

        (var h, w) = ((int)img.shape[0], (int) img.shape[1]);
        var ratio = (float)w / h;
        int resized_w;
        if(Math.Ceiling(imgH * ratio) > imgW){
            resized_w = imgW;
        } else {
            resized_w = (int)Math.Ceiling(imgH * ratio);
        }

        var resized_image = (NDArray) cv2.resize(img, (resized_w, imgH));
        resized_image = resized_image.astype(TF_DataType.TF_FLOAT);
        resized_image = new NDArray(tf.transpose(resized_image, new Axis(2, 0, 1))) / 255;
        resized_image -= 0.5;
        resized_image /= 0.5;
        var padding_im = np.zeros((imgC, imgH, imgW), dtype: np.float32);
        padding_im[new Slice(":"), new Slice(":"), new Slice(0, resized_w)] = resized_image;
        return padding_im;
    }
}