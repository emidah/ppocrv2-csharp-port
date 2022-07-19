using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SharpCV;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace PaddleOCR;

public class TextClassifier {
    private readonly int[] cls_image_shape;
    private readonly int cls_batch_num;
    private readonly float cls_thresh;
    private readonly ClsPostProcess postprocess_op;
    private readonly Args args;
    private readonly InferenceSession predictor;

    public TextClassifier(Args args) {
        this.cls_image_shape = args.cls_image_shape.Split(",").Select(int.Parse).ToArray();
        this.cls_batch_num = args.cls_batch_num;
        this.cls_thresh = args.cls_thresh;
        this.postprocess_op = new ClsPostProcess(args.label_list);
        this.args = args;

        var model_dir = args.cls_model_dir;
        var sess = new InferenceSession(model_dir);
        this.predictor = sess;
    }

    public (IList<NDArray>, object) Classify(IList<NDArray> img_list) {
        var img_num = img_list.Count;
        //# Calculate the aspect ratio of all text bars
        var width_list = new List<float>();
        foreach (var img in img_list) {
            width_list.Add(img.shape[1] / (float)img.shape[0]);
        }

        //# Sorting can speed up the cls process
        var indices = np.argsort(new NDArray(width_list.ToArray()));

        var cls_res = new List<(string, float)>(img_num);
        for (var i = 0; i < img_num; i++) {
            cls_res.Add(("", 0.0f));
        }

        var batch_num = this.cls_batch_num;

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
            //input_dict = {}
            //input_dict[self.predictor.get_inputs()[0].name] = norm_img_batch
            //outputs = self.predictor.run(None, input_dict)
            //prob_out = outputs[0]
            var mem = new Memory<float>(arr_norm_img_batch.ToArray<float>());
            var inputTensor = new DenseTensor<float>(mem, arr_norm_img_batch.shape.as_int_list());
            var input = new List<NamedOnnxValue>
                { NamedOnnxValue.CreateFromTensor(this.predictor.InputMetadata.Keys.First(), inputTensor) };
            var outputs = this.predictor.Run(input).ToList();
            var prob_out = outputs[0];
            var tensor = outputs[0].AsTensor<float>();
            var prob_out_array = new NDArray(tensor.ToArray(), new Shape(tensor.Dimensions.ToArray()));

            var cls_result = this.postprocess_op.PostProcess(prob_out_array);
            for (var rno = 0; rno < cls_result.Count; rno++) {
                var (label, score) = cls_result[rno];
                cls_res[indices[beg_img_no + rno]] = (label, score);
                if (label.Contains("180") && score > this.cls_thresh) {
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], (RotateFlags)1);
                }
            }
        }

        return (img_list, cls_res);
    }

    private NDArray resize_norm_img(NDArray img, float max_wh_ratio) {
        var (imgC, imgH, imgW) = (this.cls_image_shape[0], this.cls_image_shape[1], this.cls_image_shape[2]);
        //assert imgC == img.shape[2]
        imgW = (int)(32 * max_wh_ratio);
        var w = this.predictor.InputMetadata.First().Value.Dimensions[3]; //TODO
        if (w > 0) {
            imgW = w;
        }

        (var h, w) = ((int)img.shape[0], (int)img.shape[1]);
        var ratio = (float)w / h;
        int resized_w;
        if (Math.Ceiling(imgH * ratio) > imgW) {
            resized_w = imgW;
        } else {
            resized_w = (int)Math.Ceiling(imgH * ratio);
        }

        var resized_image = (NDArray)cv2.resize(img, (resized_w, imgH));
        resized_image = resized_image.astype(TF_DataType.TF_FLOAT);
        resized_image = new NDArray(tf.transpose(resized_image, new Axis(2, 0, 1))) / 255;
        resized_image -= 0.5;
        resized_image /= 0.5;
        var padding_im = np.zeros((imgC, imgH, imgW), np.float32);
        padding_im[new Slice(":"), new Slice(":"), new Slice(0, resized_w)] = resized_image;
        return padding_im;
    }

    public class ClsPostProcess {
        private readonly int[] label_list;
        //""" Convert between text-label and text-index """

        public ClsPostProcess(int[] label_list) {
            //super(ClsPostProcess, this).__init__()
            this.label_list = label_list;
        }

        public List<(string, float)> PostProcess(NDArray preds) {
            var pred_idxs = np.argmax(preds, 1);
            var decode_out = pred_idxs.Select((idx, i) => (this.label_list[idx].ToString(), (float)preds[i, idx])).ToList();
            return decode_out;
        }
    }
}