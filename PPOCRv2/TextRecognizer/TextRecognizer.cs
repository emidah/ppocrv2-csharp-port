using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PPOCRv2.Helpers;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace PPOCRv2.TextRecognizer;

public class TextRecognizer {
    private readonly CtcLabelDecode postprocessOp;
    private readonly InferenceSession predictor;
    private readonly int recBatchNum;
    private readonly List<int> recImageShape;

    public TextRecognizer(Args args) {
        recImageShape = args.rec_image_shape.Split(',').Select(s => int.Parse(s.Trim())).ToList();
        recBatchNum = args.rec_batch_num;
        postprocessOp = new CtcLabelDecode(args.rec_char_dict_path,
            args.use_space_char);

        var modelDir = args.rec_model_dir;
        var sess = new InferenceSession(modelDir);
        predictor = sess;
    }

    public List<(string, float)> Recognize(List<NDArray> imgList) {
        var imgNum = imgList.Count;
        //# Calculate the aspect ratio of all text bars
        var widthList = imgList.Select(img => (float)img.shape[1] / img.shape[0]).ToList();

        //# Sorting can speed up the recognition process
        var indices = np.argsort(np.array(widthList.ToArray()));

        var recRes = new List<(string, float)>(imgNum);
        for (var i = 0; i < imgNum; i++) {
            recRes.Add(("", 0.0f));
        }

        //[['', 0.0]] *img_num
        var batchNum = recBatchNum;

        for (var begImgNo = 0; begImgNo < imgNum; begImgNo += batchNum) {
            var endImgNo = Math.Min(imgNum, begImgNo + batchNum);
            var normImgBatch = new List<NDArray>();
            var maxWhRatio = 0.0f;
            for (var ino = begImgNo; ino < endImgNo; ino++) {
                var (h, w) = (imgList[indices[ino]].shape[0], imgList[indices[ino]].shape[1]);
                var whRatio = w * 1.0f / h;
                maxWhRatio = Math.Max(maxWhRatio, whRatio);
            }

            for (var ino = begImgNo; ino < endImgNo; ino++) {
                var normImg = ResizeNormImg(imgList[indices[ino]],
                    maxWhRatio);
                normImg = normImg[np.newaxis, new Slice(":")];
                normImgBatch.Add(normImg);
            }

            var arrNormImgBatch = np.concatenate(normImgBatch.ToArray());
            arrNormImgBatch = arrNormImgBatch.Copy();

            //var input_dict = new Dictionary<int, NDArray>();
            //input_dict[this.predictor.get_inputs()[0].name] = arr_norm_img_batch;
            //outputs = this.predictor.run(None, input_dict)
            //preds = outputs[0]

            var mem = new Memory<float>(arrNormImgBatch.ToArray<float>());
            var inputTensor = new DenseTensor<float>(mem, arrNormImgBatch.shape.as_int_list());
            var input = new List<NamedOnnxValue>
                { NamedOnnxValue.CreateFromTensor(predictor.InputMetadata.Keys.First(), inputTensor) };
            var outputs = predictor.Run(input).ToList();
            var tensor = outputs[0].AsTensor<float>();
            var predsArray = new NDArray(tensor.ToArray(), new Shape(tensor.Dimensions.ToArray()));

            var recResult = postprocessOp.DoDecode(predsArray);
            for (var rno = 0; rno < recResult.Count; rno++) {
                recRes[indices[begImgNo + rno]] = recResult[rno];
            }
        }

        return recRes;
    }

    private NDArray ResizeNormImg(NDArray img, float maxWhRatio) {
        var (imgC, imgH, imgW) = (recImageShape[0], recImageShape[1], recImageShape[2]);
        //assert imgC == img.shape[2]
        imgW = (int)(32 * maxWhRatio);
        var w = predictor.InputMetadata.First().Value.Dimensions[3]; //TODO
        if (w > 0) {
            imgW = w;
        }

        (var h, w) = ((int)img.shape[0], (int)img.shape[1]);
        var ratio = (float)w / h;
        int resizedW;
        if (Math.Ceiling(imgH * ratio) > imgW) {
            resizedW = imgW;
        } else {
            resizedW = (int)Math.Ceiling(imgH * ratio);
        }

        var resizedImage = (NDArray)cv2.resize(img, (resizedW, imgH));
        resizedImage = resizedImage.astype(TF_DataType.TF_FLOAT);
        resizedImage = new NDArray(tf.transpose(resizedImage, new Axis(2, 0, 1))) / 255;
        resizedImage -= 0.5;
        resizedImage /= 0.5;
        var paddingIm = np.zeros((imgC, imgH, imgW), np.float32);
        paddingIm[new Slice(":"), new Slice(":"), new Slice(0, resizedW)] = resizedImage;
        return paddingIm;
    }
}