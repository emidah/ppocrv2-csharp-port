﻿using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PPOCRv2.Helpers;
using SharpCV;
using Tensorflow;
using Tensorflow.NumPy;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace PPOCRv2.AngleClassifier;

public class TextClassifier {
    private readonly int clsBatchNum;
    private readonly int[] clsImageShape;
    private readonly float clsThresh;
    private readonly ClsPostProcess postprocessOp;
    private readonly InferenceSession predictor;

    public TextClassifier(Args args) {
        clsImageShape = args.ClsImageShape.Split(",").Select(int.Parse).ToArray();
        clsBatchNum = args.ClsBatchNum;
        clsThresh = args.ClsThresh;
        postprocessOp = new ClsPostProcess(args.LabelList);

        var modelDir = args.ClsModelDir;
        var sess = new InferenceSession(modelDir);
        predictor = sess;
    }

    public (IList<NDArray>, object) Classify(IList<NDArray> imgList) {
        var imgNum = imgList.Count;
        //# Calculate the aspect ratio of all text bars
        var widthList = new List<float>();
        foreach (var img in imgList) {
            widthList.Add(img.shape[1] / (float)img.shape[0]);
        }

        //# Sorting can speed up the cls process
        var indices = np.argsort(new NDArray(widthList.ToArray()));

        var clsRes = new List<(string, float)>(imgNum);
        for (var i = 0; i < imgNum; i++) {
            clsRes.Add(("", 0.0f));
        }

        var batchNum = clsBatchNum;

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
            var mem = new Memory<float>(arrNormImgBatch.ToArray<float>());
            var inputTensor = new DenseTensor<float>(mem, arrNormImgBatch.shape.as_int_list());
            var input = new List<NamedOnnxValue>
                { NamedOnnxValue.CreateFromTensor(predictor.InputMetadata.Keys.First(), inputTensor) };
            var outputs = predictor.Run(input).ToList();
            var tensor = outputs[0].AsTensor<float>();
            var probOutArray = new NDArray(tensor.ToArray(), new Shape(tensor.Dimensions.ToArray()));

            var clsResult = postprocessOp.PostProcess(probOutArray);
            for (var rno = 0; rno < clsResult.Count; rno++) {
                var (label, score) = clsResult[rno];
                clsRes[indices[begImgNo + rno]] = (label, score);
                if (label.Contains("180") && score > clsThresh) {
                    imgList[indices[begImgNo + rno]] = cv2.rotate(
                        imgList[indices[begImgNo + rno]], (RotateFlags)1);
                }
            }
        }

        return (imgList, clsRes);
    }

    private NDArray ResizeNormImg(NDArray img, float maxWhRatio) {
        var (imgC, imgH, imgW) = (clsImageShape[0], clsImageShape[1], clsImageShape[2]);
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