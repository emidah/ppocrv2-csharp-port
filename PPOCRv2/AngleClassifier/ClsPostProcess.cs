using Tensorflow.NumPy;

namespace PPOCRv2.AngleClassifier;

public class ClsPostProcess {
    private readonly int[] labelList;

    public ClsPostProcess(int[] labelList) {
        this.labelList = labelList;
    }

    public List<(string, float)> PostProcess(NDArray preds) {
        var predIdxs = np.argmax(preds, 1);
        var decodeOut = predIdxs.Select((idx, i) => (labelList[idx].ToString(), (float)preds[i, idx])).ToList();
        return decodeOut;
    }
}