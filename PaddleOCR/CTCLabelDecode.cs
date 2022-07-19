using Tensorflow;
using Tensorflow.NumPy;

namespace PaddleOCR;

using static Binding;

public class CtcLabelDecode : BaseRecLabelDecode {
    public CtcLabelDecode(string argsRecCharDictPath, bool argsUseSpaceChar) : base(argsRecCharDictPath, argsUseSpaceChar) {
    }

    public List<(string, float)> DoDecode(NDArray preds) {
        var predsIdx = np.argmax(preds, 2);
        var predsProb = new NDArray(tf.max(preds, new Axis(2)));
        var text = this.Decode(predsIdx, predsProb, true);
        return text;
    }

    public override string[] add_special_char(string[] dictCharacter) {
        dictCharacter = dictCharacter.Prepend("").ToArray();
        return dictCharacter;
    }
}