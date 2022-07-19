using Tensorflow;
using Tensorflow.NumPy;

namespace PaddleOCR;
using static Tensorflow.Binding;

public class CTCLabelDecode : BaseRecLabelDecode {
    public CTCLabelDecode(string argsRecCharDictPath, bool argsUseSpaceChar) : base(argsRecCharDictPath, argsUseSpaceChar) {
    }

    public List<(string, float)> DoDecode(NDArray preds) {
        var preds_idx = np.argmax(preds, 2);
        var preds_prob = new NDArray(tf.max(preds, new Axis(2)));
        var text = this.decode(preds_idx, preds_prob, is_remove_duplicate: true);
        //if label is None:
        return text;
        //label = self.decode(label)
        //return text, label
    }

    public override string[] add_special_char(string[] dict_character) {
        dict_character = dict_character.Prepend("").ToArray();
        return dict_character;
    }
}