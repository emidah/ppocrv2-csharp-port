using System.Text;
using Tensorflow;
using Tensorflow.NumPy;

namespace PPOCRv2.TextRecognizer;

/// <summary>
///     Convert between text-label and text-index
/// </summary>
public abstract class BaseRecLabelDecode {
    private readonly string[] character;


    protected BaseRecLabelDecode(string characterDictPath = null, bool useSpaceChar = false) {
        var characterStr = "";
        string[] dictCharacter;
        if (characterDictPath == null) {
            characterStr = "0123456789abcdefghijklmnopqrstuvwxyz";
            dictCharacter = characterStr.ToCharArray().Select(c => c.ToString()).ToArray();
        } else {
            //using ... (character_dict_path, "rb") as fin:
            var lines = File.ReadLines(characterDictPath, Encoding.UTF8);
            foreach (var line in lines) {
                var line2 = line.Trim('\n').Trim('\r');
                characterStr += line2;
            }

            if (useSpaceChar) {
                characterStr += " ";
            }

            dictCharacter = characterStr.ToCharArray().Select(c => c.ToString()).ToArray();
        }

        // ReSharper disable once VirtualMemberCallInConstructor
        dictCharacter = add_special_char(dictCharacter);
        character = dictCharacter;
    }

    public abstract string[] add_special_char(string[] dictCharacter);

    /// <summary>
    ///     convert text-index into text-label
    /// </summary>
    /// <param name="textIndex"></param>
    /// <param name="textProb"></param>
    /// <param name="isRemoveDuplicate"></param>
    /// <returns></returns>
    public List<(string, float)> Decode(NDArray textIndex, NDArray textProb = null, bool isRemoveDuplicate = false) {
        var resultList = new List<(string, float)>();
        var ignoredTokens = get_ignored_tokens();
        var batchSize = (int)textIndex.shape[0];
        for (var batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            var charList = new List<string>();
            var confList = new List<float>();

            for (var idx = 0; idx < (int)textIndex[batchIdx].shape[0]; idx++) {
                if (ignoredTokens.Contains(textIndex[batchIdx][idx])) {
                    continue;
                }

                if (isRemoveDuplicate) {
                    //only for predict
                    if (idx > 0 && textIndex[batchIdx][idx - 1] == textIndex[
                            batchIdx][idx]) {
                        continue;
                    }
                }

                charList.Add(character[(int)textIndex[batchIdx][
                    idx]]);
                if (textProb is not null) {
                    confList.Add(textProb[batchIdx][idx]);
                } else {
                    confList.append(1);
                }
            }

            var text = string.Join("", charList);
            resultList.append((text, np.mean(new NDArray(confList.ToArray()))));
        }

        return resultList;
    }

    public List<int> get_ignored_tokens() {
        return new List<int>();
    }
}