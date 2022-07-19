using System.Text;
using Tensorflow;
using Tensorflow.NumPy;

namespace PaddleOCR;

public abstract class BaseRecLabelDecode {
    private string beg_str;

    private string end_str;

    private string character_str;

    private readonly Dictionary<string, int> dict;

    private readonly string[] character;
    //""" Convert between text-label and text-index """

    protected BaseRecLabelDecode(string character_dict_path=null, bool use_space_char=false){
        this.beg_str = "sos";
        this.end_str = "eos";

        this.character_str = "";
        string[] dict_character;
        if (character_dict_path == null) {
            this.character_str = "0123456789abcdefghijklmnopqrstuvwxyz";
            dict_character = this.character_str.ToCharArray().Select(c => c.ToString()).ToArray();
        } else {
            //using ... (character_dict_path, "rb") as fin:
            var lines = File.ReadLines(character_dict_path, Encoding.UTF8);
            foreach (var line in lines) {
                var line2 = line.Trim('\n').Trim('\r');
                this.character_str += line2;
            }

            if (use_space_char) {
                this.character_str += " ";
            }

            dict_character = this.character_str.ToCharArray().Select(c => c.ToString()).ToArray();
        }

        // ReSharper disable once VirtualMemberCallInConstructor
        dict_character = this.add_special_char(dict_character);
        this.dict = new Dictionary<string, int>(){ };
        for (int i = 0; i < dict_character.Length; i++) {
            var ch = dict_character[i];
            this.dict[ch] = i;
        }

        this.character = dict_character;
    }

    public abstract string[] add_special_char(string[] dict_character);
        
    public List<(string, float)> decode(NDArray text_index, NDArray text_prob=null, bool is_remove_duplicate=false){
        //""" convert text-index into text-label. """
        var result_list = new List<(string,float)>();
        var ignored_tokens = this.get_ignored_tokens();
        var batch_size = (int)text_index.shape[0];
        for (var batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            var char_list = new List<string>();
            var conf_list = new List<float>();

            for (var idx = 0; idx < (int)text_index[batch_idx].shape[0]; idx++) {
                if(ignored_tokens.Contains(text_index[batch_idx][idx])){
                    continue;
                }
                if (is_remove_duplicate) {
//# only for predict
                    if (idx > 0 && text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]) {
                        continue;
                    }
                }

                char_list.Add(this.character[(int)text_index[batch_idx][
                    idx]]);
                if (text_prob is not null) {
                    conf_list.Add(text_prob[batch_idx][idx]);
                } else {
                    conf_list.append(1);
                }
            }

            var text =  string.Join("", char_list);
            result_list.append((text, np.mean(new NDArray(conf_list.ToArray()))));
        }

        return result_list;
    }

    public List<int> get_ignored_tokens() {
        return new List<int>();
    }
}