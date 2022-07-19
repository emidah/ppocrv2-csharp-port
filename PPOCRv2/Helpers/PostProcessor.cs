using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace PPOCRv2.Helpers;

internal static class PostProcessor {
    public static (NDArray dt_boxes, IList<(string, float)> rec_res) PostProcess(Args flags, NDArray dtBoxes,
        IList<(string, float)> recRes) {
        var (filterBoxes, filterRecRes) = (new List<NDArray>(), new List<(string, float)>());
        foreach (var (box, recResult) in zip(dtBoxes, recRes)) {
            var (text, score) = recResult;
            if (score >= flags.drop_score) {
                filterBoxes.append(box);
                filterRecRes.append(recResult);
            }
        }

        return (NdArrayExtensions.FromArray(filterBoxes.ToArray()), filterRecRes);
    }
}