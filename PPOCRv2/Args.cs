

namespace PPOCRv2;

public record Args {
    public int ClsBatchNum = 6;
    public string ClsImageShape = "3, 48, 192";
    public string ClsModelDir;
    public float ClsThresh = 0.9f;
    public float DetDbBoxThresh = 0.6f;
    public string DetDbScoreMode = "fast";

    public float DetDbThresh = 0.3f;
    public float DetDbUnclipRatio = 1.5f;
    public float DetLimitSideLen = 1920;
    public string DetLimitType = "max";
    public string DetModelDir;
    public float DropScore = 0.5f;
    public string ImagePath;
    public int[] LabelList = { 0, 180 };

    public int RecBatchNum = 6;
    public string RecCharDictPath = "./doc/ppocr_keys_v1.txt";
    public string RecImageShape = "3, 32, 320";
    public string RecModelDir;

    public bool UseAngleCls = true;
    public bool UseDilation = false;
    public bool UseSpaceChar = true;
}