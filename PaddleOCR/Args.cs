// ReSharper disable InconsistentNaming

namespace PaddleOCR;

public record class Args {
    public int cls_batch_num = 6;
    public string cls_image_shape = "3, 48, 192";
    public string cls_model_dir;
    public float cls_thresh = 0.9f;
    public float det_db_box_thresh = 0.6f;
    public string det_db_score_mode = "fast";

    public float det_db_thresh = 0.3f;
    public float det_db_unclip_ratio = 1.5f;
    public float det_limit_side_len = 2*960;
    public string det_limit_type = "max";
    public string det_model_dir;
    public float drop_score = 0.5f;
    public string image_path;
    public int[] label_list = { 0, 180 };

    public int rec_batch_num = 6;
    public string rec_char_dict_path = "./doc/ppocr_keys_v1.txt";
    public string rec_image_shape = "3, 32, 320";
    public string rec_model_dir;

    public bool use_angle_cls = true;
    public bool use_dilation = false;

    public bool use_paddle_predict = false;
    public bool use_space_char = true;
}