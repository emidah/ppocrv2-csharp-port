// ReSharper disable InconsistentNaming
namespace PaddleOCR; 

public record class Args {
    public string image_path;
    public string det_algorithm = "DB";
    public string det_model_dir;
    public float det_limit_side_len = 960;
    public string det_limit_type = "max";

    public float det_db_thresh = 0.3f;
    public float det_db_box_thresh = 0.6f;
    public float det_db_unclip_ratio = 1.5f;
    public int max_batch_size = 10;
    public bool use_dilation = false;
    public string det_db_score_mode = "fast";

    public string rec_algorithm = "CRNN";
    public string rec_model_dir;
    public string rec_image_shape = "3, 32, 320";
    public int rec_batch_num = 6;
    public int max_text_length = 25;
    public string rec_char_dict_path = "./doc/ppocr_keys_v1.txt";
    public bool use_space_char = true;
    public string vis_font_path = "./utils/doc/fonts/simfang.ttf";
    public float drop_score = 0.5f;

    public bool use_angle_cls = true;
    public string cls_model_dir;
    public string cls_image_shape = "3, 48, 192";
    public int[] label_list = {0, 180};
    public int cls_batch_num = 6;
    public float cls_thresh = 0.9f;

    public bool use_paddle_predict = false;
    
}