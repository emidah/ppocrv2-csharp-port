namespace PaddleOCR; 

public class Args {
    public string det_db_thresh;
    public string det_db_box_thresh;
    public string det_db_unclip_ratio;
    public bool use_dilation;
    public string det_db_score_mode;
    public string det_model_dir;
    public bool use_paddle_predict;
    public int det_limit_side_len;
    public string det_limit_type;
    public string image_path;
    public bool use_angle_cls;
}