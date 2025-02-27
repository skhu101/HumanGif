exp_folder=$1
# pred_folder="${2:-${exp_folder}/pred_gs1.5_scale-cond1.0-ref1.0}"
# gt_folder="${3:-${exp_folder}/gt}"
pred_folder="${2:-${exp_folder}/pred_np}"
gt_folder="${3:-${exp_folder}/gt_np}"

echo "pred_folder" ${pred_folder}
echo "gt_folder" ${gt_folder}
# # L1 SSIM LPIPS and PSNR
# python  tool/metrics/metric_center.py --root_dir /group2/3d/shoukanghu/champ --path_gen ${pred_folder}/ --path_gt ${gt_folder} --type l1 psnr ssim lpips --write_metric_to ${exp_folder}/metrics_np_l1_ssim_lpips_psnr.json
# python  tool/metrics/metric_center.py --root_dir /group/3d/shoukanghu/DisCo --path_gen ${pred_folder}/ --path_gt ${gt_folder} --type psnr  --write_metric_to ${exp_folder}/metrics_l1_ssim_lpips_psnr.json

# Pytorch-FID
# python -m pytorch_fid ${pred_folder} ${gt_folder} --device cuda:0

# DEPRECITED due to unstable gif generation codes
#  Generate GIFs of 16 frames, 3 fps

#  FVD eval
# the root dir should be the dir containing the fvd pretrain model (resnet-50-kinetics and i3d)
# python  tool/metrics/metric_center.py --root_dir /group/3d/shoukanghu/champ --path_gen ${pred_folder}/ --path_gt ${gt_folder}/ --type fid-vid fvd  --write_metric_to ${exp_folder}/metrics_fid-vid_fvd.json --number_sample_frames 16 --sample_duration 16
python  tool/metrics/metric_center.py --root_dir /group2/3d/shoukanghu/champ --path_gen ${pred_folder}/ --path_gt ${gt_folder}/ --type fid-vid fvd  --write_metric_to ${exp_folder}/metrics_fid-vid_fvd.json --number_sample_frames 24 --sample_duration 24