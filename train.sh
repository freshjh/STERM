python train.py --name=MLP-forgeryEncoder --DisWeight 0.2 --niter 5 --seed 2026 --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone --gpu_ids 0 --batch_size 256 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100
