CUDA_VISIBLE_DEVICES=0 python demo_yolo.py \
--center-num 25 \
--anchor-num 7 \
--anchor-k 6 \
--anchor-w 50 \
--anchor-z 20 \
--grid-size 8 \
--all-points-num 256000 \
--group-num 512 \
--local-k 10 \
--ratio 8 \
--input-h 360 \
--input-w 640 \
--local-thres 0.01 \
--heatmap-thres 0.01 \
--checkpoint '/home/johnny/goal_grasp_projects/mine_HGGD_ws/src/HGGD/weights/middle' \
--rgb-path '/home/johnny/goal_grasp_projects/mine_HGGD_ws/src/HGGD/images/test/seen/seen1.png' \
--depth-path '/home/johnny/goal_grasp_projects/mine_HGGD_ws/src/HGGD/images/test/seen/seen1D.png'


