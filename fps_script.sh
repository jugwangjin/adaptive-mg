python simple_trainer_custom_train.py default --data_dir ../tat_dataproecssing/dataset_3/truck/ --dataset_type nerf --data_factor 1 --result_dir ./results/simple_trainer_original/truck_type_nerf_factor_1  --colmap_dir ../tat_dataproecssing/dataset_3/truck/sparse/0

python simple_trainer_custom_train.py default --data_dir ../tat_dataproecssing/dataset_3/horse/ --dataset_type nerf --data_factor 1 --result_dir ./results/simple_trainer_original/horse_type_nerf_factor_1 --colmap_dir ../tat_dataproecssing/dataset_3/horse/sparse/0

python simple_trainer_custom_train.py default --data_dir ../tat_dataproecssing/dataset_3/truck/ --dataset_type nerf --data_factor 1 --custom_train_json transforms_fps_50.json --result_dir ./results/simple_trainer_original/truck_type_nerf_factor_1_fps50 --colmap_dir ../tat_dataproecssing/dataset_3/truck/sparse/0

python simple_trainer_custom_train.py default --data_dir ../tat_dataproecssing/dataset_3/horse/ --dataset_type nerf --data_factor 1 --custom_train_json transforms_fps_50.json --result_dir ./results/simple_trainer_original/horse_type_nerf_factor_1_fps50 --colmap_dir ../tat_dataproecssing/dataset_3/horse/sparse/0