python hierarchy_trainer_vcycle_v3.py --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 8
python hierarchy_trainer_vcycle_v2.py --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 8
python hierarchy_trainer_vcycle.py --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 8

exit 0


python hierarchy_trainer_simple.py default --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 8
python hierarchy_trainer_simple.py default --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 8 --use_coarse_to_fine



python hierarchy_trainer_vcycle.py --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 8


python hierarchy_trainer_simple.py default --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 4
python hierarchy_trainer_simple.py default --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 4 --use_coarse_to_fine



python hierarchy_trainer_vcycle.py --hierarchy_path ./results/simple_trainer_original/garden_type_colmap_factor_8/ply/point_cloud_29999_hierarchy/hierarchy.pt --data_dir ./dataset/360_v2/garden/ --data_factor 4