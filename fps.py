import json
import argparse
import numpy as np
from pathlib import Path


def fps(points, n_samples, init_selected=0):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]
 
    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]
 
    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]
 
    # Select a point from points by its index, save it
    selected = init_selected
    sample_inds[0] = points_left[selected]
 
    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]
 
    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]
 
        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]
 
        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
 
        # Update points_left
        points_left = np.delete(points_left, selected)
 
    return sample_inds


def sample_cameras_fps(input_json: str, output_json: str, n_samples: int, init_selected: int = 0):
    """
    Sample cameras from NeRF-style JSON using Farthest Point Sampling (FPS).
    
    Args:
        input_json: Path to input transforms JSON file (e.g., transforms_train.json)
        output_json: Path to output JSON file with sampled cameras
        n_samples: Number of cameras to sample
        init_selected: Index of initial camera to select (default: 0)
    """
    # Load input JSON
    print(f"Loading cameras from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    frames = data.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"No frames found in {input_json}")
    
    print(f"Found {len(frames)} cameras in input JSON")
    
    # Extract camera positions from transform_matrix
    # transform_matrix is a 4x4 camera-to-world matrix
    # Camera position is in the last column (first 3 elements)
    camera_positions = []
    for frame in frames:
        transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
        # Extract camera position (last column, first 3 elements)
        cam_pos = transform_matrix[:3, 3]
        camera_positions.append(cam_pos)
    
    camera_positions = np.array(camera_positions)  # [N, 3]
    print(f"Extracted camera positions: shape {camera_positions.shape}")
    
    # Clamp n_samples to available cameras
    n_samples = min(n_samples, len(frames))
    if n_samples < len(frames):
        print(f"Sampling {n_samples} cameras from {len(frames)} using FPS...")
        # Apply FPS sampling
        sampled_indices = fps(camera_positions, n_samples, init_selected=init_selected)
        print(f"FPS sampling complete. Selected {len(sampled_indices)} cameras")
    else:
        print(f"n_samples ({n_samples}) >= total cameras ({len(frames)}), using all cameras")
        # If n_samples >= total, just use all frames
        sampled_indices = np.arange(len(frames))
    
    # Create new JSON with sampled frames
    sampled_frames = [frames[i] for i in sampled_indices]
    
    output_data = {
        "camera_angle_x": data.get("camera_angle_x"),
        "frames": sampled_frames
    }
    
    # Copy other metadata if present
    for key in data.keys():
        if key not in ["camera_angle_x", "frames"]:
            output_data[key] = data[key]
    
    # Save output JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(sampled_frames)} sampled cameras to {output_json}")
    print(f"  Original cameras: {len(frames)}")
    print(f"  Sampled cameras: {len(sampled_frames)}")
    print(f"  Sampling ratio: {len(sampled_frames) / len(frames):.2%}")
    
    return sampled_indices


def main():
    parser = argparse.ArgumentParser(
        description="Sample cameras from NeRF-style JSON using Farthest Point Sampling (FPS)"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input transforms JSON file (e.g., transforms_train.json)"
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Path to output JSON file with sampled cameras"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of cameras to sample"
    )
    parser.add_argument(
        "--init_selected",
        type=int,
        default=0,
        help="Index of initial camera to select (default: 0)"
    )
    
    args = parser.parse_args()
    
    sample_cameras_fps(
        input_json=args.input_json,
        output_json=args.output_json,
        n_samples=args.n_samples,
        init_selected=args.init_selected
    )


if __name__ == "__main__":
    main() 