import os
import numpy as np
import cv2
import pandas as pd
from typing import Dict, List, Tuple, Any

class DatasetHandler:
    """Handler for loading and managing datasets"""
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.camera_data_path = os.path.join(dataset_path, 'camera')
        self.radar_data_path = os.path.join(dataset_path, 'radar')
        self.lidar_data_path = os.path.join(dataset_path, 'lidar')
        
    def download_dataset(self, dataset_name: str = 'kitti'):
        """
        Download the specified dataset
        """
        if dataset_name == 'kitti':
            print("Downloading KITTI dataset (simulated)...")
            # In a real implementation, this would download from the KITTI website
            self._create_dummy_data()
            
            # Add option to download actual KITTI data
            print("\nNote: For actual KITTI data, download from: http://www.cvlibs.net/datasets/kitti/")
            print("Place the data in the 'data/kitti' directory with the following structure:")
            print("  - data/kitti/image_2/ (for camera images)")
            print("  - data/kitti/velodyne/ (for LIDAR data)")
            print("  - data/kitti/label_2/ (for annotations)")
            
        elif dataset_name == 'nuscenes':
            print("Downloading nuScenes dataset (simulated)...")
            # In a real implementation, this would download from nuScenes
            self._create_dummy_data()
            
            # Add option to download actual nuScenes data
            print("\nNote: For actual nuScenes data, download from: https://www.nuscenes.org/download")
            print("Place the data in the 'data/nuscenes' directory")
            
        elif dataset_name == 'custom':
            print("Using custom dataset...")
            self._check_custom_dataset()
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _check_custom_dataset(self):
        """Check if custom dataset exists and create directory structure if not"""
        custom_dirs = ['camera', 'radar', 'lidar', 'annotations']
        custom_path = os.path.join(self.dataset_path, 'custom')
        
        # Create directory structure
        for directory in custom_dirs:
            dir_path = os.path.join(custom_path, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
                
        # Check if data exists
        camera_files = os.listdir(os.path.join(custom_path, 'camera')) if os.path.exists(os.path.join(custom_path, 'camera')) else []
        
        if not camera_files:
            print("No custom dataset found. Please add your data files to the following directories:")
            print(f"  - {os.path.join(custom_path, 'camera')}: Camera images (.jpg, .png)")
            print(f"  - {os.path.join(custom_path, 'radar')}: Radar data (.csv)")
            print(f"  - {os.path.join(custom_path, 'lidar')}: LIDAR point clouds (.npy, .bin)")
            print(f"  - {os.path.join(custom_path, 'annotations')}: Annotations (.json, .txt)")
            print("\nAlternatively, use the generated dummy data for testing")
            
            # Create minimal dummy data
            self._create_dummy_data(custom_path)
        else:
            print(f"Found {len(camera_files)} files in custom dataset")

    def _create_dummy_data(self, custom_path: str = None):
        """Create dummy data for demonstration purposes"""
        if custom_path:
            os.makedirs(custom_path, exist_ok=True)
            self.camera_data_path = os.path.join(custom_path, 'camera')
            self.radar_data_path = os.path.join(custom_path, 'radar')
            self.lidar_data_path = os.path.join(custom_path, 'lidar')
        
        os.makedirs(self.camera_data_path, exist_ok=True)
        os.makedirs(self.radar_data_path, exist_ok=True)
        os.makedirs(self.lidar_data_path, exist_ok=True)
        
        # Create dummy camera images (10 frames)
        for i in range(10):
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            # Add some sample objects (cars, pedestrians)
            if i % 2 == 0:
                # Draw a car-like rectangle
                cv2.rectangle(img, (100, 300), (300, 400), (0, 0, 255), -1)
            if i % 3 == 0:
                # Draw a pedestrian-like rectangle
                cv2.rectangle(img, (400, 350), (430, 450), (0, 255, 0), -1)
                
            cv2.imwrite(os.path.join(self.camera_data_path, f'frame_{i:04d}.jpg'), img)
        
        # Create dummy radar data (CSV format)
        radar_data = pd.DataFrame({
            'frame_id': list(range(10)),
            'object_id': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            'distance': [10.5, 5.2, 8.7, 9.8, 4.9, 8.2, 9.1, 4.5, 7.8, 8.5],
            'velocity': [2.1, 0.5, 1.2, 2.0, 0.3, 1.1, 1.9, 0.1, 1.0, 1.8],
            'angle': [0.1, 0.3, -0.2, 0.15, 0.25, -0.25, 0.2, 0.2, -0.3, 0.25]
        })
        radar_data.to_csv(os.path.join(self.radar_data_path, 'radar_data.csv'), index=False)
        
        # Create dummy lidar data (numpy arrays)
        for i in range(10):
            # Create random point cloud with 1000 points (x, y, z, intensity)
            points = np.random.rand(1000, 4)
            # Save as numpy file
            np.save(os.path.join(self.lidar_data_path, f'lidar_{i:04d}.npy'), points)
    
    def load_camera_frame(self, frame_id: int) -> np.ndarray:
        """Load a camera frame by ID"""
        frame_path = os.path.join(self.camera_data_path, f'frame_{frame_id:04d}.jpg')
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            raise FileNotFoundError(f"Camera frame {frame_id} not found")
    
    def load_radar_data(self, frame_id: int) -> Dict:
        """Load radar data for a specific frame"""
        radar_file = os.path.join(self.radar_data_path, 'radar_data.csv')
        if os.path.exists(radar_file):
            radar_df = pd.read_csv(radar_file)
            frame_data = radar_df[radar_df['frame_id'] == frame_id]
            
            result = {
                'objects': [],
                'distances': [],
                'velocities': [],
                'angles': []
            }
            
            for _, row in frame_data.iterrows():
                result['objects'].append(row['object_id'])
                result['distances'].append(row['distance'])
                result['velocities'].append(row['velocity'])
                result['angles'].append(row['angle'])
                
            return result
        else:
            raise FileNotFoundError("Radar data file not found")
    
    def load_lidar_data(self, frame_id: int) -> np.ndarray:
        """Load LIDAR point cloud for a specific frame"""
        lidar_file = os.path.join(self.lidar_data_path, f'lidar_{frame_id:04d}.npy')
        if os.path.exists(lidar_file):
            return np.load(lidar_file)
        else:
            raise FileNotFoundError(f"LIDAR data for frame {frame_id} not found")

    def get_dataset_frame(self, frame_id: int) -> Dict:
        """Get all sensor data for a specific frame"""
        try:
            camera = self.load_camera_frame(frame_id)
            radar = self.load_radar_data(frame_id)
            lidar = self.load_lidar_data(frame_id)
            
            return {
                'camera': camera,
                'radar': radar,
                'lidar': lidar,
                'frame_id': frame_id
            }
        except Exception as e:
            print(f"Error loading frame {frame_id}: {e}")
            return None

    def load_dataset_from_directory(self, directory_path: str):
        """
        Load a dataset from a specific directory
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        # Check if directory contains required data
        # This can be extended based on dataset format
        files = os.listdir(directory_path)
        image_files = [f for f in files if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            raise ValueError(f"No image files found in {directory_path}")
            
        print(f"Found {len(image_files)} image files in {directory_path}")
        
        # Update paths
        self.camera_data_path = directory_path
        
        return image_files

if __name__ == "__main__":
    # Test the dataset handler
    handler = DatasetHandler("../data")
    handler.download_dataset("kitti")
    
    # Load a test frame
    frame_data = handler.get_dataset_frame(0)
    if frame_data:
        print(f"Successfully loaded frame {frame_data['frame_id']}")
        print(f"Camera shape: {frame_data['camera'].shape}")
        print(f"Radar objects: {len(frame_data['radar']['objects'])}")
        print(f"LIDAR points: {frame_data['lidar'].shape}") 