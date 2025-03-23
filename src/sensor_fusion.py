import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    """Simple Kalman filter implementation for tracking objects"""
    def __init__(self, dt=0.1, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1):
        """
        Initialize Kalman filter
        
        Args:
            dt: time step
            u_x: acceleration in x direction
            u_y: acceleration in y direction
            std_acc: standard deviation of acceleration
            x_std_meas: standard deviation of measurement in x direction
            y_std_meas: standard deviation of measurement in y direction
        """
        # Define the state transition matrix (constant velocity model)
        self.A = np.array([
            [1, 0, dt, 0],   # x position
            [0, 1, 0, dt],   # y position
            [0, 0, 1, 0],    # x velocity
            [0, 0, 0, 1]     # y velocity
        ])
        
        # Define the control input matrix
        self.B = np.array([
            [dt**2/2, 0],
            [0, dt**2/2],
            [dt, 0],
            [0, dt]
        ])
        
        # Define measurement mapping matrix
        self.H = np.array([
            [1, 0, 0, 0],    # We measure only position
            [0, 1, 0, 0]
        ])
        
        # Initialize state and process noise covariance matrices
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * std_acc**2
        
        # Measurement noise covariance
        self.R = np.array([
            [x_std_meas**2, 0],
            [0, y_std_meas**2]
        ])
        
        # Initial state
        self.x = np.zeros((4, 1))
        
        # Initial state covariance
        self.P = np.eye(4) * 1000
        
        # Control input
        self.u = np.array([[u_x], [u_y]])
        
    def predict(self):
        """Predict the next state"""
        # Predict state
        self.x = self.A @ self.x + self.B @ self.u
        
        # Update covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x[:2]  # Return predicted position
    
    def update(self, z):
        """Update state based on measurement"""
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update error covariance
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:2]  # Return updated position
        

class SensorFusionSystem:
    """Implements sensor fusion for camera, radar, and LIDAR data"""
    def __init__(self):
        self.tracked_objects = {}  # Dictionary to store tracked objects with their Kalman filters
        self.next_object_id = 0    # Counter for assigning unique IDs to objects
        
    def _iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes
        box format: [x1, y1, x2, y2]
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # If the boxes don't overlap, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def _process_yolo_results(self, yolo_results):
        """Process YOLO detection results to extract bounding boxes and classes"""
        processed_results = []
        
        # Extract detections from YOLO results
        for result in yolo_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, box in enumerate(boxes):
                processed_results.append({
                    'box': box,
                    'class': int(classes[i]),
                    'confidence': confidences[i],
                    'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                    'source': 'camera'
                })
                
        return processed_results
    
    def _process_radar_data(self, radar_data):
        """Process radar data into a common format"""
        processed_results = []
        
        # Convert radar detections to bounding boxes (estimated)
        for i in range(len(radar_data['objects'])):
            # Use radar distance and angle to estimate object position
            distance = radar_data['distances'][i]
            angle = radar_data['angles'][i]
            velocity = radar_data['velocities'][i]
            
            # Convert polar to cartesian coordinates (simplified)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            # Create a small bounding box at the object location
            # This is a simplified estimation
            box_size = 50  # pixels
            box = [x - box_size/2, y - box_size/2, x + box_size/2, y + box_size/2]
            
            processed_results.append({
                'box': box,
                'class': -1,  # Radar doesn't provide class information
                'confidence': 0.9,  # High confidence for radar detection
                'center': [x, y],
                'distance': distance,
                'velocity': velocity,
                'source': 'radar'
            })
                
        return processed_results
    
    def _process_lidar_data(self, lidar_data):
        """Process LIDAR point cloud data into a common format"""
        processed_results = []
        
        # Simplified: use clustering to find objects in point cloud
        # In a real implementation, this would use more sophisticated algorithms
        
        # Example: use a simple threshold-based clustering
        if len(lidar_data) > 0:
            # Extract x, y coordinates
            points = lidar_data[:, :2]
            
            # Use DBSCAN clustering
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
                labels = clustering.labels_
                
                # Process clusters
                unique_labels = set(labels)
                for label in unique_labels:
                    if label == -1:  # Skip noise
                        continue
                        
                    # Get points in cluster
                    cluster_points = points[labels == label]
                    
                    # Calculate min/max coordinates for bounding box
                    min_x, min_y = np.min(cluster_points, axis=0)
                    max_x, max_y = np.max(cluster_points, axis=0)
                    
                    # Create bounding box
                    box = [min_x, min_y, max_x, max_y]
                    center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
                    
                    # Calculate distance from origin (simplified)
                    distance = np.sqrt(center[0]**2 + center[1]**2)
                    
                    processed_results.append({
                        'box': box,
                        'class': -1,  # LIDAR doesn't provide class information
                        'confidence': 0.85,
                        'center': center,
                        'distance': distance,
                        'source': 'lidar'
                    })
            except ImportError:
                print("DBSCAN not available, skipping LIDAR processing")
                
        return processed_results
    
    def _associate_detections(self, camera_objects, radar_objects, lidar_objects):
        """Associate objects from different sensors based on spatial proximity"""
        # Combine all detections
        all_objects = camera_objects + radar_objects + lidar_objects
        
        # If no objects, return empty list
        if not all_objects:
            return []
            
        # Group objects by spatial proximity
        grouped_objects = []
        used_indices = set()
        
        for i, obj1 in enumerate(all_objects):
            if i in used_indices:
                continue
                
            group = [obj1]
            used_indices.add(i)
            
            for j, obj2 in enumerate(all_objects):
                if j in used_indices or i == j:
                    continue
                    
                # Check if objects are close
                if obj1.get('box') and obj2.get('box'):
                    iou_score = self._iou(obj1['box'], obj2['box'])
                    if iou_score > 0.1:  # Low threshold for association
                        group.append(obj2)
                        used_indices.add(j)
                elif 'center' in obj1 and 'center' in obj2:
                    # Calculate Euclidean distance between centers
                    dist = np.sqrt((obj1['center'][0] - obj2['center'][0])**2 + 
                                  (obj1['center'][1] - obj2['center'][1])**2)
                    if dist < 100:  # Distance threshold in pixels
                        group.append(obj2)
                        used_indices.add(j)
            
            grouped_objects.append(group)
            
        # For any remaining objects, add them as single-element groups
        for i, obj in enumerate(all_objects):
            if i not in used_indices:
                grouped_objects.append([obj])
                used_indices.add(i)
                
        return grouped_objects
    
    def _fuse_object_group(self, group):
        """Fuse information from a group of associated objects"""
        fused_object = {
            'box': None,
            'class': -1,
            'confidence': 0,
            'center': [0, 0],
            'distance': None,
            'velocity': None,
            'sources': []
        }
        
        # Prioritize camera for classification
        camera_objects = [obj for obj in group if obj['source'] == 'camera']
        radar_objects = [obj for obj in group if obj['source'] == 'radar']
        lidar_objects = [obj for obj in group if obj['source'] == 'lidar']
        
        # Add source information
        for obj in group:
            fused_object['sources'].append(obj['source'])
            
        # Use camera for class and box if available
        if camera_objects:
            best_camera_obj = max(camera_objects, key=lambda x: x['confidence'])
            fused_object['box'] = best_camera_obj['box']
            fused_object['class'] = best_camera_obj['class']
            fused_object['confidence'] = best_camera_obj['confidence']
        elif lidar_objects:
            # If no camera, use LIDAR for box
            best_lidar_obj = lidar_objects[0]
            fused_object['box'] = best_lidar_obj['box']
            
        # Use radar for distance and velocity if available
        if radar_objects:
            best_radar_obj = radar_objects[0]
            fused_object['distance'] = best_radar_obj.get('distance')
            fused_object['velocity'] = best_radar_obj.get('velocity')
        elif lidar_objects:
            # If no radar, use LIDAR for distance
            best_lidar_obj = lidar_objects[0]
            fused_object['distance'] = best_lidar_obj.get('distance')
            
        # Calculate weighted center position
        weights = []
        centers = []
        
        for obj in group:
            if obj['source'] == 'camera':
                weight = 1.0
            elif obj['source'] == 'radar':
                weight = 1.5  # Radar is more accurate for position
            elif obj['source'] == 'lidar':
                weight = 1.2  # LIDAR is also accurate
            else:
                weight = 1.0
                
            weights.append(weight)
            centers.append(obj['center'])
            
        if centers:
            total_weight = sum(weights)
            weighted_x = sum(w * c[0] for w, c in zip(weights, centers)) / total_weight
            weighted_y = sum(w * c[1] for w, c in zip(weights, centers)) / total_weight
            fused_object['center'] = [weighted_x, weighted_y]
            
        return fused_object
    
    def _update_tracks(self, fused_objects):
        """Update object tracks using Kalman filter"""
        # If no tracked objects yet, initialize new tracks
        if not self.tracked_objects:
            for obj in fused_objects:
                obj_id = self.next_object_id
                self.next_object_id += 1
                
                # Initialize Kalman filter for this object
                kf = KalmanFilter()
                if 'center' in obj:
                    kf.x = np.array([[obj['center'][0]], [obj['center'][1]], [0], [0]])
                    
                self.tracked_objects[obj_id] = {
                    'filter': kf,
                    'data': obj,
                    'age': 1,
                    'unmatched_count': 0
                }
            return list(self.tracked_objects.items())
            
        # Predict new locations for all tracked objects
        for obj_id, track in self.tracked_objects.items():
            track['filter'].predict()
            
        # Calculate cost matrix for matching
        cost_matrix = np.zeros((len(self.tracked_objects), len(fused_objects)))
        track_ids = list(self.tracked_objects.keys())
        
        for i, obj_id in enumerate(track_ids):
            track = self.tracked_objects[obj_id]
            predicted_pos = track['filter'].x[:2].flatten()
            
            for j, obj in enumerate(fused_objects):
                if 'center' in obj:
                    # Calculate distance between predicted position and detected position
                    distance = np.sqrt((predicted_pos[0] - obj['center'][0])**2 + 
                                      (predicted_pos[1] - obj['center'][1])**2)
                    cost_matrix[i, j] = distance
                else:
                    cost_matrix[i, j] = 1000  # Large cost for unlikely matches
                    
        # Perform assignment using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Mark all tracks as unmatched initially
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_detections = set(range(len(fused_objects)))
        
        # Process matches
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] > 100:  # Maximum distance threshold
                continue
                
            track_id = track_ids[row]
            track = self.tracked_objects[track_id]
            obj = fused_objects[col]
            
            # Update Kalman filter with new measurement
            if 'center' in obj:
                track['filter'].update(np.array([obj['center']]).T)
                
            # Update track data
            track['data'] = obj
            track['age'] += 1
            track['unmatched_count'] = 0
            
            # Remove from unmatched sets
            unmatched_tracks.discard(row)
            unmatched_detections.discard(col)
            
        # Handle unmatched detections - create new tracks
        for idx in unmatched_detections:
            obj = fused_objects[idx]
            obj_id = self.next_object_id
            self.next_object_id += 1
            
            # Initialize Kalman filter
            kf = KalmanFilter()
            if 'center' in obj:
                kf.x = np.array([[obj['center'][0]], [obj['center'][1]], [0], [0]])
                
            self.tracked_objects[obj_id] = {
                'filter': kf,
                'data': obj,
                'age': 1,
                'unmatched_count': 0
            }
            
        # Handle unmatched tracks - increment unmatched count
        for idx in unmatched_tracks:
            track_id = track_ids[idx]
            self.tracked_objects[track_id]['unmatched_count'] += 1
            
            # Remove track if it hasn't been matched for too long
            if self.tracked_objects[track_id]['unmatched_count'] > 5:
                del self.tracked_objects[track_id]
                
        # Return current tracks
        return list(self.tracked_objects.items())
    
    def process_sensor_data(self, camera_data, radar_data, lidar_data):
        """
        Process and fuse data from multiple sensors
        """
        # Process data from each sensor
        camera_objects = self._process_yolo_results(camera_data) if camera_data else []
        radar_objects = self._process_radar_data(radar_data) if radar_data else []
        lidar_objects = self._process_lidar_data(lidar_data) if lidar_data is not None else []
        
        # Associate objects from different sensors
        grouped_objects = self._associate_detections(camera_objects, radar_objects, lidar_objects)
        
        # Fuse associated objects
        fused_objects = [self._fuse_object_group(group) for group in grouped_objects]
        
        # Update tracking
        tracks = self._update_tracks(fused_objects)
        
        return {
            'objects': fused_objects,
            'tracks': tracks
        }
        
    def get_collision_risks(self, fused_data):
        """
        Calculate collision risks based on fused sensor data
        """
        risks = []
        
        for obj_id, track in fused_data['tracks']:
            obj = track['data']
            
            # Skip if no distance information
            if not obj.get('distance'):
                continue
                
            # Basic collision risk calculation based on distance and velocity
            distance = obj['distance']
            velocity = obj.get('velocity', 0)
            
            # Ensure velocity is a number, default to 0 if None
            if velocity is None:
                velocity = 0
                
            # Time to collision (seconds) - simplified calculation
            ttc = float('inf') if velocity <= 0 else distance / velocity
            
            # Risk score based on time to collision
            if ttc < 1.0:
                risk_score = 1.0  # Imminent collision
            elif ttc < 2.0:
                risk_score = 0.8  # High risk
            elif ttc < 3.0:
                risk_score = 0.5  # Medium risk
            elif ttc < 5.0:
                risk_score = 0.3  # Low risk
            else:
                risk_score = 0.1  # Very low risk
                
            # Add to risks list
            risks.append({
                'object_id': obj_id,
                'risk_score': risk_score,
                'ttc': ttc,
                'distance': distance,
                'velocity': velocity
            })
            
        return risks 