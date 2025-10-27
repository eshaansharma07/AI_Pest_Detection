import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Simulated ML libraries (replace with actual tensorflow/pytorch imports)
# from tensorflow.keras.models import load_model
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC


@dataclass
class SensorData:
    """Container for environmental sensor data"""
    temperature: float
    humidity: float
    soil_moisture: float
    light_intensity: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DetectionResult:
    """Container for detection results"""
    disease_status: str
    pest_presence: str
    recommended_action: str
    confidence_score: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalize and enhance image"""
        # Resize image
        resized = cv2.resize(image, self.target_size)
        
        # Noise reduction using Gaussian blur
        denoised = cv2.GaussianBlur(resized, (5, 5), 0)
        
        # Contrast enhancement using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Normalize to [0, 1]
        normalized = enhanced.astype(np.float32) / 255.0
        
        return normalized


class FeatureExtractor:
    """Extracts features from images and sensor data"""
    
    @staticmethod
    def extract_visual_features(image: np.ndarray) -> Dict:
        """Extract color, texture, and edge features from image"""
        features = {}
        
        # Color features (mean and std per channel)
        features['color_mean'] = np.mean(image, axis=(0, 1))
        features['color_std'] = np.std(image, axis=(0, 1))
        
        # Texture features using Local Binary Pattern simulation
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        features['texture_variance'] = np.var(gray)
        features['texture_entropy'] = -np.sum(
            (gray / 255.0) * np.log2((gray / 255.0) + 1e-10)
        ) / (gray.shape[0] * gray.shape[1])
        
        # Edge features using Canny
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    @staticmethod
    def extract_sensor_features(sensor_data: SensorData) -> Dict:
        """Derive environmental indicators from sensor data"""
        features = {
            'temp_mean': sensor_data.temperature,
            'humidity_mean': sensor_data.humidity,
            'soil_moisture_mean': sensor_data.soil_moisture,
            'light_intensity_mean': sensor_data.light_intensity,
            # Derived features
            'temp_humidity_ratio': sensor_data.temperature / (sensor_data.humidity + 1e-5),
            'moisture_light_product': sensor_data.soil_moisture * sensor_data.light_intensity,
        }
        return features


class DiseaseDetectionCNN:
    """Simulated CNN model for disease detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # In production: self.model = load_model(model_path)
        self.disease_classes = ['Healthy', 'Leaf_Blight', 'Rust', 'Powdery_Mildew', 'Bacterial_Spot']
    
    def predict(self, image_features: np.ndarray) -> Tuple[str, float]:
        """Predict disease from image features"""
        # Simulated prediction (replace with actual model.predict)
        confidence = np.random.uniform(0.7, 0.99)
        disease_idx = np.random.randint(0, len(self.disease_classes))
        
        return self.disease_classes[disease_idx], confidence


class PestPredictionModel:
    """Simulated ML model for pest prediction using sensor data"""
    
    def __init__(self, model_type: str = 'RandomForest'):
        self.model_type = model_type
        # In production: 
        # if model_type == 'RandomForest':
        #     self.model = RandomForestClassifier()
        # else:
        #     self.model = SVC()
        self.pest_levels = ['None', 'Low', 'Medium', 'High']
    
    def predict(self, sensor_features: Dict) -> Tuple[str, float]:
        """Predict pest presence from sensor features"""
        # Simulated prediction (replace with actual model.predict)
        confidence = np.random.uniform(0.65, 0.95)
        
        # Simple rule-based simulation
        temp = sensor_features.get('temp_mean', 25)
        humidity = sensor_features.get('humidity_mean', 50)
        
        if temp > 30 and humidity > 70:
            pest_level = 'High'
        elif temp > 25 and humidity > 60:
            pest_level = 'Medium'
        else:
            pest_level = 'Low'
        
        return pest_level, confidence


class KnowledgeBase:
    """Stores detection history and outcomes for continuous learning"""
    
    def __init__(self, db_path: str = 'knowledge_base.json'):
        self.db_path = db_path
        self.records = []
    
    def store(self, image_path: str, sensor_data: SensorData, 
              result: DetectionResult, actual_outcome: Optional[str] = None):
        """Store detection data and outcomes"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'sensor_data': {
                'temperature': sensor_data.temperature,
                'humidity': sensor_data.humidity,
                'soil_moisture': sensor_data.soil_moisture,
                'light_intensity': sensor_data.light_intensity,
            },
            'prediction': {
                'disease_status': result.disease_status,
                'pest_presence': result.pest_presence,
                'recommended_action': result.recommended_action,
                'confidence': result.confidence_score,
            },
            'actual_outcome': actual_outcome
        }
        self.records.append(record)
    
    def save(self):
        """Persist knowledge base to disk"""
        with open(self.db_path, 'w') as f:
            json.dump(self.records, f, indent=2)
    
    def load(self):
        """Load knowledge base from disk"""
        try:
            with open(self.db_path, 'r') as f:
                self.records = json.load(f)
        except FileNotFoundError:
            self.records = []


class DiseaseDetectionSystem:
    """Main system implementing Algorithm 1"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.cnn_model = DiseaseDetectionCNN()
        self.pest_model = PestPredictionModel()
        self.knowledge_base = KnowledgeBase()
    
    def impute_missing_values(self, sensor_data: SensorData) -> SensorData:
        """Perform data imputation for missing sensor values"""
        # Use historical mean or interpolation
        if sensor_data.temperature is None:
            sensor_data.temperature = 25.0  # Default/historical mean
        if sensor_data.humidity is None:
            sensor_data.humidity = 60.0
        if sensor_data.soil_moisture is None:
            sensor_data.soil_moisture = 40.0
        if sensor_data.light_intensity is None:
            sensor_data.light_intensity = 500.0
        return sensor_data
    
    def analyze_decision(self, disease_status: str, pest_presence: str,
                        disease_conf: float, pest_conf: float) -> Tuple[str, bool]:
        """Step 6: Decision Analysis"""
        alert_needed = False
        
        if disease_status != 'Healthy' and disease_conf > 0.7:
            alert_needed = True
            if disease_status in ['Leaf_Blight', 'Rust']:
                action = "Apply biological pesticide (Bacillus subtilis)"
            elif disease_status in ['Powdery_Mildew']:
                action = "Apply sulfur-based fungicide and improve air circulation"
            else:
                action = "Isolate infected area and consult agronomist"
        
        elif pest_presence in ['High', 'Medium'] and pest_conf > 0.65:
            alert_needed = True
            if pest_presence == 'High':
                action = "Apply neem-based pesticide immediately and monitor closely"
            else:
                action = "Deploy pheromone traps and increase monitoring frequency"
        
        else:
            action = "No action required - Continue regular monitoring"
        
        return action, alert_needed
    
    def send_alert(self, message: str):
        """Send alert notification to farmer"""
        print(f"\n{'='*60}")
        print(f"🚨 ALERT NOTIFICATION")
        print(f"{'='*60}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Message: {message}")
        print(f"{'='*60}\n")
    
    def process(self, crop_image: np.ndarray, sensor_data: SensorData,
                image_path: str = 'current_image.jpg') -> DetectionResult:
        """
        Main processing pipeline (Algorithm 1)
        
        Args:
            crop_image: Input crop image
            sensor_data: Environmental sensor readings
            image_path: Path/identifier for the image
            
        Returns:
            DetectionResult with disease status, pest presence, and recommendations
        """
        print("=" * 60)
        print("AI-BASED DISEASE DETECTION AND PEST CONTROL SYSTEM")
        print("=" * 60)
        
        # Step 1: Data Collection (already provided as input)
        print("\n[Step 1] Data Collection: ✓ Complete")
        
        # Step 2: Preprocessing
        print("[Step 2] Preprocessing...")
        processed_image = self.preprocessor.preprocess(crop_image)
        sensor_data = self.impute_missing_values(sensor_data)
        print("         - Image resized, denoised, and enhanced")
        print("         - Sensor data normalized and cleaned")
        
        # Step 3: Feature Extraction
        print("[Step 3] Feature Extraction...")
        visual_features = self.feature_extractor.extract_visual_features(processed_image)
        sensor_features = self.feature_extractor.extract_sensor_features(sensor_data)
        print(f"         - Extracted {len(visual_features)} visual features")
        print(f"         - Extracted {len(sensor_features)} sensor features")
        
        # Step 4: Model Selection (already initialized)
        print("[Step 4] Model Selection: ✓ Models loaded")
        
        # Step 5: Prediction
        print("[Step 5] Running Predictions...")
        disease_status, disease_conf = self.cnn_model.predict(processed_image)
        pest_presence, pest_conf = self.pest_model.predict(sensor_features)
        print(f"         - Disease: {disease_status} (confidence: {disease_conf:.2%})")
        print(f"         - Pest Level: {pest_presence} (confidence: {pest_conf:.2%})")
        
        # Step 6: Decision Analysis
        print("[Step 6] Decision Analysis...")
        recommended_action, alert_needed = self.analyze_decision(
            disease_status, pest_presence, disease_conf, pest_conf
        )
        
        if alert_needed:
            self.send_alert(f"Detected: {disease_status} | Pest Level: {pest_presence}")
        
        # Step 7: Continuous Learning
        print("[Step 7] Continuous Learning...")
        avg_confidence = (disease_conf + pest_conf) / 2
        result = DetectionResult(
            disease_status=disease_status,
            pest_presence=pest_presence,
            recommended_action=recommended_action,
            confidence_score=avg_confidence
        )
        self.knowledge_base.store(image_path, sensor_data, result)
        print("         - Data stored in knowledge base")
        
        # Step 8: Output
        print("\n[Step 8] Results:")
        print(f"         Disease Status: {result.disease_status}")
        print(f"         Pest Presence: {result.pest_presence}")
        print(f"         Recommended Action: {result.recommended_action}")
        print(f"         Confidence: {result.confidence_score:.2%}")
        print("=" * 60)
        
        return result


class DroneScanner:
    """Implements Algorithm 2: Drone-Based Field Scanning"""
    
    def __init__(self):
        self.geotagged_images = []
    
    def scan_field(self, flight_path: List[Tuple[float, float]], 
                   crop_field_area: str) -> List[Dict]:
        """
        Execute drone scanning routine
        
        Args:
            flight_path: List of GPS coordinates (lat, lon)
            crop_field_area: Identifier for the crop field
            
        Returns:
            List of geotagged crop images with metadata
        """
        print("\n" + "=" * 60)
        print("DRONE-BASED FIELD SCANNING AND MONITORING")
        print("=" * 60)
        print(f"Field Area: {crop_field_area}")
        print(f"Flight Path Points: {len(flight_path)}")
        print("-" * 60)
        
        # Initialize drone cameras
        print("\n[Initialize] Drone cameras (RGB + Multispectral) ready")
        
        geotagged_images = []
        
        # Scan each GPS point
        for idx, (lat, lon) in enumerate(flight_path, 1):
            print(f"\n[Point {idx}/{len(flight_path)}] GPS: ({lat:.6f}, {lon:.6f})")
            
            # Capture high-resolution image (simulated)
            print("  - Capturing high-resolution RGB image...")
            image_data = self._capture_image()
            
            # Record environmental parameters
            print("  - Recording environmental parameters...")
            env_params = self._read_sensors()
            
            # Tag with GPS and timestamp
            timestamp = datetime.now()
            print(f"  - Tagging with timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create geotagged record
            record = {
                'image_id': f"{crop_field_area}_{idx:04d}",
                'gps_coordinates': {'latitude': lat, 'longitude': lon},
                'timestamp': timestamp.isoformat(),
                'image_data': image_data,
                'environmental_data': env_params,
                'field_area': crop_field_area
            }
            
            geotagged_images.append(record)
            
            # Upload to central server (simulated)
            print("  - Uploading to AI server... ✓")
        
        print("\n" + "=" * 60)
        print(f"Scan Complete: {len(geotagged_images)} images collected")
        print("=" * 60)
        
        self.geotagged_images = geotagged_images
        return geotagged_images
    
    def _capture_image(self) -> np.ndarray:
        """Simulate image capture"""
        # In production: interface with actual drone camera
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def _read_sensors(self) -> Dict:
        """Simulate sensor readings"""
        return {
            'temperature': np.random.uniform(20, 35),
            'humidity': np.random.uniform(40, 85),
            'thermal_signature': np.random.uniform(25, 40)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Demonstration of the complete system"""
    
    # Initialize the disease detection system
    detection_system = DiseaseDetectionSystem()
    
    # Simulate crop image (replace with actual image loading)
    crop_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # Simulate sensor data
    sensor_data = SensorData(
        temperature=32.5,
        humidity=75.0,
        soil_moisture=45.0,
        light_intensity=850.0
    )
    
    # Process the data
    result = detection_system.process(crop_image, sensor_data)
    
    # Save knowledge base
    detection_system.knowledge_base.save()
    
    print("\n\n")
    
    # Demonstrate drone scanning
    drone = DroneScanner()
    
    # Define flight path (GPS coordinates)
    flight_path = [
        (28.6139, 77.2090),  # Example coordinates
        (28.6145, 77.2095),
        (28.6150, 77.2100),
        (28.6155, 77.2105),
    ]
    
    # Execute field scan
    scanned_data = drone.scan_field(flight_path, "Field_A_Section_1")
    
    # Process scanned images through detection system
    print("\n\nProcessing scanned images through AI detection system...\n")
    for record in scanned_data[:2]:  # Process first 2 as example
        sensor_data = SensorData(
            temperature=record['environmental_data']['temperature'],
            humidity=record['environmental_data']['humidity'],
            soil_moisture=40.0,  # Would come from ground sensors
            light_intensity=800.0
        )
        
        result = detection_system.process(
            record['image_data'],
            sensor_data,
            record['image_id']
        )
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()