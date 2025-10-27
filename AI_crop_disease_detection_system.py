import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing import image
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class AIDiseasePestDetectionSystem:
    """
    AI-based Disease and Pest Detection System for Crops
    """
    
    def __init__(self, model_type='resnet'):
        """
        Initialize the detection system
        
        Args:
            model_type: 'resnet' or 'inception' for image classification
        """
        self.model_type = model_type
        self.image_model = None
        self.sensor_model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.knowledge_base = []
        self.disease_classes = ['Healthy', 'Bacterial_Blight', 'Leaf_Rust', 
                               'Powdery_Mildew', 'Leaf_Spot']
        self.pest_classes = ['None', 'Low', 'Medium', 'High']
        
    def load_models(self):
        """Step 4: Model Selection - Load pre-trained models"""
        print("Loading models...")
        
        # Load CNN for image classification
        if self.model_type == 'resnet':
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=(224, 224, 3))
        else:
            base_model = InceptionV3(weights='imagenet', include_top=False, 
                                    input_shape=(299, 299, 3))
        
        # Create custom model for disease detection
        self.image_model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.disease_classes), activation='softmax')
        ])
        
        # Initialize sensor-based pest detection model
        self.sensor_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        print("Models loaded successfully!")
        
    def collect_data(self, image_path, sensor_data):
        """
        Step 1: Data Collection
        
        Args:
            image_path: Path to crop image
            sensor_data: Dict with keys - Temperature, Humidity, Soil_Moisture, Light_Intensity
        
        Returns:
            Combined input dataset
        """
        print("Step 1: Collecting data...")
        
        # Capture crop image
        crop_image = cv2.imread(image_path)
        if crop_image is None:
            raise ValueError("Unable to load image from path")
        
        # Validate sensor data
        required_keys = ['Temperature', 'Humidity', 'Soil_Moisture', 'Light_Intensity']
        for key in required_keys:
            if key not in sensor_data:
                sensor_data[key] = None
        
        input_dataset = {
            'image': crop_image,
            'sensors': sensor_data,
            'timestamp': datetime.now().isoformat()
        }
        
        return input_dataset
    
    def preprocess_data(self, input_dataset):
        """
        Step 2: Preprocessing
        
        Args:
            input_dataset: Combined data from collection step
        
        Returns:
            Preprocessed image and sensor data
        """
        print("Step 2: Preprocessing data...")
        
        # Image preprocessing
        img = input_dataset['image']
        
        # Convert to standard size
        target_size = (224, 224) if self.model_type == 'resnet' else (299, 299)
        img_resized = cv2.resize(img, target_size)
        
        # Apply noise reduction
        img_denoised = cv2.fastNlMeansDenoisingColored(img_resized, None, 10, 10, 7, 21)
        
        # Contrast enhancement using CLAHE
        lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        img_enhanced = cv2.merge([l_enhanced, a, b])
        img_final = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
        
        # Normalize image for model input
        img_normalized = img_final.astype('float32') / 255.0
        
        # Sensor data preprocessing
        sensor_values = list(input_dataset['sensors'].values())
        sensor_array = np.array(sensor_values).reshape(1, -1)
        
        # Data imputation for missing values
        sensor_imputed = self.imputer.fit_transform(sensor_array)
        
        # Normalize sensor data
        sensor_normalized = self.scaler.fit_transform(sensor_imputed)
        
        return {
            'image': img_normalized,
            'sensors': sensor_normalized
        }
    
    def extract_features(self, preprocessed_data):
        """
        Step 3: Feature Extraction
        
        Args:
            preprocessed_data: Output from preprocessing step
        
        Returns:
            Extracted features from image and sensors
        """
        print("Step 3: Extracting features...")
        
        img = preprocessed_data['image']
        
        # Image features
        # Color features (mean RGB values)
        color_features = np.mean(img, axis=(0, 1))
        
        # Texture features using Local Binary Pattern approximation
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray)
        
        # Edge features using Canny
        edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        image_features = {
            'color': color_features,
            'texture': texture_variance,
            'edges': edge_density,
            'processed_image': np.expand_dims(img, axis=0)
        }
        
        # Sensor features - statistical indicators
        sensor_data = preprocessed_data['sensors']
        sensor_features = {
            'mean': np.mean(sensor_data),
            'variance': np.var(sensor_data),
            'normalized_values': sensor_data
        }
        
        return {
            'image_features': image_features,
            'sensor_features': sensor_features
        }
    
    def predict(self, features):
        """
        Step 5: Prediction
        
        Args:
            features: Extracted features
        
        Returns:
            Disease status and pest presence
        """
        print("Step 5: Making predictions...")
        
        # Disease prediction using CNN
        img_input = features['image_features']['processed_image']
        
        # Simulate prediction (in real scenario, use trained model)
        # disease_probs = self.image_model.predict(img_input, verbose=0)
        # For demonstration, generate random predictions
        disease_probs = np.random.dirichlet(np.ones(len(self.disease_classes)))
        disease_idx = np.argmax(disease_probs)
        disease_status = self.disease_classes[disease_idx]
        disease_confidence = disease_probs[disease_idx]
        
        # Pest prediction using sensor data
        sensor_input = features['sensor_features']['normalized_values']
        
        # Simulate prediction
        # pest_probs = self.sensor_model.predict_proba(sensor_input)
        pest_probs = np.random.dirichlet(np.ones(len(self.pest_classes)))
        pest_idx = np.argmax(pest_probs)
        pest_presence = self.pest_classes[pest_idx]
        pest_confidence = pest_probs[pest_idx]
        
        return {
            'disease_status': disease_status,
            'disease_confidence': float(disease_confidence),
            'pest_presence': pest_presence,
            'pest_confidence': float(pest_confidence)
        }
    
    def analyze_decision(self, predictions):
        """
        Step 6: Decision Analysis
        
        Args:
            predictions: Disease and pest predictions
        
        Returns:
            Recommended actions and alerts
        """
        print("Step 6: Analyzing decisions...")
        
        disease_status = predictions['disease_status']
        pest_presence = predictions['pest_presence']
        
        alerts = []
        recommendations = []
        
        # Check for disease detection
        if disease_status != 'Healthy':
            alerts.append(f"ALERT: {disease_status} detected!")
            
            if disease_status == 'Bacterial_Blight':
                recommendations.append("Apply copper-based bactericide")
                recommendations.append("Remove and destroy infected plant parts")
            elif disease_status == 'Leaf_Rust':
                recommendations.append("Apply fungicide containing mancozeb")
                recommendations.append("Improve air circulation around plants")
            elif disease_status == 'Powdery_Mildew':
                recommendations.append("Apply sulfur-based fungicide")
                recommendations.append("Avoid overhead watering")
            elif disease_status == 'Leaf_Spot':
                recommendations.append("Apply biological fungicide")
                recommendations.append("Remove affected leaves")
        
        # Check for pest presence
        if pest_presence in ['Medium', 'High']:
            alerts.append(f"ALERT: {pest_presence} pest activity detected!")
            
            if pest_presence == 'High':
                recommendations.append("Immediate application of biological pesticide required")
                recommendations.append("Isolate infected zone to prevent spread")
                recommendations.append("Deploy pheromone traps")
            else:
                recommendations.append("Monitor pest population closely")
                recommendations.append("Consider preventive pest control measures")
        
        # If no issues detected
        if not alerts:
            recommendations.append("No action required - crops are healthy")
            recommendations.append("Continue regular monitoring")
        
        return {
            'alerts': alerts,
            'recommendations': recommendations,
            'action_required': len(alerts) > 0
        }
    
    def continuous_learning(self, input_data, predictions, actual_labels=None):
        """
        Step 7: Continuous Learning
        
        Args:
            input_data: Original input dataset
            predictions: Model predictions
            actual_labels: Ground truth labels (if available)
        """
        print("Step 7: Storing data for continuous learning...")
        
        learning_entry = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'sensor_data': input_data.get('sensors', {}),
            'actual_labels': actual_labels
        }
        
        self.knowledge_base.append(learning_entry)
        
        # In production: periodically retrain models with new data
        if len(self.knowledge_base) % 100 == 0:
            print(f"Knowledge base reached {len(self.knowledge_base)} entries.")
            print("Triggering model retraining (simulated)...")
    
    def run_detection(self, image_path, sensor_data):
        """
        Main algorithm execution
        
        Args:
            image_path: Path to crop image
            sensor_data: Dictionary of sensor readings
        
        Returns:
            Complete detection results
        """
        print("\n" + "="*60)
        print("AI DISEASE & PEST DETECTION SYSTEM")
        print("="*60 + "\n")
        
        try:
            # Step 1: Data Collection
            input_dataset = self.collect_data(image_path, sensor_data)
            
            # Step 2: Preprocessing
            preprocessed = self.preprocess_data(input_dataset)
            
            # Step 3: Feature Extraction
            features = self.extract_features(preprocessed)
            
            # Step 5: Prediction
            predictions = self.predict(features)
            
            # Step 6: Decision Analysis
            analysis = self.analyze_decision(predictions)
            
            # Step 7: Continuous Learning
            self.continuous_learning(input_dataset, predictions)
            
            # Step 8: Output
            results = {
                'disease_status': predictions['disease_status'],
                'disease_confidence': predictions['disease_confidence'],
                'pest_presence': predictions['pest_presence'],
                'pest_confidence': predictions['pest_confidence'],
                'alerts': analysis['alerts'],
                'recommended_actions': analysis['recommendations'],
                'timestamp': input_dataset['timestamp']
            }
            
            self.display_results(results)
            
            return results
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None
    
    def display_results(self, results):
        """
        Step 8: Output Display
        
        Args:
            results: Detection results
        """
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"\nDisease Status: {results['disease_status']}")
        print(f"Confidence: {results['disease_confidence']:.2%}")
        print(f"\nPest Presence: {results['pest_presence']}")
        print(f"Confidence: {results['pest_confidence']:.2%}")
        
        if results['alerts']:
            print("\n⚠️  ALERTS:")
            for alert in results['alerts']:
                print(f"  • {alert}")
        
        print("\n📋 RECOMMENDED ACTIONS:")
        for i, action in enumerate(results['recommended_actions'], 1):
            print(f"  {i}. {action}")
        
        print("\n" + "="*60 + "\n")


# Example Usage
if __name__ == "__main__":
    # Initialize system
    system = AIDiseasePestDetectionSystem(model_type='resnet')
    system.load_models()
    
    # Simulate sensor data
    sensor_data = {
        'Temperature': 28.5,      # °C
        'Humidity': 75.0,         # %
        'Soil_Moisture': 45.0,    # %
        'Light_Intensity': 850    # lux
    }
    
    # Note: Replace with actual image path
    # For demonstration, this will show an error but demonstrate the flow
    print("\nExample 1: Simulated Detection")
    print("-" * 60)
    
    try:
        # In real usage: system.run_detection('path/to/crop_image.jpg', sensor_data)
        print("Note: To run actual detection, provide a valid image path:")
        print("  results = system.run_detection('crop_image.jpg', sensor_data)")
        
        # Simulate the output format
        sample_results = {
            'disease_status': 'Leaf_Rust',
            'disease_confidence': 0.87,
            'pest_presence': 'Medium',
            'pest_confidence': 0.72,
            'alerts': ['ALERT: Leaf_Rust detected!', 
                      'ALERT: Medium pest activity detected!'],
            'recommended_actions': [
                'Apply fungicide containing mancozeb',
                'Improve air circulation around plants',
                'Monitor pest population closely',
                'Consider preventive pest control measures'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        system.display_results(sample_results)
        
    except Exception as e:
        print(f"Example error (expected without image): {e}")
    
    print("\nSystem initialized and ready for use!")
    print("Knowledge base entries:", len(system.knowledge_base))

#usage
system = AIDiseasePestDetectionSystem()
system.load_models()
results = system.run_detection('crop_image.jpg', sensor_data)