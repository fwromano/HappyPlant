from flask import Flask, render_template, request, jsonify
import os
import requests
import cv2
import numpy as np
from PIL import Image
import io
from werkzeug.utils import secure_filename
import json
from datetime import datetime
from dotenv import load_dotenv
import platform

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get API keys from environment variables
PLANTNET_API_KEY = os.getenv('PLANTNET_API_KEY')
TREFLE_API_KEY = os.getenv('TREFLE_API_KEY')

# Check if API keys are configured
if not PLANTNET_API_KEY:
    print("Warning: PLANTNET_API_KEY not found in environment variables")
    print("Please set up your .env file with your API key")

class SmartPlantWateringSystem:
    def __init__(self, plantnet_api_key=None, trefle_api_key=None):
        self.plantnet_api_key = plantnet_api_key
        self.trefle_api_key = trefle_api_key
        
        # Default moisture levels for plant families
        self.family_moisture = {
            "Araceae": "high",      # Pothos, Peace Lily, etc.
            "Asparagaceae": "low",  # Snake plant, Spider plant
            "Cactaceae": "very_low", # All cacti
            "Ferns": "very_high",
            "Orchidaceae": "medium", # Orchids
            "Rosaceae": "medium",    # Roses
            "default": "medium"
        }
        
        self.moisture_to_ratio = {
            "very_low": 0.1,
            "low": 0.2,
            "medium": 0.35,
            "high": 0.5,
            "very_high": 0.6
        }
    
    def identify_plant(self, image_path):
        """Identify plant using PlantNet API"""
        if not self.plantnet_api_key:
            return None
            
        try:
            with open(image_path, 'rb') as image_file:
                files = {'images': image_file}
                data = {'organs': ['leaf', 'flower', 'fruit']}
                
                # Updated API endpoint
                url = f"https://my-api.plantnet.org/v2/identify/all?api-key={self.plantnet_api_key}"
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    results = response.json()
                    if results['results']:
                        best_match = results['results'][0]
                        return {
                            'common_name': best_match['species']['commonNames'][0] if best_match['species'].get('commonNames') else None,
                            'scientific_name': best_match['species']['scientificName'],
                            'family': best_match['species']['family']['scientificName'],
                            'confidence': best_match['score']
                        }
        except Exception as e:
            print(f"Error in plant identification: {e}")
        return None
    
    def estimate_pot_size(self, image_path):
        """Estimate pot size using computer vision"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Fallback to default pot size if no contours found
                return {
                    'diameter_cm': 20.0,
                    'estimated_volume_liters': 3.0,
                    'confidence': 'low'
                }
            
            # Find the largest contour (assuming it's the pot)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Estimate pot diameter in pixels
            diameter_pixels = max(w, h)
            
            # Scale factor (calibrated for typical photos)
            scale_factor = 0.5  # cm per pixel
            diameter_cm = diameter_pixels * scale_factor
            
            # Convert to volume (assuming cylindrical pot)
            radius_cm = diameter_cm / 2
            volume_liters = (3.14159 * radius_cm**2 * diameter_cm) / 1000
            
            return {
                'diameter_cm': round(diameter_cm, 1),
                'estimated_volume_liters': round(volume_liters, 2),
                'confidence': 'medium'
            }
        except Exception as e:
            print(f"Error in pot size estimation: {e}")
            # Return default values on error
            return {
                'diameter_cm': 20.0,
                'estimated_volume_liters': 3.0,
                'confidence': 'low'
            }
    
    def get_watering_schedule(self, plant_info, pot_info):
        """Calculate watering schedule based on plant and pot info"""
        family = plant_info.get('family', 'default')
        moisture_level = self.family_moisture.get(family, 'medium')
        
        soil_volume = pot_info['estimated_volume_liters'] * 0.7
        water_ratio = self.moisture_to_ratio[moisture_level]
        water_amount = soil_volume * water_ratio
        
        frequency_days = {
            'very_low': 21,
            'low': 14,
            'medium': 7,
            'high': 5,
            'very_high': 3
        }[moisture_level]
        
        return {
            'plant': plant_info['common_name'] or plant_info['scientific_name'],
            'water_amount_ml': round(water_amount * 1000),
            'frequency_days': frequency_days,
            'schedule': f"Water {round(water_amount * 1000)}ml every {frequency_days} days",
            'moisture_level': moisture_level
        }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual-calculate', methods=['POST'])
def manual_calculate():
    """Calculate watering schedule from manual input"""
    data = request.get_json()
    
    plant_name = data.get('plant_name', '')
    pot_volume_liters = float(data.get('pot_volume', 0))
    
    if not plant_name or pot_volume_liters <= 0:
        return jsonify({'error': 'Please provide valid plant name and pot volume'}), 400
    
    # Create manual plant info
    plant_info = {
        'common_name': plant_name,
        'scientific_name': plant_name,
        'family': 'unknown',
        'confidence': 1.0  # 100% confidence since it's manual input
    }
    
    # Create manual pot info  
    pot_info = {
        'diameter_cm': (pot_volume_liters * 1000 / 3.14159) ** (1/3) * 2,  # Rough diameter calculation
        'estimated_volume_liters': pot_volume_liters,
        'confidence': 'user_provided'
    }
    
    # Process the manual input
    system = SmartPlantWateringSystem(
        plantnet_api_key=PLANTNET_API_KEY,
        trefle_api_key=TREFLE_API_KEY
    )
    
    # Try to identify plant family for common plants
    common_plant_families = {
        'snake plant': 'Asparagaceae',
        'pothos': 'Araceae',
        'peace lily': 'Araceae',
        'spider plant': 'Asparagaceae',
        'aloe vera': 'Asphodelaceae',
        'monstera': 'Araceae',
        'philodendron': 'Araceae',
        'rubber plant': 'Moraceae',
        'fiddle leaf fig': 'Moraceae',
        'zz plant': 'Araceae',
        'prayer plant': 'Marantaceae',
        'bird of paradise': 'Strelitziaceae',
        'boston fern': 'Ferns',
        'jade plant': 'Crassulaceae',
        'cactus': 'Cactaceae',
        'orchid': 'Orchidaceae'
    }
    
    # Check if we know the family for this plant
    plant_name_lower = plant_name.lower()
    for known_plant, family in common_plant_families.items():
        if known_plant in plant_name_lower:
            plant_info['family'] = family
            break
    else:
        # Try PlantNet API only if we didn't find the family
        if PLANTNET_API_KEY:
            try:
                # Use PlantNet search API for species lookup
                url = f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}&q={plant_name}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    results = response.json()
                    if results and 'results' in results and results['results']:
                        first_result = results['results'][0]
                        if 'species' in first_result and 'family' in first_result['species']:
                            plant_info['family'] = first_result['species']['family']['scientificName']
            except Exception as e:
                print(f"PlantNet API error: {e}")
                # Use default family if API fails
    
    # Get watering schedule
    watering_schedule = system.get_watering_schedule(plant_info, pot_info)
    
    return jsonify({
        'plant_info': plant_info,
        'pot_info': pot_info,
        'watering_schedule': watering_schedule
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        system = SmartPlantWateringSystem(
            plantnet_api_key=PLANTNET_API_KEY,
            trefle_api_key=TREFLE_API_KEY
        )
        
        try:
            # Identify plant
            plant_info = system.identify_plant(filepath)
            if not plant_info:
                return jsonify({'error': 'Could not identify plant. Please try a clearer photo.'}), 400
            
            # Estimate pot size
            pot_info = system.estimate_pot_size(filepath)
            if not pot_info:
                return jsonify({'error': 'Could not estimate pot size. Using default values.'}), 400
            
            # Get watering schedule
            watering_schedule = system.get_watering_schedule(plant_info, pot_info)
            
            # Clean up - remove the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'plant_info': plant_info,
                'pot_info': pot_info,
                'watering_schedule': watering_schedule
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Check for required API keys
    if not PLANTNET_API_KEY:
        print("\n⚠️  WARNING: PlantNet API key not found!")
        print("Please create a .env file with your PLANTNET_API_KEY")
        print("The application will run but plant identification will not work.\n")
    
    # Always use port 5001
    port = 5001
    print(f"\n🌱 Starting Plant Watering Calculator on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)