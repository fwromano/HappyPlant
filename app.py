# FILE: app.py (Modified for testing PlantNet 400 error)
# ================================================
from flask import Flask, render_template, request, jsonify
import os
import requests
import cv2
import numpy as np
from PIL import Image # Pillow import, although not directly used in this version, kept for potential future image ops
import io # Kept, but not directly used in current logic
from werkzeug.utils import secure_filename
import json # Kept, but maybe only needed if loading json files directly
from datetime import datetime
from dotenv import load_dotenv
import platform # Kept, but not directly used in app logic
import traceback # Import traceback for more detailed error logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'} # Added webp
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a-strong-dev-secret-key-please-change') # Use a default, remind to change

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get API keys from environment variables
PLANTNET_API_KEY = os.getenv('PLANTNET_API_KEY')
TREFLE_API_KEY = os.getenv('TREFLE_API_KEY') # Note: Trefle API key isn't used in current logic, but loaded

# Check if API keys are configured at startup
if not PLANTNET_API_KEY:
    print("\n⚠️ WARNING: PLANTNET_API_KEY not found in environment variables.")
    print("   Please set it up in your .env file.")
    print("   Photo identification will likely fail.\n")

class SmartPlantWateringSystem:
    def __init__(self, plantnet_api_key=None, trefle_api_key=None):
        self.plantnet_api_key = plantnet_api_key
        self.trefle_api_key = trefle_api_key # Not currently used but kept

        # Default moisture levels for plant families (lowercase keys for easier matching)
        self.family_moisture = {
            "araceae": "high",         # Pothos, Peace Lily, Monstera, ZZ Plant, Philodendron
            "asparagaceae": "low",     # Snake plant, Spider plant
            "cactaceae": "very_low",   # All cacti
            "nephrolepidaceae": "very_high", # Boston Fern often falls here
            "polypodiaceae": "very_high", # Other common ferns
            "dryopteridaceae": "very_high", # Yet more ferns
            "aspleniaceae": "very_high", # Ferns
            "pteridaceae": "very_high", # Ferns
            "orchidaceae": "medium",   # Orchids
            "rosaceae": "medium",      # Roses (if grown indoors)
            "asphodelaceae": "low",    # Aloe Vera, Haworthia
            "moraceae": "medium",      # Rubber Plant, Fiddle Leaf Fig
            "marantaceae": "high",     # Prayer Plants (Maranta, Calathea, Stromanthe)
            "strelitziaceae": "high",  # Bird of Paradise
            "crassulaceae": "low",     # Jade Plant, Echeveria, Sedum, Kalanchoe (most succulents)
            "euphorbiaceae": "low",    # Some succulents, Poinsettia (medium if flowering)
            "begoniaceae": "high",     # Begonias
            "commelinaceae": "medium", # Spiderwort, Tradescantia
            "lamiaceae": "medium",     # Coleus, Mint, Basil (herbs often medium)
            "gesneriaceae": "high",    # African Violets
            "piperaceae": "medium",    # Peperomia
            "default": "medium"        # Fallback if family unknown or unlisted
        }
        # Mapping from moisture level description to a watering ratio (fraction of soil volume)
        self.moisture_to_ratio = {
            "very_low": 0.10, # Water sparingly, ~10% of soil volume
            "low": 0.20,      # Water ~20% of soil volume
            "medium": 0.30,   # Water ~30% of soil volume (adjust down from 0.35)
            "high": 0.40,     # Water ~40% of soil volume (adjust down from 0.5)
            "very_high": 0.50 # Water thoroughly, ~50% (adjust down from 0.6)
        }

    def identify_plant(self, image_path):
        """Identify plant using PlantNet API"""
        if not self.plantnet_api_key:
            print("Error: PlantNet API key is required for identification.")
            # Return an error structure consistent with other parts of the app
            return {'error': 'PlantNet API key not configured on server'}

        try:
            with open(image_path, 'rb') as image_file:
                files = {'images': image_file} # API expects the file part named 'images'

                # *** MODIFICATION FOR TESTING 400 ERROR ***
                # Simplify the 'organs' parameter. Try sending only 'auto'.
                # Original line:
                # data = {'organs': ['leaf', 'flower', 'fruit', 'bark', 'auto']}
                # Test line:
                data = {'organs': ['auto']}
                # *******************************************

                api_url = f"https://my-api.plantnet.org/v2/identify/all?api-key={self.plantnet_api_key}"
                # Optional parameters (can be added if needed, like include-related-images=false)
                params = {'include-related-images': 'false'}

                print(f"Calling PlantNet API: {api_url} with organs: {data['organs']}") # Log the call

                response = requests.post(api_url, params=params, files=files, data=data, timeout=30) # Add timeout

                print(f"PlantNet API Response Status: {response.status_code}") # Log status code

                # Check status code BEFORE trying to parse JSON
                if response.status_code == 400:
                     print(f"PlantNet returned 400 Bad Request. Response text: {response.text}")
                     # Provide a more specific error message
                     return {'error': f'PlantNet rejected the request (400 Bad Request). This might be due to the image format/content or API parameters. Response: {response.text[:200]}...'} # Show beginning of error text
                elif response.status_code == 404 and "Subscription not found" in response.text:
                     print(f"PlantNet API Key invalid or subscription issue. Response: {response.text}")
                     return {'error': 'PlantNet API key seems invalid or subscription not found.'}
                # Use raise_for_status for other non-OK codes (like 401, 403, 5xx)
                response.raise_for_status()

                results = response.json()
                # print(f"PlantNet Raw Results: {json.dumps(results, indent=2)}") # DEBUG: Log raw results

                if results and 'results' in results and results['results']:
                    best_match = results['results'][0]
                    family_info = best_match['species'].get('family', {})
                    common_names = best_match['species'].get('commonNames', [])
                    score = best_match.get('score', 0)

                    plant_info = {
                        'common_name': common_names[0] if common_names else 'Unknown',
                        'scientific_name': best_match['species'].get('scientificNameWithoutAuthor', 'Unknown'),
                        'family': family_info.get('scientificNameWithoutAuthor', 'Unknown'),
                        'confidence': round(score * 100, 1) # Score to percentage
                    }
                    # Use scientific name if common name is 'Unknown'
                    if plant_info['common_name'] == 'Unknown' and plant_info['scientific_name'] != 'Unknown':
                         plant_info['common_name'] = plant_info['scientific_name']
                    print(f"PlantNet Identification Success: {plant_info}") # Log success
                    return plant_info
                else:
                    print("PlantNet returned success status but no results found in response.")
                    return {'error': 'Plant identified successfully, but no matching species found by PlantNet.'} # More specific message

        except requests.exceptions.Timeout:
             print("Error: PlantNet API request timed out.")
             return {'error': 'The request to PlantNet timed out. Please try again later.'}
        except requests.exceptions.RequestException as e:
            # Catch connection errors, HTTP errors (already handled 400 above, but good fallback)
            print(f"Error calling PlantNet API: {e}")
            # Check if response object exists in exception args for more details
            err_msg = f'API request failed: {e}'
            if e.response is not None:
                err_msg += f" Status Code: {e.response.status_code}. Response: {e.response.text[:200]}..."
            return {'error': err_msg}
        except Exception as e:
            # Catch unexpected errors (e.g., file reading issues, JSON parsing if format changes)
            print(f"Unexpected error during plant identification: {e}")
            print(traceback.format_exc()) # Log the full traceback for debugging
            return {'error': f'An unexpected error occurred during identification: {e}'}



    def estimate_pot_size(self, image_path):
        """Estimate pot size using computer vision (EXPERIMENTAL - Improved Heuristics)"""
        print(f"Estimating pot size for: {image_path}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return {'error': 'Failed to load image file.'}

            img_height, img_width = image.shape[:2]
            img_area = img_height * img_width
            max_img_dim = max(img_height, img_width) # Used for scaling heuristic

            # --- Basic Image Processing ---
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 50, 150) # Parameters might need tuning

            # --- Contour Finding and Filtering ---
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("Warning: No contours found.")
                return {'diameter_cm': 15.0, 'estimated_volume_liters': 1.5, 'confidence': 'low (no contours)'}

            # Filter 1: Minimum Area
            min_contour_area = img_area * 0.01 # Require > 1% of image area
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            if not large_contours:
                print(f"Warning: No contours found larger than {min_contour_area:.0f} pixels. Using largest overall.")
                # Fallback to largest contour if any exist
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    initial_confidence = 'low (using small contour)'
                else: # Should not happen if contours existed
                     return {'diameter_cm': 15.0, 'estimated_volume_liters': 1.5, 'confidence': 'low (no contours)'}
            else:
                 # Filter 2: Aspect Ratio (among large contours)
                 reasonable_contours = []
                 for cnt in large_contours:
                     x, y, w, h = cv2.boundingRect(cnt)
                     if w > 0 and h > 0:
                         aspect_ratio = w / float(h)
                         # Keep contours that aren't excessively skinny or wide (adjust range as needed)
                         if 0.3 < aspect_ratio < 3.0:
                             reasonable_contours.append(cnt)

                 if not reasonable_contours:
                      print("Warning: No large contours with reasonable aspect ratio found. Using largest overall.")
                      largest_contour = max(large_contours, key=cv2.contourArea) # Fallback to largest of the large ones
                      initial_confidence = 'low (bad aspect ratio)'
                 else:
                      # Select the largest contour among the reasonably shaped ones
                      largest_contour = max(reasonable_contours, key=cv2.contourArea)
                      initial_confidence = 'medium (good contour found)'

            # --- Dimension Estimation (Heuristic) ---
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Use max dimension of bounding box as characteristic size in pixels
            dimension_pixels = max(w, h)

            # Heuristic Scaling: Map fraction of max image dimension to cm range (e.g., 5cm to 45cm)
            # This assumes the main part of the pot takes up this bounding box dimension.
            pot_dimension_fraction = dimension_pixels / max_img_dim
            diameter_cm = 5.0 + pot_dimension_fraction * 40.0 # Maps 0.0->5cm, 1.0->45cm

            # Clamp diameter to a reasonable range
            min_diameter = 5.0
            max_diameter = 50.0
            clamped_diameter_cm = max(min_diameter, min(diameter_cm, max_diameter))

            final_confidence = initial_confidence
            if clamped_diameter_cm != diameter_cm: # If clamping occurred, reduce confidence
                print(f"Warning: Heuristic diameter {diameter_cm:.1f}cm was outside range [{min_diameter}-{max_diameter}]cm, clamped.")
                final_confidence = 'low (heuristic out of range)'
            diameter_cm = clamped_diameter_cm

            # Estimate volume: Assume height ~= diameter (still a big assumption)
            radius_cm = diameter_cm / 2
            # Alternative: Estimate height from 'h' using the same flawed scaling?
            # scaled_h_cm = 5.0 + (h / max_img_dim) * 40.0
            # height_cm = max(5.0, min(scaled_h_cm, 50.0)) # Clamp height too
            height_cm = diameter_cm # Keep simple assumption for now
            volume_liters = (np.pi * (radius_cm ** 2) * height_cm) / 1000.0

            # Clamp volume
            min_volume = 0.1
            max_volume = 75.0 # Increased max slightly
            clamped_volume_liters = max(min_volume, min(volume_liters, max_volume))
            if clamped_volume_liters != volume_liters and final_confidence != 'low (heuristic out of range)':
                 print(f"Warning: Calculated volume {volume_liters:.2f}L was outside range [{min_volume}-{max_volume}]L, clamped.")
                 # Only downgrade confidence if it wasn't already low due to diameter clamping
                 if 'medium' in final_confidence: final_confidence = 'low (volume out of range)'
            volume_liters = clamped_volume_liters


            result = {
                'diameter_cm': round(diameter_cm, 1),
                'estimated_volume_liters': round(volume_liters, 2),
                'confidence': final_confidence
            }
            print(f"Pot Size Estimation Result: {result}")
            return result

        except cv2.error as e:
            print(f"OpenCV Error during pot size estimation: {e}")
            print(traceback.format_exc())
            return {'error': f'OpenCV processing failed: {e}'}
        except Exception as e:
            print(f"Unexpected error during pot size estimation: {e}")
            print(traceback.format_exc())
            return {
                'diameter_cm': 15.0, 'estimated_volume_liters': 1.5,
                'confidence': f'error ({e})'
            }


    def get_watering_schedule(self, plant_info, pot_info):
        """Calculate watering schedule based on plant and pot info"""
        # Input validation
        if not isinstance(plant_info, dict) or not isinstance(pot_info, dict):
             return {'error': 'Invalid input: plant_info and pot_info must be dictionaries.'}
        if 'estimated_volume_liters' not in pot_info:
             # Maybe try using diameter if volume is missing? For now, require volume.
             return {'error': 'Missing "estimated_volume_liters" in pot_info.'}

        # Determine plant family's moisture needs
        family_input = plant_info.get('family', 'default').lower().strip() # Lowercase and strip whitespace
        # Find the family in our dictionary keys (case-insensitive)
        matched_family_key = 'default' # Start with default
        if family_input and family_input != 'unknown':
             # Check exact lowercase match first
             if family_input in self.family_moisture:
                  matched_family_key = family_input
             else:
                  # Optional: Could add partial matching here if needed, but exact is safer
                  print(f"Family '{family_input}' not in known list, using default.")
                  matched_family_key = 'default'
        else:
             matched_family_key = 'default' # Use default if family is missing or 'unknown'

        moisture_level = self.family_moisture.get(matched_family_key, self.family_moisture['default'])

        # Get pot volume, handle potential errors/invalid values
        try:
            pot_volume_liters = float(pot_info['estimated_volume_liters'])
            if pot_volume_liters <= 0:
                print(f"Warning: Pot volume {pot_volume_liters} is invalid, using default 1.5L.")
                pot_volume_liters = 1.5
        except (ValueError, TypeError, KeyError):
            print(f"Warning: Invalid or missing pot volume, using default 1.5L.")
            pot_volume_liters = 1.5 # Default if key missing, not a number, or zero/negative

        # --- Watering Calculation ---
        # Assume effective soil volume is a fraction of pot volume (e.g., 70-80%)
        soil_volume_factor = 0.75
        soil_volume_liters = pot_volume_liters * soil_volume_factor

        # Get watering ratio based on moisture level
        water_ratio = self.moisture_to_ratio.get(moisture_level, self.moisture_to_ratio['medium'])

        # Calculate water amount in liters, then convert to ml
        water_amount_liters = soil_volume_liters * water_ratio
        # Ensure minimum water amount (e.g., > 10ml) to avoid negligible amounts for tiny pots
        min_water_ml = 10
        water_amount_ml = max(min_water_ml, round(water_amount_liters * 1000))

        # Determine frequency based on moisture level and maybe pot size? (Simple version uses moisture level only)
        frequency_days_map = {
            'very_low': 21,  # ~3 weeks
            'low': 14,       # ~2 weeks
            'medium': 7,     # ~1 week
            'high': 4,       # ~Twice a week (adjust down from 5)
            'very_high': 2   # ~Every other day (adjust down from 3)
        }
        # Optional: Adjust frequency slightly based on pot size (larger pots dry slower)
        # This is a simple heuristic, could be more complex
        if pot_volume_liters > 5.0: # Large pot (e.g., > 5L)
            frequency_adjustment = 1.2 # Increase interval by 20%
        elif pot_volume_liters < 1.0: # Small pot (e.g., < 1L)
             frequency_adjustment = 0.8 # Decrease interval by 20%
        else:
             frequency_adjustment = 1.0

        base_frequency = frequency_days_map.get(moisture_level, frequency_days_map['medium'])
        frequency_days = round(base_frequency * frequency_adjustment)
        # Ensure frequency is at least 1 day
        frequency_days = max(1, frequency_days)

        # Get plant display name
        plant_display_name = plant_info.get('common_name', 'Unknown Plant')
        if plant_display_name in ['Unknown', 'N/A', 'Unknown Plant'] and plant_info.get('scientific_name'):
             plant_display_name = plant_info.get('scientific_name') # Fallback to scientific name

        # Construct result
        schedule_result = {
            'plant_name': plant_display_name,
            'plant_family': plant_info.get('family', 'Unknown'),
            'pot_volume_liters': round(pot_volume_liters, 2),
            'moisture_level_required': moisture_level,
            'water_amount_ml': water_amount_ml,
            'frequency_days': frequency_days,
            'schedule_summary': f"Water approx. {water_amount_ml}ml every {frequency_days} days."
        }
        print(f"Calculated Schedule: {schedule_result}")
        return schedule_result


# Initialize the system once (reuse for all requests)
system = SmartPlantWateringSystem(
    plantnet_api_key=PLANTNET_API_KEY,
    trefle_api_key=TREFLE_API_KEY # Pass even if not used by system currently
)

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Main Application Routes ---

@app.route('/')
def index():
    """Serves the main calculator page."""
    # Pass API key status to the template if needed (e.g., to show warnings)
    plantnet_ready = bool(PLANTNET_API_KEY)
    return render_template('index.html', plantnet_ready=plantnet_ready)

@app.route('/manual-calculate', methods=['POST'])
def manual_calculate():
    """Handles manual calculation requests from the main page."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request: No JSON data received.'}), 400

    plant_name = data.get('plant_name', '').strip()
    try:
        pot_volume_liters_input = data.get('pot_volume')
        if pot_volume_liters_input is None: # Check if key exists
             raise ValueError("Missing 'pot_volume'")
        pot_volume_liters = float(pot_volume_liters_input)
        if pot_volume_liters <= 0:
             raise ValueError("Pot volume must be positive")
    except (ValueError, TypeError):
        # More specific error message based on exception type might be good
        print(f"Invalid pot volume received: {data.get('pot_volume')}")
        return jsonify({'error': 'Invalid pot volume provided. Please enter a positive number.'}), 400

    if not plant_name:
        return jsonify({'error': 'Plant name cannot be empty.'}), 400

    # --- Try to determine family from common name dictionary ---
    plant_family = 'default' # Start with default
    common_plant_families = { # Use lowercase keys
        'snake plant': 'asparagaceae', 'sansevieria': 'asparagaceae',
        'pothos': 'araceae', 'devils ivy': 'araceae',
        'peace lily': 'araceae', 'spathiphyllum': 'araceae',
        'spider plant': 'asparagaceae', 'chlorophytum comosum': 'asparagaceae',
        'aloe vera': 'asphodelaceae',
        'monstera': 'araceae', 'swiss cheese plant': 'araceae',
        'philodendron': 'araceae',
        'rubber plant': 'moraceae', 'ficus elastica': 'moraceae',
        'fiddle leaf fig': 'moraceae', 'ficus lyrata': 'moraceae',
        'zz plant': 'araceae', 'zamioculcas zamiifolia': 'araceae',
        'prayer plant': 'marantaceae', 'maranta': 'marantaceae', 'calathea': 'marantaceae',
        'bird of paradise': 'strelitziaceae',
        'boston fern': 'nephrolepidaceae', # Be more specific if possible
        'fern': 'default', # Generic fern fallback to medium/high default based on family mapping
        'jade plant': 'crassulaceae', 'crassula ovata': 'crassulaceae',
        'cactus': 'cactaceae', # Generic cactus
        'orchid': 'orchidaceae', # Generic orchid
        'succulent': 'default', # Use medium default? Or low? Let's use low via Crassulaceae
        'african violet': 'gesneriaceae',
        'begonia': 'begoniaceae',
        'peperomia': 'piperaceae',
        'coleus': 'lamiaceae',
        'tradescantia': 'commelinaceae',
        # Add more common names and map them to keys in system.family_moisture
    }
    plant_name_lower = plant_name.lower()
    found_family = False
    for common_name, family_key in common_plant_families.items():
        if common_name in plant_name_lower:
            plant_family = family_key
            found_family = True
            print(f"Manual input: Matched '{plant_name}' to family '{plant_family}' via common name '{common_name}'.")
            break
    if not found_family:
         print(f"Manual input: Could not determine family for '{plant_name}', using '{plant_family}'.")

    # Create plant_info dictionary
    plant_info = {
        'common_name': plant_name,
        'scientific_name': plant_name, # Use input name as placeholder if not looked up
        'family': plant_family, # Use determined family or 'default'
        'confidence': 1.0 if found_family else 0.5 # Confidence based on whether we mapped family
    }

    # Estimate diameter roughly from volume for display purposes (same logic as test page)
    try:
        radius_cm = ( (pot_volume_liters * 1000) / (2 * np.pi) )**(1/3)
        diameter_cm = round(2 * radius_cm, 1)
    except ValueError:
        diameter_cm = None # Calculation failed

    # Create pot_info dictionary
    pot_info = {
        'diameter_cm': diameter_cm, # Can be null if calculation failed
        'estimated_volume_liters': pot_volume_liters,
        'confidence': 'user_provided'
    }

    # Get watering schedule using the global system instance
    watering_schedule = system.get_watering_schedule(plant_info, pot_info)
    if 'error' in watering_schedule:
         # Don't expose internal errors directly, log them and return generic message
         print(f"Error calculating schedule for manual input: {watering_schedule['error']}")
         return jsonify({'error': 'Failed to calculate watering schedule based on input.'}), 500

    # Return all gathered info
    return jsonify({
        'plant_info': plant_info,
        'pot_info': pot_info,
        'watering_schedule': watering_schedule
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, performs full analysis, returns results."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Use timestamp for unique filename to prevent clashes
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)
            print(f"File saved successfully to: {filepath}")

            # --- Perform Full Workflow ---
            # 1. Identify plant
            print("Step 1: Identifying plant...")
            plant_info = system.identify_plant(filepath)
            # Check for error dictionary specifically
            if isinstance(plant_info, dict) and 'error' in plant_info:
                 error_msg = plant_info['error']
                 print(f"Plant identification failed: {error_msg}")
                 # Return 4xx or 5xx based on error type? 400 for client-side issues (bad key, bad image), 500 for server issues
                 status_code = 400 if "API key" in error_msg or "Bad Request" in error_msg else 500
                 return jsonify({'error': f'Plant ID Error: {error_msg}'}), status_code

            if not plant_info: # Should not happen if error dict is returned, but safeguard
                 print("Plant identification returned empty result unexpectedly.")
                 return jsonify({'error': 'Plant identification failed unexpectedly.'}), 500

            # 2. Estimate pot size
            print("Step 2: Estimating pot size...")
            pot_info = system.estimate_pot_size(filepath)
            if isinstance(pot_info, dict) and 'error' in pot_info:
                 # Pot size estimation failure might be less critical, maybe proceed with defaults?
                 # For now, let's return an error but maybe make it less severe (e.g., 400)
                 error_msg = pot_info['error']
                 print(f"Pot size estimation failed: {error_msg}")
                 return jsonify({'error': f'Pot Size Error: {error_msg}'}), 400 # Indicate client-side issue (bad photo?) or processing fail

            if not pot_info:
                 print("Pot size estimation returned empty result unexpectedly.")
                 return jsonify({'error': 'Pot size estimation failed unexpectedly.'}), 500

            # 3. Get watering schedule
            print("Step 3: Calculating watering schedule...")
            watering_schedule = system.get_watering_schedule(plant_info, pot_info)
            if isinstance(watering_schedule, dict) and 'error' in watering_schedule:
                 error_msg = watering_schedule['error']
                 print(f"Watering schedule calculation failed: {error_msg}")
                 return jsonify({'error': f'Schedule Calc Error: {error_msg}'}), 500 # Internal calculation error


            print("Full analysis successful.")
            # Combine all results into one JSON response
            return jsonify({
                'plant_info': plant_info,
                'pot_info': pot_info,
                'watering_schedule': watering_schedule
            })

        except FileNotFoundError:
             print(f"Error: Uploaded file not found after saving? Path: {filepath}")
             return jsonify({'error': 'Internal server error: Uploaded file disappeared.'}), 500
        except Exception as e:
             # Catch-all for any other unexpected errors during processing
             print(f"Unhandled exception during upload processing: {e}")
             print(traceback.format_exc())
             return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500
        finally:
            # Clean up the uploaded file in all cases (success or failure)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Cleaned up temporary file: {filepath}")
                except OSError as e:
                    # Log error but don't prevent response from being sent
                    print(f"Error removing temporary file {filepath}: {e}")

    else:
        # File type not allowed
        allowed_types = ", ".join(app.config['ALLOWED_EXTENSIONS'])
        return jsonify({'error': f'Invalid file type. Allowed types: {allowed_types}'}), 400


# --- Routes for Browser-Based Testing Tools ---

@app.route('/test/identify')
def test_identify_page():
    """Serves the HTML page for testing Plant Identification."""
    plantnet_ready = bool(PLANTNET_API_KEY)
    return render_template('test_identify.html', plantnet_ready=plantnet_ready)

@app.route('/test/pot-size')
def test_pot_size_page():
    """Serves the HTML page for testing Pot Size Estimation."""
    return render_template('test_pot_size.html')

@app.route('/test/schedule')
def test_schedule_page():
    """Serves the HTML page for testing Schedule Calculation."""
    # Provide known families (keys from the dict) for the datalist helper
    known_families = sorted(list(system.family_moisture.keys())) # Sort alphabetically
    # Moisture levels not directly needed by template, but could be passed if helpful
    # moisture_levels = list(system.moisture_to_ratio.keys())
    return render_template('test_schedule.html', known_families=known_families)


# --- API Endpoints for Testing Individual Components ---

@app.route('/test/identify-plant', methods=['POST'])
def test_identify_plant_api():
    """API endpoint to test only the plant identification."""
    if 'image' not in request.files:
        return jsonify({'error': 'No "image" file part in the request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Use unique temp file path
    filepath = None # Initialize to prevent reference before assignment in finally
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            print(f"Test Identify: File saved to {filepath}")

            # Call the identification method
            plant_info = system.identify_plant(filepath)

            # Return the result (which might be an error dictionary itself)
            # Determine status code based on result content
            status_code = 200
            if isinstance(plant_info, dict) and 'error' in plant_info:
                 # Use 400 for client-side errors (bad key, bad image format) or 500 otherwise
                 error_msg = plant_info['error']
                 status_code = 400 if "API key" in error_msg or "Bad Request" in error_msg or "rejected" in error_msg else 500
            elif not plant_info: # Should not happen with current logic, but safeguard
                 plant_info = {'error': 'Identification failed unexpectedly.'}
                 status_code = 500

            print(f"Test Identify Result (Status {status_code}): {plant_info}")
            return jsonify(plant_info), status_code
        else:
            # Invalid file type
            allowed_types = ", ".join(app.config['ALLOWED_EXTENSIONS'])
            return jsonify({'error': f'Invalid file type. Allowed: {allowed_types}'}), 400
    except Exception as e:
        # Catch unexpected errors during the test process
        print(f"Error in /test/identify-plant API endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500
    finally:
        # Ensure cleanup happens
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Test Identify: Cleaned up {filepath}")
            except OSError as e:
                print(f"Error removing test file {filepath}: {e}")


@app.route('/test/estimate-pot-size', methods=['POST'])
def test_estimate_pot_size_api():
    """API endpoint to test only the pot size estimation."""
    if 'image' not in request.files:
        return jsonify({'error': 'No "image" file part in the request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    filepath = None
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            print(f"Test Pot Size: File saved to {filepath}")

            # Call the estimation method
            pot_info = system.estimate_pot_size(filepath)

            status_code = 200
            if isinstance(pot_info, dict) and 'error' in pot_info:
                 status_code = 400 # Assume bad image or OpenCV issue is client/input related
            elif not pot_info:
                 pot_info = {'error': 'Pot size estimation failed unexpectedly.'}
                 status_code = 500

            print(f"Test Pot Size Result (Status {status_code}): {pot_info}")
            return jsonify(pot_info), status_code
        else:
            allowed_types = ", ".join(app.config['ALLOWED_EXTENSIONS'])
            return jsonify({'error': f'Invalid file type. Allowed: {allowed_types}'}), 400
    except Exception as e:
        print(f"Error in /test/estimate-pot-size API endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Test Pot Size: Cleaned up {filepath}")
            except OSError as e:
                print(f"Error removing test file {filepath}: {e}")


@app.route('/test/calculate-schedule', methods=['POST'])
def test_calculate_schedule_api():
    """API endpoint to test only the watering schedule calculation."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request: No JSON data received.'}), 400

    plant_info = data.get('plant_info')
    pot_info = data.get('pot_info')

    # Basic validation of input structure
    if not isinstance(plant_info, dict) or not isinstance(pot_info, dict):
        return jsonify({'error': 'Invalid data format: "plant_info" and "pot_info" must be objects.'}), 400
    # Could add more specific validation for required keys if needed

    try:
        # Call the calculation method
        schedule = system.get_watering_schedule(plant_info, pot_info)

        status_code = 200
        if isinstance(schedule, dict) and 'error' in schedule:
             status_code = 400 # Assume error is due to invalid input data

        print(f"Test Schedule Calc Result (Status {status_code}): {schedule}")
        return jsonify(schedule), status_code
    except Exception as e:
        # Catch unexpected errors during calculation
        print(f"Error in /test/calculate-schedule API endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An unexpected server error occurred during calculation: {e}'}), 500


# --- Main Execution ---

if __name__ == '__main__':
    # Use port 5001 as default, allow override via environment variable
    port = int(os.getenv('PORT', 5001))
    # Determine debug mode from FLASK_ENV or default to True if not set
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'

    print(f"\n🌱 Starting Plant Watering Calculator in {'DEBUG' if debug_mode else 'PRODUCTION'} mode")
    print(f"   Access the UI at: http://localhost:{port} or http://<your-ip>:{port}")
    print(f"   API testing UI available under /test/*")
    if not PLANTNET_API_KEY and debug_mode:
         print("   ⚠️ PlantNet API Key missing - photo features will fail.")

    # Run the Flask development server
    # For production, use a proper WSGI server like Gunicorn or Waitress
    # Example: gunicorn -w 4 'app:app' -b 0.0.0.0:5001
    app.run(debug=debug_mode, host='0.0.0.0', port=port)