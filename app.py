# FILE: app.py
# ================================================
from flask import Flask, render_template, request, jsonify
import os
import requests
import cv2
import numpy as np
import math
from PIL import Image 
import pillow_heif
from pillow_heif import open_heif
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
import platform # Keep for potential future use
import traceback # For detailed error logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp', "heic", "heif"} # Added webp
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a-strong-dev-secret-key-please-change')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get API keys from environment variables
PLANTNET_API_KEY = os.getenv('PLANTNET_API_KEY')
TREFLE_API_KEY = os.getenv('TREFLE_API_KEY') # Not currently used

# Check if API keys are configured at startup
if not PLANTNET_API_KEY:
    print("\n⚠️ WARNING: PLANTNET_API_KEY not found in environment variables.")
    print("   Please set it up in your .env file.")
    print("   Photo identification will likely fail.\n")




def load_image_with_optional_depth(path: str):
    """
    Returns   (bgr_image, depth_map_m | None)
    • Handles .heic/.heif with embedded Apple portrait‑depth
    • Falls back to cv2.imread for all other formats
    • No pyheif dependency (pillow‑heif wheel contains libheif)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in ('.heic', '.heif'):
        return cv2.imread(path), None

    try:
        heif = open_heif(path, load_metadata=True, convert_hdr_to_8bit=True)
    except Exception as e:
        print(f"HEIF read error: {e}")
        return cv2.imread(path), None

    # RGB frame -------------------------------------------------------
    pil_img = heif.to_pillow()
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Depth map (if any) ---------------------------------------------
    depth = None
    meta_blocks = getattr(heif, "metadata", None)
    if meta_blocks is None:
        meta_blocks = heif.info.get("metadata", [])


    for m in meta_blocks:
        # Apple tags depth aux images with “…aux:depth” (sometimes “…auxl” older)
        if b"depth" in m.get("identifier", b""):
            w, h = m["width"], m["height"]
            raw = np.frombuffer(m["data"], np.uint16).reshape(h, w)
            # Very rough disparity→depth (meters). 57 mm baseline for iPhone 13 Pro
            depth = 0.057 / (raw.astype(np.float32) + 1e-5)
            break  # take first depth block

    return bgr, depth
    
class SmartPlantWateringSystem:
    def __init__(self, plantnet_api_key=None, trefle_api_key=None):
        self.plantnet_api_key = plantnet_api_key
        self.trefle_api_key = trefle_api_key

        # Default moisture levels (lowercase keys)
        self.family_moisture = {
            "araceae": "high", "asparagaceae": "low", "cactaceae": "very_low",
            "nephrolepidaceae": "very_high", "polypodiaceae": "very_high",
            "dryopteridaceae": "very_high", "aspleniaceae": "very_high",
            "pteridaceae": "very_high", "orchidaceae": "medium", "rosaceae": "medium",
            "asphodelaceae": "low", "moraceae": "medium", "marantaceae": "high",
            "strelitziaceae": "high", "crassulaceae": "low", "euphorbiaceae": "low",
            "begoniaceae": "high", "commelinaceae": "medium", "lamiaceae": "medium",
            "gesneriaceae": "high", "piperaceae": "medium",
            "default": "medium"
        }
        # Moisture level to watering ratio
        self.moisture_to_ratio = {
            "very_low": 0.10, "low": 0.20, "medium": 0.30,
            "high": 0.40, "very_high": 0.50
        }

    def identify_plant(self, image_path):
        """Identify plant using PlantNet API"""
        if not self.plantnet_api_key:
            print("Error: PlantNet API key is required for identification.")
            return {'error': 'PlantNet API key not configured on server'}
        try:
            with open(image_path, 'rb') as image_file:
                files = {'images': image_file}
                # Using 'auto' often works well and simplifies the request
                data = {'organs': ['auto']}
                api_url = f"https://my-api.plantnet.org/v2/identify/all?api-key={self.plantnet_api_key}"
                params = {'include-related-images': 'false'}
                print(f"Calling PlantNet API: {api_url} with organs: {data['organs']}")
                response = requests.post(api_url, params=params, files=files, data=data, timeout=30)
                print(f"PlantNet API Response Status: {response.status_code}")

                if response.status_code == 400:
                     print(f"PlantNet returned 400 Bad Request. Response text: {response.text}")
                     return {'error': f'PlantNet rejected the request (400 Bad Request). May be image format/content issue. Response: {response.text[:200]}...'}
                elif response.status_code == 404 and "Subscription not found" in response.text:
                     print(f"PlantNet API Key invalid or subscription issue. Response: {response.text}")
                     return {'error': 'PlantNet API key seems invalid or subscription not found.'}
                response.raise_for_status() # Handle other errors (401, 403, 5xx)

                results = response.json()
                if results and 'results' in results and results['results']:
                    best_match = results['results'][0]
                    family_info = best_match['species'].get('family', {})
                    common_names = best_match['species'].get('commonNames', [])
                    score = best_match.get('score', 0)
                    plant_info = {
                        'common_name': common_names[0] if common_names else 'Unknown',
                        'scientific_name': best_match['species'].get('scientificNameWithoutAuthor', 'Unknown'),
                        'family': family_info.get('scientificNameWithoutAuthor', 'Unknown'),
                        'confidence': round(score * 100, 1)
                    }
                    if plant_info['common_name'] == 'Unknown' and plant_info['scientific_name'] != 'Unknown':
                         plant_info['common_name'] = plant_info['scientific_name']
                    print(f"PlantNet Identification Success: {plant_info}")
                    return plant_info
                else:
                    print("PlantNet returned success status but no results found.")
                    return {'error': 'Plant identified successfully, but no matching species found by PlantNet.'}
        except requests.exceptions.Timeout:
             print("Error: PlantNet API request timed out.")
             return {'error': 'The request to PlantNet timed out.'}
        except requests.exceptions.RequestException as e:
            print(f"Error calling PlantNet API: {e}")
            err_msg = f'API request failed: {e}'
            if e.response is not None: err_msg += f" Status: {e.response.status_code}. Response: {e.response.text[:200]}..."
            return {'error': err_msg}
        except Exception as e:
            print(f"Unexpected error during plant identification: {e}")
            print(traceback.format_exc())
            return {'error': f'An unexpected error occurred during identification: {e}'}


    def estimate_pot_size(self, image_path):
        """
        Robust pot‑geometry estimator.
        • Uses HEIC depth if available     • Never raises 'Could not isolate pot'
        • Returns legacy 'diameter_cm' for UI plus rich fields.
        """
        img, depth = load_image_with_optional_depth(image_path)
        if img is None:
            return {"error": "Failed to load image."}

        H, W = img.shape[:2]
        max_dim = max(H, W)

        # --- pre‑process -------------------------------------------------
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
        gray  = clahe.apply(gray)
        blur  = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 40, 120)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {"error": "No contours detected."}

        # --- heuristic passes to choose the pot contour ------------------
        def pick_contour(th_area, th_ratio):
            for c in sorted(cnts, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(c)
                if area < th_area * H * W:          # area too small
                    continue
                x,y,w,h = cv2.boundingRect(c)
                ar = w/h if h else 0
                if th_ratio[0] <= ar <= th_ratio[1]:
                    return c
            return None

        pot = (pick_contour(0.08, (0.5, 3.0)) or      # normal close‑up
            pick_contour(0.03, (0.3, 4.0)) or      # zoomed‑out
            max(cnts, key=cv2.contourArea))        # last resort

        conf_note = "medium"
        if pot is cnts[-1]:   # resort case
            conf_note = "low (fallback contour)"

        # --- ellipse fit -------------------------------------------------
        ys = pot[:,0,1]
        y_min, y_max = ys.min(), ys.max()
        h_px = y_max - y_min
        top_pts = pot[ys < y_min + 0.25*h_px]
        bot_pts = pot[ys > y_max - 0.25*h_px]
        if len(top_pts) < 5 or len(bot_pts) < 5:
            return quick_cylinder(h_px, max_dim, conf_note)

        try:
            top_ell = cv2.fitEllipse(top_pts)
            bot_ell = cv2.fitEllipse(bot_pts)
        except cv2.error:
            return quick_cylinder(h_px, max_dim, "low (ellipse fail)")

        d_top_px = max(top_ell[1]);  d_bot_px = max(bot_ell[1])
        r_top_px, r_bot_px = d_top_px/2, d_bot_px/2
        slope_deg = math.degrees(math.atan(abs(r_top_px - r_bot_px) / h_px))
        ratio = d_bot_px / d_top_px
        shape = "cylinder" if 0.9<=ratio<=1.1 else ("cone" if ratio<=0.4 else "frustum")

        # --- scale conversion -------------------------------------------
        if depth is not None and not np.all(np.isnan(depth)):
            d1 = depth[int(top_ell[0][1]), int(top_ell[0][0])]
            d2 = depth[int(bot_ell[0][1]), int(bot_ell[0][0])]
            depth_m = np.nanmean([d1, d2])
            m_per_px = depth_m / 2770.0          # iPhone 13 Pro equiv. focal length
            cm = lambda px: px * m_per_px * 100
            conf_note = "high (depth)"
        else:
            cm = lambda px: np.clip(5 + px/max_dim*40, 5, 50)

        d_top_cm = cm(d_top_px); d_bot_cm = cm(d_bot_px); h_cm = cm(h_px)

        # --- volume ------------------------------------------------------
        if shape == "cylinder":
            vol_cm3 = math.pi*(d_top_cm/2)**2*h_cm
        elif shape == "cone":
            vol_cm3 = math.pi*(d_bot_cm/2)**2*h_cm/3
        else:
            R,r = d_top_cm/2, d_bot_cm/2
            vol_cm3 = math.pi*h_cm*(R**2+R*r+r**2)/3
        vol_L = round(np.clip(vol_cm3/1000, 0.1, 75), 2)

        return {
            "shape": shape,
            "top_diameter_cm": round(d_top_cm,1),
            "bottom_diameter_cm": round(d_bot_cm,1),
            "height_cm": round(h_cm,1),
            "slope_deg": round(slope_deg,1),
            "estimated_volume_liters": vol_L,
            "diameter_cm": round((d_top_cm+d_bot_cm)/2,1),  # legacy field
            "confidence": conf_note
        }

    # --------------------------------------------------------------------
    # helper: quick cylinder when ellipse fails
    # --------------------------------------------------------------------
    def quick_cylinder(h_px, max_dim, note):
        px2cm = lambda px: np.clip(5 + px/max_dim*40, 5, 50)
        d_cm = px2cm(h_px*0.8)         # assume height≈diameter
        h_cm = px2cm(h_px)
        vol_L = round(np.clip(math.pi*(d_cm/2)**2*h_cm/1000, 0.1, 75), 2)
        return {
            "shape": "cylinder", "top_diameter_cm": round(d_cm,1),
            "bottom_diameter_cm": round(d_cm,1), "height_cm": round(h_cm,1),
            "slope_deg": 0.0, "estimated_volume_liters": vol_L,
            "diameter_cm": round(d_cm,1), "confidence": note
        }
    
    def get_watering_schedule(self, plant_info, pot_info):
        """Calculate watering schedule based on plant and pot info"""
        if not isinstance(plant_info, dict) or not isinstance(pot_info, dict): return {'error': 'Invalid input: plant_info/pot_info must be dictionaries.'}
        if 'estimated_volume_liters' not in pot_info: return {'error': 'Missing "estimated_volume_liters" in pot_info.'}

        family_input = plant_info.get('family', 'default').lower().strip()
        matched_family_key = 'default'
        if family_input and family_input != 'unknown':
             if family_input in self.family_moisture: matched_family_key = family_input
             else: print(f"Family '{family_input}' not known, using default."); matched_family_key = 'default'
        else: matched_family_key = 'default'
        moisture_level = self.family_moisture.get(matched_family_key, self.family_moisture['default'])

        try:
            pot_volume_liters = float(pot_info['estimated_volume_liters'])
            if pot_volume_liters <= 0: print(f"Warning: Invalid pot volume {pot_volume_liters}, using 1.5L."); pot_volume_liters = 1.5
        except (ValueError, TypeError, KeyError): print(f"Warning: Invalid/missing pot volume, using 1.5L."); pot_volume_liters = 1.5

        soil_volume_factor = 0.75; soil_volume_liters = pot_volume_liters * soil_volume_factor
        water_ratio = self.moisture_to_ratio.get(moisture_level, self.moisture_to_ratio['medium'])
        water_amount_liters = soil_volume_liters * water_ratio; min_water_ml = 10
        water_amount_ml = max(min_water_ml, round(water_amount_liters * 1000))

        frequency_days_map = { 'very_low': 21, 'low': 14, 'medium': 7, 'high': 4, 'very_high': 2 }
        if pot_volume_liters > 5.0: frequency_adjustment = 1.2
        elif pot_volume_liters < 1.0: frequency_adjustment = 0.8
        else: frequency_adjustment = 1.0
        base_frequency = frequency_days_map.get(moisture_level, frequency_days_map['medium'])
        frequency_days = max(1, round(base_frequency * frequency_adjustment))

        plant_display_name = plant_info.get('common_name', 'Unknown Plant')
        if plant_display_name in ['Unknown', 'N/A', 'Unknown Plant'] and plant_info.get('scientific_name'): plant_display_name = plant_info.get('scientific_name')

        schedule_result = {
            'plant_name': plant_display_name, 'plant_family': plant_info.get('family', 'Unknown'),
            'pot_volume_liters': round(pot_volume_liters, 2), 'moisture_level_required': moisture_level,
            'water_amount_ml': water_amount_ml, 'frequency_days': frequency_days,
            'schedule_summary': f"Water approx. {water_amount_ml}ml every {frequency_days} days."
        }
        print(f"Calculated Schedule: {schedule_result}")
        return schedule_result


# Initialize the system once
system = SmartPlantWateringSystem(plantnet_api_key=PLANTNET_API_KEY, trefle_api_key=TREFLE_API_KEY)

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Main Application Routes ---

@app.route('/')
def index():
    """Serves the main calculator page."""
    plantnet_ready = bool(PLANTNET_API_KEY)
    return render_template('index.html', plantnet_ready=plantnet_ready)

@app.route('/manual-calculate', methods=['POST'])
def manual_calculate():
    """Handles manual calculation requests from the main page."""
    data = request.get_json()
    if not data: return jsonify({'error': 'Invalid request: No JSON data received.'}), 400

    plant_name = data.get('plant_name', '').strip()
    try:
        pot_volume_liters_input = data.get('pot_volume');
        if pot_volume_liters_input is None: raise ValueError("Missing 'pot_volume'")
        pot_volume_liters = float(pot_volume_liters_input)
        if pot_volume_liters <= 0: raise ValueError("Pot volume must be positive")
    except (ValueError, TypeError): print(f"Invalid pot volume received: {data.get('pot_volume')}"); return jsonify({'error': 'Invalid pot volume provided. Please enter a positive number.'}), 400
    if not plant_name: return jsonify({'error': 'Plant name cannot be empty.'}), 400

    # Determine family from common name dictionary
    plant_family = 'default'; found_family = False
    # (Keep the common_plant_families dictionary as defined in previous versions of app.py)
    common_plant_families = { # Use lowercase keys matching system.family_moisture
        'snake plant': 'asparagaceae', 'sansevieria': 'asparagaceae', 'pothos': 'araceae', 'devils ivy': 'araceae',
        'peace lily': 'araceae', 'spathiphyllum': 'araceae', 'spider plant': 'asparagaceae', 'chlorophytum comosum': 'asparagaceae',
        'aloe vera': 'asphodelaceae', 'monstera': 'araceae', 'swiss cheese plant': 'araceae', 'philodendron': 'araceae',
        'rubber plant': 'moraceae', 'ficus elastica': 'moraceae', 'fiddle leaf fig': 'moraceae', 'ficus lyrata': 'moraceae',
        'zz plant': 'araceae', 'zamioculcas zamiifolia': 'araceae', 'prayer plant': 'marantaceae', 'maranta': 'marantaceae', 'calathea': 'marantaceae',
        'bird of paradise': 'strelitziaceae', 'boston fern': 'nephrolepidaceae', 'fern': 'default', 'jade plant': 'crassulaceae', 'crassula ovata': 'crassulaceae',
        'cactus': 'cactaceae', 'orchid': 'orchidaceae', 'succulent': 'crassulaceae', # Map generic succulent to crassulaceae/low
        'african violet': 'gesneriaceae', 'begonia': 'begoniaceae', 'peperomia': 'piperaceae', 'coleus': 'lamiaceae', 'tradescantia': 'commelinaceae',
    }
    plant_name_lower = plant_name.lower()
    for common_name, family_key in common_plant_families.items():
        if common_name in plant_name_lower: plant_family = family_key; found_family = True; print(f"Manual input: Matched '{plant_name}' to family '{plant_family}'."); break
    if not found_family: print(f"Manual input: Could not determine family for '{plant_name}', using '{plant_family}'.")

    plant_info = {'common_name': plant_name, 'scientific_name': plant_name, 'family': plant_family, 'confidence': 1.0 if found_family else 0.5 }
    try: radius_cm = ( (pot_volume_liters * 1000) / (2 * np.pi) )**(1/3); diameter_cm = round(2 * radius_cm, 1)
    except ValueError: diameter_cm = None
    pot_info = {'diameter_cm': diameter_cm, 'estimated_volume_liters': pot_volume_liters, 'confidence': 'user_provided' }
    watering_schedule = system.get_watering_schedule(plant_info, pot_info)
    if 'error' in watering_schedule: print(f"Error calculating schedule for manual input: {watering_schedule['error']}"); return jsonify({'error': 'Failed to calculate watering schedule based on input.'}), 500

    return jsonify({'plant_info': plant_info, 'pot_info': pot_info, 'watering_schedule': watering_schedule})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, performs full analysis, returns results."""
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected for upload.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename); timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{timestamp}_{filename}"; filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(filepath); print(f"File saved successfully to: {filepath}")
            print("Step 1: Identifying plant..."); plant_info = system.identify_plant(filepath)
            if isinstance(plant_info, dict) and 'error' in plant_info: error_msg = plant_info['error']; print(f"Plant ID failed: {error_msg}"); status_code = 400 if "API key" in error_msg or "Bad Request" in error_msg else 500; return jsonify({'error': f'Plant ID Error: {error_msg}'}), status_code
            if not plant_info: print("Plant ID returned empty."); return jsonify({'error': 'Plant identification failed unexpectedly.'}), 500

            print("Step 2: Estimating pot size..."); pot_info = system.estimate_pot_size(filepath)
            if isinstance(pot_info, dict) and 'error' in pot_info: error_msg = pot_info['error']; print(f"Pot size failed: {error_msg}"); return jsonify({'error': f'Pot Size Error: {error_msg}'}), 400
            if not pot_info: print("Pot size returned empty."); return jsonify({'error': 'Pot size estimation failed unexpectedly.'}), 500

            print("Step 3: Calculating schedule..."); watering_schedule = system.get_watering_schedule(plant_info, pot_info)
            if isinstance(watering_schedule, dict) and 'error' in watering_schedule: error_msg = watering_schedule['error']; print(f"Schedule calc failed: {error_msg}"); return jsonify({'error': f'Schedule Calc Error: {error_msg}'}), 500

            print("Full analysis successful.")
            return jsonify({'plant_info': plant_info, 'pot_info': pot_info, 'watering_schedule': watering_schedule})
        except FileNotFoundError: print(f"Error: Uploaded file not found? Path: {filepath}"); return jsonify({'error': 'Internal server error: Uploaded file missing.'}), 500
        except Exception as e: print(f"Unhandled exception during upload: {e}"); print(traceback.format_exc()); return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500
        finally:
            if 'filepath' in locals() and os.path.exists(filepath):
                try: os.remove(filepath); print(f"Cleaned up temp file: {filepath}")
                except OSError as e: print(f"Error removing temp file {filepath}: {e}")
    else:
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
    known_families = sorted(list(system.family_moisture.keys()))
    return render_template('test_schedule.html', known_families=known_families)


# --- API Endpoints for Testing Individual Components ---

@app.route('/test/identify-plant', methods=['POST'])
def test_identify_plant_api():
    """API endpoint to test only the plant identification."""
    if 'image' not in request.files: return jsonify({'error': 'No "image" file part in the request.'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No file selected.'}), 400
    filepath = None
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename); timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{filename}"; filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath); print(f"Test Identify: File saved to {filepath}")
            plant_info = system.identify_plant(filepath)
            status_code = 200
            if isinstance(plant_info, dict) and 'error' in plant_info: error_msg = plant_info['error']; status_code = 400 if "API key" in error_msg or "Bad Request" in error_msg or "rejected" in error_msg else 500
            elif not plant_info: plant_info = {'error': 'Identification failed unexpectedly.'}; status_code = 500
            print(f"Test Identify Result (Status {status_code}): {plant_info}")
            return jsonify(plant_info), status_code
        else: allowed_types = ", ".join(app.config['ALLOWED_EXTENSIONS']); return jsonify({'error': f'Invalid file type. Allowed: {allowed_types}'}), 400
    except Exception as e: print(f"Error in /test/identify-plant API: {e}"); print(traceback.format_exc()); return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try: os.remove(filepath); print(f"Test Identify: Cleaned up {filepath}")
            except OSError as e: print(f"Error removing test file {filepath}: {e}")

@app.route('/test/estimate-pot-size', methods=['POST'])
def test_estimate_pot_size_api():
    """API endpoint to test only the pot size estimation."""
    if 'image' not in request.files: return jsonify({'error': 'No "image" file part in the request.'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'No file selected.'}), 400
    filepath = None
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename); timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{filename}"; filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath); print(f"Test Pot Size: File saved to {filepath}")
            pot_info = system.estimate_pot_size(filepath)
            status_code = 200
            if isinstance(pot_info, dict) and 'error' in pot_info: status_code = 400
            elif not pot_info: pot_info = {'error': 'Pot size estimation failed unexpectedly.'}; status_code = 500
            print(f"Test Pot Size Result (Status {status_code}): {pot_info}")
            return jsonify(pot_info), status_code
        else: allowed_types = ", ".join(app.config['ALLOWED_EXTENSIONS']); return jsonify({'error': f'Invalid file type. Allowed: {allowed_types}'}), 400
    except Exception as e: print(f"Error in /test/estimate-pot-size API: {e}"); print(traceback.format_exc()); return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try: os.remove(filepath); print(f"Test Pot Size: Cleaned up {filepath}")
            except OSError as e: print(f"Error removing test file {filepath}: {e}")

@app.route('/test/calculate-schedule', methods=['POST'])
def test_calculate_schedule_api():
    """API endpoint to test only the watering schedule calculation."""
    data = request.get_json()
    if not data: return jsonify({'error': 'Invalid request: No JSON data received.'}), 400
    plant_info = data.get('plant_info'); pot_info = data.get('pot_info')
    if not isinstance(plant_info, dict) or not isinstance(pot_info, dict): return jsonify({'error': 'Invalid data format: "plant_info" / "pot_info" must be objects.'}), 400
    try:
        schedule = system.get_watering_schedule(plant_info, pot_info)
        status_code = 200
        if isinstance(schedule, dict) and 'error' in schedule: status_code = 400
        print(f"Test Schedule Calc Result (Status {status_code}): {schedule}")
        return jsonify(schedule), status_code
    except Exception as e: print(f"Error in /test/calculate-schedule API: {e}"); print(traceback.format_exc()); return jsonify({'error': f'An unexpected server error occurred during calculation: {e}'}), 500


# --- Route for displaying QR code data (kept in case needed later) ---
@app.route('/display')
def display_qr_data():
    """Displays plant care data received via URL parameters."""
    plant_name = request.args.get('p', 'Unknown Plant')
    water_amount = request.args.get('w', 'N/A')
    frequency = request.args.get('f', 'N/A')
    # moisture = request.args.get('m', 'N/A') # Optional
    return render_template('display_qr_data.html', plant_name=plant_name, water_amount=water_amount, frequency=frequency)


# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    print(f"\n🌱 Starting Plant Watering Calculator in {'DEBUG' if debug_mode else 'PRODUCTION'} mode")
    print(f"   Access the UI at: http://localhost:{port} or http://<your-ip>:{port}")
    print(f"   Individual feature pages available under /test/*")
    if not PLANTNET_API_KEY and debug_mode: print("   ⚠️ PlantNet API Key missing - photo features will fail.")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)