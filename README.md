# 🌱 HappyPlant - Plant Watering Calculator

HappyPlant is a smart web application designed to help you determine the optimal watering schedule for your houseplants. It uses plant identification, pot size estimation (experimental), and details about plant families to provide personalized recommendations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 💧 **Personalized Watering Schedules**: Calculates water amount (ml) and frequency (days) based on plant type and estimated pot volume.
- 📸 **Photo Upload**: Upload a photo for automatic plant identification (via PlantNet API) and pot size estimation (via OpenCV).
- ✏️ **Manual Entry**: Option to manually input plant name and pot volume if a photo isn't available or identification fails.
- 🌿 **Plant Identification Tool**: Separate interface to test plant identification by uploading an image.
- 📏 **Pot Size Estimation Tool**: Separate interface to test the experimental pot size estimation by uploading an image.
- 📅 **Schedule Calculation Tool**: Separate interface to manually input plant family and pot volume to test the scheduling algorithm.
- 💾 **Data Export**: Export the generated plant profile (plant info, pot info, schedule) as TXT or JSON files.
- 📲 **Calendar QR Code**: Generate a QR code containing a recurring calendar event (vEvent) for your watering schedule, scannable by most calendar apps.

## Quick Start

The easiest way to run the application is using the provided `run.py` script. This script automates the setup process.

1.  **Navigate** to the project directory in your terminal.
2.  **Run the script**:
    ```bash
    python run.py
    ```
3.  The script will:
    - Check your Python version (3.9+ recommended).
    - Create a Python virtual environment (`venv`).
    - Install all required dependencies from `requirements.txt`.
    - Check for necessary files/directories (`.env`, `templates`).
    - Launch the Flask web application (usually on `http://localhost:5001`).
    - Attempt to open the application in your default web browser.

## Requirements

- **Python**: Version 3.9 or higher is recommended. The `run.py` script attempts to find a suitable version.
- **Dependencies**: Listed in `requirements.txt` (installed automatically by `run.py` or manually via `pip install -r requirements.txt`). Key dependencies include Flask, OpenCV (headless), Requests, Numpy.
- **PlantNet API Key**: **Required** for the core plant identification feature via photo upload. You need to obtain a free key from [my.plantnet.org](https://my.plantnet.org/account/api).
- **Operating System**: Tested on macOS and Linux. Should work on Windows (ensure Python is correctly installed and accessible).
- **Internet Connection**: Required for PlantNet API calls.

## Manual Setup

If you prefer not to use `run.py`:

1.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    ```
2.  **Activate Environment**:
    - macOS/Linux: `source venv/bin/activate`
    - Windows: `venv\Scripts\activate`
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Up Environment Variables**:
    - Copy `.env.example` to a new file named `.env`.
    - Open `.env` and paste your PlantNet API key:
      ```dotenv
      PLANTNET_API_KEY=your_plantnet_api_key_here
      ```
    - (Optional) Set other variables like `SECRET_KEY`.
5.  **Run the Application**:
    ```bash
    python app.py
    ```
6.  Open your browser to `http://localhost:5001` (or the address shown in the terminal).

## How It Works

The application follows these main steps when processing a photo:

1.  **Plant Identification**: The uploaded image is sent to the PlantNet API (if a key is configured) to identify the plant's species and family.
2.  **Pot Size Estimation**: OpenCV techniques (edge detection, contour analysis) are used to estimate the pot's diameter and volume from the image. _Note: This is experimental and accuracy can vary significantly._
3.  **Watering Algorithm**:
    - The identified plant **family** determines the general moisture requirement (e.g., Cactaceae need less water than Araceae).
    - The estimated pot **volume** helps determine the _amount_ of water needed per watering session.
    - These factors are combined to calculate a recommended watering amount (ml) and frequency (days).

For manual entry, the user provides the plant name (used to guess the family from common names) and the pot volume directly.

## Core Components

- **Main Calculator (`/`)**: The primary interface with "Upload Photo" and "Manual Entry" tabs. Displays the full profile card results.
- **Plant Identifier (`/test/identify`)**: A standalone tool to upload an image and view the detailed identification result from PlantNet.
- **Pot Size Estimator (`/test/pot-size`)**: A standalone tool to upload an image and view the estimated pot dimensions and confidence level.
- **Schedule Calculator (`/test/schedule`)**: A standalone tool to manually input plant family and pot volume to see the resulting schedule calculation.

## Export Options

When results are generated on the main page, you can export the data:

- **Save as TXT**: Downloads a formatted text file with Plant Info, Pot Info, and Schedule details.
- **Save as JSON**: Downloads the complete data structure as a JSON file.
- **QR: Add Reminder**: Generates a QR code containing calendar event data (`vEvent`). Scanning this with a compatible calendar app (iOS, Google Calendar) should prompt you to add the recurring watering schedule to your calendar.

_(Deprecated QR options for vCard and Raw JSON still exist in the code but are hidden from the UI)._

## API Keys

A **PlantNet API key is essential** for identifying plants from photos.

1.  Register and obtain a key from [my.plantnet.org/account/api](https://my.plantnet.org/account/api).
2.  Create a `.env` file in the project root (or copy `.env.example`).
3.  Add your key: `PLANTNET_API_KEY=YOUR_ACTUAL_KEY_HERE`.

Without a valid key, the photo upload identification will fail. Manual entry will still work based on common name matching.

## Pot Size Estimation Notes

The current pot size estimation uses basic computer vision (edge detection, contour analysis, heuristics) and **does not use a scale reference**. Its accuracy is **highly dependent** on:

- Clear, well-lit photos.
- The pot being reasonably centered and distinct from the background.
- The viewing angle (works best with straight-on shots).

Results should be treated as rough estimates. For improved accuracy, providing volume manually or modifying the code to use a reference object (like a coin) in the photo would be necessary.

## Limitations

- Plant identification requires a working internet connection and a valid PlantNet API key.
- Pot size estimation accuracy is limited and experimental.
- Watering recommendations are general guidelines based on family and volume; actual needs may vary based on environment (light, humidity, temperature), soil type, and specific plant health. Always check soil moisture before watering.

## Contributing

Contributions are welcome! Please feel free to:

- Report bugs or suggest features by opening an issue.
- Submit pull requests with improvements or bug fixes.
- Share feedback on the accuracy of the algorithms.

## License

This project is licensed under the **MIT License**. See the LICENSE file (if included) or the header notice for details. You are free to use, modify, and distribute the code.

## Acknowledgments

- Plant identification powered by the [PlantNet API](https://my.plantnet.org/).
- Computer vision components utilize the [OpenCV](https://opencv.org/) library.
- Web application framework: [Flask](https://flask.palletsprojects.com/).
- QR Code generation: [qrcode.js](https://github.com/davidshimjs/qrcodejs).
