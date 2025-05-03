# 🌱 HappyPlant

A smart application that helps you determine the optimal watering schedule for your houseplants based on plant type and pot size.

## Features

- 📸 **Photo Upload**: Upload a photo of your plant for automatic identification
- ✏️ **Manual Entry**: Enter plant information and pot size manually
- 💧 **Smart Recommendations**: Get customized watering amounts and schedules
- 🔍 **Plant Recognition**: Powered by PlantNet API for accurate plant identification
- 📏 **Pot Size Estimation**: Computer vision algorithms estimate pot dimensions
- 📱 **Mobile-Friendly**: Responsive interface works on all devices

## Quick Start

Run the application with a single command:

```bash
python run.py
```

The script automatically:

- Sets up a Python virtual environment
- Installs all required dependencies
- Creates necessary directories
- Launches the application in your browser

## Requirements

- Python 3.9+ (automatically detected and used by run.py)
- Internet connection (for plant identification)
- PlantNet API key (optional - for photo identification)

## Manual Setup

If you prefer manual setup, follow these steps:

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file and add:
     ```
     PLANTNET_API_KEY=your_key_here
     ```

5. Run the application:
   ```bash
   python app.py
   ```

## How It Works

1. **Plant Identification**:

   - Upload a photo or enter plant name manually
   - The app identifies the plant species and family

2. **Pot Size Calculation**:

   - Photo: Computer vision estimates pot dimensions
   - Manual: User provides pot volume

3. **Watering Algorithm**:
   - Plant family determines moisture requirements
   - Pot volume determines water amount
   - Combined to create a personalized watering schedule

## Common Plant Families

The application recognizes many common houseplants, including:

- Araceae (Pothos, Peace Lily, Monstera)
- Asparagaceae (Snake Plant, Spider Plant)
- Cactaceae (All cacti)
- Marantaceae (Prayer Plants)
- Moraceae (Rubber Plants, Fiddle Leaf Fig)
- And many more!

## API Keys

For photo identification, you'll need a PlantNet API key:

1. Sign up at [PlantNet](https://my.plantnet.org/account/api)
2. Create a `.env` file in the project root
3. Add your key: `PLANTNET_API_KEY=your_key_here`

## Limitations

- Plant identification requires an internet connection and API key
- Pot size estimation works best with clear photos of the entire pot
- Recommendations are general guidelines and may need adjustment based on specific conditions

## Contributing

Contributions are welcome! Feel free to:

- Report bugs or request features via issues
- Submit pull requests for improvements
- Share feedback for better plant care algorithms

## License

MIT License - Feel free to use, modify, and distribute as needed

## Acknowledgments

- Plant identification powered by [PlantNet](https://my.plantnet.org/)
- Computer vision components built with OpenCV
- Interface created with Flask and vanilla JavaScript
