#!/usr/bin/env python3
"""
All-in-one script to set up and run the Plant Watering Calculator
"""
import os
import sys
import subprocess
import platform
import time
import webbrowser
import shutil
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

def print_colored(message, color):
    """Print colored text to terminal"""
    print(f"{color}{message}{Colors.NC}")

def check_python_version():
    """Check if current Python version is suitable"""
    if sys.version_info < (3, 7):
        print_colored("Error: Python 3.7+ is required", Colors.RED)
        sys.exit(1)
    return sys.version_info

def find_python_39_plus():
    """Find Python 3.9+ installation on the system"""
    python_commands = ['python3.12', 'python3.11', 'python3.10', 'python3.9', 'python3', 'python']
    
    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version_parts = result.stdout.strip().split('.')
                major, minor = int(version_parts[0]), int(version_parts[1])
                if major >= 3 and minor >= 9:
                    print_colored(f"Found {cmd} version {major}.{minor}", Colors.GREEN)
                    return cmd
        except FileNotFoundError:
            continue
    
    return None

def check_dependencies_installed():
    """Check if required dependencies are installed"""
    try:
        # Check if requirements.txt packages are installed
        result = subprocess.run([sys.executable, "-m", "pip", "show", "Flask", "opencv-python-headless", "numpy", "Pillow", "requests", "Werkzeug", "python-dotenv"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    print_colored("Installing dependencies...", Colors.YELLOW)
    
    # First, upgrade pip
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    
    # Then install from requirements.txt if it exists
    if os.path.exists('requirements.txt'):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    else:
        print_colored("Error: requirements.txt not found!", Colors.RED)
        sys.exit(1)

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        # Check if we need Python 3.9+
        current_version = sys.version_info
        if current_version < (3, 9):
            print_colored(f"Current Python version {current_version.major}.{current_version.minor} is too old", Colors.YELLOW)
            python_cmd = find_python_39_plus()
            
            if python_cmd:
                print_colored(f"Creating virtual environment with {python_cmd}...", Colors.YELLOW)
                subprocess.run([python_cmd, "-m", "venv", "venv"], check=True)
            else:
                print_colored("Error: No Python 3.9+ found on system", Colors.RED)
                print_colored("Please install Python 3.9 or higher", Colors.RED)
                print_colored("Visit: https://www.python.org/downloads/", Colors.GREEN)
                sys.exit(1)
        else:
            print_colored("Creating virtual environment...", Colors.YELLOW)
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
    # Get path to virtual environment Python
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    return str(python_exe)

def create_project_structure():
    """Create necessary directories"""
    directories = ['templates', 'uploads']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_and_move_files():
    """Check if files are in correct locations"""
    # Check if index.html needs to be moved to templates
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        print_colored("Moving index.html to templates directory...", Colors.YELLOW)
        shutil.move('index.html', 'templates/index.html')
    
    # Create .env file from .env.example if it doesn't exist
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        print_colored("Creating .env file from .env.example...", Colors.YELLOW)
        shutil.copy('.env.example', '.env')
        print_colored("Please edit .env file with your API keys", Colors.YELLOW)

def check_api_keys():
    """Check if API keys are configured"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
            if 'PLANTNET_API_KEY=your_plantnet_api_key_here' in content:
                print_colored("⚠️  Warning: Please update your API keys in .env file", Colors.YELLOW)
                time.sleep(2)

def launch_browser(port):
    """Launch web browser with the application URL"""
    url = f"http://localhost:{port}"
    print_colored(f"Opening browser to {url}", Colors.YELLOW)
    time.sleep(2)  # Give Flask time to start
    webbrowser.open(url)

def main():
    """Main execution function"""
    print_colored("🌱 Plant Watering Calculator", Colors.GREEN)
    print_colored("==============================", Colors.GREEN)
    
    # Check Python version
    current_version = check_python_version()
    
    # Create project structure
    create_project_structure()
    
    # Check if we need to set up virtual environment
    if not os.path.exists('venv'):
        python_exe = setup_virtual_environment()
        
        # Run this script again with the virtual environment Python
        print_colored("Restarting with virtual environment...", Colors.YELLOW)
        subprocess.run([python_exe, __file__], check=True)
        sys.exit(0)
    else:
        # Check if we're actually running in the virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if not in_venv:
            # We're not in the virtual environment - get the venv python and restart
            if platform.system() == "Windows":
                python_exe = Path("venv") / "Scripts" / "python.exe"
            else:
                python_exe = Path("venv") / "bin" / "python"
            
            print_colored("Activating virtual environment...", Colors.YELLOW)
            subprocess.run([str(python_exe), __file__], check=True)
            sys.exit(0)
        
        # Already in virtual environment, continue setup
        
        # Check if dependencies are installed
        if not check_dependencies_installed():
            install_dependencies()
        
        # Check and move files
        check_and_move_files()
        
        # Check API keys
        check_api_keys()
        
        # Check if app.py exists
        if not os.path.exists('app.py'):
            print_colored("Error: app.py not found!", Colors.RED)
            print_colored("Please ensure app.py is in the current directory.", Colors.RED)
            sys.exit(1)
        
        # Always use port 5001 as requested
        port = 5001
        
        # Launch browser after a delay
        from threading import Thread
        t = Thread(target=launch_browser, args=(port,))
        t.daemon = True
        t.start()
        
        # Start the Flask application
        print_colored("\nStarting Plant Watering Calculator...", Colors.GREEN)
        print_colored(f"Running on port {port}", Colors.GREEN)
        
        # Set environment variables for Flask
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = 'True'
        
        # Run the Flask app
        import app
        app.app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)

if __name__ == "__main__":
    main()