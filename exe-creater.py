import os
import sys
import subprocess

def package_executable(script_name):
    # Check if the provided script exists
    if not os.path.isfile(script_name):
        print(f"Error: {script_name} does not exist.")
        return
    
    # Prepare the command for PyInstaller
    command = [
        'pyinstaller',
        '--onefile',  # Create a single executable
        '--windowed',  # Disable the console window (useful for GUI apps)
        '--distpath', '.',  # Set the output directory to the current directory
        script_name
    ]

    # Run the PyInstaller command
    print(f"Packaging {script_name} into an executable...")
    try:
        subprocess.run(command, check=True)
        print("Packaging complete. Check the current directory for the executable.")
    except subprocess.CalledProcessError as e:
        print(f"Error during packaging: {e}")
        sys.exit(1)

if __name__ == "__main__":
    script_to_package = 'StockyHub.py'  # Specify the main script
    package_executable(script_to_package)
