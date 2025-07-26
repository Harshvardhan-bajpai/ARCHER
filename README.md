# ARCHER – Active Response & Control Hub for Emergency Reconnaissance
ARCHER is an AI-powered drone and rover system that detects threats like violence or distress in real time and alerts police. It deploys ground rovers with non-lethal tools, reducing response time and enhancing urban safety through smart surveillance.

# System Overview
ARCHER consists of three primary components:

1. Drone System (Aerial Unit)
Runs on a Raspberry Pi 4 mounted on the drone, using an ESP32-CAM for video feed and servo motor control.
Camera Module: ESP32-CAM
Servo Control: GPIO 14 on ESP32
Communication: Wi-Fi LAN between Raspberry Pi and ESP32-CAM
Purpose: Aerial surveillance and threat detection

2. Rover System (Ground Unit)
Runs on a Raspberry Pi 4, uses a USB webcam for vision, and communicates with an onboard Arduino for motor control and gimbal movement.
Camera Module: USB webcam
Gimbal Control: Servo connected to Arduino on D10
Communication: LAN over USB between Raspberry Pi and Arduino
Purpose: Target tracking and defense action

3. Ground Station
Runs on a Raspberry Pi Zero, equipped with an LCD touchscreen for user interaction.
Functions:
Rover docking station
First aid deployment
Remote launch of the rover

# Download all the files in the respective Mini Computers (RPI's)

# Drone System Setup (ESP32-CAM + Raspberry Pi 4)
Install Python 3.10 or higher (Python 3.13+ recommended).

Open the drone/ folder in your terminal.

Install required dependencies:
pip install -r requirements.txt

Run the drone detection system:
python app.py

# Rover System Setup (Raspberry Pi 4 + Arduino)
Install Python 3.10 or higher (Python 3.13+ recommended).

Open the rover/ folder in your terminal.

Install required dependencies:
pip install -r requirements.txt

Run the tracking and defense system:
python app.py

# Ground Station Setup (Raspberry Pi Zero + LCD Touchscreen)
Open the ground_station/ folder in your terminal.

Install required dependencies:
pip install -r requirements.txt

Run the ground station interface:
python app.py


# Or, if using a virtual python environment in above files 

path/to/venv/bin/python app.py
