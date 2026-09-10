"""
FocusSense Hardware Abstraction & Sensor Driver Interface
=========================================================
This module handles sensor interactions, desk presence detection, smart study
light control, and RFID identification.

CURRENT PHASE: SAFE SOFTWARE SIMULATION MODE
- No physical hardware required.
- No hardware-specific C-extensions or external device libraries required.
- Provides realistic PRESENT and ABSENT states for testing and UI development.
- Architected with clean extension hooks for physical sensor integration.

FUTURE HARDWARE INTEGRATION:
- When real mmWave radar (HLK-LD2410, LD2420, DFRobot SEN0395, or ESP32)
  is connected, replace or implement `_read_physical_radar()` and set
  `simulation_mode = False`.
"""

from datetime import datetime


class SensorController:
    """
    Controller for FocusSense IoT Sensors & Actuators.
    Supports both software simulation and physical sensor hardware.
    """

    def __init__(self):
        # By default, FocusSense runs in software simulation mode
        self.simulation_mode: bool = True
        self.hardware_connected: bool = False
        
        # Internal simulated presence state (False = ABSENT, True = PRESENT)
        self._simulated_presence: bool = False

    # -------------------------------------------------------------------------
    # Simulation Management Interface
    # -------------------------------------------------------------------------
    def is_simulation_mode(self) -> bool:
        """Return True if currently operating in software simulation mode."""
        return self.simulation_mode

    def set_simulation_mode(self, enabled: bool):
        """Enable or disable simulation mode."""
        self.simulation_mode = bool(enabled)

    def get_simulated_presence(self) -> bool:
        """Return the current simulated presence boolean (True = PRESENT)."""
        return self._simulated_presence

    def set_simulated_presence(self, is_present: bool) -> dict:
        """
        Explicitly set the simulated presence state.
        
        Args:
            is_present: True for PRESENT, False for ABSENT
            
        Returns:
            Dict containing the updated radar status dictionary.
        """
        self._simulated_presence = bool(is_present)
        return self.check_radar_presence()

    def toggle_simulated_presence(self) -> dict:
        """
        Toggle between PRESENT and ABSENT simulated states.
        
        Returns:
            Dict containing the updated radar status dictionary.
        """
        self._simulated_presence = not self._simulated_presence
        return self.check_radar_presence()

    # -------------------------------------------------------------------------
    # Radar Presence Detection Interface
    # -------------------------------------------------------------------------
    def check_radar_presence(self, is_session_active: bool = False) -> dict:
        """
        Check student desk presence using mmWave radar sensor.
        
        In SIMULATION MODE:
            Returns the current simulated presence state (`_simulated_presence`).
            
        In PHYSICAL HARDWARE MODE:
            Polls the physical radar sensor driver via `_read_physical_radar()`.
            
        Returns:
            dict: {
                "present": bool,
                "status": "PRESENT" | "ABSENT",
                "simulation_mode": bool,
                "timestamp": ISO-8601 string
            }
        """
        if self.simulation_mode:
            # Safe software simulation: use internal state
            presence = self._simulated_presence
        else:
            # Physical hardware mode
            presence = self._read_physical_radar()

        status_str = "PRESENT" if presence else "ABSENT"

        return {
            "present": presence,
            "status": status_str,
            "simulation_mode": self.simulation_mode,
            "timestamp": datetime.now().isoformat()
        }

    # -------------------------------------------------------------------------
    # Smart Study Light Control Interface
    # -------------------------------------------------------------------------
    def get_study_light_state(self, is_session_active: bool = False) -> dict:
        """
        Get the current state and color of the smart study light.
        
        Rule:
            - When student presence is PRESENT (or session active), light is ON
              with optimal circadian focus temperature (#38bdf8 / cyan).
            - When ABSENT and no active session, light is OFF.
            
        Returns:
            dict: {
                "state": "ON" | "OFF",
                "color": hex string,
                "timestamp": ISO-8601 string
            }
        """
        radar_result = self.check_radar_presence(is_session_active)
        presence = radar_result["present"]

        # Light turns ON when presence is detected or study session is active
        light_on = presence or is_session_active
        state = "ON" if light_on else "OFF"
        color = "#38bdf8" if light_on else "#000000"

        if not self.simulation_mode and self.hardware_connected:
            self._write_physical_light(state, color)

        return {
            "state": state,
            "color": color,
            "timestamp": datetime.now().isoformat()
        }

    # -------------------------------------------------------------------------
    # RFID Card Reader Interface
    # -------------------------------------------------------------------------
    def scan_rfid_card(self, mock_uid: str = None) -> dict:
        """
        Read student RFID badge UID for contactless identification.
        """
        if self.simulation_mode:
            uid = mock_uid or ("FS-SIM-CARD" if self._simulated_presence else "NONE")
            detected = bool(mock_uid or self._simulated_presence)
        else:
            detected, uid = self._read_physical_rfid()

        return {
            "card_detected": detected,
            "uid": uid,
            "simulation_mode": self.simulation_mode,
            "timestamp": datetime.now().isoformat()
        }

    # -------------------------------------------------------------------------
    # FUTURE PHYSICAL SENSOR HOOKS
    # -------------------------------------------------------------------------
    def _read_physical_radar(self) -> bool:
        """
        =======================================================================
        [FUTURE HARDWARE CONNECTION POINT - MMWAVE RADAR SENSOR]
        =======================================================================
        When the physical mmWave radar sensor arrives:
        
        Supported Sensors:
          1. HLK-LD2410 / LD2420 (24GHz human presence radar via UART / GPIO Out)
          2. DFRobot SEN0395 (24GHz mmWave radar)
          3. ESP32 microcontroller publishing over serial/USB, WebSockets, or MQTT
        
        Wiring Guide (Example for HLK-LD2410 via USB-UART adapter):
          - Radar VCC -> 5V
          - Radar GND -> GND
          - Radar TX  -> RX (COM port / /dev/ttyUSB0)
          - Radar RX  -> TX
          - Radar OUT -> GPIO pin (HIGH = Presence, LOW = Absent)
          
        Implementation steps:
          1. Connect sensor to USB-Serial or GPIO pin.
          2. Install `pyserial` (`pip install pyserial`).
          3. Read the presence frame or query the digital OUT pin:
             Example:
                 import serial
                 # with serial.Serial('COM3', 256000, timeout=0.1) as ser:
                 #     frame = ser.read(32)
                 #     return parse_ld2410_presence(frame)
                 
        For now, this returns False as no physical hardware is attached.
        =======================================================================
        """
        return False

    def _write_physical_light(self, state: str, color: str):
        """
        =======================================================================
        [FUTURE HARDWARE CONNECTION POINT - SMART FOCUS LIGHT]
        =======================================================================
        When the physical smart light / NeoPixel LED strip / relay is connected:
        
        Example:
          - Send serial command or HTTP/MQTT payload to ESP32:
            `ser.write(b"LIGHT:ON\n" if state == "ON" else b"LIGHT:OFF\n")`
        =======================================================================
        """
        pass

    def _read_physical_rfid(self) -> tuple:
        """
        =======================================================================
        [FUTURE HARDWARE CONNECTION POINT - RC522 / PN532 RFID READER]
        =======================================================================
        When physical RFID reader is connected via SPI or UART:
          Read card UID and return (detected: bool, uid: str).
        =======================================================================
        """
        return False, "NONE"


# Singleton sensor controller instance for easy import across the application
sensor_controller = SensorController()


def check_radar(is_session_active: bool = False) -> str:
    """
    Convenience helper returning 'PRESENT' or 'ABSENT'.
    Respects current simulation state or physical sensor reading.
    """
    return sensor_controller.check_radar_presence(is_session_active)["status"]


def check_light(is_session_active: bool = False) -> str:
    """
    Convenience helper returning 'ON' or 'OFF'.
    """
    return sensor_controller.get_study_light_state(is_session_active)["state"]
