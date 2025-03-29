import random
import logging
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import base64
import os

# Configure logging for auditing purposes
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("security_audit.log"), logging.StreamHandler()])

class SecurityManager:
    def __init__(self):
        self.threats = [
            "market manipulation",
            "cyber attack",
            "data breach",
            "phishing attempt",
            "insider threat",
            "malware infection"
        ]
        self.detection_history = []
        self.audit_log = []
        self.secret_key = os.urandom(32)  # AES-256 secret key (randomly generated)
        self.iv = os.urandom(16)  # Initialization vector (should be unique for each encryption)

    def audit_system(self):
        """Audit the system and log potential security risks"""
        # Example audit functionality
        print("Auditing the system for potential security risks...")

        # Simulating some checks or audits (this can be replaced with real security checks)
        audit_result = "System audit completed successfully with no issues detected."

        # Log the audit result
        self.audit_log.append(audit_result)

        # Optionally return the audit result
        return audit_result

    def view_audit_log(self):
        """Return the audit log"""
        return self.audit_log

    def check_security_vulnerabilities(self):
        """Check for security vulnerabilities (just an example function)"""
        # Simulate checking for vulnerabilities
        print("Checking for vulnerabilities...")
        # Example return
        return "No vulnerabilities found."


    def encrypt(self, data: str) -> str:
        """Encrypt data using AES encryption."""
        cipher = AES.new(self.secret_key, AES.MODE_CBC, self.iv)
        padded_data = pad(data.encode(), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted_data).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data using AES encryption."""
        cipher = AES.new(self.secret_key, AES.MODE_CBC, self.iv)
        encrypted_data = base64.b64decode(encrypted_data)
        decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
        return decrypted_data.decode()

    def scan_for_threats(self) -> str:
        """Simulate scanning for threats, encrypt the results, and log them."""
        threat = random.choice(self.threats)
        threat_detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Encrypt threat and timestamp
        encrypted_threat = self.encrypt(threat)
        encrypted_time = self.encrypt(threat_detected_time)
        
        # Store encrypted threat in history
        self.detection_history.append({
            "threat": encrypted_threat,
            "timestamp": encrypted_time
        })

        # Log the audit trail
        logging.info(f"🔐 Scanning for threats at {threat_detected_time}...")
        logging.warning(f"⚠ Threat detected: {threat}")

        return threat

    def show_detection_history(self) -> None:
        """Display the history of detected threats, decrypting for visibility."""
        if not self.detection_history:
            logging.info("No threats detected so far.")
            return
        
        logging.info("Detection History:")
        for entry in self.detection_history:
            decrypted_threat = self.decrypt(entry['threat'])
            decrypted_time = self.decrypt(entry['timestamp'])
            print(f"At {decrypted_time}, detected threat: {decrypted_threat}")

    def reset_detection_history(self) -> None:
        """Reset the threat detection history and audit the action."""
        self.detection_history = []
        logging.info("Detection history has been reset.")

    def perform_security_audit(self) -> None:
        """Perform a security audit by scanning multiple times."""
        logging.info("🔒 Performing a full security audit...")

        # Simulate multiple threat scans and record them in the encrypted history
        for _ in range(3):  # Change the range to adjust the number of scans
            self.scan_for_threats()

        # Show audit results
        self.show_detection_history()

    def generate_unique_id(self, input_data: str) -> str:
        """Generate a unique hash-based ID for each threat detection event."""
        hash_object = hashlib.sha256(input_data.encode())
        unique_id = hash_object.hexdigest()
        logging.info(f"Generated unique ID for threat: {unique_id}")
        return unique_id


if __name__ == "__main__":
    # Example usage
    security_manager = SecurityManager()
    
    # Perform a security audit
    security_manager.perform_security_audit()

    # Optionally, reset the history after the audit
    security_manager.reset_detection_history()

