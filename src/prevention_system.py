#!/usr/bin/env python3
"""
Prevention System - Automated response to ransomware detection
Implements proactive measures to stop ransomware when detected
"""

import os
import stat
import subprocess
import psutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreventionSystem:
    """Automated prevention actions for ransomware response"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('prevention', {}).get('enabled', False)
        self.auto_terminate = config.get('prevention', {}).get('auto_terminate', False)
        self.lock_filesystem = config.get('prevention', {}).get('lock_filesystem', False)
        self.disable_network = config.get('prevention', {}).get('disable_network', False)
        self.early_warning = config.get('prevention', {}).get('early_warning_detection', False)
        
        logger.info(f"Prevention System initialized (enabled: {self.enabled})")
    
    def handle_critical_alert(self, alert_details):
        """
        Handle CRITICAL alert with prevention actions
        
        Args:
            alert_details: Dictionary containing alert information
        """
        if not self.enabled:
            logger.info("Prevention system disabled - taking no action")
            return
        
        logger.warning(f"CRITICAL ALERT DETECTED - Initiating prevention measures")
        logger.warning(f"Alert details: {alert_details}")
        
        actions_taken = []
        
        # 1. Terminate malicious process
        if self.auto_terminate and alert_details.get('pid'):
            if self.terminate_process(alert_details['pid']):
                actions_taken.append("Process terminated")
        
        # 2. Lock file system
        if self.lock_filesystem:
            folders = self.config.get('monitoring', {}).get('monitor_folders', [])
            if self.lock_file_system(folders):
                actions_taken.append("File system locked")
        
        # 3. Disable network
        if self.disable_network:
            if self.isolate_network():
                actions_taken.append("Network isolated")
        
        logger.warning(f"Prevention actions taken: {actions_taken}")
    
    def terminate_process(self, pid):
        """
        Terminate process by PID
        
        Args:
            pid: Process ID to terminate
            
        Returns:
            bool: True if successful
        """
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            
            logger.warning(f"Terminating process: {process_name} (PID: {pid})")
            
            # Terminate process
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=5)
                logger.warning(f"Process {pid} terminated successfully")
                return True
            except psutil.TimeoutExpired:
                # Force kill if terminate doesn't work
                process.kill()
                process.wait(timeout=2)
                logger.warning(f"Process {pid} force killed")
                return True
                
        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} not found")
            return False
        except psutil.AccessDenied:
            logger.error(f"Access denied to process {pid} - run as Administrator")
            return False
        except Exception as e:
            logger.error(f"Failed to terminate process {pid}: {e}")
            return False
    
    def lock_file_system(self, directories):
        """
        Set directories to read-only to prevent further encryption
        
        Args:
            directories: List of directory paths to lock
            
        Returns:
            bool: True if successful
        """
        try:
            logger.warning(f"Locking file system for directories: {directories}")
            
            for directory in directories:
                if not os.path.exists(directory):
                    logger.warning(f"Directory not found: {directory}")
                    continue
                
                for root, dirs, files in os.walk(directory):
                    # Lock directories
                    for d in dirs:
                        try:
                            dir_path = os.path.join(root, d)
                            os.chmod(dir_path, stat.S_IRUSR | stat.S_IXUSR)
                        except Exception as e:
                            logger.debug(f"Could not lock directory {d}: {e}")
                    
                    # Lock files
                    for f in files:
                        try:
                            file_path = os.path.join(root, f)
                            os.chmod(file_path, stat.S_IRUSR)
                        except Exception as e:
                            logger.debug(f"Could not lock file {f}: {e}")
            
            logger.warning("File system locked successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to lock file system: {e}")
            return False
    
    def isolate_network(self):
        """
        Disable all network adapters to prevent lateral movement
        
        Returns:
            bool: True if successful
        """
        try:
            logger.warning("Disabling network adapters for isolation")
            
            # Disable all network interfaces
            result = subprocess.run(
                ['netsh', 'interface', 'set', 'interface', '*', 'disable'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.warning("Network adapters disabled successfully")
                return True
            else:
                logger.error(f"Failed to disable network: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to isolate network: {e}")
            return False
    
    def check_early_warning_indicators(self, event_aggregator):
        """
        Check for early warning signs of ransomware
        
        Args:
            event_aggregator: EventAggregator instance with current events
            
        Returns:
            list: List of warning messages
        """
        warnings = []
        
        if not self.early_warning:
            return warnings
        
        try:
            features = event_aggregator.get_aggregated_features()
            
            # Check for high file scan rate
            file_operations = features[0] + features[1]  # file_read + file_created
            if file_operations > 100:  # More than 100 file operations in window
                warnings.append(f"High file operation rate: {file_operations}")
            
            # Check for high registry modification rate
            reg_modifications = features[2]  # regkey_written
            if reg_modifications > 20:
                warnings.append(f"High registry modification rate: {reg_modifications}")
            
            # Check for high API call rate
            api_calls = features[4]  # apistats
            if api_calls > 50:
                warnings.append(f"High API call rate: {api_calls}")
            
            # Check for directory enumeration
            dir_enum = features[6]  # directory_enumerated
            if dir_enum > 50:
                warnings.append(f"High directory enumeration: {dir_enum}")
            
            if warnings:
                logger.warning(f"Early warning indicators detected: {warnings}")
            
        except Exception as e:
            logger.error(f"Error checking early warning indicators: {e}")
        
        return warnings
    
    def enable_controlled_folder_access(self):
        """
        Enable Windows Defender Controlled Folder Access
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Enabling Windows Defender Controlled Folder Access")
            
            result = subprocess.run([
                'powershell',
                '-Command',
                'Set-MpPreference -EnableControlledFolderAccess Enabled'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Controlled Folder Access enabled successfully")
                return True
            else:
                logger.error(f"Failed to enable Controlled Folder Access: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to enable Controlled Folder Access: {e}")
            return False
    
    def add_protected_folder(self, folder_path):
        """
        Add folder to Windows Defender protected folders list
        
        Args:
            folder_path: Path to folder to protect
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Adding folder to protected list: {folder_path}")
            
            result = subprocess.run([
                'powershell',
                '-Command',
                f'Add-MpPreference -ControlledFolderAccessProtectedFolders "{folder_path}"'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Folder added to protected list: {folder_path}")
                return True
            else:
                logger.error(f"Failed to add protected folder: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add protected folder: {e}")
            return False


# Utility functions for standalone use
def terminate_process(pid):
    """Standalone function to terminate a process"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        return True
    except:
        return False


def isolate_network():
    """Standalone function to isolate network"""
    try:
        subprocess.run(['netsh', 'interface', 'set', 'interface', '*', 'disable'])
        return True
    except:
        return False


if __name__ == "__main__":
    # Test prevention system
    import json
    
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Initialize prevention system
    prevention = PreventionSystem(config)
    
    # Test early warning indicators
    print("Testing prevention system...")
    print(f"Prevention enabled: {prevention.enabled}")
    print(f"Auto-terminate: {prevention.auto_terminate}")
    print(f"Lock filesystem: {prevention.lock_filesystem}")
    print(f"Disable network: {prevention.disable_network}")
