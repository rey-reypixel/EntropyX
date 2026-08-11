import winreg
import time
import threading
from event_collector import collector


# ==========================================================
# REGISTRY MONITOR
# ==========================================================

class RegistryMonitor:
    """
    Monitor Windows registry changes using polling.
    Note: Real-time registry monitoring requires complex hooks.
    This implementation uses polling for key registry hives.
    """
    
    def __init__(self, poll_interval=5):
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None
        
        # Registry hives to monitor
        self.hives = {
            "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
            "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
            "HKEY_CLASSES_ROOT": winreg.HKEY_CLASSES_ROOT,
            "HKEY_USERS": winreg.HKEY_USERS
        }
        
        # Track registry state
        self.registry_state = {}
        self._initialize_state()
    
    def _initialize_state(self):
        """
        Initialize registry state by taking a snapshot.
        """
        for hive_name, hive_key in self.hives.items():
            try:
                self.registry_state[hive_name] = self._get_key_count(hive_key)
            except Exception as e:
                print(f"[WARNING] Could not initialize {hive_name}: {e}")
                self.registry_state[hive_name] = 0
    
    def _get_key_count(self, hive_key):
        """
        Get approximate count of keys in a hive (sampling).
        """
        count = 0
        try:
            # Sample a few common paths to estimate activity
            sample_paths = [
                r"SOFTWARE",
                r"SYSTEM",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
            ]
            
            for path in sample_paths:
                try:
                    key = winreg.OpenKey(hive_key, path)
                    count += winreg.QueryInfoKey(key)[0]  # Subkey count
                    winreg.CloseKey(key)
                except:
                    pass
        except:
            pass
        return count
    
    def _monitor_changes(self):
        """
        Monitor for registry changes by polling.
        """
        while self.running:
            try:
                for hive_name, hive_key in self.hives.items():
                    try:
                        current_count = self._get_key_count(hive_key)
                        previous_count = self.registry_state.get(hive_name, 0)
                        
                        # Detect significant changes
                        if current_count != previous_count:
                            change_type = "written" if current_count > previous_count else "read"
                            
                            collector.add_event(
                                source="registry",
                                event=change_type,
                                process="Unknown",
                                pid=-1,
                                details={
                                    "hive": hive_name,
                                    "previous_count": previous_count,
                                    "current_count": current_count,
                                    "change": current_count - previous_count
                                }
                            )
                            
                            self.registry_state[hive_name] = current_count
                    except Exception as e:
                        continue
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                print(f"Registry Monitor Error: {e}")
                time.sleep(self.poll_interval)
    
    def start(self):
        """
        Start registry monitoring in a separate thread.
        """
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_changes, daemon=True)
        self.thread.start()
        print("Registry Monitor Started")
    
    def stop(self):
        """
        Stop registry monitoring.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


# ==========================================================
# START FUNCTION (for compatibility with other monitors)
# ==========================================================

def start():
    """
    Start registry monitoring (blocking call for compatibility).
    """
    print("\n===================================")
    print(" Registry Monitor Started")
    print("===================================\n")
    
    monitor = RegistryMonitor(poll_interval=5)
    monitor.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nRegistry Monitor Stopped.")
        monitor.stop()


if __name__ == "__main__":
    start()
