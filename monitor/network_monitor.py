import psutil
import socket
import time

from event_collector import collector


# ==========================================================
# NETWORK MONITOR
# ==========================================================

def start():

    print("\n===================================")
    print(" Network Monitor Started")
    print("===================================\n")

    # Initialize with all currently active network connections to avoid startup false-positive surge
    seen_connections = set()
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.raddr:
                pid = conn.pid if conn.pid else -1
                local_ip = conn.laddr.ip
                local_port = conn.laddr.port
                remote_ip = conn.raddr.ip
                remote_port = conn.raddr.port
                status = conn.status
                protocol = "TCP"
                try:
                    if conn.type == socket.SOCK_DGRAM:
                        protocol = "UDP"
                except:
                    pass
                connection_id = (pid, local_ip, local_port, remote_ip, remote_port, protocol)
                seen_connections.add(connection_id)
    except Exception:
        pass

    while True:

        try:

            connections = psutil.net_connections(kind="inet")

            for conn in connections:

                try:

                    # Ignore connections without remote endpoint
                    if not conn.raddr:
                        continue

                    pid = conn.pid if conn.pid else -1

                    process_name = "Unknown"

                    if pid != -1:

                        try:
                            process_name = psutil.Process(pid).name()
                        except:
                            process_name = "Unknown"

                    local_ip = conn.laddr.ip
                    local_port = conn.laddr.port

                    remote_ip = conn.raddr.ip
                    remote_port = conn.raddr.port

                    status = conn.status

                    protocol = "TCP"

                    try:

                        if conn.type == socket.SOCK_DGRAM:
                            protocol = "UDP"

                    except:
                        pass

                    connection_id = (
                        pid,
                        local_ip,
                        local_port,
                        remote_ip,
                        remote_port,
                        protocol
                    )

                    if connection_id in seen_connections:
                        continue

                    seen_connections.add(connection_id)

                    collector.add_event(

                        source="network",

                        event="connection",

                        process=process_name,

                        pid=pid,

                        details={

                            "protocol": protocol,

                            "status": status,

                            "local_ip": local_ip,

                            "local_port": local_port,

                            "remote_ip": remote_ip,

                            "remote_port": remote_port

                        }

                    )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess
                ):
                    continue

                except Exception:
                    continue

            time.sleep(3)

        except KeyboardInterrupt:

            print("\nNetwork Monitor Stopped.")

            break

        except Exception as e:

            print(f"Network Monitor Error: {e}")

            time.sleep(3)