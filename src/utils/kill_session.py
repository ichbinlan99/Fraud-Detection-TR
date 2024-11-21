import os
import signal

def kill_process_on_port(port):
    # Find the process ID (PID) using the port
    command = f"lsof -t -i:{port}"
    process = os.popen(command).read().strip()
    
    if process:
        # Kill the process
        os.kill(int(process), signal.SIGKILL)
        print(f"Process on port {port} has been terminated.")
    else:
        print(f"No process found on port {port}.")
