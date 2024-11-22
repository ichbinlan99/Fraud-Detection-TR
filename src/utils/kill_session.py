import os
import signal

def kill_process_on_port(port):
    """
    Terminates the process running on the specified port.
    This function finds the process ID (PID) of the process using the given port
    and then kills the process.
    Args:
        port (int): The port number on which the process is running.
    Raises:
        ProcessLookupError: If no process is found on the specified port.
        OSError: If an error occurs while attempting to kill the process.
    """

    # Find the process ID (PID) using the port
    command = f"lsof -t -i:{port}"
    process = os.popen(command).read().strip()
    
    if process:
        # Kill the process
        os.kill(int(process), signal.SIGKILL)
        print(f"Process on port {port} has been terminated.")
    else:
        print(f"No process found on port {port}.")
