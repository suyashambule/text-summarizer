import os
import sys
import logging

# Directory to store log files
log_dirs = "logs"

# Create the logs directory if it doesn't exist
os.makedirs(log_dirs, exist_ok=True)

# Define the logging format
logging_str = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'

# Set up logging to output to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(os.path.join(log_dirs, "app.log")),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Example logging message
logging.info("Logging system initialized successfully.")

