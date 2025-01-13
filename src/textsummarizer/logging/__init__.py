import os
import sys
import logging

log_dirs = "logs"

os.makedirs(log_dirs, exist_ok=True)

logging_str = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(os.path.join(log_dirs, "app.log")), 
        logging.StreamHandler(sys.stdout)  
    ]
)


logging.info("Logging system initialized successfully.")

