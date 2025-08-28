import logging

# Standard logger configuration with file and line information
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    ),
)

# Reusable logger instance for the entire project
logger = logging.getLogger("recruitment_ai")
