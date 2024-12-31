import logging

# Logger 1: codejournal
logger = logging.getLogger("codejournal")
logger.setLevel(logging.INFO)

codejournal_format = "%(asctime)s - %(levelname)s - %(message)s"
console_handler = logging.StreamHandler()  # Output to console
console_formatter = logging.Formatter(codejournal_format)
console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)

# Logger 2: trainer
train_logger = logging.getLogger("trainer")
train_logger.setLevel(logging.INFO)

train_log_format = f'\n{"*"*25}\n%(asctime)s %(levelname)s:\n\n%(message)s\n{"*"*25}\n'
file_handler = logging.FileHandler("train.log")  # Output to file
file_formatter = logging.Formatter(train_log_format)
file_handler.setFormatter(file_formatter)

train_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()  # Output to console
console_formatter = logging.Formatter(train_log_format)
console_handler.setFormatter(console_formatter)
train_logger.addHandler(console_handler)
