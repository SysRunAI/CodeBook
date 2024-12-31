import os
import gc
import cv2
import sys
import time
import json
import math
import wandb
import joblib
import shutil
import logging
import random
from Crypto.Random import random as crandom # pip install pycryptodome
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

# Typing
from typing import Union, List, Dict, Any, Tuple, Optional, Callable, Iterable


# Importing libraries
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# Dev
sns.set(style="darkgrid")
warnings.filterwarnings("ignore")
tqdm.pandas()

# env
from dotenv import load_dotenv
load_dotenv('./codebook/.env')

# Video
from moviepy import *  # latest version

from .loggers import logger
# Tests
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA devices: {torch.cuda.device_count()}")

# environ check
if not os.environ.get("SLACK_WEBHOOK_URL"):
    logger.warning("Slack webhook URL (SLACK_WEBHOOK_URL) not found. Slack notifications will not be sent.")

if not os.environ.get("HUGGINGFACE_TOKEN"):
    logger.warning("HuggingFace token (HUGGINGFACE_TOKEN) not found. Pushing to the Hub will not be possible.")