"""

--------------------------------------------------------------------------------
from .imports import *

# Importing custom libraries hack
path = "your_path"
sys.path.insert(0, path)


--------------------------------------------------------------------------------
Linux:

du <path> -h # Find how much space is used by a file or folder
df -h # Find disk usage-overall
ls <pat> | wc -l # Count the number of files in a folder

tmux attach -t <session_name> # Attach to a session
Ctrl-b + [ # tmux copy mode, enter q to exit
Ctrl-b + d # Detach from a session
--------------------------------------------------------------------------------
Jupyter:

# matplotlib retina mode
%config InlineBackend.figure_format = 'retina'

# Capture the output of a cell
%%capture
--------------------------------------------------------------------------------
Better randomness:
from Crypto.Random import random # pip install pycryptodome

# drop in replacement for random.randint, random.random, random.randrange, etc
--------------------------------------------------------------------------------

"""

__all__ = []


