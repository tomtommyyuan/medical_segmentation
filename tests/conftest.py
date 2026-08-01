"""Put src/ on the import path so tests can import the scripts directly."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
