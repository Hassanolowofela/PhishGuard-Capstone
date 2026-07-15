"""Test configuration: make the webapp modules importable.

The serving code lives in webapp/, so add it to the path before the tests
import features and inference.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webapp"))
