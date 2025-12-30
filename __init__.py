"""
ComfyUI-Soprano-TTS
A ComfyUI custom node for Soprano TTS - Ultra-lightweight, high-fidelity text-to-speech.
"""

from .soprano_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Web directory for frontend JavaScript (audio player)
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

