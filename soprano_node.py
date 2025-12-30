"""
Soprano TTS Node for ComfyUI

Integrates Soprano TTS (https://github.com/ekwek1/soprano) with ComfyUI's
memory management and interrupt systems.

Includes integrated audio saving and playback (based on SCG Save Audio).
"""

import os
import gc
import glob
import random
import string
import re
import torch
import numpy as np
from pathlib import Path
import folder_paths

# Try to import pydub for audio export
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("[Soprano TTS] Warning: pydub not installed. Audio saving/playback disabled.")
    print("[Soprano TTS] Install with: pip install pydub")

# Apply compatibility patches BEFORE importing soprano
try:
    from .soprano_compat import apply_all_patches
    apply_all_patches()
except Exception as e:
    print(f"[Soprano TTS] Warning: Could not apply compatibility patches: {e}")
    print("[Soprano TTS] This may cause issues if using PyTorch < 2.8")

# Try to import ComfyUI's model management
try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None
    print("[Soprano TTS] Warning: Could not import comfy.model_management")


class SopranoTTSNode:
    """
    Soprano TTS Node for ComfyUI
    
    Ultra-lightweight (80M params), real-time text-to-speech synthesis.
    Generates high-fidelity 32kHz audio at ~2000x real-time speed.
    
    Features integrated audio saving and playback.
    """
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = self._get_models_directory()
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        
    def _get_models_directory(self):
        """Get or create the TTS models directory"""
        # Use ComfyUI's models directory structure
        base_models_dir = folder_paths.models_dir
        tts_dir = os.path.join(base_models_dir, "TTS")
        
        # Create directory if it doesn't exist
        os.makedirs(tts_dir, exist_ok=True)
        
        return tts_dir
    
    def _check_soprano_installed(self):
        """Check if soprano-tts package is installed"""
        try:
            import soprano
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"[Soprano TTS] Warning: soprano package found but error importing: {e}")
            return False
    
    def _load_model(self):
        """Load the Soprano TTS model"""
        if self.model is not None:
            return self.model
        
        # Check if soprano-tts is installed
        if not self._check_soprano_installed():
            raise RuntimeError(
                "soprano-tts is not installed.\n\n"
                "⚠️  WARNING: Soprano TTS requires PyTorch 2.8.0+ which may conflict with ComfyUI!\n"
                "ComfyUI typically uses PyTorch 2.5.1, and upgrading will break ComfyUI.\n\n"
                "See the README.md for installation options:\n"
                "1. Try: pip install soprano-tts --no-deps && pip install unidecode scipy transformers\n"
                "2. Or use a dedicated Python environment for Soprano\n"
                "3. See: ComfyUI/custom_nodes/ComfyUI-Soprano-TTS/README.md for details"
            )
        
        print("[Soprano TTS] Loading Soprano TTS model...")
        
        try:
            from soprano import SopranoTTS
            
            # Create model with optimized settings
            # The model will automatically download to HuggingFace cache if not present
            self.model = SopranoTTS(
                backend='auto',  # Will use LMDeploy if available, else transformers
                device=self.device,
                cache_size_mb=10,
                decoder_batch_size=1
            )
            
            print(f"[Soprano TTS] Model loaded successfully on {self.device}")
            return self.model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Soprano TTS model: {e}")
    
    def _unload_model(self):
        """Unload model and free memory"""
        if self.model is not None:
            # Move model to CPU if possible
            try:
                if hasattr(self.model, 'model'):
                    if hasattr(self.model.model, 'to'):
                        self.model.model.to('cpu')
                if hasattr(self.model, 'decoder'):
                    if hasattr(self.model.decoder, 'to'):
                        self.model.decoder.to('cpu')
            except Exception:
                pass
            
            self.model = None
            
            # Clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    def _get_next_counter(self, folder, base_filename, extension):
        """Find the next available counter for the filename pattern."""
        pattern = os.path.join(folder, f"{base_filename}_*.{extension}")
        existing_files = glob.glob(pattern)
        
        max_counter = 0
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            try:
                name_without_ext = os.path.splitext(filename)[0]
                parts = name_without_ext.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    counter = int(parts[1])
                    max_counter = max(max_counter, counter)
            except (ValueError, IndexError):
                continue
        
        return max_counter + 1
    
    def _sanitize_filename(self, name_string):
        """Sanitize a string for use in filenames."""
        if not name_string:
            return ""
        name_string = re.sub(r'[\s/:*?"<>|]+', '_', name_string)
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', name_string)
        return sanitized[:100]
    
    def _export_audio(self, audio_tensor, sample_rate, filename_prefix, format, bitrate, save):
        """Export audio to file and return UI payload for player."""
        if not PYDUB_AVAILABLE:
            print("[Soprano TTS] pydub not available, skipping audio export")
            return None
        
        try:
            # Convert tensor to numpy for pydub
            if audio_tensor.dim() == 3:
                waveform = audio_tensor[0].cpu()  # Remove batch dim
            elif audio_tensor.dim() == 2:
                waveform = audio_tensor.cpu()
            else:
                waveform = audio_tensor.unsqueeze(0).cpu()
            
            # Convert to int16 for AudioSegment
            waveform_np = waveform.numpy().T
            audio_data_int16 = (waveform_np * 32767).astype(np.int16)
            num_channels = audio_data_int16.shape[1] if audio_data_int16.ndim > 1 else 1
            
            audio_segment = AudioSegment(
                data=audio_data_int16.tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=num_channels
            )
            
            # Determine output path
            if save:
                num_samples = int(audio_segment.duration_seconds * audio_segment.frame_rate)
                resolved_folder, base_filename, _, subfolder, _ = folder_paths.get_save_image_path(
                    filename_prefix, self.output_dir, num_samples, num_channels
                )
                counter = self._get_next_counter(resolved_folder, base_filename, format)
                final_filename = f"{base_filename}_{counter:05}.{format}"
                full_path = os.path.join(resolved_folder, final_filename)
                output_type = "output"
            else:
                random_prefix = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
                final_filename = f"soprano_temp_{random_prefix}.{format}"
                subfolder = ""
                full_path = os.path.join(self.temp_dir, final_filename)
                output_type = "temp"
            
            # Build export parameters
            export_params = {}
            if format in ["mp3", "ogg"]:
                if bitrate == "VBR":
                    export_params['parameters'] = ["-q:a", "4"]
                else:
                    export_params['bitrate'] = bitrate
            elif format == "flac":
                export_params['parameters'] = ["-compression_level", "5"]
            
            # Export
            audio_segment.export(full_path, format=format, **export_params)
            
            if save:
                print(f"[Soprano TTS] Saved audio to: {full_path}")
            else:
                print(f"[Soprano TTS] Exported preview to: {full_path}")
            
            return {
                "filename": final_filename,
                "subfolder": subfolder,
                "type": output_type
            }
            
        except Exception as e:
            print(f"[Soprano TTS] Error exporting audio: {e}")
            return None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! I am Soprano, an ultra-lightweight text to speech model."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False
                }),
                # Audio output settings
                "filename_prefix": ("STRING", {
                    "default": "audio/soprano_tts"
                }),
                "format": (["wav", "mp3", "flac", "ogg"], {
                    "default": "wav"
                }),
                "bitrate": (["128k", "192k", "256k", "320k", "VBR"], {
                    "default": "320k"
                }),
                "save": ("BOOLEAN", {
                    "default": True
                }),
                "autoplay": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    OUTPUT_NODE = True
    CATEGORY = "audio/generation"
    
    def generate_speech(self, text, temperature, top_p, repetition_penalty, seed, keep_model_loaded,
                        filename_prefix, format, bitrate, save, autoplay):
        """
        Generate speech from text using Soprano TTS
        
        Args:
            text: Input text to synthesize
            temperature: Sampling temperature (lower = more consistent, higher = more varied)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            seed: Random seed (for ComfyUI caching - not used in generation)
            keep_model_loaded: Keep model in memory after generation
            filename_prefix: Prefix for saved audio files
            format: Audio format (wav, mp3, flac, ogg)
            bitrate: Bitrate for compressed formats
            save: Whether to save to output folder (False = temp only)
            autoplay: Whether to autoplay in the UI
            
        Returns:
            Audio dictionary and UI payload for player
        """
        
        # Note: seed parameter is not used in generation, it's just here to force
        # ComfyUI to re-execute the node when the seed changes (ComfyUI quirk)
        
        try:
            # Load model
            model = self._load_model()
            
            # Check for interrupts before generation
            if model_management is not None:
                try:
                    model_management.throw_exception_if_processing_interrupted()
                except Exception as e:
                    print("[Soprano TTS] Generation interrupted by user")
                    if not keep_model_loaded:
                        self._unload_model()
                    raise e
            
            print(f"[Soprano TTS] Generating speech for text: '{text[:50]}...'")
            print(f"[Soprano TTS] Parameters - temp: {temperature}, top_p: {top_p}, rep_penalty: {repetition_penalty}")
            
            # Generate audio
            # Soprano returns a tensor of shape [channels, samples]
            audio_tensor = model.infer(
                text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Check for interrupts after generation
            if model_management is not None:
                try:
                    model_management.throw_exception_if_processing_interrupted()
                except Exception as e:
                    print("[Soprano TTS] Post-generation interrupted by user")
                    if not keep_model_loaded:
                        self._unload_model()
                    raise e
            
            # Convert to ComfyUI audio format
            # Soprano outputs [channels, samples], we need [batch, channels, samples]
            if audio_tensor.dim() == 1:
                # Mono audio [samples] -> [1, 1, samples]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Soprano generates at 32kHz
            sample_rate = 32000
            
            # Create ComfyUI audio dictionary
            audio_dict = {
                "waveform": audio_tensor,
                "sample_rate": sample_rate
            }
            
            print(f"[Soprano TTS] Generated audio shape: {audio_tensor.shape}, sample_rate: {sample_rate}")
            
            # Export audio and get UI payload for player
            export_result = self._export_audio(
                audio_tensor, sample_rate, filename_prefix, format, bitrate, save
            )
            
            # Unload model if not keeping it loaded
            if not keep_model_loaded:
                self._unload_model()
            
            # Build response with both output and UI
            if export_result:
                export_result["autoplay"] = autoplay
                ui_payload = {"audio": [export_result]}
                return {"result": (audio_dict,), "ui": ui_payload}
            else:
                # No export (pydub not available), just return audio
                return (audio_dict,)
            
        except Exception as e:
            print(f"[Soprano TTS] Error during generation: {e}")
            # Always try to unload on error
            if not keep_model_loaded:
                self._unload_model()
            raise e


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SopranoTTS": SopranoTTSNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SopranoTTS": "Soprano TTS"
}

