"""
Soprano TTS Compatibility Layer

This module provides compatibility patches to run soprano-tts with PyTorch 2.5.1
instead of the officially required PyTorch 2.8.0+.

The only PyTorch 2.8 feature used by soprano-tts is torch.compiler.disable,
which we can safely stub out for older PyTorch versions.
"""

import torch
import sys


def patch_torch_compiler():
    """
    Patch torch.compiler.disable decorator for PyTorch < 2.8
    
    The @torch.compiler.disable decorator was added in PyTorch 2.8.
    For older versions, we create a no-op decorator that simply returns
    the function unchanged.
    """
    
    # Check if torch.compiler exists and has disable
    if hasattr(torch, 'compiler'):
        if hasattr(torch.compiler, 'disable'):
            print("[Soprano Compat] torch.compiler.disable already exists, no patching needed")
            return True
    
    # Create torch.compiler if it doesn't exist
    if not hasattr(torch, 'compiler'):
        print("[Soprano Compat] Creating torch.compiler module for PyTorch < 2.8")
        
        # Create a minimal compiler module
        class CompilerModule:
            @staticmethod
            def disable(fn):
                """No-op decorator for PyTorch < 2.8"""
                return fn
        
        torch.compiler = CompilerModule()
    else:
        # torch.compiler exists but doesn't have disable
        print("[Soprano Compat] Adding disable method to torch.compiler")
        
        def disable_decorator(fn):
            """No-op decorator for PyTorch < 2.8"""
            return fn
        
        torch.compiler.disable = disable_decorator
    
    print(f"[Soprano Compat] Successfully patched torch.compiler.disable for PyTorch {torch.__version__}")
    return True


def check_soprano_compatibility():
    """
    Check if soprano-tts can work with current PyTorch version
    
    Returns:
        tuple: (is_compatible, message)
    """
    pytorch_version = torch.__version__
    major, minor = pytorch_version.split('.')[:2]
    major, minor = int(major), int(minor)
    
    # PyTorch 2.5+ should work with our patches
    if major >= 2 and minor >= 5:
        return True, f"PyTorch {pytorch_version} is compatible with patches"
    else:
        return False, f"PyTorch {pytorch_version} is too old, need at least 2.5.0"


def apply_all_patches():
    """
    Apply all compatibility patches
    
    Returns:
        bool: True if all patches applied successfully
    """
    print("[Soprano Compat] Applying compatibility patches for soprano-tts...")
    
    # Check basic compatibility
    compatible, msg = check_soprano_compatibility()
    if not compatible:
        print(f"[Soprano Compat] ERROR: {msg}")
        return False
    
    print(f"[Soprano Compat] {msg}")
    
    # Apply torch.compiler patch
    if not patch_torch_compiler():
        return False
    
    print("[Soprano Compat] All patches applied successfully!")
    return True


# Auto-apply patches when module is imported
if __name__ != "__main__":
    apply_all_patches()

