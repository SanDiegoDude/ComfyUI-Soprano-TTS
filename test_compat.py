#!/usr/bin/env python3
"""
Test script to verify soprano-tts compatibility with PyTorch 2.5.1

Run this BEFORE installing soprano-tts to verify your environment,
or AFTER installing to test if everything works.
"""

import sys


def test_pytorch_version():
    """Test PyTorch version"""
    print("=" * 60)
    print("Testing PyTorch Version")
    print("=" * 60)
    
    try:
        import torch
        version = torch.__version__
        print(f"✅ PyTorch installed: {version}")
        
        major, minor = version.split('.')[:2]
        major, minor = int(major), int(minor)
        
        if major >= 2 and minor >= 5:
            print(f"✅ PyTorch {version} is compatible with our patches")
            return True
        else:
            print(f"❌ PyTorch {version} is too old (need 2.5.0+)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False


def test_compatibility_patches():
    """Test compatibility patches"""
    print("\n" + "=" * 60)
    print("Testing Compatibility Patches")
    print("=" * 60)
    
    try:
        from soprano_compat import apply_all_patches
        result = apply_all_patches()
        
        if result:
            print("✅ All compatibility patches applied successfully")
            return True
        else:
            print("❌ Failed to apply compatibility patches")
            return False
    except ImportError:
        print("❌ soprano_compat.py not found")
        return False
    except Exception as e:
        print(f"❌ Error applying patches: {e}")
        return False


def test_soprano_import():
    """Test soprano-tts import"""
    print("\n" + "=" * 60)
    print("Testing Soprano TTS Import")
    print("=" * 60)
    
    try:
        from soprano import SopranoTTS
        print("✅ soprano-tts imported successfully")
        return True
    except ImportError as e:
        print(f"⚠️  soprano-tts not installed: {e}")
        print("   This is expected if you haven't installed it yet")
        print("   Install with: pip install soprano-tts --no-deps")
        return None  # Not a failure, just not installed
    except Exception as e:
        print(f"❌ Error importing soprano-tts: {e}")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "=" * 60)
    print("Testing Dependencies")
    print("=" * 60)
    
    deps = {
        'unidecode': 'pip install unidecode',
        'scipy': 'pip install scipy',
        'transformers': 'pip install transformers',
        'torch': 'pip install torch',
    }
    
    all_ok = True
    for dep, install_cmd in deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"❌ {dep} not installed - run: {install_cmd}")
            all_ok = False
    
    return all_ok


def test_torch_compiler():
    """Test torch.compiler.disable patch"""
    print("\n" + "=" * 60)
    print("Testing torch.compiler.disable Patch")
    print("=" * 60)
    
    try:
        import torch
        
        if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
            print("✅ torch.compiler.disable is available")
            
            # Test that it works as a decorator
            @torch.compiler.disable
            def test_func():
                return "test"
            
            result = test_func()
            if result == "test":
                print("✅ torch.compiler.disable decorator works correctly")
                return True
            else:
                print("❌ torch.compiler.disable decorator not working properly")
                return False
        else:
            print("❌ torch.compiler.disable not available")
            return False
    except Exception as e:
        print(f"❌ Error testing torch.compiler: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Soprano TTS Compatibility Test")
    print("=" * 60)
    print()
    
    results = {}
    
    # Run tests
    results['pytorch'] = test_pytorch_version()
    results['patches'] = test_compatibility_patches()
    results['torch_compiler'] = test_torch_compiler()
    results['dependencies'] = test_dependencies()
    results['soprano'] = test_soprano_import()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        
        print(f"{test_name:20s}: {status}")
    
    # Overall result and recommendations
    print("\n" + "=" * 60)
    
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    
    if not failed:
        if skipped:
            print("✅ All tests passed (some skipped)")
            print("\n" + "=" * 60)
            print("NEXT STEPS")
            print("=" * 60)
            if 'soprano' in skipped:
                print("\n📦 Install soprano-tts:")
                print("   pip install soprano-tts --no-deps")
                print("   pip install unidecode")
                print("\n🔄 Then restart ComfyUI")
        else:
            print("✅ ALL TESTS PASSED! Soprano TTS should work!")
            print("\nYou're ready to use the Soprano TTS node in ComfyUI!")
    else:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        print("\n" + "=" * 60)
        print("HOW TO FIX")
        print("=" * 60)
        
        # Specific fix recommendations for each failure
        if 'pytorch' in failed:
            print("\n🔧 PyTorch version issue:")
            print("   Your PyTorch is too old (need 2.5.0+)")
            print("   Fix: pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121")
            
        if 'dependencies' in failed:
            print("\n🔧 Missing dependencies:")
            print("   Fix: pip install unidecode scipy transformers")
            
        if 'patches' in failed or 'torch_compiler' in failed:
            print("\n🔧 Compatibility patch issue:")
            print("   This usually means PyTorch is broken or incompatible")
            print("   Fix: Reinstall PyTorch:")
            print("      pip uninstall torch")
            print("      pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121")
            
        if 'soprano' in failed:
            print("\n🔧 Soprano import failed:")
            print("   soprano-tts is installed but broken")
            print("   Fix:")
            print("      pip uninstall soprano-tts")
            print("      pip install soprano-tts --no-deps")
            print("      pip install unidecode")
        
        print("\n💡 If ComfyUI won't start (undefined symbol: ncclMemFree):")
        print("   Your PyTorch got upgraded by lmdeploy. Restore it:")
        print("      pip uninstall lmdeploy torch triton")
        print("      pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \\")
        print("        --extra-index-url https://download.pytorch.org/whl/cu121")
        print("      pip install xformers==0.0.29.post1")
        
        print("\n📖 For detailed help, see README.md")
    
    print("=" * 60)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

