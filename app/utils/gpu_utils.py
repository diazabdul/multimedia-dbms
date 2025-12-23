"""
GPU Utility Module - PyTorch Version
Handles GPU detection and configuration for feature extraction
"""
import os
import logging

logger = logging.getLogger(__name__)

# Global GPU configuration
_gpu_enabled = None
_gpu_info = None
_device = None

# Known unsupported GPU architectures (compute capabilities)
# sm_120 = RTX 50 series (Blackwell) - not yet supported in PyTorch 2.6
UNSUPPORTED_SM = ['sm_120', 'sm_121', 'sm_122']


def configure_gpu(force_cpu: bool = False):
    """
    Configure GPU/CPU usage for PyTorch
    
    Args:
        force_cpu: If True, disable GPU even if available
    
    Returns:
        dict with GPU configuration info
    """
    global _gpu_enabled, _gpu_info, _device
    
    try:
        import torch
        
        if force_cpu:
            _device = torch.device('cpu')
            _gpu_enabled = False
            _gpu_info = {'enabled': False, 'reason': 'Forced CPU mode', 'device': 'cpu'}
            logger.info("GPU disabled - using CPU mode")
            return _gpu_info
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            # Check GPU compute capability
            props = torch.cuda.get_device_properties(0)
            compute_cap = f"sm_{props.major}{props.minor}"
            
            # Check if architecture is supported
            if compute_cap in UNSUPPORTED_SM or props.major >= 12:
                logger.warning(f"GPU {device_name} ({compute_cap}) not yet supported by PyTorch")
                _device = torch.device('cpu')
                _gpu_enabled = False
                _gpu_info = {
                    'enabled': False, 
                    'reason': f'{compute_cap} not supported by PyTorch yet',
                    'device': 'cpu',
                    'gpu_detected': device_name,
                    'compute_capability': compute_cap
                }
                logger.info(f"Using CPU fallback for {device_name}")
                return _gpu_info
            
            # Test if GPU actually works
            try:
                test_device = torch.device('cuda')
                test_tensor = torch.zeros(1, device=test_device)
                del test_tensor
                
                _device = test_device
                _gpu_enabled = True
                _gpu_info = {
                    'enabled': True,
                    'device': 'cuda',
                    'device_count': device_count,
                    'device_name': device_name,
                    'cuda_version': cuda_version,
                    'compute_capability': compute_cap
                }
                logger.info(f"GPU enabled: {device_name} (CUDA {cuda_version})")
                
            except RuntimeError as e:
                logger.warning(f"GPU detected but not compatible: {e}")
                _device = torch.device('cpu')
                _gpu_enabled = False
                _gpu_info = {
                    'enabled': False, 
                    'reason': f'GPU error: {str(e)[:50]}',
                    'device': 'cpu',
                    'gpu_detected': device_name
                }
        else:
            _device = torch.device('cpu')
            _gpu_enabled = False
            _gpu_info = {'enabled': False, 'reason': 'CUDA not available', 'device': 'cpu'}
            logger.info("CUDA not available - using CPU mode")
            
    except ImportError:
        _gpu_enabled = False
        _gpu_info = {'enabled': False, 'reason': 'PyTorch not installed', 'device': 'cpu'}
        logger.warning("PyTorch not installed")
    except Exception as e:
        _gpu_enabled = False
        _gpu_info = {'enabled': False, 'reason': str(e), 'device': 'cpu'}
        logger.error(f"GPU detection error: {e}")
    
    return _gpu_info


def get_device():
    """Get the PyTorch device (cuda or cpu)"""
    global _device
    
    if _device is None:
        configure_gpu()
    
    return _device


def is_gpu_available():
    """Check if GPU is available and enabled"""
    global _gpu_enabled
    
    if _gpu_enabled is None:
        configure_gpu()
    
    return _gpu_enabled


def get_gpu_info():
    """Get detailed GPU information"""
    global _gpu_info
    
    if _gpu_info is None:
        configure_gpu()
    
    return _gpu_info


def get_device_summary():
    """Get a human-readable summary of the compute device"""
    info = get_gpu_info()
    
    if info.get('enabled'):
        name = info.get('device_name', 'Unknown GPU')
        cuda_ver = info.get('cuda_version', '')
        return f"🚀 GPU: {name} (CUDA {cuda_ver})"
    else:
        reason = info.get('reason', 'Unknown')
        return f"💻 CPU only ({reason})"


# Auto-configure on import (can be overridden)
def init_gpu(force_cpu: bool = None):
    """
    Initialize GPU configuration
    
    Args:
        force_cpu: If True, use CPU only. If None, auto-detect from environment.
    
    Returns:
        GPU info dictionary
    """
    # Check environment variable for CPU-only mode
    if force_cpu is None:
        force_cpu = os.environ.get('MMDB_FORCE_CPU', '').lower() in ('1', 'true', 'yes')
    
    return configure_gpu(force_cpu=force_cpu)


# Convenience function to print status
def print_gpu_status():
    """Print GPU status to console"""
    info = get_gpu_info()
    
    print("\n" + "=" * 50)
    print("🖥️  PyTorch Compute Device Status")
    print("=" * 50)
    
    if info.get('enabled'):
        print(f"✅ GPU Enabled")
        print(f"   Device: {info.get('device_name', 'Unknown')}")
        print(f"   CUDA Version: {info.get('cuda_version', 'Unknown')}")
        print(f"   Device Count: {info.get('device_count', 1)}")
    else:
        print(f"❌ GPU Not Available")
        print(f"   Reason: {info.get('reason', 'Unknown')}")
        print(f"   Running on: CPU")
    
    print("=" * 50 + "\n")


if __name__ == '__main__':
    # Test GPU detection
    init_gpu()
    print_gpu_status()
