"""
Resource Manager for Brain MRI Tumor Classification.
Manages GPU/CPU resources, memory optimization, and hardware monitoring.
"""

import os
import sys
import gc
import time
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import torch
import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVICE, BATCH_SIZE


# ==================== ENUMS AND DATA CLASSES ====================

class DeviceType(Enum):
    """Device types for computation."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class GPUInfo:
    """GPU information data class."""
    name: str
    index: int
    total_memory_mb: float
    free_memory_mb: float
    used_memory_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None


@dataclass
class SystemInfo:
    """System information data class."""
    os_name: str
    os_version: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    total_ram_gb: float
    available_ram_gb: float
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    gpu_count: int
    gpus: List[GPUInfo]


# ==================== RESOURCE MANAGER ====================

class ResourceManager:
    """
    Manages computational resources for training and inference.
    
    Features:
    - GPU/CPU detection and monitoring
    - Memory management and optimization
    - Automatic batch size recommendation
    - Resource cleanup
    """
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize the resource manager.
        
        Args:
            preferred_device: Preferred device ('cuda', 'cpu', 'auto')
        """
        self.preferred_device = preferred_device or 'auto'
        self._device = None
        self._gpu_info_cache = None
        self._cache_time = 0
        self._cache_ttl = 5  # Cache TTL in seconds
    
    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            self._device = self._select_device()
        return self._device
    
    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if self.preferred_device == 'cpu':
            return torch.device('cpu')
        
        if self.preferred_device == 'cuda' or self.preferred_device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        
        return torch.device('cpu')
    
    # ==================== SYSTEM INFO ====================
    
    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information."""
        # CPU info
        cpu_name = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1
        
        # RAM info
        ram = psutil.virtual_memory()
        total_ram_gb = ram.total / (1024 ** 3)
        available_ram_gb = ram.available / (1024 ** 3)
        
        # GPU info
        gpus = self.get_all_gpu_info()
        
        # CUDA info
        cuda_version = None
        cudnn_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if torch.backends.cudnn.is_available():
                cudnn_version = str(torch.backends.cudnn.version())
        
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            python_version=platform.python_version(),
            pytorch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_count=len(gpus),
            gpus=gpus
        )
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all available GPUs."""
        gpus = []
        
        if not torch.cuda.is_available():
            return gpus
        
        for i in range(torch.cuda.device_count()):
            gpus.append(self.get_gpu_info(i))
        
        return gpus
    
    def get_gpu_info(self, device_index: int = 0) -> GPUInfo:
        """
        Get detailed GPU information.
        
        Args:
            device_index: GPU device index
            
        Returns:
            GPUInfo object
        """
        if not torch.cuda.is_available():
            return GPUInfo(
                name="No GPU",
                index=0,
                total_memory_mb=0,
                free_memory_mb=0,
                used_memory_mb=0,
                utilization_percent=0
            )
        
        props = torch.cuda.get_device_properties(device_index)
        total_mem = props.total_memory / (1024 ** 2)
        
        # Get current memory usage
        torch.cuda.synchronize(device_index)
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
        free = total_mem - reserved
        
        # Try to get utilization (requires nvidia-smi)
        utilization = 0.0
        temperature = None
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits', f'-i={device_index}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                utilization = float(parts[0].strip())
                temperature = float(parts[1].strip())
        except:
            pass
        
        return GPUInfo(
            name=props.name,
            index=device_index,
            total_memory_mb=total_mem,
            free_memory_mb=free,
            used_memory_mb=reserved,
            utilization_percent=utilization,
            temperature_celsius=temperature
        )
    
    # ==================== MEMORY MANAGEMENT ====================
    
    def get_available_memory(self) -> float:
        """
        Get available memory in MB.
        
        Returns:
            Available memory in MB
        """
        if self.device.type == 'cuda':
            gpu_info = self.get_gpu_info(self.device.index if self.device.index else 0)
            return gpu_info.free_memory_mb
        else:
            ram = psutil.virtual_memory()
            return ram.available / (1024 ** 2)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB
        """
        stats = {
            "ram_used_mb": psutil.virtual_memory().used / (1024 ** 2),
            "ram_available_mb": psutil.virtual_memory().available / (1024 ** 2),
            "ram_percent": psutil.virtual_memory().percent,
        }
        
        if self.device.type == 'cuda':
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return stats
    
    def clear_memory(self, aggressive: bool = False) -> None:
        """
        Clear unused memory.
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        # Python garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
            if aggressive:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
    
    def optimize_memory(self) -> None:
        """Optimize memory usage for training."""
        # Enable memory-efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable TF32 for faster computation on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
    
    # ==================== BATCH SIZE OPTIMIZATION ====================
    
    def recommend_batch_size(
        self,
        model_name: str,
        image_size: int = 224,
        target_memory_usage: float = 0.8
    ) -> int:
        """
        Recommend optimal batch size based on available resources.
        
        Args:
            model_name: Name of the model
            image_size: Input image size
            target_memory_usage: Target GPU memory usage (0-1)
            
        Returns:
            Recommended batch size
        """
        # Model memory estimates (approximate, in MB per image)
        model_memory_per_image = {
            "resnet50": 25,
            "resnet101": 35,
            "densenet121": 30,
            "efficientnet_b4": 45,
            "mobilenetv2": 15,
        }
        
        # Base memory overhead for model weights
        model_base_memory = {
            "resnet50": 200,
            "resnet101": 350,
            "densenet121": 100,
            "efficientnet_b4": 150,
            "mobilenetv2": 50,
        }
        
        available_memory = self.get_available_memory()
        
        # Reserve memory for model and overhead
        base_usage = model_base_memory.get(model_name, 200)
        usable_memory = (available_memory * target_memory_usage) - base_usage
        
        # Calculate batch size
        per_image_memory = model_memory_per_image.get(model_name, 30)
        recommended = int(usable_memory / per_image_memory)
        
        # Clamp to reasonable range
        recommended = max(1, min(recommended, 128))
        
        # Round to power of 2 for efficiency
        powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
        for p in reversed(powers_of_2):
            if p <= recommended:
                return p
        
        return 1
    
    def get_optimal_workers(self) -> int:
        """
        Get optimal number of data loader workers.
        
        Returns:
            Recommended number of workers
        """
        cpu_count = psutil.cpu_count(logical=False) or 2
        
        # Use half of available cores, minimum 2, maximum 8
        return max(2, min(cpu_count // 2, 8))
    
    # ==================== CONTEXT MANAGERS ====================
    
    def training_context(self):
        """Context manager for training optimization."""
        return TrainingContext(self)
    
    def inference_context(self):
        """Context manager for inference optimization."""
        return InferenceContext(self)
    
    # ==================== REPORTING ====================
    
    def print_system_info(self) -> None:
        """Print formatted system information."""
        info = self.get_system_info()
        
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        
        print(f"\n📋 Operating System")
        print(f"   {info.os_name} {info.os_version}")
        
        print(f"\n💻 CPU")
        print(f"   {info.cpu_name}")
        print(f"   Cores: {info.cpu_cores} | Threads: {info.cpu_threads}")
        
        print(f"\n🧠 RAM")
        print(f"   Total: {info.total_ram_gb:.1f} GB")
        print(f"   Available: {info.available_ram_gb:.1f} GB")
        
        print(f"\n🐍 Python & PyTorch")
        print(f"   Python: {info.python_version}")
        print(f"   PyTorch: {info.pytorch_version}")
        
        print(f"\n🎮 GPU")
        if info.cuda_available:
            print(f"   CUDA: {info.cuda_version}")
            print(f"   cuDNN: {info.cudnn_version}")
            for gpu in info.gpus:
                print(f"\n   GPU {gpu.index}: {gpu.name}")
                print(f"   Memory: {gpu.free_memory_mb:.0f} MB free / {gpu.total_memory_mb:.0f} MB total")
                if gpu.temperature_celsius:
                    print(f"   Temperature: {gpu.temperature_celsius}°C")
                print(f"   Utilization: {gpu.utilization_percent}%")
        else:
            print("   No CUDA GPU available")
        
        print(f"\n🔧 Selected Device: {self.device}")
        print("=" * 60)


# ==================== CONTEXT MANAGERS ====================

class TrainingContext:
    """Context manager for training optimization."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.rm = resource_manager
        self.original_settings = {}
    
    def __enter__(self):
        # Store original settings
        if torch.cuda.is_available():
            self.original_settings['benchmark'] = torch.backends.cudnn.benchmark
        
        # Optimize for training
        self.rm.optimize_memory()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original settings
        if 'benchmark' in self.original_settings:
            torch.backends.cudnn.benchmark = self.original_settings['benchmark']
        
        # Clean up
        self.rm.clear_memory()
        
        return False


class InferenceContext:
    """Context manager for inference optimization."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.rm = resource_manager
    
    def __enter__(self):
        # Optimize for inference
        torch.set_grad_enabled(False)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Re-enable gradients
        torch.set_grad_enabled(True)
        
        # Clean up
        self.rm.clear_memory()
        
        return False


# ==================== MEMORY MONITOR ====================

class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self, resource_manager: ResourceManager, log_interval: int = 10):
        """
        Initialize memory monitor.
        
        Args:
            resource_manager: ResourceManager instance
            log_interval: Logging interval in seconds
        """
        self.rm = resource_manager
        self.log_interval = log_interval
        self.history = []
        self.start_time = None
    
    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.time()
        self.history = []
        self._log_memory()
    
    def _log_memory(self) -> None:
        """Log current memory usage."""
        stats = self.rm.get_memory_usage()
        stats['timestamp'] = time.time() - self.start_time if self.start_time else 0
        self.history.append(stats)
    
    def log(self) -> None:
        """Log memory if interval has passed."""
        if not self.history:
            self._log_memory()
            return
        
        elapsed = time.time() - self.start_time
        last_log = self.history[-1]['timestamp']
        
        if elapsed - last_log >= self.log_interval:
            self._log_memory()
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.history:
            return {}
        
        peaks = {}
        for key in self.history[0].keys():
            if key != 'timestamp':
                peaks[f"peak_{key}"] = max(h[key] for h in self.history)
        
        return peaks
    
    def print_summary(self) -> None:
        """Print memory usage summary."""
        peaks = self.get_peak_memory()
        
        print("\n📊 Memory Usage Summary")
        print("-" * 40)
        for key, value in peaks.items():
            print(f"   {key}: {value:.1f} MB")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("Resource Manager - System Check\n")
    
    rm = ResourceManager()
    rm.print_system_info()
    
    print("\n📦 Batch Size Recommendations:")
    for model in ["resnet50", "resnet101", "densenet121", "efficientnet_b4", "mobilenetv2"]:
        bs = rm.recommend_batch_size(model)
        print(f"   {model}: {bs}")
    
    print(f"\n👷 Optimal Workers: {rm.get_optimal_workers()}")
