#!/usr/bin/env python3
"""
Wandb monitoring and observability for GRPO training
Comprehensive logging, metrics tracking, and visualization
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class WandbMonitor:
    """Comprehensive Wandb monitoring for GRPO training"""
    
    def __init__(self, project_name: str = "grpo-unsloth-training", config: Dict = None):
        self.project_name = project_name
        self.config = config or {}
        self.run = None
        self.step = 0
        self.epoch = 0
        
        # Initialize wandb
        self._init_wandb()
        
        # System monitoring
        self.gpu_utilization = []
        self.memory_usage = []
        self.training_losses = []
        self.rewards = []
        self.advantages = []
        
    def _init_wandb(self):
        """Initialize Wandb run"""
        try:
            self.run = wandb.init(
                project=self.project_name,
                config=self.config,
                name=f"grpo-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags=["grpo", "unsloth", "20b", "nvidia-4080-super"],
                notes="GRPO training with Unsloth on NVIDIA 4080 Super 16GB VRAM"
            )
            print(f"✅ Wandb initialized: {self.run.url}")
        except Exception as e:
            print(f"❌ Failed to initialize Wandb: {e}")
            self.run = None
    
    def log_system_info(self):
        """Log system information"""
        if not self.run:
            return
            
        try:
            # GPU information
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version()
                }
                wandb.log({"system/gpu": gpu_info})
            
            # CPU and memory info
            cpu_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total": psutil.virtual_memory().total / 1024**3,
                "memory_available": psutil.virtual_memory().available / 1024**3
            }
            wandb.log({"system/cpu": cpu_info})
            
            # Python environment
            import sys
            env_info = {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
            wandb.log({"system/environment": env_info})
            
        except Exception as e:
            print(f"Warning: Failed to log system info: {e}")
    
    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics"""
        if not self.run:
            return
            
        step = step or self.step
        
        # Add step and epoch info
        log_data = {
            "step": step,
            "epoch": self.epoch,
            **metrics
        }
        
        # Log to wandb
        wandb.log(log_data)
        
        # Store for analysis
        if "loss" in metrics:
            self.training_losses.append(metrics["loss"])
        if "rewards_mean" in metrics:
            self.rewards.append(metrics["rewards_mean"])
        if "advantages_mean" in metrics:
            self.advantages.append(metrics["advantages_mean"])
    
    def log_memory_usage(self, step: Optional[int] = None):
        """Log GPU and system memory usage"""
        if not self.run:
            return
            
        step = step or self.step
        
        try:
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = {
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_memory_free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
                }
                wandb.log({f"memory/gpu": gpu_memory}, step=step)
                
                # Store for plotting
                self.memory_usage.append({
                    "step": step,
                    "allocated": gpu_memory["gpu_memory_allocated"],
                    "reserved": gpu_memory["gpu_memory_reserved"],
                    "free": gpu_memory["gpu_memory_free"]
                })
            
            # System memory
            system_memory = {
                "system_memory_used": psutil.virtual_memory().used / 1024**3,
                "system_memory_available": psutil.virtual_memory().available / 1024**3,
                "system_memory_percent": psutil.virtual_memory().percent
            }
            wandb.log({f"memory/system": system_memory}, step=step)
            
        except Exception as e:
            print(f"Warning: Failed to log memory usage: {e}")
    
    def log_gpu_utilization(self, step: Optional[int] = None):
        """Log GPU utilization"""
        if not self.run:
            return
            
        step = step or self.step
        
        try:
            # Get GPU utilization using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                
                gpu_metrics = {
                    "gpu_utilization": gpu_util.gpu,
                    "gpu_memory_utilization": gpu_util.memory,
                    "gpu_temperature": gpu_temp,
                    "gpu_power_usage": gpu_power
                }
                
                wandb.log({f"gpu/utilization": gpu_metrics}, step=step)
                
                # Store for plotting
                self.gpu_utilization.append({
                    "step": step,
                    "gpu_util": gpu_util.gpu,
                    "memory_util": gpu_util.memory,
                    "temperature": gpu_temp,
                    "power": gpu_power
                })
                
            except (ImportError, Exception) as e:
                # Fallback: estimate utilization from memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    gpu_metrics = {
                        "gpu_memory_utilization": memory_used * 100,
                        "gpu_utilization": min(memory_used * 100, 100)  # Rough estimate
                    }
                    wandb.log({f"gpu/utilization": gpu_metrics}, step=step)
                    
        except Exception as e:
            print(f"Warning: Failed to log GPU utilization: {e}")
    
    def log_generation_samples(self, prompts: List[str], responses: List[str], 
                             rewards: List[float], step: Optional[int] = None):
        """Log sample generations for analysis"""
        if not self.run:
            return
            
        step = step or self.step
        
        try:
            # Create a table of samples
            sample_data = []
            for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards)):
                sample_data.append([
                    f"Sample {i+1}",
                    prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    response[:200] + "..." if len(response) > 200 else response,
                    f"{reward:.4f}"
                ])
            
            # Log as wandb table
            table = wandb.Table(
                columns=["Sample", "Prompt", "Response", "Reward"],
                data=sample_data
            )
            wandb.log({f"generations/samples_step_{step}": table}, step=step)
            
            # Log individual samples as text
            for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards)):
                wandb.log({
                    f"generations/sample_{i+1}_reward": reward,
                    f"generations/sample_{i+1}_prompt": prompt,
                    f"generations/sample_{i+1}_response": response
                }, step=step)
                
        except Exception as e:
            print(f"Warning: Failed to log generation samples: {e}")
    
    def create_plots(self):
        """Create and log plots for analysis"""
        if not self.run:
            return
            
        try:
            # Training loss plot
            if self.training_losses:
                plt.figure(figsize=(10, 6))
                plt.plot(self.training_losses)
                plt.title("Training Loss Over Time")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.grid(True)
                wandb.log({"plots/training_loss": wandb.Image(plt)})
                plt.close()
            
            # Rewards plot
            if self.rewards:
                plt.figure(figsize=(10, 6))
                plt.plot(self.rewards)
                plt.title("Rewards Over Time")
                plt.xlabel("Step")
                plt.ylabel("Mean Reward")
                plt.grid(True)
                wandb.log({"plots/rewards": wandb.Image(plt)})
                plt.close()
            
            # Memory usage plot
            if self.memory_usage:
                plt.figure(figsize=(12, 8))
                
                steps = [m["step"] for m in self.memory_usage]
                allocated = [m["allocated"] for m in self.memory_usage]
                reserved = [m["reserved"] for m in self.memory_usage]
                free = [m["free"] for m in self.memory_usage]
                
                plt.subplot(2, 1, 1)
                plt.plot(steps, allocated, label="Allocated", color="red")
                plt.plot(steps, reserved, label="Reserved", color="orange")
                plt.plot(steps, free, label="Free", color="green")
                plt.title("GPU Memory Usage")
                plt.xlabel("Step")
                plt.ylabel("Memory (GB)")
                plt.legend()
                plt.grid(True)
                
                # GPU utilization plot
                if self.gpu_utilization:
                    plt.subplot(2, 1, 2)
                    gpu_steps = [g["step"] for g in self.gpu_utilization]
                    gpu_util = [g["gpu_util"] for g in self.gpu_utilization]
                    memory_util = [g["memory_util"] for g in self.gpu_utilization]
                    
                    plt.plot(gpu_steps, gpu_util, label="GPU Utilization", color="blue")
                    plt.plot(gpu_steps, memory_util, label="Memory Utilization", color="purple")
                    plt.title("GPU Utilization")
                    plt.xlabel("Step")
                    plt.ylabel("Utilization (%)")
                    plt.legend()
                    plt.grid(True)
                
                plt.tight_layout()
                wandb.log({"plots/memory_and_utilization": wandb.Image(plt)})
                plt.close()
            
            # Rewards distribution
            if self.rewards:
                plt.figure(figsize=(10, 6))
                plt.hist(self.rewards, bins=20, alpha=0.7, edgecolor='black')
                plt.title("Rewards Distribution")
                plt.xlabel("Reward Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                wandb.log({"plots/rewards_distribution": wandb.Image(plt)})
                plt.close()
                
        except Exception as e:
            print(f"Warning: Failed to create plots: {e}")
    
    def log_model_artifacts(self, model_path: str, step: Optional[int] = None):
        """Log model artifacts"""
        if not self.run:
            return
            
        step = step or self.step
        
        try:
            # Log model as artifact
            artifact = wandb.Artifact(
                name=f"grpo-model-step-{step}",
                type="model",
                description=f"GRPO model checkpoint at step {step}"
            )
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
            
        except Exception as e:
            print(f"Warning: Failed to log model artifacts: {e}")
    
    def log_configuration_summary(self, config: Dict):
        """Log configuration summary"""
        if not self.run:
            return
            
        try:
            # Create a summary table
            config_data = []
            for key, value in config.items():
                config_data.append([key, str(value)])
            
            table = wandb.Table(
                columns=["Parameter", "Value"],
                data=config_data
            )
            wandb.log({"configuration/summary": table})
            
        except Exception as e:
            print(f"Warning: Failed to log configuration summary: {e}")
    
    def update_step(self, step: int):
        """Update current step"""
        self.step = step
    
    def update_epoch(self, epoch: int):
        """Update current epoch"""
        self.epoch = epoch
    
    def finish(self):
        """Finish wandb run"""
        if self.run:
            # Create final plots
            self.create_plots()
            
            # Log final summary
            if self.training_losses:
                final_metrics = {
                    "final/loss": self.training_losses[-1],
                    "final/avg_loss": np.mean(self.training_losses),
                    "final/min_loss": np.min(self.training_losses)
                }
                if self.rewards:
                    final_metrics.update({
                        "final/avg_reward": np.mean(self.rewards),
                        "final/max_reward": np.max(self.rewards),
                        "final/min_reward": np.min(self.rewards)
                    })
                
                wandb.log(final_metrics)
            
            wandb.finish()
            print("✅ Wandb run finished")


def test_wandb_monitoring():
    """Test wandb monitoring functionality"""
    print("Testing Wandb monitoring...")
    
    # Test configuration
    config = {
        "model_name": "unsloth/gpt-2",
        "max_seq_length": 1024,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "num_generations": 4
    }
    
    # Initialize monitor
    monitor = WandbMonitor("grpo-test", config)
    
    if not monitor.run:
        print("❌ Wandb not available. Please check your setup.")
        return False
    
    try:
        # Test system info logging
        print("Testing system info logging...")
        monitor.log_system_info()
        
        # Test memory logging
        print("Testing memory logging...")
        monitor.log_memory_usage()
        
        # Test GPU utilization
        print("Testing GPU utilization...")
        monitor.log_gpu_utilization()
        
        # Test training metrics
        print("Testing training metrics...")
        for step in range(5):
            metrics = {
                "loss": 1.0 - step * 0.1,
                "rewards_mean": 0.5 + step * 0.1,
                "advantages_mean": 0.2 + step * 0.05
            }
            monitor.log_training_metrics(metrics, step)
            monitor.update_step(step)
        
        # Test generation samples
        print("Testing generation samples...")
        prompts = ["What is 2+2?", "What is 3+3?"]
        responses = ["The answer is 4", "The answer is 6"]
        rewards = [0.8, 0.9]
        monitor.log_generation_samples(prompts, responses, rewards)
        
        # Test configuration summary
        print("Testing configuration summary...")
        monitor.log_configuration_summary(config)
        
        # Create plots
        print("Creating plots...")
        monitor.create_plots()
        
        print("✅ Wandb monitoring test completed successfully!")
        print(f"View your run at: {monitor.run.url}")
        
        # Finish the test run
        monitor.finish()
        
        return True
        
    except Exception as e:
        print(f"❌ Wandb monitoring test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_wandb_monitoring()
    sys.exit(0 if success else 1)
