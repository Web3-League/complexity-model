"""
Complexity - Modern Transformer Architecture with Token-Routed MLP

Innovations:
- Token-Routed MLP: Routes tokens to specialized experts based on token ID
- CGGR Triton kernels: 5-6x speedup for Token-Routed MLP
- Flash Attention via SDPA (PyTorch 2.0+)
- QK Normalization for training stability
- Sliding Window Attention (optional)

v0.6.0: Added comprehensive CUDA optimizations + Robotics module
  - Fused QK Norm + Flash Attention (~15-20% faster)
  - Fused RMSNorm + MLP (~20-30% faster)
  - Persistent CGGR kernels (~10-15% faster)
  - Fused Residual + RMSNorm (~5-10% faster)
  - INT8 quantization for inference (~40-50% faster)
  - Robotics module for real-time robot control
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="complexity-model",
    version="0.6.3",
    description="Complexity transformer with CUDA-optimized kernels, Token-Routed MLP, and Robotics support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pacific-Prime",
    author_email="",
    url="https://github.com/Web3-League/complexity-model",
    project_urls={
        "GitHub": "https://github.com/Web3-League/complexity-model",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
    ],
    extras_require={
        "training": ["datasets>=2.0.0", "tensorboard", "tqdm"],
        "cuda": ["triton>=2.0.0"],  # Fused CUDA kernels (40-60% speedup)
        "robotics": ["numpy", "mujoco"],  # Robotics module
        "all": ["triton>=2.0.0", "datasets>=2.0.0", "tensorboard", "tqdm", "numpy", "mujoco"],
        "dev": ["pytest", "black", "isort"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm transformer token-routed-mlp flash-attention qk-norm complexity cggr triton robotics cuda",
)
