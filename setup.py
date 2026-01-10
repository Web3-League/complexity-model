"""
Complexity - Modern Transformer Architecture with Token-Routed MLP

Innovations:
- Token-Routed MLP: Routes tokens to specialized experts based on token ID
- CGGR Triton kernels: 5-6x speedup for Token-Routed MLP
- Flash Attention via SDPA (PyTorch 2.0+)
- QK Normalization for training stability
- Sliding Window Attention (optional)

v0.3.0: Added CGGR (Coalesced Grouped Gemm with Ragged) Triton optimization
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="complexity-model",
    version="0.5.0",
    description="Complexity transformer with CGGR Triton Token-Routed MLP, Flash Attention, QK Norm",
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
        "cuda": ["triton>=2.0.0"],  # CGGR Triton kernels (5-6x speedup)
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
    keywords="llm transformer token-routed-mlp flash-attention qk-norm complexity cggr triton",
)
