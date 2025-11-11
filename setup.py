
from setuptools import setup, find_packages

setup(
    name="tts_project",
    version="0.1.0",
    author="Nick Ward",
    description="A TTS project with PyTorch models and utilities",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "scipy",
    ],
)
