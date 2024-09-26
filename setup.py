from setuptools import setup, find_packages

setup(
    name="pygame_puzzle_project",
    version="1.0.0",
    description="Sliding puzzle game using Pygame and MediaPipe for hand detection",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pygame",
        "opencv-python",
        "mediapipe"
    ],
)
