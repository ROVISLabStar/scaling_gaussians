from setuptools import setup, find_packages

setup(
    name="gs_vs",                  # Replace with your actual package name
    version="0.1.0",
    packages=find_packages(),           # Automatically finds packages in folders with __init__.py
    install_requires=[],                # Add dependencies here, or use requirements.txt
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    url="https://github.com/yourusername/my_project",  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
