from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="codejournal",
    version="0.1.1",
    author="Your Name",
    author_email="your_email@example.com",
    description="A Python package code snip package for faster pytorch-AI development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/codejournal",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,  # Use the dynamically loaded requirements
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)