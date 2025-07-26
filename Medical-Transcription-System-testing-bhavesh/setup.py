from setuptools import setup, find_packages

setup(
    name="medical-transcription-pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A pipeline for medical transcription, including ASR, NER, normalization, and note generation.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "whisper",
        "scipy",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "fuzzywuzzy",
        "nltk",
        "flask",  # if you plan to create a web interface
        "requests",  # for API calls if needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)