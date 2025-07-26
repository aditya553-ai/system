from setuptools import setup, find_packages

setup(
    name="medical_nlp_pipeline",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Medical NLP pipeline for entity recognition and relation extraction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "spacy>=3.0.0",
        "transformers>=4.0.0",
        "torch>=1.7.0",
        "fuzzywuzzy>=0.18.0",
        "nltk>=3.5",
        "requests>=2.25.1",
        "networkx>=2.5",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0"
    ],
)