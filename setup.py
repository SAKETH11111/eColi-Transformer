from setuptools import setup, find_packages

setup(
    name="ecoli_transformer",
    version="0.1.0",
    description="E. coli Transformer for codon analysis",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "transformers",
        "scikit-learn",
    ],
)
