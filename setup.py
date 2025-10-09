from setuptools import setup, find_packages

setup(
    name="explainbench",
    version="0.1.0",
    description="ExplainBench: Benchmarking XAI Methods on Time-Series",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy","pandas","PyYAML","torch>=2.0.0","scikit-learn","matplotlib","tqdm","captum"
    ],
)
