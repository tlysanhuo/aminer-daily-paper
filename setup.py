from setuptools import find_packages, setup


setup(
    name="aminer-rec",
    version="0.1.0",
    description="AMiner-powered personalized paper recommendation pipeline.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="tly",
    author_email="liyan.tie@aminer.cn",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["aminer_rec*", "scripts*"]),
    install_requires=[
        "openai>=1.30.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "aminer-rec=aminer_rec.cli:main",
        ],
    },
)
