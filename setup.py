from setuptools import setup, find_packages

def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name="causal-component-analysis",
    author="Armin Kekić",
    author_email="armin.kekic@mailbox.org",
    packages=find_packages(),
    description="Causal Component Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9.0",
    install_requires=parse_requirements("./requirements.txt"),
    license="MIT",
)
