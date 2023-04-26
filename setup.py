import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spiralflow",
    version="0.0.3",
    author="Travis Hammond",
    description="A framework for creating chat spirals for Large Language Models finetuned for conversations. Currently work-in-progress.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tiger767/Spiralflow",
    packages=setuptools.find_packages(),
    install_requires=[
        "regex",
        "openai",
        "pandas",
        "tiktoken",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    license="MIT License",
)
