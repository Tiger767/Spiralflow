import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chat-spiral",
    version="0.0.1",
    author="Travis Hammond",
    description="A framework for creating chat spirals for Large Language Models finetuned for conversations. Currently work-in-progress.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tiger767/spiral",
    packages=setuptools.find_packages(),
    install_requires=[
        "regex",
        "openai",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    license="MIT License",
)