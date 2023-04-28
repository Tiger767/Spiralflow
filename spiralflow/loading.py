import os
import regex as re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import fitz
from bs4 import BeautifulSoup


class Loader(ABC):
    def __init__(self, chunker=None):
        """
        :param chunker: The chunker to split text into chunks (default: None).
        """
        self.chunker = chunker

    @abstractmethod
    def load(filepath: str) -> List[Dict[str, Any]]:
        pass

    def handle_chunking(
        self, loaded_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        :param loaded_items: List of dictionaries containing the path to the file and its content with other possible metadata.
        """
        if self.chunker is None:
            return loaded_items

        chunked_items = []
        for item in loaded_items:
            contents = self.chunker.chunk(item["content"])
            del item["content"]
            chunked_items.extend([{"content": content, **item} for content in contents])
        return chunked_items


class TextLoader(Loader):
    def __init__(self, encoding: str = "utf-8", chunker=None):
        """
        :param encoding: The encoding of the text file (default: 'utf-8').
        :param chunker: The chunker to split text into chunks (default: None).
        """
        super().__init__(chunker)
        self.encoding = encoding

    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """
        :param filepath: Path to the text file to be loaded.
        :return: Dictionary containing the path to the file and its content.
        """
        with open(filepath, "r", encoding=self.encoding) as file:
            content = file.read()

        return self.handle_chunking([{"content": content, "path": filepath}])


class PDFLoader(TextLoader):
    """
    PDFLoader that uses fitz (PyMuPDF) to load PDF files.
    """

    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """
        :param filepath: Path to the PDF file to be loaded.
        :return: List of dictionaries containing the path to the file, its content, and the page number.
        """
        doc = fitz.open(filepath)
        pages = []

        for page in doc:
            pages.append(
                {
                    "content": page.get_text().encode(self.encoding),
                    "path": filepath,
                    "page_number": page.number,
                }
            )

        return self.handle_chunking(pages)


class HTMLLoader(Loader):
    """
    HTMLLoader that uses beautiful soup to load HTML files.
    """

    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """
        :param filepath: Path to the HTML file to be loaded.
        :return: Dictionary containing the path to the file, its content, and the soup title.
        """
        with open(filepath, "r", encoding=self.encoding) as file:
            content = file.read()

        soup = BeautifulSoup(content, "html.parser")

        return self.handle_chunking(
            [
                {
                    "content": soup.get_text(),
                    "path": filepath,
                    "title": soup.title.string if soup.title else None,
                }
            ]
        )


class DirectoryLoader(Loader):
    def __init__(
        self,
        path: str,
        loader: Loader,
        should_recurse: bool = True,
        filter_regex: Optional[str] = None,
    ):
        """
        :param path: Path to the directory to load files from.
        :param loader: Class that actually loads the specific file.
        :param should_recurse: Whether to recursively load files from subdirectories (default: True).
        :param filter_regex: Regular expression to filter files by (default: None).
        """
        self.path = path
        self.should_recurse = should_recurse
        self.filter_regex = filter_regex
        self.loader = loader

    def load(self, current_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        :param current_path: Current path while recursively loading directories (default: None).
        :return: List of dictionaries containing the path to the file, its content, and possibly other metadata.
        """
        if current_path is None:
            current_path = self.path

        files = []
        for entry in os.scandir(current_path):
            if entry.is_dir() and self.should_recurse:
                files.extend(self.load(entry.path))
            elif entry.is_file() and (
                self.filter_regex is None or re.match(self.filter_regex, entry.name)
            ):
                files.extend(self.loader.load(entry.path))

        return files


class DirectoryMultiLoader(Loader):
    def __init__(
        self,
        path: str,
        loader: Dict[str, Loader],
        should_recurse: bool = True,
    ):
        """
        :param path: Path to the directory to load files from.
        :param loader: Dictionary of loader instances, with keys being filter regexes.
        :param should_recurse: Whether to recursively load files from subdirectories (default: True).
        """
        self.path = path
        self.should_recurse = should_recurse
        self.loader = loader

    def load(self, current_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        :param current_path: Current path while recursively loading directories (default: None).
        """
        if current_path is None:
            current_path = self.path

        files = []
        for entry in os.scandir(current_path):
            if entry.is_dir() and self.should_recurse:
                files += self.load(entry.path)
            elif entry.is_file():
                for filter_regex, loader in self.loader.items():
                    if re.match(filter_regex, entry.name):
                        files += loader.load(entry.path)
                        break

        return files
