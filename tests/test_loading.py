import pytest
import os
import shutil
from unittest.mock import MagicMock
from spiralflow.loading import (
    Loader,
    TextLoader,
    PDFLoader,
    HTMLLoader,
    DirectoryLoader,
    DirectoryMultiLoader,
)


# Helper function to create temporary directories and files
def setup_temp_files():
    os.makedirs("temp_dir")
    with open("temp_dir/sample.txt", "w") as f:
        f.write("Sample text content.")

    # Create other sample files (PDF, HTML) in temp_dir


# Helper function to delete temporary directories and files
def teardown_temp_files():
    shutil.rmtree("temp_dir")


# TestLoader
class TestLoader:
    def test_handle_chunking(self):
        # Test with no chunker
        loader = Loader()
        loaded_items = [{"content": "Sample content", "path": "sample.txt"}]
        result = loader.handle_chunking(loaded_items)
        assert result == loaded_items

        # Test with a custom chunker
        chunker = MagicMock()
        chunker.chunk.return_value = ["Chunk 1", "Chunk 2"]
        loader = Loader(chunker=chunker)
        result = loader.handle_chunking(loaded_items)
        expected_result = [
            {"content": "Chunk 1", "path": "sample.txt"},
            {"content": "Chunk 2", "path": "sample.txt"},
        ]
        assert result == expected_result


# TestTextLoader
class TestTextLoader:
    def test_init(self):
        # Test with default parameters
        text_loader = TextLoader()
        assert text_loader.encoding == "utf-8"

        # Test with custom parameters
        text_loader = TextLoader(encoding="iso-8859-1")
        assert text_loader.encoding == "iso-8859-1"

    def test_load(self):
        setup_temp_files()
        text_loader = TextLoader()
        result = text_loader.load("temp_dir/sample.txt")
        teardown_temp_files()

        assert result == [
            {"content": "Sample text content.", "path": "temp_dir/sample.txt"}
        ]


# TestPDFLoader
class TestPDFLoader:
    def test_load(self):
        setup_temp_files()  # Make sure you have a sample PDF file in the temp_dir
        pdf_loader = PDFLoader()
        result = pdf_loader.load("temp_dir/sample.pdf")
        teardown_temp_files()

        # Customize the expected result based on the sample PDF file
        expected_result = [
            {
                "content": "Sample PDF content.",
                "path": "temp_dir/sample.pdf",
                "page_number": 0,
            }
        ]
        assert result == expected_result


# TestHTMLLoader
class TestHTMLLoader:
    def test_load(self):
        setup_temp_files()  # Make sure you have a sample HTML file in the temp_dir
        html_loader = HTMLLoader()
        result = html_loader.load("temp_dir/sample.html")
        teardown_temp_files()

        # Customize the expected result based on the sample HTML file
        expected_result = [
            {
                "content": "Sample HTML content.",
                "path": "temp_dir/sample.html",
                "title": "Sample HTML Title",
            }
        ]
        assert result == expected_result


# TestDirectoryLoader
class TestDirectoryLoader:
    def test_init(self):
        # Test with default parameters
        dir_loader = DirectoryLoader("temp_dir", TextLoader())
        assert dir_loader.should_recurse is True
        assert dir_loader.filter_regex is None

        # Test with custom parameters
        dir_loader = DirectoryLoader("temp_dir", TextLoader(), False, r"sample\.txt")
        assert dir_loader.should_recurse is False
        assert dir_loader.filter_regex == r"sample\.txt"

    def test_load(self):
        setup_temp_files()  # Make sure you have a sample text file in the temp_dir
        dir_loader = DirectoryLoader("temp_dir", TextLoader())
        result = dir_loader.load()
        teardown_temp_files()

        expected_result = [
            {"content": "Sample text content.", "path": "temp_dir/sample.txt"}
        ]
        assert result == expected_result

        # Test with recursion disabled
        setup_temp_files()  # Create sample text file in temp_dir and a subdirectory with another text file
        dir_loader = DirectoryLoader("temp_dir", TextLoader(), should_recurse=False)
        result = dir_loader.load()
        teardown_temp_files()

        # Customize the expected result based on the files in temp_dir
        expected_result = [
            {"content": "Sample text content.", "path": "temp_dir/sample.txt"}
        ]
        assert result == expected_result


# TestDirectoryMultiLoader
class TestDirectoryMultiLoader:
    def test_init(self):
        # Test with default parameters
        dir_multi_loader = DirectoryMultiLoader("temp_dir", {"txt": TextLoader()})
        assert dir_multi_loader.should_recurse is True

        # Test with custom parameters
        dir_multi_loader = DirectoryMultiLoader(
            "temp_dir", {"txt": TextLoader()}, False
        )
        assert dir_multi_loader.should_recurse is False

    def test_load(self):
        setup_temp_files()  # Make sure you have sample files for each loader in the temp_dir
        dir_multi_loader = DirectoryMultiLoader(
            "temp_dir", {"txt": TextLoader(), "pdf": PDFLoader(), "html": HTMLLoader()}
        )
        result = dir_multi_loader.load()
        teardown_temp_files()

        # Customize the expected result based on the files in temp_dir
        expected_result = [
            {"content": "Sample text content.", "path": "temp_dir/sample.txt"},
            {
                "content": "Sample PDF content.",
                "path": "temp_dir/sample.pdf",
                "page_number": 0,
            },
            {
                "content": "Sample HTML content.",
                "path": "temp_dir/sample.html",
                "title": "Sample HTML Title",
            },
        ]
        assert result == expected_result

        # Test with recursion disabled
        setup_temp_files()  # Create sample files in temp_dir and a subdirectory with another set of files
        dir_multi_loader = DirectoryMultiLoader(
            "temp_dir",
            {"txt": TextLoader(), "pdf": PDFLoader(), "html": HTMLLoader()},
            should_recurse=False,
        )
        result = dir_multi_loader.load()
        teardown_temp_files()

        # Customize the expected result based on the files in temp_dir
        expected_result = [
            {"content": "Sample text content.", "path": "temp_dir/sample.txt"},
            {
                "content": "Sample PDF content.",
                "path": "temp_dir/sample.pdf",
                "page_number": 0,
            },
            {
                "content": "Sample HTML content.",
                "path": "temp_dir/sample.html",
                "title": "Sample HTML Title",
            },
        ]
        assert result == expected_result
