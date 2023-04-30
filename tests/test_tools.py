import os
import tempfile
import shutil
from typing import Dict, Union
import pytest

from spiralflow.tools import GoogleSearchTool, FileTool


class TestGoogleSearchTool:
    @pytest.fixture(scope="class")
    def google_search_tool(self) -> GoogleSearchTool:
        api_key = "your_api_key"
        cse_id = "your_cse_id"
        return GoogleSearchTool(api_key=api_key, cse_id=cse_id)

    def test_search(self, google_search_tool: GoogleSearchTool) -> None:
        query = "Python"
        results = google_search_tool.search(query)
        assert results is not None
        assert len(results) > 0

    def test_use(self, google_search_tool: GoogleSearchTool) -> None:
        inputs: Dict[str, str] = {"query": "Python"}
        output = google_search_tool.use(inputs)
        assert output is not None
        assert len(output) > 0


class TestFileTool:
    @pytest.fixture(scope="class")
    def file_tool(self) -> FileTool:
        temp_dir = tempfile.mkdtemp()
        yield FileTool(path=temp_dir)
        shutil.rmtree(temp_dir)

    def test_read(self, file_tool: FileTool) -> None:
        file_name = "test_read.txt"
        file_path = os.path.join(file_tool.path, file_name)
        with open(file_path, "w") as f:
            f.write("Hello, world!")

        inputs: Dict[str, str] = {"command": "read", "file": file_name}
        output = file_tool.use(inputs)
        assert output == "Hello, world!"

    def test_write(self, file_tool: FileTool) -> None:
        file_name = "test_write.txt"
        inputs: Dict[str, str] = {
            "command": "write",
            "file": file_name,
            "data": "Hello, world!",
        }
        file_tool.use(inputs)

        file_path = os.path.join(file_tool.path, file_name)
        with open(file_path, "r") as f:
            content = f.read()
        assert content == "Hello, world!"

    def test_append(self, file_tool: FileTool) -> None:
        file_name = "test_append.txt"
        file_path = os.path.join(file_tool.path, file_name)
        with open(file_path, "w") as f:
            f.write("Hello, ")

        inputs: Dict[str, str] = {
            "command": "append",
            "file": file_name,
            "data": "world!",
        }
        file_tool.use(inputs)

        with open(file_path, "r") as f:
            content = f.read()
        assert content == "Hello, world!"

    def test_change_directory(self, file_tool: FileTool) -> None:
        new_dir = "test_dir"
        new_path = os.path.join(file_tool.path, new_dir)
        os.mkdir(new_path)

        inputs: Dict[str, str] = {"command": "cd", "path": new_dir}
        file_tool.use(inputs)

        assert file_tool.path == new_path

    def test_list(self, file_tool: FileTool) -> None:
        file_name = "test_list.txt"
        file_path = os.path.join(file_tool.path, file_name)
        with open(file_path, "w") as f:
            f.write("Hello, world!")

        inputs: Dict[str, str] = {"command": "list"}
        output = file_tool.use(inputs)

        assert file_name in output

    def test_make_directory(self, file_tool: FileTool) -> None:
        new_dir = "test_make_dir"
        inputs: Dict[str, str] = {"command": "mkdir", "directory": new_dir}
        file_tool.use(inputs)

        new_path = os.path.join(file_tool.path, new_dir)
        assert os.path.isdir(new_path)

    def test_invalid_command(self, file_tool: FileTool) -> None:
        inputs: Dict[str, str] = {"command": "invalid"}
        with pytest.raises(ValueError) as excinfo:
            file_tool.use(inputs)
        assert "Invalid command" in str(excinfo.value)
