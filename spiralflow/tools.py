from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np

import os

from googleapiclient.discovery import build


class BaseTool(ABC):
    @abstractmethod
    def use(self, inputs: Dict[str, str]) -> Union[str, list[str]]:
        pass


class GoogleSearchTool(BaseTool):
    def __init__(
        self,
        api_key: str,
        cse_id: str,
        num_results: int = 10,
        failed_search_result: str = "No google search results found.",
        join_snippets: Optional[str] = "\n",
    ) -> None:
        """
        Initialize the GoogleSearchTool.

        :param api_key: The Google API key.
        :param cse_id: The Google Custom Search Engine ID.
        :param num_results: The max number of results to return.
        :param failed_search_result: The result to return if the search fails.
        :param join_snippets: The string to join the snippets with. If None, the snippets will be returned as a list.
        """
        self.cse_id = cse_id
        self.num_results = num_results
        self.failed_search_result = failed_search_result
        self.join_snippets = join_snippets
        self.engine = build("customsearch", "v1", developerKey=api_key)

    def search(self, query: str) -> Optional[list]:
        """
        :param query: The query to search for.
        :return: The search results.
        """
        results = (
            self.engine.cse()
            .list(q=query, cx=self.cse_id, num=self.num_results)
            .execute()
        )
        return results.get("items")

    def use(self, inputs: Dict[str, str]) -> Union[str, list[str]]:
        """
        :param inputs: The inputs to the tool. Must contain a 'query' key.
        :return: The output of the tool: Google search snippets.
        """
        query = inputs["query"]
        results = self.search(query)
        if results is None:
            return self.failed_search_result

        output = [result["snippet"] for result in results if "snippet" in result]
        if self.join_snippets is not None:
            output = self.join_snippets.join(output)
        return output


class FileTool(BaseTool):
    def __init__(self, path: str) -> None:
        """
        Initialize the FileTool.

        :param path: The path to the file.
        """
        self.path = os.path.abspath(path)
        self.commands = [
            "read",
            "r",
            "write",
            "w",
            "append",
            "a",
            "cd",
            "chdir",
            "change_directory",
            "ls",
            "list",
            "mkdir",
            "make_directory",
        ]

    def check_required_inputs(
        self, required_inputs: list[str], inputs: list[str]
    ) -> str:
        """
        Checks if the required inputs are in the inputs list. If not, raises a ValueError.

        :param required_inputs: The required inputs.
        :param inputs: The inputs to check.
        """
        missing_inputs = []
        for input in required_inputs:
            if input not in inputs:
                missing_inputs.append(input)
        if len(missing_inputs) > 0:
            err = "Missing input(s): {"
            for i in range(len(missing_inputs)):
                missed_input = missing_inputs[i]
                err += missed_input
                if i != len(missing_inputs) - 1:
                    err += ", "
                else:
                    err += "}"
            raise ValueError(err)

    def get_closest_command(self, incorrect_command: str):
        """
        Returns the closest command to an incorrectly inputted command

        :param incorrect_command: The command that was inputted incorrectly.
        :return: The closest possible command.
        """
        distances = []
        for command in self.commands:
            matrix = np.zeros((len(incorrect_command) + 1, len(command) + 1))
            for i in range(len(incorrect_command) + 1):
                matrix[i][0] = i
            for i in range(len(command) + 1):
                matrix[0][i] = i
            for i in range(1, len(incorrect_command) + 1):
                for j in range(1, len(command) + 1):
                    matrix[i][j] = min(
                        matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1]
                    )
                    if incorrect_command[i - 1] != command[j - 1]:
                        matrix[i][j] += 1
            distances.append(matrix[len(incorrect_command)][len(command)])
        return self.commands[distances.index(min(distances))]

    def use(self, inputs: Dict[str, str]) -> Union[str, list[str]]:
        """
        :param inputs: The inputs to the tool. Must contain a 'command' key.
                        Depending on the 'command' value, other keys will be required as follows:
                        [read, r]: file
                        [write, w]: file, data
                        [append, a]: file, data
                        [cd, chdir, change directory]: path
                        [ls, list]: path
                        [mkdir, make directory]: path
        :return: The output of the tool: The contents of the file.
        """
        command = inputs["command"]
        if command in ["read", "r"]:
            self.check_required_inputs(required_inputs=["file"], inputs=inputs.keys())

            file_path = os.path.join(self.path, inputs["file"])
            with open(file_path, "r") as f:
                return f.read()
        elif command in ["write", "w"]:
            self.check_required_inputs(
                required_inputs=["file", "data"], inputs=inputs.keys()
            )

            file_path = os.path.join(self.path, inputs["file"])
            with open(file_path, "w") as f:
                f.write(inputs["data"])
                return "WROTE: " + inputs["data"] + "\nTO: " + file_path
        elif command in ["append", "a"]:
            self.check_required_inputs(
                required_inputs=["file", "data"], inputs=inputs.keys()
            )

            file_path = os.path.join(self.path, inputs["file"])
            with open(file_path, "a") as f:
                f.write(inputs["data"])
                return "APPENDED: " + inputs["data"] + "\nTO: " + file_path
        elif command in ["cd", "chdir", "change_directory"]:
            self.check_required_inputs(required_inputs=["path"], inputs=inputs.keys())

            self.path = os.path.abspath(inputs["path"])
            return "CHANGED DIRECTORY TO: " + self.path
        elif command in ["ls", "list"]:
            return os.listdir(self.path)
        elif command in ["mkdir", "make_directory"]:
            self.check_required_inputs(
                required_inputs=["directory"], inputs=inputs.keys()
            )

            directory_path = os.path.join(self.path, inputs["directory"])
            if os.path.isdir(directory_path):
                return "DIRECTORY ALREADY EXISTS: " + directory_path
            os.mkdir(directory_path)
            return "CREATED DIRECTORY: " + directory_path
        else:
            raise ValueError(
                f"Invalid command: {command} \n\tDid you mean: {self.get_closest_command(command)}?"
            )
