from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

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
    pass


class PythonREPLTool(BaseTool):
    pass
