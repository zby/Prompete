from typing import Protocol, List, Union, Callable
from llm_easy_tools import LLMFunction

class ToolManager(Protocol):
    def get_tools(self) -> List[Union[Callable, LLMFunction]]:
        """Return list of available tools as functions"""
        ...

class ToolList:
    def __init__(self, tools: List[Union[Callable, LLMFunction]]):
        self._tools = tools

    def get_tools(self) -> List[Union[Callable, LLMFunction]]:
        return self._tools

if __name__ == '__main__':
    def search(argument: str) -> str:
        """Search for text in documents"""
        return f"Searching for '{argument}'"

    def calculate(argument: str) -> float:
        """Evaluate mathematical expression"""
        return f"Evaluating '{argument}'"

    tool_manager = ToolList([
        search,
        calculate
    ])

    for tool in tool_manager.get_tools():
        argument = 'test'
        print("Result of tool execution: ", tool(argument))
