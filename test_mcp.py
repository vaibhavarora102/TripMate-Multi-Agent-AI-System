import asyncio
from mcp_client import get_mcp_tools


# print(asyncio.run(get_mcp_tools("tavily")))


async def main():
    tavily_tools = await get_mcp_tools("tavily")
    for tool in tavily_tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())