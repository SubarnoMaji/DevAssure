import os
import logging
from typing import TypedDict, Annotated
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools.retriever_tool import vector_store_search

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def get_llm(model: str = "gpt-5-mini", temperature: float = 0.7) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key
    )


tools = [vector_store_search]

SYSTEM_PROMPT = """You are a helpful assistant with access to a document search tool.

You have access to the following tool:
- vector_store_search: Search for relevant documents in the vector store collection.

Use the vector_store_search tool when:
- The user asks about information that might be in the document collection
- The user asks about specific documents, files, or content
- You need to find information from indexed documents

Do NOT use the tool when:
- The question is about general knowledge that doesn't require document search
- The question is a simple factual question you can answer directly
- The question is about your capabilities or how to use you

When you use the tool, analyze the retrieved documents and provide a comprehensive answer based on the retrieved information.
When you don't use the tool, answer directly and helpfully."""


def create_agent_node(state: AgentState) -> dict:
    logger.info("=" * 50)
    logger.info("AGENT NODE: Processing")

    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    messages = state["messages"]
    logger.info(f"Current message count: {len(messages)}")

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content_preview = str(msg.content)[:100] if msg.content else "[no content]"
        logger.info(f"  Message {i+1} [{msg_type}]: {content_preview}...")

    if not any(isinstance(m, BaseMessage) and m.type == "system" for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    logger.info("Calling LLM...")
    response = llm_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"LLM Response: Requesting tool calls")
        for tc in response.tool_calls:
            logger.info(f"  Tool: {tc['name']}, Args: {tc['args']}")
    else:
        content_preview = str(response.content)[:200] if response.content else "[no content]"
        logger.info(f"LLM Response: {content_preview}...")

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("ROUTER: Directing to TOOLS node")
        return "tools"
    logger.info("ROUTER: Directing to END")
    return END


def create_tool_node(state: AgentState) -> dict:
    logger.info("=" * 50)
    logger.info("TOOL NODE: Executing tools")

    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)

    if "messages" in result:
        for msg in result["messages"]:
            content_preview = str(msg.content)[:300] if msg.content else "[no content]"
            logger.info(f"Tool Result: {content_preview}...")

    return result


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", create_agent_node)
    workflow.add_node("tools", create_tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


graph = create_graph()


def process_query(query: str, use_retrieval: bool = None) -> str:
    logger.info("=" * 60)
    logger.info("NEW QUERY RECEIVED")
    logger.info(f"Query: {query[:100]}...")
    logger.info(f"Use retrieval: {use_retrieval}")
    logger.info("=" * 60)

    if use_retrieval:
        query = f"[IMPORTANT: You must use the vector_store_search tool for this query] {query}"

    initial_state = {
        "messages": [HumanMessage(content=query)]
    }

    logger.info("Starting graph execution...")
    result = graph.invoke(initial_state)

    messages = result.get("messages", [])
    logger.info(f"Graph completed. Total messages: {len(messages)}")

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            logger.info("=" * 60)
            logger.info("FINAL RESPONSE GENERATED")
            logger.info(f"Response: {msg.content[:200]}...")
            logger.info("=" * 60)
            return msg.content

    logger.warning("No response generated from graph")
    return "No response generated"

if __name__ == "__main__":
    response = process_query("What documents do you have?", use_retrieval=True)
    print(response)
