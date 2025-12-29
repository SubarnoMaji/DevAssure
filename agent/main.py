import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.retriever_tool import vector_store_search

load_dotenv()


def get_llm(model: str = "gpt-5-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Initialize OpenAI LLM."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key
    )


def create_rag_agent(llm=None, tools=None):
    """
    Create a RAG agent that can use the retriever tool when needed.

    Args:
        llm: Optional ChatOpenAI instance
        tools: Optional list of tools (default: uses vector_store_search)

    Returns:
        AgentExecutor instance
    """
    if llm is None:
        llm = get_llm()

    if tools is None:
        tools = [vector_store_search]

    # Create a prompt that instructs the agent when to use the tool
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to a document search tool.

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
        When you don't use the tool, answer directly and helpfully."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


def process_query(query: str, use_retrieval: bool = None) -> str:
    """
    Process a query using the RAG flow.
    The agent will automatically decide if retrieval is needed.

    Args:
        query: The user's question/query
        use_retrieval: If True, adds instruction to force retrieval. If None, agent decides.

    Returns:
        Processed answer from the agent
    """
    # Modify query if retrieval should be forced
    if use_retrieval:
        query = f"[IMPORTANT: You must use the vector_store_search tool for this query] {query}"

    # Create agent and execute
    agent_executor = create_rag_agent()

    # Invoke the agent
    result = agent_executor.invoke({"input": query})
    return result.get("output", "No response generated")


def simple_openai_call(prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
    """Simple OpenAI call without tools (for comparison)."""
    llm = get_llm(model, temperature)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
