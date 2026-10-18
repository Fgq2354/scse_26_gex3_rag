"""A LangChain tool for retrieval from the course policy index."""
from langchain_core.tools import tool
from index_store import checked_store
from settings import BASE_URL, cool_down


def make_search_tool(base_url=BASE_URL):
    store = checked_store(base_url)

    @tool
    def search_policies(question: str) -> list[dict]:
        """Find two relevant passages in the supplied university IT policy file."""
        if not question.strip():
            raise ValueError("Provide a non-empty question")
        cool_down()
        matches = store.similarity_search(question, k=2)
        if not matches:
            raise RuntimeError("Retrieval returned no policy passages")
        return [{"policy_id": doc.metadata["policy_id"], "source": doc.metadata["source"],
                 "text": doc.page_content} for doc in matches]

    return search_policies
