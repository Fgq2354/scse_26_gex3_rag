"""Build and verify a persistent index from the unchanged course loader."""
import hashlib
import json

from policy_loader import POLICY_FILE, load_policy_documents
from settings import BASE_URL, EMBEDDING, INDEX, cool_down, server_json


def source_digest():
    return hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()


def embedding_identity(base_url):
    entries = server_json(base_url + "/api/tags")["models"]
    expected = EMBEDDING if ":" in EMBEDDING else EMBEDDING + ":latest"
    for entry in entries:
        if entry["name"] == expected:
            return entry["digest"]
    raise RuntimeError("Install nomic-embed-text before building the index")


def open_store(base_url=BASE_URL, index=INDEX):
    from chromadb.config import Settings
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    embedding = OllamaEmbeddings(
        model=EMBEDDING, base_url=base_url, num_gpu=99, num_thread=2,
        keep_alive=120, client_kwargs={"timeout": 120, "trust_env": False})
    return Chroma(collection_name="course_it_policy",
                  embedding_function=embedding, persist_directory=str(index),
                  client_settings=Settings(anonymized_telemetry=False))


def inspect_contents(store, documents, complete=True):
    actual = store.get(include=["documents", "metadatas"])
    expected = {doc.metadata["policy_id"]: doc for doc in documents}
    found = set(actual["ids"])
    if found - expected.keys():
        raise ValueError("Index contains documents outside the single course source")
    for identifier, text, metadata in zip(actual["ids"], actual["documents"], actual["metadatas"]):
        doc = expected[identifier]
        if text != doc.page_content or metadata != doc.metadata:
            raise ValueError("Stored text or metadata differs from the supplied policy")
    if complete and found != expected.keys():
        raise ValueError("Index is incomplete; run rag.py before asking questions")
    return found


def build_index(base_url=BASE_URL, index=INDEX):
    documents = load_policy_documents()
    index.mkdir(parents=True, exist_ok=True)
    manifest_path = index / "source.json"
    identity = {"source_sha256": source_digest(), "embedding_model": EMBEDDING,
                "embedding_digest": embedding_identity(base_url), "count": len(documents)}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(k) != value for k, value in identity.items()):
            raise ValueError("Source or embedding changed; use a fresh index directory")
    else:
        if (index / "chroma.sqlite3").exists():
            raise ValueError("Existing index has no source manifest; use a fresh directory")
        manifest_path.write_text(json.dumps({**identity, "complete": False}), encoding="utf-8")
    store = open_store(base_url, index)
    present = inspect_contents(store, documents, complete=False)
    missing = [doc for doc in documents if doc.metadata["policy_id"] not in present]
    for offset in range(0, len(missing), 16):
        cool_down()
        batch = missing[offset:offset + 16]
        store.add_documents(batch, ids=[doc.metadata["policy_id"] for doc in batch])
        print(f"Indexed {len(present) + min(offset + 16, len(missing))}/{len(documents)}", flush=True)
    inspect_contents(store, documents)
    manifest_path.write_text(json.dumps({**identity, "complete": True}, indent=2), encoding="utf-8")
    return len(documents)


def checked_store(base_url=BASE_URL, index=INDEX):
    manifest = json.loads((index / "source.json").read_text(encoding="utf-8"))
    documents = load_policy_documents()
    if (not manifest["complete"] or manifest["source_sha256"] != source_digest()
            or manifest["count"] != len(documents)
            or manifest["embedding_model"] != EMBEDDING
            or manifest["embedding_digest"] != embedding_identity(base_url)):
        raise ValueError("Index manifest does not match the source or embedding model")
    store = open_store(base_url, index)
    inspect_contents(store, documents)
    return store
