"""Index the single supplied policy document."""
import argparse
from index_store import build_index
from settings import BASE_URL, INDEX


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    count = build_index(args.base_url)
    print(f"Index ready with {count} policy sections")
