"""Build and query a FAISS vector index over clinical note text.

This script converts note text into sentence embeddings, stores the vectors
and note metadata on disk, and supports semantic search queries over the
embedded note corpus.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

import extract_clinical_entities


# -----------------
# Configuration
# -----------------
DATA_DIR = "datasets"
NOTES_FILE = "clinical_notes.csv"
METADATA_FILE = "note_vector_metadata.csv"
EMBEDDINGS_FILE = "clinical_note_embeddings.npy"
INDEX_FILE = "clinical_note_index.faiss"
CONFIG_FILE = "vector_index_config.json"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for index building and querying."""
    parser = argparse.ArgumentParser(
        description="Build a vector index over clinical notes and run semantic search."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Semantic search query. If omitted, the script only builds the index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of semantic search matches to return.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model to use for embeddings.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    """Ensure vector search dependencies are installed before continuing."""
    missing_packages = []

    if sentence_transformers is None:
        missing_packages.append("sentence-transformers")
    if faiss is None:
        missing_packages.append("faiss-cpu")

    if missing_packages:
        print(f"Error: missing required package(s): {', '.join(missing_packages)}")
        print("Install them with: pip install sentence-transformers faiss-cpu")
        sys.exit(1)


def get_notes_path() -> str:
    """Return the absolute path to the note dataset."""
    return os.path.join(DATA_DIR, NOTES_FILE)


def load_or_collect_notes() -> pd.DataFrame:
    """Load note data from CSV, or collect it from FHIR bundles if needed."""
    notes_path = get_notes_path()

    if os.path.exists(notes_path):
        notes_df = pd.read_csv(notes_path)
    else:
        if not os.path.isdir(extract_clinical_entities.DATA_DIR):
            print(
                f"Error: note dataset '{notes_path}' not found and "
                f"FHIR directory '{extract_clinical_entities.DATA_DIR}' is missing."
            )
            sys.exit(1)

        print("Clinical notes CSV not found. Collecting notes from FHIR bundles...")
        note_records = extract_clinical_entities.collect_all_notes(extract_clinical_entities.DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
        notes_df = pd.DataFrame(note_records, columns=extract_clinical_entities.NOTE_COLUMNS)
        notes_df.to_csv(notes_path, index=False)
        print(f"Saved note dataset to '{notes_path}'")

    if notes_df.empty:
        return pd.DataFrame(columns=extract_clinical_entities.NOTE_COLUMNS)

    notes_df = notes_df.fillna("")
    notes_df["note_text"] = notes_df["note_text"].astype(str)
    notes_df = notes_df[notes_df["note_text"].str.strip() != ""].copy()
    notes_df = notes_df.drop_duplicates(subset=["note_id"], keep="first")
    notes_df = notes_df.reset_index(drop=True)

    return notes_df


def load_embedding_model(model_name: str):
    """Load the requested sentence-transformer model."""
    return sentence_transformers.SentenceTransformer(model_name)


def encode_texts(model, texts: list[str]) -> np.ndarray:
    """Encode note text into float32 embedding vectors."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 10,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray):
    """Create a flat L2 FAISS index from embedding vectors."""
    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatL2(dimension)
    if len(embeddings) > 0:
        index.add(embeddings)
    return index


def save_vector_artifacts(
    notes_df: pd.DataFrame,
    embeddings: np.ndarray,
    index,
    model_name: str,
) -> None:
    """Persist vector search artifacts to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)

    metadata_path = os.path.join(DATA_DIR, METADATA_FILE)
    embeddings_path = os.path.join(DATA_DIR, EMBEDDINGS_FILE)
    index_path = os.path.join(DATA_DIR, INDEX_FILE)
    config_path = os.path.join(DATA_DIR, CONFIG_FILE)

    notes_df.to_csv(metadata_path, index=False)
    np.save(embeddings_path, embeddings)
    faiss.write_index(index, index_path)

    config = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "note_count": int(len(notes_df)),
        "metadata_file": METADATA_FILE,
        "embeddings_file": EMBEDDINGS_FILE,
        "index_file": INDEX_FILE,
    }
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2)


def build_vector_store(model_name: str) -> None:
    """Build note embeddings and save the FAISS vector store."""
    ensure_dependencies()
    notes_df = load_or_collect_notes()

    if notes_df.empty:
        print("No note text available to embed. Vector index was not created.")
        return

    print(f"Building vector store from {len(notes_df)} clinical note(s)...")
    model = load_embedding_model(model_name)
    embeddings = encode_texts(model, notes_df["note_text"].tolist())
    index = build_faiss_index(embeddings)
    save_vector_artifacts(notes_df, embeddings, index, model_name)

    print(f"Stored {len(notes_df)} embedding vector(s) in '{os.path.join(DATA_DIR, INDEX_FILE)}'")
    print(f"Saved vector metadata to '{os.path.join(DATA_DIR, METADATA_FILE)}'")


def vector_artifacts_exist() -> bool:
    """Return True when the required vector store artifacts are present."""
    required_files = [METADATA_FILE, EMBEDDINGS_FILE, INDEX_FILE, CONFIG_FILE]
    return all(os.path.exists(os.path.join(DATA_DIR, filename)) for filename in required_files)


def load_vector_artifacts() -> tuple[pd.DataFrame, object, dict]:
    """Load note metadata, the FAISS index, and config from disk."""
    metadata_path = os.path.join(DATA_DIR, METADATA_FILE)
    index_path = os.path.join(DATA_DIR, INDEX_FILE)
    config_path = os.path.join(DATA_DIR, CONFIG_FILE)

    metadata_df = pd.read_csv(metadata_path)
    index = faiss.read_index(index_path)

    with open(config_path, encoding="utf-8") as file_obj:
        config = json.load(file_obj)

    return metadata_df.fillna(""), index, config


def semantic_search(query: str, top_k: int) -> pd.DataFrame:
    """Run a semantic search query against the note vector store."""
    ensure_dependencies()

    if not vector_artifacts_exist():
        print("Vector index not found. Building it now before searching...")
        build_vector_store(DEFAULT_MODEL_NAME)

    if not vector_artifacts_exist():
        return pd.DataFrame()

    metadata_df, index, config = load_vector_artifacts()
    if metadata_df.empty or index.ntotal == 0:
        return pd.DataFrame()

    model = load_embedding_model(config["model_name"])
    query_embedding = encode_texts(model, [query])
    distances, indices = index.search(query_embedding, min(top_k, index.ntotal))

    results = []
    for rank, (distance, row_index) in enumerate(zip(distances[0], indices[0]), start=1):
        if row_index < 0:
            continue

        note_row = metadata_df.iloc[int(row_index)].to_dict()
        note_row["rank"] = rank
        note_row["distance"] = float(distance)
        results.append(note_row)

    return pd.DataFrame(results)


def print_search_results(results_df: pd.DataFrame, query: str) -> None:
    """Print semantic search results in a readable terminal format."""
    if results_df.empty:
        print(f"No semantic matches found for query: '{query}'")
        return

    print(f"\nSemantic matches for: '{query}'\n")
    for _, row in results_df.iterrows():
        note_preview = str(row["note_text"]).replace("\n", " ").strip()
        if len(note_preview) > 180:
            note_preview = f"{note_preview[:177]}..."

        print(f"[{int(row['rank'])}] patient_id={row['patient_id']} distance={row['distance']:.4f}")
        print(
            f"    resource={row['resource_type']}:{row['resource_id']} "
            f"source={row['note_source']} note_id={row['note_id']}"
        )
        print(f"    note={note_preview}")


def main() -> None:
    """Main entry point: build the vector store and optionally search it."""
    args = parse_args()
    build_vector_store(args.model_name)

    if args.query:
        results_df = semantic_search(args.query, args.top_k)
        print_search_results(results_df, args.query)


if __name__ == "__main__":
    main()
