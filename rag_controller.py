"""RAG Controller — orchestrates hybrid retrieval + LLM generation.

Ties together the hybrid retriever (FAISS + BM25), dual prompt templates
(clinician / patient), and the Ollama LLM client into a single query
interface for the clinical decision-support system.
"""

import argparse
import os
import sys
import time

import pandas as pd

import hybrid_retriever
import llm_client
import prompt_templates
import verifier


# -----------------
# Configuration
# -----------------
DATA_DIR = "datasets"
DEFAULT_TOP_K = 8
DEFAULT_MODE = "both"  # "clinician", "patient", or "both"


class RAGController:
    """End-to-end RAG controller for clinical decision support.

    Connects:
    1. HybridRetriever — retrieves relevant clinical note chunks.
    2. PromptTemplates — formats dual-mode prompts.
    3. OllamaClient — generates LLM responses.

    Attributes:
        retriever: HybridRetriever instance.
        llm: OllamaClient instance.
    """

    def __init__(
        self,
        model: str = llm_client.DEFAULT_MODEL,
        base_url: str = llm_client.DEFAULT_BASE_URL,
    ) -> None:
        print("Initializing RAG Controller...")

        # Initialize hybrid retriever
        print("  Loading hybrid retriever (FAISS + BM25)...")
        self.retriever = hybrid_retriever.HybridRetriever()
        print(f"  Loaded {len(self.retriever.metadata_df)} note(s)")

        # Initialize LLM client
        self.llm = llm_client.OllamaClient(model=model, base_url=base_url)
        print(f"  LLM model: {self.llm.model}")
        print(f"  Ollama server: {self.llm.base_url}")

        if self.llm.is_available():
            print("  Ollama status: connected")
        else:
            print("  Ollama status: NOT CONNECTED (generation will fail)")
            print("  Start with: ollama serve")

        print("  Initializing safety verifiers...")
        self.numeric_verifier = verifier.NumericVerifier()
        self.semantic_verifier = verifier.SemanticVerifier()

        print("RAG Controller ready.\n")

    def _load_patient_context(self, patient_id: str) -> dict:
        """Load patient demographics, conditions, and medications from CSVs.

        Args:
            patient_id: The patient ID to look up.

        Returns:
            Dict with patient info, conditions list, and medications list.
        """
        context = {"patient_id": patient_id}

        # Load patient demographics
        patients_path = os.path.join(DATA_DIR, "patients.csv")
        if os.path.exists(patients_path):
            patients_df = pd.read_csv(patients_path)
            match = patients_df[patients_df["patient_id"] == patient_id]
            if not match.empty:
                row = match.iloc[0].to_dict()
                context.update({
                    "first_name": row.get("first_name", ""),
                    "last_name": row.get("last_name", ""),
                    "age": row.get("age", ""),
                    "gender": row.get("gender", ""),
                })

        # Load active conditions
        conditions_path = os.path.join(DATA_DIR, "conditions.csv")
        if os.path.exists(conditions_path):
            cond_df = pd.read_csv(conditions_path)
            if "clinical_status" in cond_df.columns:
                cond_df = cond_df[cond_df["clinical_status"].fillna("").str.lower() == "active"]
            patient_conds = cond_df[cond_df["patient_id"] == patient_id]
            context["conditions"] = patient_conds["condition"].tolist()

        # Load current medications
        meds_path = os.path.join(DATA_DIR, "medications.csv")
        if os.path.exists(meds_path):
            meds_df = pd.read_csv(meds_path)
            if "status" in meds_df.columns:
                meds_df = meds_df[meds_df["status"].fillna("").str.lower() == "active"]
            patient_meds = meds_df[meds_df["patient_id"] == patient_id]
            context["medications"] = patient_meds["medication"].tolist()

        return context

    def query(
        self,
        query_text: str,
        mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict:
        """Run a complete RAG query: retrieve → prompt → generate.

        Args:
            query_text: Natural-language clinical query.
            mode: Output mode — "clinician", "patient", or "both".
            top_k: Number of chunks to retrieve (target 5-8).
            temperature: LLM sampling temperature.
            max_tokens: Maximum tokens for LLM generation.

        Returns:
            Dict with keys: query, mode, chunks, clinician_response,
            patient_response, total_duration_ms, retrieval_duration_ms.
        """
        overall_start = time.perf_counter()

        # 1. Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        chunks = self.retriever.retrieve(query=query_text, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        print(f"Retrieved {len(chunks)} chunk(s) in {retrieval_ms:.0f}ms")

        # 2. Determine patient context from top-ranked chunk
        patient_context = None
        if chunks:
            top_patient_id = chunks[0].get("patient_id", "")
            if top_patient_id:
                patient_context = self._load_patient_context(top_patient_id)

        # 3. Generate responses based on mode
        result = {
            "query": query_text,
            "mode": mode,
            "chunks_retrieved": len(chunks),
            "chunks": chunks,
            "clinician_response": None,
            "patient_response": None,
            "retrieval_duration_ms": round(retrieval_ms),
            "verification_flags": {},
        }

        if mode in ("clinician", "both"):
            print("Generating clinician response...")
            clinician_prompt = prompt_templates.build_clinician_prompt(
                query=query_text,
                retrieved_chunks=chunks,
                patient_context=patient_context,
            )
            clinician_result = self.llm.generate(
                prompt=clinician_prompt,
                system_prompt=prompt_templates.CLINICIAN_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result["clinician_response"] = clinician_result
            if clinician_result.get("error"):
                print(f"  Clinician generation error: {clinician_result['error']}")
            else:
                print(f"  Clinician response: {clinician_result['duration_ms']}ms")

        if mode in ("patient", "both"):
            print("Generating patient response...")
            patient_prompt = prompt_templates.build_patient_prompt(
                query=query_text,
                retrieved_chunks=chunks,
                patient_context=patient_context,
            )
            patient_result = self.llm.generate(
                prompt=patient_prompt,
                system_prompt=prompt_templates.PATIENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result["patient_response"] = patient_result
            if patient_result.get("error"):
                print(f"  Patient generation error: {patient_result['error']}")
            else:
                print(f"  Patient response: {patient_result['duration_ms']}ms")

        # 4. Verify outputs and Log Provenance
        total_ms = (time.perf_counter() - overall_start) * 1000
        result["total_duration_ms"] = round(total_ms)

        print("Running verification checks...")
        for res_mode, res_key in [("clinician", "clinician_response"), ("patient", "patient_response")]:
            res_obj = result.get(res_key)
            if res_obj and not res_obj.get("error"):
                gen_text = res_obj.get("response", "")
                
                # Check numeric and semantic consistency
                numeric_findings = self.numeric_verifier.verify_detailed(
                    generated_text=gen_text,
                    retrieved_chunks=chunks,
                    patient_context=patient_context,
                )
                unsup_nums = [finding["normalized_value"] for finding in numeric_findings]
                semantic_findings = self.semantic_verifier.verify_detailed(
                    generated_text=gen_text,
                    retrieved_chunks=chunks,
                    patient_context=patient_context,
                    numeric_findings=numeric_findings,
                )
                unsup_claims = [finding["claim_text"] for finding in semantic_findings]
                
                result["verification_flags"][res_key] = {
                    "unsupported_numbers": unsup_nums,
                    "numeric_findings": numeric_findings,
                    "semantic_findings": semantic_findings,
                    "unsupported_claims": unsup_claims,
                }
                
                # Log to audit trail
                verifier.log_provenance(
                    query=query_text,
                    mode=res_mode,
                    chunks=chunks,
                    generated_text=gen_text,
                    unsupported_numbers=unsup_nums,
                    unsupported_claims=unsup_claims,
                    retrieval_ms=round(retrieval_ms),
                    generation_ms=res_obj.get("duration_ms", 0),
                    numeric_findings=numeric_findings,
                    semantic_findings=semantic_findings,
                )

        print(f"\nTotal pipeline time: {total_ms:.0f}ms")

        return result


def _print_numeric_warnings(flags: dict) -> None:
    """Print structured numeric verifier warnings when present."""
    numeric_findings = flags.get("numeric_findings", [])
    if numeric_findings:
        print("\n[WARNING] Numeric Verification Findings:")
        for finding in numeric_findings:
            candidate = finding.get("best_candidate_source") or "no close source match"
            reason = finding.get("reason", "unsupported")
            print(
                f"  - claim={finding.get('claim_text', '')} | "
                f"closest={candidate} | reason={reason}"
            )
        return

    if flags.get("unsupported_numbers"):
        print(f"\n[WARNING] Unsupported Numbers: {flags['unsupported_numbers']}")


def _print_semantic_warnings(flags: dict) -> None:
    """Print structured semantic verifier warnings when present."""
    semantic_findings = flags.get("semantic_findings", [])
    if semantic_findings:
        print("\n[WARNING] Semantic Verification Findings:")
        for finding in semantic_findings:
            candidate = finding.get("best_candidate_source") or "no close source match"
            reason = finding.get("reason", "unsupported")
            print(
                f"  - claim={finding.get('claim_text', '')} | "
                f"closest={candidate} | reason={reason}"
            )
        return

    if flags.get("unsupported_claims"):
        print("\n[WARNING] Unsupported Claims (Hallucinations?):")
        for claim in flags["unsupported_claims"]:
            print(f"  - {claim}")


def print_rag_result(result: dict) -> None:
    """Print a formatted RAG result to the terminal."""
    print("\n" + "=" * 70)
    print(f"QUERY: {result['query']}")
    print(f"MODE: {result['mode']}")
    print(f"CHUNKS RETRIEVED: {result['chunks_retrieved']}")
    print(f"RETRIEVAL TIME: {result['retrieval_duration_ms']}ms")
    print(f"TOTAL TIME: {result['total_duration_ms']}ms")
    print("=" * 70)

    # Show retrieved chunks summary
    if result["chunks"]:
        print("\n--- Retrieved Chunks ---")
        for chunk in result["chunks"]:
            preview = chunk["note_text"][:120].replace("\n", " ")
            print(
                f"  [{chunk['rank']}] patient={chunk['patient_id']} "
                f"fused={chunk['fused_score']:.4f} | {preview}..."
            )

    # Show clinician response
    if result.get("clinician_response"):
        cr = result["clinician_response"]
        print("\n--- Clinician Summary ---")
        if cr.get("error"):
            print(f"ERROR: {cr['error']}")
        else:
            print(f"(generated in {cr['duration_ms']}ms)\n")
            print(cr["response"])
            
            # Show flags
            flags = result.get("verification_flags", {}).get("clinician_response", {})
            _print_numeric_warnings(flags)
            _print_semantic_warnings(flags)

    # Show patient response
    if result.get("patient_response"):
        pr = result["patient_response"]
        print("\n--- Patient Summary ---")
        if pr.get("error"):
            print(f"ERROR: {pr['error']}")
        else:
            print(f"(generated in {pr['duration_ms']}ms)\n")
            print(pr["response"])
            
            # Show flags
            flags = result.get("verification_flags", {}).get("patient_response", {})
            _print_numeric_warnings(flags)
            _print_semantic_warnings(flags)

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RAG controller."""
    parser = argparse.ArgumentParser(
        description="RAG Controller — Clinical Decision Support with Hybrid Retrieval + LLM"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="patient with chest pain and diabetes",
        help="Clinical query to process.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["clinician", "patient", "both"],
        default=DEFAULT_MODE,
        help="Output mode: clinician, patient, or both.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=llm_client.DEFAULT_MODEL,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: initialize RAG controller and process a query."""
    args = parse_args()

    controller = RAGController(model=args.model)

    result = controller.query(
        query_text=args.query,
        mode=args.mode,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    print_rag_result(result)


if __name__ == "__main__":
    main()
