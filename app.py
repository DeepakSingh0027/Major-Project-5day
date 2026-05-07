"""Streamlit frontend for the clinical RAG knowledge base.

Provides patient profile selection, clinician-facing RAG summaries, and
patient-facing explanations over the local clinical knowledge base.
"""

import os
import sys
from datetime import datetime
from importlib import metadata

try:
    import streamlit as st
except ModuleNotFoundError:
    print("Error: streamlit is not installed. Install it with: pip install streamlit")
    sys.exit(1)

try:
    import pandas as pd
except ModuleNotFoundError:
    st.error("pandas is not installed. Install it with: pip install pandas")
    st.stop()

import llm_client


# -----------------
# Configuration
# -----------------
DATA_DIR = "datasets"
PATIENTS_FILE = "patients.csv"
PATIENTS_PATH = os.path.join(DATA_DIR, PATIENTS_FILE)
WORKFLOW_IMAGE_PATH = "workflow.jpeg"
OLLAMA_MODEL = llm_client.DEFAULT_MODEL
OLLAMA_BASE_URL = llm_client.DEFAULT_BASE_URL
VECTOR_ARTIFACTS = [
    "clinical_note_embeddings.npy",
    "clinical_note_index.faiss",
    "note_vector_metadata.csv",
    "vector_index_config.json",
]

SESSION_DEFAULTS = {
    "active_patient_id": None,
    "clinician_summary": None,
    "patient_explanation": None,
    "patient_selector": None,
    "batch_id": "",          # set once per session for audit log grouping
    "active_model": "",     # Task 11: active Ollama model (defaults to OLLAMA_MODEL)
    "tuned_top_k": None,    # Task 11: chunk reduction override (None = use UI default)
}

PATIENT_PROFILE_COLUMNS = [
    "patient_id",
    "first_name",
    "last_name",
    "age",
    "gender",
    "race",
    "ethnicity",
    "marital_status",
    "city",
    "state",
    "country",
]

OPTIONAL_PROFILE_COLUMNS = [
    "first_name",
    "last_name",
    "age",
    "gender",
    "race",
    "ethnicity",
    "marital_status",
    "city",
    "state",
    "country",
]


# -----------------
# Startup Checks
# -----------------
def validate_environment() -> None:
    """Validate lightweight UI dependencies without loading RAG components."""
    dependency_names = ["streamlit", "pandas"]
    missing = []

    for dependency in dependency_names:
        try:
            metadata.version(dependency)
        except metadata.PackageNotFoundError:
            missing.append(dependency)

    if missing:
        st.error(f"Missing required package(s): {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}", language="bash")
        st.stop()


def initialize_session_state() -> None:
    """Create stable state keys used by current and future dashboard tasks."""
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # batch_id is set once per browser session and never overwritten on rerun
    if not st.session_state["batch_id"]:
        st.session_state["batch_id"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # active_model is set once per session; never reset on rerun
    if not st.session_state["active_model"]:
        st.session_state["active_model"] = OLLAMA_MODEL


# -----------------
# Cached Resources
# -----------------
@st.cache_data(show_spinner=False)
def load_patients(path: str) -> pd.DataFrame:
    """Load only patient profile columns needed for selection and demographics."""
    return pd.read_csv(path, usecols=lambda column: column in PATIENT_PROFILE_COLUMNS)


@st.cache_resource(show_spinner=False)
def load_workflow_image(path: str) -> bytes:
    """Load the workflow diagram once to avoid repeated image reads."""
    with open(path, "rb") as image_file:
        return image_file.read()


@st.cache_resource(show_spinner=False)
def load_rag_controller(model: str = OLLAMA_MODEL):
    """Load the RAG controller, keyed by model name for runtime switching."""
    import rag_controller
    return rag_controller.RAGController(model=model, base_url=OLLAMA_BASE_URL)


@st.cache_data(show_spinner=False, ttl=5)
def check_ollama_readiness(model_name: str, base_url: str) -> dict:
    """Check Ollama readiness without loading the full RAG stack."""
    return llm_client.get_ollama_readiness(model_name=model_name, base_url=base_url)


# -----------------
# Data Helpers
# -----------------
def patients_file_exists() -> bool:
    """Return True when the patient profile CSV is available."""
    return os.path.exists(PATIENTS_PATH)


def vector_artifacts_exist() -> bool:
    """Return True when FAISS vector artifacts are available."""
    return all(os.path.exists(os.path.join(DATA_DIR, filename)) for filename in VECTOR_ARTIFACTS)


def normalize_patients_df(patients_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected optional columns exist and patient IDs are displayable."""
    if "patient_id" not in patients_df.columns:
        st.error("datasets/patients.csv is missing the required patient_id column.")
        st.stop()

    normalized_df = patients_df.copy()
    normalized_df["patient_id"] = normalized_df["patient_id"].astype(str)

    for column in OPTIONAL_PROFILE_COLUMNS:
        if column not in normalized_df.columns:
            normalized_df[column] = ""

    return normalized_df


def clean_value(value: object, fallback: str = "Not available") -> str:
    """Return a readable value for optional CSV fields."""
    if pd.isna(value):
        return fallback

    value_text = str(value).strip()
    if not value_text or value_text.lower() == "nan":
        return fallback

    return value_text


def build_patient_label(row: pd.Series) -> str:
    """Build a human-friendly selector label with a patient ID fallback."""
    first_name = clean_value(row.get("first_name", ""), fallback="")
    last_name = clean_value(row.get("last_name", ""), fallback="")
    patient_id = clean_value(row.get("patient_id", ""), fallback="unknown")
    full_name = " ".join(part for part in [first_name, last_name] if part)

    if full_name:
        return f"{full_name} ({patient_id})"
    return patient_id


def get_active_patient(patients_df: pd.DataFrame) -> pd.Series:
    """Return the active patient row, defaulting to the first row."""
    patient_ids = patients_df["patient_id"].tolist()
    active_patient_id = st.session_state.get("active_patient_id")

    if active_patient_id not in patient_ids:
        active_patient_id = patient_ids[0]
        st.session_state["active_patient_id"] = active_patient_id
        st.session_state["patient_selector"] = active_patient_id

    return patients_df[patients_df["patient_id"] == active_patient_id].iloc[0]


def handle_patient_change() -> None:
    """Update active patient state when the sidebar selector changes."""
    selected_patient_id = st.session_state.get("patient_selector")
    previous_patient_id = st.session_state.get("active_patient_id")

    if selected_patient_id != previous_patient_id:
        st.session_state["active_patient_id"] = selected_patient_id
        st.session_state["clinician_summary"] = None
        st.session_state["patient_explanation"] = None


def generation_state_key(mode: str, patient_id: str, suffix: str) -> str:
    """Build a mode- and patient-specific session key for generated output."""
    safe_patient_id = str(patient_id).replace(" ", "_")
    return f"{mode}_{suffix}_{safe_patient_id}"


# -----------------
# Rendering
# -----------------
def render_missing_data_message() -> None:
    """Render a clear recovery path when patient data is unavailable."""
    st.error(f"Missing required dataset: {PATIENTS_PATH}")
    st.write("Create patient data before launching the dashboard.")
    st.code("python extract_data.py", language="bash")
    st.write("For standalone testing without Synthea data, use:")
    st.code("python generate_mock_data.py", language="bash")


def render_sidebar(patients_df: pd.DataFrame) -> None:
    """Render patient selection and optional workflow context."""
    labels_by_id = {
        row["patient_id"]: build_patient_label(row)
        for _, row in patients_df.iterrows()
    }
    patient_ids = list(labels_by_id.keys())

    if st.session_state.get("active_patient_id") not in patient_ids:
        st.session_state["active_patient_id"] = patient_ids[0]
    if st.session_state.get("patient_selector") not in patient_ids:
        st.session_state["patient_selector"] = st.session_state["active_patient_id"]

    st.sidebar.header("Patient Profile")
    st.sidebar.selectbox(
        "Select patient",
        options=patient_ids,
        key="patient_selector",
        format_func=lambda patient_id: labels_by_id.get(patient_id, patient_id),
        on_change=handle_patient_change,
    )

    st.sidebar.caption(f"{len(patient_ids)} patient profiles loaded")

    if os.path.exists(WORKFLOW_IMAGE_PATH):
        with st.sidebar.expander("Project workflow"):
            st.image(load_workflow_image(WORKFLOW_IMAGE_PATH), use_container_width=True)


def render_patient_metrics(patient: pd.Series) -> None:
    """Render core demographics as dashboard metrics."""
    city = clean_value(patient.get("city", ""), fallback="")
    state = clean_value(patient.get("state", ""), fallback="")
    country = clean_value(patient.get("country", ""), fallback="")
    location_parts = [part for part in [city, state, country] if part]
    location = ", ".join(location_parts) if location_parts else "Not available"

    metric_columns = st.columns(3)
    metric_columns[0].metric("Age", clean_value(patient.get("age", "")))
    metric_columns[1].metric("Gender", clean_value(patient.get("gender", "")))
    metric_columns[2].metric("Location", location)


def render_patient_profile(patient: pd.Series) -> None:
    """Render selected patient demographics."""
    first_name = clean_value(patient.get("first_name", ""), fallback="")
    last_name = clean_value(patient.get("last_name", ""), fallback="")
    full_name = " ".join(part for part in [first_name, last_name] if part)
    patient_title = full_name or clean_value(patient.get("patient_id", ""))

    st.subheader(patient_title)
    st.caption(f"Patient ID: {clean_value(patient.get('patient_id', ''))}")
    render_patient_metrics(patient)

    detail_columns = st.columns(3)
    detail_columns[0].write(f"Race: {clean_value(patient.get('race', ''))}")
    detail_columns[1].write(f"Ethnicity: {clean_value(patient.get('ethnicity', ''))}")
    detail_columns[2].write(f"Marital status: {clean_value(patient.get('marital_status', ''))}")


def build_default_clinician_query(patient: pd.Series) -> str:
    """Build a useful default query for the active patient."""
    patient_id = clean_value(patient.get("patient_id", ""))
    first_name = clean_value(patient.get("first_name", ""), fallback="")
    last_name = clean_value(patient.get("last_name", ""), fallback="")
    full_name = " ".join(part for part in [first_name, last_name] if part)

    if full_name:
        return f"Generate a clinician summary for {full_name}, patient {patient_id}."
    return f"Generate a clinician summary for patient {patient_id}."


def build_default_patient_query(patient: pd.Series) -> str:
    """Build a useful default patient-facing query for the active patient."""
    patient_id = clean_value(patient.get("patient_id", ""))
    first_name = clean_value(patient.get("first_name", ""), fallback="")
    last_name = clean_value(patient.get("last_name", ""), fallback="")
    full_name = " ".join(part for part in [first_name, last_name] if part)

    if full_name:
        return f"Explain {full_name}'s results and care plan in simple terms."
    return f"Explain the results and care plan for patient {patient_id} in simple terms."


def render_setup_commands() -> None:
    """Show setup commands needed before running the RAG dashboard."""
    st.code("python generate_mock_data.py", language="bash")
    st.write("For real Synthea data, run:")
    st.code("python extract_clinical_entities.py\npython vector_search.py", language="bash")


def render_ollama_setup_commands() -> None:
    """Show setup commands for the local Ollama dependency."""
    st.write("Start Ollama and ensure the configured model is available:")
    st.code(f"ollama serve\nollama pull {OLLAMA_MODEL}", language="bash")
    st.write("For a longer-running local session, start Ollama in the background:")
    st.code("nohup ollama serve > ollama.log 2>&1 &", language="bash")


def run_generation_query(
    patient_id: str,
    mode: str,
    query: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
    label: str,
) -> dict | None:
    """Run a RAG query for a single output mode with Streamlit status updates."""
    if not vector_artifacts_exist():
        st.error("Vector artifacts are missing. Build the note index before generation.")
        render_setup_commands()
        return None

    active_model = st.session_state.get("active_model", OLLAMA_MODEL)
    readiness = check_ollama_readiness(active_model, OLLAMA_BASE_URL)
    if not readiness.get("ready"):
        st.error(readiness.get("error") or "Ollama is not ready for generation.")
        render_ollama_setup_commands()
        return None

    try:
        controller = load_rag_controller(model=active_model)
    except SystemExit:
        st.error("RAG dependencies or vector artifacts are unavailable.")
        render_setup_commands()
        return None
    except Exception as exc:
        st.error(f"Unable to initialize RAG controller: {exc}")
        return None

    with st.status(f"Generating {label.lower()}...", expanded=True) as status:
        def write_progress(message: str) -> None:
            st.write(message)

        try:
            result = controller.query(
                query_text=query,
                mode=mode,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                patient_id=patient_id,
                progress_callback=write_progress,
                batch_id=st.session_state.get("batch_id", ""),
            )
        except Exception as exc:
            status.update(label=f"{label} generation failed", state="error", expanded=True)
            st.error(f"Generation failed: {exc}")
            return None

        status.update(label=f"{label} complete", state="complete", expanded=False)

    return result


def store_generation_result(patient_id: str, mode: str, result: dict) -> None:
    """Persist generated output under mode- and patient-specific session keys."""
    response_key = f"{mode}_response"
    response = result.get(response_key) or {}
    generated_text = response.get("response", "")

    st.session_state[generation_state_key(mode, patient_id, "result")] = result
    st.session_state[generation_state_key(mode, patient_id, "summary")] = generated_text
    st.session_state[generation_state_key(mode, patient_id, "evidence")] = result.get("chunks", [])
    st.session_state[generation_state_key(mode, patient_id, "flags")] = (
        result.get("verification_flags", {}).get(response_key, {})
    )

    if mode == "clinician":
        st.session_state[generation_state_key(mode, patient_id, "edited_summary")] = generated_text
        st.session_state[generation_state_key(mode, patient_id, "override_ack")] = False


def render_pipeline_metrics(result: dict, mode: str) -> None:
    """Render RAG timing and retrieval metrics."""
    response = result.get(f"{mode}_response") or {}
    metrics = st.columns(4)
    metrics[0].metric("Chunks", result.get("chunks_retrieved", 0))
    metrics[1].metric("Retrieval", f"{result.get('retrieval_duration_ms', 0)} ms")
    metrics[2].metric("Generation", f"{response.get('duration_ms', 0)} ms")
    metrics[3].metric("Total", f"{result.get('total_duration_ms', 0)} ms")


def render_verifier_findings(flags: dict) -> None:
    """Render numeric and semantic verifier warnings."""
    numeric_findings = flags.get("numeric_findings", [])
    semantic_findings = flags.get("semantic_findings", [])

    if not numeric_findings and not semantic_findings:
        st.success("No verifier warnings were raised for this summary.")
        return

    if numeric_findings:
        st.warning("Numeric verifier warnings")
        for finding in numeric_findings:
            claim = finding.get("claim_text", "")
            reason = finding.get("reason", "unsupported")
            source = finding.get("best_candidate_source") or "no close source match"
            st.write(f"- {claim} | reason: {reason} | closest source: {source}")

    if semantic_findings:
        st.warning("Semantic verifier warnings")
        for finding in semantic_findings:
            claim = finding.get("claim_text", "")
            reason = finding.get("reason", "unsupported")
            source = finding.get("best_candidate_source") or "no close source match"
            st.write(f"- {claim} | reason: {reason} | closest source: {source}")


def render_evidence_chunks(chunks: list[dict]) -> None:
    """Render retrieved evidence chunks with traceable expanders."""
    if not chunks:
        st.info("No patient-matched evidence chunks were retrieved.")
        return

    for chunk in chunks:
        rank = chunk.get("rank", "")
        resource_type = chunk.get("resource_type", "Note") or "Note"
        note_id = chunk.get("note_id", "unknown")
        date = chunk.get("date", "")
        fused_score = chunk.get("fused_score", 0.0)
        date_label = f" | {date}" if date else ""
        title = f"Rank {rank} | {resource_type} | {note_id}{date_label} | fused {fused_score}"

        with st.expander(title):
            st.caption(
                f"Resource ID: {chunk.get('resource_id', 'unknown')} | "
                f"Dense: {chunk.get('dense_score', 0.0)} | "
                f"Lexical: {chunk.get('lexical_score', 0.0)}"
            )
            st.write(chunk.get("note_text", ""))


def log_clinician_decision(
    patient_id: str,
    query: str,
    original_text: str,
    edited_text: str,
) -> None:
    """Append a clinician decision row to the audit log (Task 12).

    clinician_accepted=True when the text was left unchanged (accepted),
    False when the clinician edited before finalising (override).
    """
    import verifier as _verifier  # noqa: PLC0415 — lazy to avoid circular startup cost
    clinician_accepted = original_text.strip() == edited_text.strip()
    _verifier.log_provenance(
        query=query,
        mode="clinician_decision",
        chunks=[],
        generated_text=edited_text,
        unsupported_numbers=[],
        unsupported_claims=[],
        retrieval_ms=0,
        generation_ms=0,
        clinician_accepted=clinician_accepted,
        batch_id=st.session_state.get("batch_id", ""),
    )


def render_clinician_override(patient_id: str, summary_text: str, flags: dict) -> None:
    """Render editable clinician-reviewed summary controls (Task 12)."""
    edited_key = generation_state_key("clinician", patient_id, "edited_summary")
    ack_key = generation_state_key("clinician", patient_id, "override_ack")
    decision_key = generation_state_key("clinician", patient_id, "decision_logged")

    if edited_key not in st.session_state:
        st.session_state[edited_key] = summary_text
    if ack_key not in st.session_state:
        st.session_state[ack_key] = False
    if decision_key not in st.session_state:
        st.session_state[decision_key] = False

    st.text_area(
        "Clinician-reviewed summary",
        key=edited_key,
        height=280,
    )

    has_warnings = bool(flags.get("numeric_findings") or flags.get("semantic_findings"))
    st.checkbox(
        "Acknowledge Verifier Warnings",
        key=ack_key,
        disabled=not has_warnings,
    )

    # Task 12 — log the clinician's final decision
    result_key = generation_state_key("clinician", patient_id, "result")
    original_query = (
        st.session_state.get(result_key, {}).get("query", "") if st.session_state.get(result_key) else ""
    )
    edited_text = st.session_state.get(edited_key, "")
    accepted = summary_text.strip() == edited_text.strip()
    btn_label = "✅ Accept Summary" if accepted else "📝 Log Override"

    if st.session_state.get(decision_key):
        st.success("Decision logged to audit trail.")
    elif st.button(btn_label, key=generation_state_key("clinician", patient_id, "log_btn")):
        log_clinician_decision(
            patient_id=patient_id,
            query=original_query,
            original_text=summary_text,
            edited_text=edited_text,
        )
        st.session_state[decision_key] = True
        st.rerun()


def generation_is_available() -> bool:
    """Render dependency warnings and return whether generation can run."""
    available = True

    if not vector_artifacts_exist():
        st.error("Vector artifacts are missing. Build the note index before generation.")
        render_setup_commands()
        available = False

    active_model = st.session_state.get("active_model", OLLAMA_MODEL)
    readiness = check_ollama_readiness(active_model, OLLAMA_BASE_URL)
    if not readiness.get("ready"):
        st.error(
            "Ollama is not running or the configured model is missing. "
            f"Details: {readiness.get('error', 'unknown readiness error')}"
        )
        render_ollama_setup_commands()
        available = False

    return available


def render_generation_ui(
    patient: pd.Series,
    mode: str,
    tab_label: str,
    button_label: str,
    default_query: str,
    empty_message: str,
    allow_override: bool = False,
) -> None:
    """Render shared generation controls and results for one RAG output mode."""
    patient_id = clean_value(patient.get("patient_id", ""))
    result_key = generation_state_key(mode, patient_id, "result")
    evidence_key = generation_state_key(mode, patient_id, "evidence")
    flags_key = generation_state_key(mode, patient_id, "flags")
    summary_key = generation_state_key(mode, patient_id, "summary")
    query_key = generation_state_key(mode, patient_id, "query")
    top_k_key = generation_state_key(mode, patient_id, "top_k")
    temperature_key = generation_state_key(mode, patient_id, "temperature")
    max_tokens_key = generation_state_key(mode, patient_id, "max_tokens")
    button_key = generation_state_key(mode, patient_id, "button")

    if query_key not in st.session_state:
        st.session_state[query_key] = default_query

    st.subheader(tab_label)
    query = st.text_input(f"{tab_label} query", key=query_key)

    control_columns = st.columns(3)
    # Task 11: respect any active chunk-reduction tune from the Latency Tuning panel
    _default_top_k = int(st.session_state.get("tuned_top_k") or 3)
    top_k = control_columns[0].number_input(
        "Evidence chunks",
        1,
        12,
        _default_top_k,
        key=top_k_key,
    )
    temperature = control_columns[1].slider(
        "Temperature",
        0.0,
        1.0,
        0.3,
        0.05,
        key=temperature_key,
    )
    max_tokens = control_columns[2].number_input(
        "Max tokens",
        256,
        2048,
        1024,
        128,
        key=max_tokens_key,
    )

    can_generate = generation_is_available()
    if st.button(button_label, type="primary", disabled=not can_generate, key=button_key):
        result = run_generation_query(
            patient_id=patient_id,
            mode=mode,
            query=query,
            top_k=int(top_k),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            label=button_label,
        )
        if result is not None:
            store_generation_result(patient_id, mode, result)

    result = st.session_state.get(result_key)
    summary_text = st.session_state.get(summary_key, "")
    flags = st.session_state.get(flags_key, {})
    evidence_chunks = st.session_state.get(evidence_key, [])

    if not result:
        st.info(empty_message)
        return

    response = result.get(f"{mode}_response") or {}
    if response.get("error"):
        st.error(f"LLM generation error: {response['error']}")
        render_ollama_setup_commands()

    render_pipeline_metrics(result, mode)

    st.markdown("### Generated Output")
    if summary_text:
        st.write(summary_text)
    else:
        st.info("No generated text was returned.")

    if allow_override:
        st.markdown("### Clinician Override")
        render_clinician_override(patient_id, summary_text, flags)

    st.markdown("### Verifier Findings")
    render_verifier_findings(flags)

    st.markdown("### Retrieved Evidence")
    render_evidence_chunks(evidence_chunks)


def render_clinician_dashboard(patient: pd.Series) -> None:
    """Render the clinician dashboard."""
    render_generation_ui(
        patient=patient,
        mode="clinician",
        tab_label="Clinician Dashboard",
        button_label="Generate Clinician Summary",
        default_query=build_default_clinician_query(patient),
        empty_message="Generate a clinician summary to view evidence and verifier findings.",
        allow_override=True,
    )


def render_patient_explanation(patient: pd.Series) -> None:
    """Render the patient-facing explanation workflow."""
    render_generation_ui(
        patient=patient,
        mode="patient",
        tab_label="Patient Explanation",
        button_label="Generate Patient Explanation",
        default_query=build_default_patient_query(patient),
        empty_message="Generate a patient explanation to view simple results and next questions.",
    )


# -----------------
# Latency Tuning Panel (Day 4 Task 11)
# -----------------

def render_latency_tuning_panel(audit_df: "pd.DataFrame") -> None:
    """Render Task 11 latency tuning controls.

    Shows current average latency vs the 2800ms target. When over target,
    surfaces two actionable options: reduce retrieved chunks or switch the
    active Ollama model to a lighter quantized variant.
    """
    st.markdown("### ⚡ Latency Tuning (Task 11 — Target: 2.8 s)")

    # Only measure generation rows (not clinician_decision rows)
    gen_df = audit_df[audit_df["mode"].isin(["clinician", "patient"])]
    active_model = st.session_state.get("active_model", OLLAMA_MODEL)
    tuned_top_k = st.session_state.get("tuned_top_k") or 3

    # ---- Status banner ----
    status_col, info_col = st.columns([3, 2])
    with info_col:
        st.info(
            f"**Active model:** `{active_model}`  \n"
            f"**Current top_k:** `{tuned_top_k}`"
        )

    if gen_df.empty or "generation_ms" not in gen_df.columns:
        with status_col:
            st.info("⌛ No generation data yet — run a clinician or patient query first.")
        return

    gen_ms = pd.to_numeric(gen_df["generation_ms"], errors="coerce").dropna()
    avg_ms = float(gen_ms.mean()) if len(gen_ms) else 0.0
    pct_passing = float((gen_ms < 2800).mean() * 100) if len(gen_ms) else 0.0

    with status_col:
        if avg_ms <= 2800:
            st.success(
                f"✅ **Target met** — Avg {avg_ms:.0f} ms < 2800 ms "
                f"({pct_passing:.0f}% of {len(gen_ms)} queries within target)"
            )
        else:
            st.error(
                f"❌ **Over target** — Avg {avg_ms:.0f} ms "
                f"(+{avg_ms - 2800:.0f} ms | {pct_passing:.0f}% within target)"
            )

    if avg_ms <= 2800:
        st.caption("No tuning required. Re-check after more queries are logged.")
        return

    st.markdown("**Tuning Options** — apply one or both to bring average under 2.8 s:")
    opt1_col, opt2_col = st.columns(2)

    # ---- Option 1: Reduce retrieved chunks ----
    with opt1_col:
        with st.container(border=True):
            st.markdown("**📏 Option 1 — Reduce Retrieved Chunks**")
            st.write(
                f"Fewer context chunks means less tokens for the model to process. "
                f"Current top_k: **{tuned_top_k}** \u2192 Suggested: **5**"
            )
            new_top_k = st.select_slider(
                "Set top_k",
                options=[3, 5, 6, 8, 10],
                value=min(tuned_top_k, 5),
                key="tune_top_k_slider",
            )
            st.caption(f"Estimated impact: ~{(tuned_top_k - new_top_k) * 150:.0f} ms saved per query.")
            def apply_chunk_reduction():
                new_val = int(st.session_state["tune_top_k_slider"])
                st.session_state["tuned_top_k"] = new_val
                for _k in list(st.session_state.keys()):
                    if "_top_k_" in _k:
                        st.session_state[_k] = new_val
                st.cache_data.clear()

            if st.button("\u26a1 Apply Chunk Reduction", key="apply_chunk_tune", type="primary", on_click=apply_chunk_reduction):
                st.success(f"top_k set to {st.session_state['tuned_top_k']}. Run a new query and refresh to measure improvement.")

    # ---- Option 2: Switch model ----
    with opt2_col:
        with st.container(border=True):
            st.markdown("**🤖 Option 2 — Switch to Lighter Model**")
            st.write(
                "A smaller or more quantized model has lower inference latency. "
                "Quantized variants (q4_0) are typically 40-60% faster than default."
            )
            readiness = check_ollama_readiness(active_model, OLLAMA_BASE_URL)
            available_models = readiness.get("models", [])
            if not available_models:
                available_models = [active_model]

            current_idx = (
                available_models.index(active_model)
                if active_model in available_models else 0
            )
            selected_model = st.selectbox(
                "Select model",
                available_models,
                index=current_idx,
                key="tune_model_select",
            )
            st.caption(
                "After switching, click \"Generate\" to measure the new latency. "
                "Pull lighter models with: `ollama pull llama3:8b-instruct-q4_0`"
            )
            if st.button("🔄 Apply Model Switch", key="apply_model_tune"):
                st.session_state["active_model"] = selected_model
                st.cache_resource.clear()  # force controller reload with new model
                st.cache_data.clear()
                st.success(f"Model switched to `{selected_model}`. Run a new query to benchmark.")
                st.rerun()


# -----------------
# Audit Log Tab (Day 4 Task 10/11)
# -----------------

@st.cache_data(show_spinner=False, ttl=30)
def load_audit_log_cached() -> pd.DataFrame:
    """Load the audit log with a short TTL cache to avoid per-render reads."""
    import verifier as _v  # noqa: PLC0415
    return _v.load_audit_log()


def render_health_scorecard(df: pd.DataFrame) -> None:
    """Render Performance Score, Grounding Score, and Override Rate cards."""
    if df.empty:
        st.info("No audit data yet. Run a generation query to populate the log.")
        return

    gen_ms = pd.to_numeric(df.get("generation_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    performance_score = float((gen_ms < 2800).mean() * 100) if len(gen_ms) else 0.0

    numeric_flag = df.get("numeric_flag", pd.Series([False] * len(df))).fillna(False).astype(bool)
    semantic_flag = df.get("semantic_flag", pd.Series([False] * len(df))).fillna(False).astype(bool)
    grounding_score = float((~(numeric_flag | semantic_flag)).mean() * 100)

    clinician_df = df[df["mode"] == "clinician_decision"]
    if not clinician_df.empty and "clinician_accepted" in clinician_df.columns:
        reviewed = clinician_df["clinician_accepted"].astype(str).isin(["True", "False", "true", "false"])
        accepted = clinician_df["clinician_accepted"].astype(str).str.lower() == "true"
        override_label = f"{accepted.sum() / max(reviewed.sum(), 1) * 100:.0f}%"
    else:
        override_label = "N/A"

    perf_delta = "✅ On target" if performance_score >= 80 else "⚠️ Needs tuning"
    grnd_delta = "✅ Clean" if grounding_score >= 90 else "⚠️ Flags present"

    cols = st.columns(4)
    cols[0].metric("Total Queries", len(df))
    cols[1].metric("⚡ Performance Score", f"{performance_score:.0f}%", perf_delta)
    cols[2].metric("🎯 Grounding Score", f"{grounding_score:.0f}%", grnd_delta)
    cols[3].metric("🩺 Override Rate", override_label)


def render_latency_chart(df: pd.DataFrame) -> None:
    """Render generation latency trend; highlight when average exceeds 2800ms."""
    if df.empty or "generation_ms" not in df.columns:
        st.info("No latency data available.")
        return

    gen_ms = pd.to_numeric(df["generation_ms"], errors="coerce").fillna(0)
    avg_ms = float(gen_ms.mean())

    if avg_ms < 2800:
        st.success(f"✅ Avg latency **{avg_ms:.0f} ms** — under 2.8 s target")
    else:
        st.error(f"❌ Avg latency **{avg_ms:.0f} ms** — {avg_ms - 2800:.0f} ms over target")

    chart_df = pd.DataFrame({
        "Generation (ms)": gen_ms.values,
        "Target 2800 ms": [2800] * len(gen_ms),
    })
    st.line_chart(chart_df, height=220)
    st.caption(f"Peak: {gen_ms.max():.0f} ms | Target: 2800 ms | n={len(gen_ms)}")


def render_flag_rate_chart(df: pd.DataFrame) -> None:
    """Render numeric and semantic verifier flag counts as a stacked bar chart."""
    if df.empty:
        st.info("No verifier data available.")
        return

    numeric_flag = df.get("numeric_flag", pd.Series([False] * len(df))).fillna(False).astype(bool).astype(int)
    semantic_flag = df.get("semantic_flag", pd.Series([False] * len(df))).fillna(False).astype(bool).astype(int)

    chart_df = pd.DataFrame({
        "Numeric Flags": numeric_flag.values,
        "Semantic Flags": semantic_flag.values,
    })
    st.bar_chart(chart_df, height=220)
    st.caption(
        f"Numeric: {int(numeric_flag.sum())}/{len(df)} | "
        f"Semantic: {int(semantic_flag.sum())}/{len(df)} queries flagged"
    )


def render_audit_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter controls and return the filtered DataFrame."""
    filter_cols = st.columns(4)

    modes = ["All"] + sorted(df["mode"].dropna().unique().tolist())
    selected_mode = filter_cols[0].selectbox("Mode", modes, key="audit_filter_mode")

    sessions = ["All"]
    if "batch_id" in df.columns:
        sessions += sorted(df["batch_id"].dropna().unique().tolist(), reverse=True)
    selected_session = filter_cols[1].selectbox("Session", sessions, key="audit_filter_session")

    over_threshold = filter_cols[2].checkbox("⚠️ Over 2.8 s only", key="audit_filter_threshold")

    has_flags_only = filter_cols[3].checkbox("🚩 Flagged only", key="audit_filter_flags")

    filtered = df.copy()
    if selected_mode != "All":
        filtered = filtered[filtered["mode"] == selected_mode]
    if selected_session != "All":
        filtered = filtered[filtered["batch_id"] == selected_session]
    if over_threshold:
        filtered = filtered[pd.to_numeric(filtered.get("generation_ms", 0), errors="coerce").fillna(0) >= 2800]
    if has_flags_only:
        nf = filtered.get("numeric_flag", pd.Series([False] * len(filtered))).fillna(False).astype(bool)
        sf = filtered.get("semantic_flag", pd.Series([False] * len(filtered))).fillna(False).astype(bool)
        filtered = filtered[nf | sf]
    return filtered.reset_index(drop=True)


def render_audit_table(df: pd.DataFrame) -> None:
    """Render audit log as a summary table with expandable row details."""
    if df.empty:
        st.info("No audit rows match the current filters.")
        return

    display_cols = [c for c in [
        "timestamp", "batch_id", "mode", "generation_ms",
        "retrieval_ms", "numeric_flag", "semantic_flag", "clinician_accepted",
    ] if c in df.columns]

    display_df = df[display_cols].copy()
    if "generation_ms" in display_df.columns:
        display_df["generation_ms"] = display_df["generation_ms"].apply(
            lambda x: f"🔴 {x} ms" if pd.notna(x) and int(x) >= 2800 else f"🟢 {x} ms"
        )
    st.dataframe(display_df, use_container_width=True)

    st.markdown("#### Row Details")
    for i, (_, row) in enumerate(df.iterrows()):
        ts = str(row.get("timestamp", ""))[:19]
        mode_label = row.get("mode", "")
        gen_ms = row.get("generation_ms", 0)
        latency_icon = "🔴" if pd.notna(gen_ms) and int(gen_ms) >= 2800 else "🟢"
        with st.expander(f"[{i + 1}] {ts} | {mode_label} | {latency_icon} {gen_ms} ms"):
            st.markdown(f"**Query:** {row.get('query', '')}")
            st.markdown(f"**Batch:** `{row.get('batch_id', '')}`")
            accepted_val = row.get("clinician_accepted", "")
            if str(accepted_val).lower() == "true":
                st.success("Clinician accepted this summary.")
            elif str(accepted_val).lower() == "false":
                st.warning("Clinician overrode this summary.")
            st.markdown("**Generated Text:**")
            st.write(row.get("generated_text", ""))
            nf_json = row.get("numeric_findings_json", "")
            sf_json = row.get("semantic_findings_json", "")
            if nf_json and nf_json not in ("", "[]"):
                with st.expander("Numeric Findings JSON"):
                    st.code(nf_json, language="json")
            if sf_json and sf_json not in ("", "[]"):
                with st.expander("Semantic Findings JSON"):
                    st.code(sf_json, language="json")


def render_audit_tab() -> None:
    """Render the Audit Log & Analytics tab (Day 4 Task 10 / 11)."""
    col_refresh, col_info = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Logs", key="audit_refresh"):
            st.cache_data.clear()
            st.rerun()

    audit_df = load_audit_log_cached()

    with col_info:
        st.caption(
            f"{len(audit_df)} total entries in `datasets/audit_log.csv` "
            f"| Current session: `{st.session_state.get('batch_id', 'unknown')}`"
        )

    if audit_df.empty:
        st.info("No audit data yet — run a generation query to populate the log.")
        return

    st.markdown("### 📊 Health Scorecard")
    render_health_scorecard(audit_df)
    st.divider()

    render_latency_tuning_panel(audit_df)
    st.divider()

    st.markdown("### Filters")
    filtered_df = render_audit_filters(audit_df)
    st.caption(f"Showing **{len(filtered_df)}** of {len(audit_df)} rows")
    st.divider()

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("#### ⚡ Generation Latency (ms)")
        render_latency_chart(filtered_df)
    with chart_cols[1]:
        st.markdown("#### 🚩 Verifier Flag Rate")
        render_flag_rate_chart(filtered_df)
    st.divider()

    st.markdown("### Audit Log")
    render_audit_table(filtered_df)


def render_main_tabs(patient: pd.Series) -> None:
    """Render three-tab dashboard: Clinician, Patient, and Audit & Analytics."""
    clinician_tab, patient_tab, audit_tab = st.tabs([
        "🩺 Clinician Dashboard",
        "👤 Patient Explanation",
        "📊 Audit & Analytics",
    ])

    with clinician_tab:
        render_clinician_dashboard(patient)

    with patient_tab:
        render_patient_explanation(patient)

    with audit_tab:
        render_audit_tab()


def render_app(patients_df: pd.DataFrame) -> None:
    """Render the Day 4 app: clinician + patient + audit & analytics tabs."""
    patients_df = normalize_patients_df(patients_df)
    if patients_df.empty:
        st.error("datasets/patients.csv does not contain any patient rows.")
        st.stop()

    render_sidebar(patients_df)
    active_patient = get_active_patient(patients_df)

    st.title("Clinical RAG Dashboard")
    st.caption("Day 4 — System Testing: latency tuning · clinician override · audit analytics")
    render_patient_profile(active_patient)
    st.divider()
    render_main_tabs(active_patient)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Clinical RAG Dashboard",
        layout="wide",
    )
    
    # Premium UI Overhaul Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #00D4FF 0%, #007BFF 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 123, 255, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 123, 255, 0.3);
            border: none;
        }

        /* Card / Container Styling */
        div[data-testid="stExpander"] {
            background-color: #151B2B;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        div[data-testid="stExpander"]:hover {
            border: 1px solid rgba(0, 212, 255, 0.3);
        }

        /* Metric Styling */
        div[data-testid="stMetricValue"] {
            font-weight: 700;
            color: #00D4FF !important;
        }

        /* Input Elements */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
            background-color: #0A0E17;
            transition: border-color 0.3s ease;
        }
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
            border-color: #00D4FF;
            box-shadow: 0 0 0 1px #00D4FF;
        }

        /* Headers */
        h1, h2, h3 {
            background: linear-gradient(90deg, #FFFFFF 0%, #B0C4DE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    validate_environment()
    initialize_session_state()

    if not patients_file_exists():
        render_missing_data_message()
        return

    try:
        patients_df = load_patients(PATIENTS_PATH)
    except ValueError as exc:
        st.error(f"Unable to load {PATIENTS_PATH}: {exc}")
        st.stop()
    except OSError as exc:
        st.error(f"Unable to read {PATIENTS_PATH}: {exc}")
        st.stop()

    render_app(patients_df)


if __name__ == "__main__":
    main()
