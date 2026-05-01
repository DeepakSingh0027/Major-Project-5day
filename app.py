"""Streamlit frontend for the clinical RAG knowledge base.

Provides patient profile selection, clinician-facing RAG summaries, and
patient-facing explanations over the local clinical knowledge base.
"""

import os
import sys
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
def load_rag_controller():
    """Load the RAG controller only when generation is requested."""
    import rag_controller

    return rag_controller.RAGController(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


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

    readiness = check_ollama_readiness(OLLAMA_MODEL, OLLAMA_BASE_URL)
    if not readiness.get("ready"):
        st.error(readiness.get("error") or "Ollama is not ready for generation.")
        render_ollama_setup_commands()
        return None

    try:
        controller = load_rag_controller()
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


def render_clinician_override(patient_id: str, summary_text: str, flags: dict) -> None:
    """Render editable clinician-reviewed summary controls."""
    edited_key = generation_state_key("clinician", patient_id, "edited_summary")
    ack_key = generation_state_key("clinician", patient_id, "override_ack")

    if edited_key not in st.session_state:
        st.session_state[edited_key] = summary_text
    if ack_key not in st.session_state:
        st.session_state[ack_key] = False

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


def generation_is_available() -> bool:
    """Render dependency warnings and return whether generation can run."""
    available = True

    if not vector_artifacts_exist():
        st.error("Vector artifacts are missing. Build the note index before generation.")
        render_setup_commands()
        available = False

    readiness = check_ollama_readiness(OLLAMA_MODEL, OLLAMA_BASE_URL)
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
    top_k = control_columns[0].number_input(
        "Evidence chunks",
        1,
        12,
        8,
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


def render_dual_output_tabs(patient: pd.Series) -> None:
    """Render the Day 3 dual-output shell without triggering RAG generation."""
    clinician_tab, patient_tab = st.tabs(["Clinician Dashboard", "Patient Explanation"])

    with clinician_tab:
        render_clinician_dashboard(patient)

    with patient_tab:
        render_patient_explanation(patient)


def render_app(patients_df: pd.DataFrame) -> None:
    """Render the complete Day 3 Task 7 app shell."""
    patients_df = normalize_patients_df(patients_df)
    if patients_df.empty:
        st.error("datasets/patients.csv does not contain any patient rows.")
        st.stop()

    render_sidebar(patients_df)
    active_patient = get_active_patient(patients_df)

    st.title("Clinical RAG Dashboard")
    st.caption("Day 3 Task 7: patient selection and application initialization")
    render_patient_profile(active_patient)
    st.divider()
    render_dual_output_tabs(active_patient)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Clinical RAG Dashboard",
        layout="wide",
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
