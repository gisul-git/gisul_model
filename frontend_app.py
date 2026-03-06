import streamlit as st
import requests
import pandas as pd

API_BASE = "http://localhost:8000/api/v1"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen Question Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Qwen Question Generator")

# ─────────────────────────────────────────────
# SESSION STATE — must be initialised before
# any widgets render so sidebar can read flags
# ─────────────────────────────────────────────
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_endpoint" not in st.session_state:
    st.session_state.last_endpoint = None

generating = st.session_state.is_generating

# ─────────────────────────────────────────────
# SIDEBAR  —  server controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Server Controls")

    if generating:
        st.warning("⏳ Generation in progress — server controls disabled.")

    # Health check
    if st.button("🔍 Check Server Health", use_container_width=True,
                 disabled=generating,
                 help="Disabled while generation is running."):
        try:
            r = requests.get(f"http://localhost:8000/health", timeout=5)
            h = r.json()
            if h.get("status") == "healthy":
                st.success("Server is healthy ✅")
                st.write(f"Model loaded: `{h.get('model_loaded')}`")
                st.write(f"GPU memory: `{round(h.get('memory_gb', 0), 2)} GB`")
                qs = h.get("queue_sizes", {})
                if any(v > 0 for v in qs.values()):
                    st.write("Queue sizes:", qs)
            else:
                st.warning("Server responded but may not be fully ready.")
        except Exception as e:
            st.error(f"Cannot reach server: {e}")

    st.divider()

    # Cache controls
    st.subheader("🗑️ Cache")
    use_cache = st.toggle("Use Cache", value=True,
                          help="Uncheck to bypass cache and always regenerate fresh questions.")

    if st.button("🧹 Clear Cache", use_container_width=True, type="secondary",
                 disabled=generating,
                 help="Disabled while generation is running."):
        try:
            r = requests.post(f"{API_BASE}/clear-cache", timeout=10)
            result = r.json()
            st.success(f"✅ {result.get('message', 'Cache cleared')}")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

    st.divider()

    # Stats
    if st.button("📊 View Stats", use_container_width=True,
                 disabled=generating,
                 help="Disabled while generation is running."):
        try:
            r = requests.get(f"http://localhost:8000/stats", timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"Could not fetch stats: {e}")

# ─────────────────────────────────────────────
# ENDPOINT SELECTOR
# ─────────────────────────────────────────────
ENDPOINTS = [
    "generate-topics",
    "generate-mcq",
    "generate-subjective",
    "generate-coding",
    "generate-sql",
    "generate-aiml",
]

ENDPOINT_LABELS = {
    "generate-topics":     "📋 Generate Topics",
    "generate-mcq":        "✅ Multiple Choice (MCQ)",
    "generate-subjective": "✍️  Subjective Questions",
    "generate-coding":     "💻 Coding Problems",
    "generate-sql":        "🗄️  SQL Problems",
    "generate-aiml":       "🤖 AI/ML Dataset Problems",
}

endpoint = st.selectbox(
    "Choose Endpoint",
    ENDPOINTS,
    format_func=lambda x: ENDPOINT_LABELS[x],
)

st.divider()

# ─────────────────────────────────────────────
# SHARED INPUTS — rendered based on endpoint
# ─────────────────────────────────────────────
needs_topic_fields   = endpoint in ("generate-mcq", "generate-subjective",
                                    "generate-coding", "generate-sql", "generate-aiml")
needs_topics_fields  = endpoint == "generate-topics"
needs_language       = endpoint == "generate-coding"
needs_db_type        = endpoint == "generate-sql"
needs_audience       = endpoint in ("generate-mcq", "generate-subjective")

st.subheader("🔧 Request Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    num_questions = st.number_input(
        "Number of Questions / Items",
        min_value=1, max_value=20, value=3,
        help="How many items to generate in a single request."
    )

if needs_topics_fields:
    col1b, col2b = st.columns(2)
    with col1b:
        assessment_title = st.text_input("Assessment Title", "Python Developer Assessment")
        job_designation  = st.text_input("Job Designation",  "Software Developer")
        skills           = st.text_input("Skills (comma separated)", "Python, FastAPI, SQL")
    with col2b:
        experience_min   = st.number_input("Min Experience (yrs)", 0, 30, 1)
        experience_max   = st.number_input("Max Experience (yrs)", 0, 30, 3)
        # num_topics is driven by num_questions above — no separate input needed

if needs_topic_fields:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        topic = st.text_input("Topic", "Python list slicing")
    with col_t2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

if needs_audience:
    target_audience = st.text_input("Target Audience", "Mid-level Software Developers")

if needs_language:
    language = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "Go", "TypeScript"])

if needs_db_type:
    database_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite", "MS SQL Server"])

if endpoint in ("generate-coding", "generate-sql"):
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        job_role = st.text_input("Job Role", "Software Engineer",
                                  help="e.g. Backend Engineer, Data Engineer, Full Stack Developer")
    with col_r2:
        experience_years = st.selectbox("Experience Level",
                                         ["0-1 (Fresher)", "1-3 (Junior)", "3-5 (Mid-level)",
                                          "5-8 (Senior)", "8+ (Staff/Principal)"],
                                         index=2)

st.divider()

# ─────────────────────────────────────────────
# PAYLOAD BUILDER — only sends relevant keys
# ─────────────────────────────────────────────
def build_payload() -> dict:
    base = {
        "num_questions": int(num_questions),
        "use_cache": use_cache,
    }

    if endpoint == "generate-topics":
        base.update({
            "assessment_title": assessment_title,
            "job_designation":  job_designation,
            "skills":           [s.strip() for s in skills.split(",") if s.strip()],
            "experience_min":   int(experience_min),
            "experience_max":   int(experience_max),
            # num_questions drives topic count — passed as num_topics to the API
            "num_topics":       int(num_questions),
        })

    elif endpoint == "generate-mcq":
        base.update({
            "topic":           topic,
            "difficulty":      difficulty,
            "target_audience": target_audience,
        })

    elif endpoint == "generate-subjective":
        base.update({
            "topic":           topic,
            "difficulty":      difficulty,
            "target_audience": target_audience,
        })

    elif endpoint == "generate-coding":
        base.update({
            "topic":            topic,
            "difficulty":       difficulty,
            "language":         language,
            "job_role":         job_role,
            "experience_years": experience_years,
        })

    elif endpoint == "generate-sql":
        base.update({
            "topic":            topic,
            "difficulty":       difficulty,
            "database_type":    database_type,
            "job_role":         job_role,
            "experience_years": experience_years,
        })

    elif endpoint == "generate-aiml":
        base.update({
            "topic":      topic,
            "difficulty": difficulty,
        })

    return base


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def show_metadata(data: dict):
    """Always show generation metadata at the bottom of a response."""
    cols = st.columns(3)
    with cols[0]:
        t = data.get("generation_time_seconds", 0)
        st.metric("⏱ Generation Time", f"{round(t, 2)}s")
    with cols[1]:
        st.metric("📦 Batch Size", data.get("batch_size", 1))
    with cols[2]:
        hit = data.get("cache_hit", False)
        st.metric("🗃 Cache Hit", "Yes" if hit else "No")


def render_mcq_question(q: dict, index: int):
    st.markdown(f"#### Question {index}")
    st.write(q.get("question", "*(no question text)*"))

    options = q.get("options", [])
    if options:
        for opt in options:
            is_correct = opt.get("isCorrect", False)
            label = opt.get("label", "?")
            text  = opt.get("text", "")
            if is_correct:
                st.markdown(f"**{label}. ✅ {text}**")
            else:
                st.write(f"{label}. {text}")

    exp = q.get("explanation", "")
    if exp:
        with st.expander("💡 Explanation"):
            st.info(exp)

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.caption(f"Difficulty: `{q.get('difficulty', '—')}`")
    with meta_cols[1]:
        st.caption(f"Bloom Level: `{q.get('bloomLevel', '—')}`")


def render_subjective_question(q: dict, index: int):
    st.markdown(f"#### Question {index}")
    st.write(q.get("question", "*(no question text)*"))

    ans = q.get("expectedAnswer", "")
    if ans:
        with st.expander("📝 Expected Answer"):
            st.success(ans)

    criteria = q.get("gradingCriteria", [])
    if criteria:
        with st.expander("📏 Grading Criteria"):
            for c in criteria:
                st.write(f"• {c}")

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.caption(f"Difficulty: `{q.get('difficulty', '—')}`")
    with meta_cols[1]:
        st.caption(f"Bloom Level: `{q.get('bloomLevel', '—')}`")


def render_coding_problem(p: dict, index: int):
    st.markdown(f"#### Problem {index}")

    st.markdown("**Problem Statement**")
    st.write(p.get("problemStatement", ""))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Input Format**")
        st.write(p.get("inputFormat", ""))
    with col_b:
        st.markdown("**Output Format**")
        st.write(p.get("outputFormat", ""))

    constraints = p.get("constraints", [])
    if constraints:
        with st.expander("📐 Constraints"):
            for c in constraints:
                st.write(f"• {c}")

    examples = p.get("examples", [])
    if examples:
        with st.expander(f"📖 Examples ({len(examples)})"):
            for ex in examples:
                st.write(f"**Input:** `{ex.get('input', '')}`")
                st.write(f"**Output:** `{ex.get('output', '')}`")
                if "explanation" in ex:
                    st.caption(ex["explanation"])
                st.divider()

    starter = p.get("starterCode", "")
    if starter:
        lang_map = {"Python": "python", "JavaScript": "javascript",
                    "Java": "java", "C++": "cpp", "Go": "go", "TypeScript": "typescript"}
        lang = lang_map.get(p.get("language", "Python"), "python")
        with st.expander("🖊 Starter Code"):
            st.code(starter, language=lang)

    test_cases = p.get("testCases", [])
    if test_cases:
        with st.expander(f"🧪 Test Cases ({len(test_cases)})"):
            try:
                st.dataframe(pd.DataFrame(test_cases), use_container_width=True)
            except Exception:
                st.json(test_cases)

    # New fields from improved prompt
    complexity = p.get("expectedComplexity", "")
    if complexity:
        st.caption(f"Expected Complexity: `{complexity}`")

    hints = p.get("hints", [])
    if hints:
        with st.expander("💡 Hints"):
            for idx, h in enumerate(hints, 1):
                st.write(f"Hint {idx}: {h}")

    st.caption(f"Difficulty: `{p.get('difficulty', '—')}`")


def render_sql_problem(p: dict, index: int):
    st.markdown(f"#### SQL Problem {index}")
    st.write(p.get("problemStatement", ""))

    schema = p.get("schema", {})
    if schema:
        with st.expander("🗂 Schema"):
            tables = schema.get("tables", [])
            for table in tables:
                st.markdown(f"**Table: `{table.get('name', '?')}`**")
                cols = table.get("columns", [])
                if cols:
                    try:
                        st.dataframe(pd.DataFrame(cols), use_container_width=True)
                    except Exception:
                        st.json(cols)

    expected_query = p.get("expectedQuery", "")
    if expected_query:
        with st.expander("✅ Expected Query"):
            st.code(expected_query, language="sql")

    explanation = p.get("explanation", "")
    if explanation:
        with st.expander("💡 Explanation"):
            st.info(explanation)

    # New fields from improved prompt
    alt = p.get("alternativeApproach", "")
    if alt:
        with st.expander("⚠️ Alternative Approach & Why It's Worse"):
            st.write(alt)

    concepts = p.get("concepts_tested", [])
    if concepts:
        st.caption(f"Concepts tested: {', '.join(concepts)}")

    expected_output = p.get("expectedOutput", [])
    if expected_output:
        with st.expander("📋 Expected Output"):
            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(expected_output), use_container_width=True)
            except Exception:
                st.json(expected_output)

    st.caption(f"Difficulty: `{p.get('difficulty', '—')}`")


def render_aiml_problem(p: dict, index: int):
    st.markdown(f"#### AI/ML Problem {index}")
    st.write(p.get("problemStatement", ""))

    dataset = p.get("dataset", {})
    if dataset:
        with st.expander("📊 Dataset Details"):
            st.write(f"**Description:** {dataset.get('description', '')}")

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Target:** `{dataset.get('target', '—')}`")
                st.write(f"**Target Type:** `{dataset.get('target_type', '—')}`")
                st.write(f"**Size:** `{dataset.get('size', '—')}`")
            with c2:
                dist = dataset.get("class_distribution", {})
                if dist:
                    st.write("**Class Distribution:**")
                    st.json(dist)

            features = dataset.get("features", [])
            if features:
                st.write(f"**Features:** {', '.join(f'`{f}`' for f in features)}")

            ft = dataset.get("feature_types", {})
            if ft:
                with st.expander("Feature Types"):
                    st.json(ft)

            data_rows = dataset.get("data", [])
            if data_rows:
                with st.expander(f"Sample Data ({len(data_rows)} rows)"):
                    try:
                        df = pd.DataFrame(data_rows)
                        st.dataframe(df.head(20), use_container_width=True)
                        if len(data_rows) > 20:
                            st.caption(f"Showing first 20 of {len(data_rows)} rows.")
                    except Exception:
                        st.json(data_rows[:5])

    approach = p.get("expectedApproach", "")
    if approach:
        with st.expander("🧪 Expected Approach"):
            st.write(approach)

    criteria = p.get("evaluationCriteria", [])
    if criteria:
        with st.expander("📏 Evaluation Criteria"):
            for c in criteria:
                st.write(f"• {c}")

    st.caption(f"Difficulty: `{p.get('difficulty', '—')}`")


def render_topic(t: dict, index: int):
    """Render a single topic dict from the topics list."""
    label = t.get("label", t) if isinstance(t, dict) else str(t)
    qtype = t.get("questionType", "")   if isinstance(t, dict) else ""
    diff  = t.get("difficulty", "")     if isinstance(t, dict) else ""
    judge = t.get("canUseJudge0", None) if isinstance(t, dict) else None

    cols = st.columns([3, 2, 2, 1])
    with cols[0]:
        st.write(f"**{index}.** {label}")
    with cols[1]:
        st.caption(f"Type: `{qtype}`" if qtype else "")
    with cols[2]:
        st.caption(f"Difficulty: `{diff}`" if diff else "")
    with cols[3]:
        if judge is not None:
            st.caption("Judge0 ✅" if judge else "")


# ─────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────
if st.button(
    "⏳ Generating…" if generating else "🚀 Generate",
    type="primary",
    use_container_width=True,
    disabled=generating,
    help="Already generating — please wait." if generating else None,
):
    payload = build_payload()
    url = f"{API_BASE}/{endpoint}"

    with st.expander("📨 Request Payload", expanded=False):
        st.json(payload)

    # Set flag so sidebar buttons disable on next rerun
    st.session_state.is_generating = True
    st.session_state.last_response = None
    st.session_state.last_endpoint = endpoint

    try:
        label = ENDPOINT_LABELS.get(endpoint, endpoint)
        with st.spinner(f"Calling {label} — generating {num_questions} item(s)…"):
            response = requests.post(url, json=payload, timeout=300)

        status = response.status_code

        if status != 200:
            st.error(f"❌ Server returned HTTP {status}")
            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)
        else:
            data = response.json()
            st.session_state.last_response = data
            st.success(f"✅ HTTP {status} — received response")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server at `localhost:8000`. Is it running?")
    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out (300 s). The model may still be generating — check server logs.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    finally:
        # Always clear the flag so sidebar re-enables after request completes
        st.session_state.is_generating = False

# ─────────────────────────────────────────────
# DISPLAY — render last_response from session state
# This survives sidebar interactions (no data loss on rerun)
# ─────────────────────────────────────────────
data = st.session_state.get("last_response")

if data:
    st.divider()
    st.header("📄 Generated Output")

    # ──────────────────────────────────────────
    # TOPICS
    # ──────────────────────────────────────────
    if "topics" in data:
        topics_list = data["topics"]
        st.subheader(f"📋 Generated Topics ({len(topics_list)})")
        for i, t in enumerate(topics_list, 1):
            render_topic(t, i)

    # ──────────────────────────────────────────
    # MCQ / SUBJECTIVE  — {"questions": [...]}
    # ──────────────────────────────────────────
    elif "questions" in data:
        questions_list = data["questions"]
        st.subheader(f"✅ Questions ({len(questions_list)})")
        for i, q in enumerate(questions_list, 1):
            with st.container(border=True):
                if "options" in q:
                    render_mcq_question(q, i)
                else:
                    render_subjective_question(q, i)

    # ──────────────────────────────────────────
    # CODING  — {"coding_problems": [...]}
    # ──────────────────────────────────────────
    elif "coding_problems" in data:
        problems = data["coding_problems"]
        st.subheader(f"💻 Coding Problems ({len(problems)})")
        for i, p in enumerate(problems, 1):
            with st.container(border=True):
                render_coding_problem(p, i)

    # ──────────────────────────────────────────
    # SQL  — {"sql_problems": [...]}
    # ──────────────────────────────────────────
    elif "sql_problems" in data:
        problems = data["sql_problems"]
        st.subheader(f"🗄️ SQL Problems ({len(problems)})")
        for i, p in enumerate(problems, 1):
            with st.container(border=True):
                render_sql_problem(p, i)

    # ──────────────────────────────────────────
    # AIML  — {"aiml_problems": [...]}
    # ──────────────────────────────────────────
    elif "aiml_problems" in data:
        problems = data["aiml_problems"]
        st.subheader(f"🤖 AI/ML Problems ({len(problems)})")
        for i, p in enumerate(problems, 1):
            with st.container(border=True):
                render_aiml_problem(p, i)

    # ──────────────────────────────────────────
    # FALLBACK
    # ──────────────────────────────────────────
    else:
        st.warning("Unrecognised response shape — showing raw JSON.")
        st.json(data)

    # ──────────────────────────────────────────
    # METADATA — always shown
    # ──────────────────────────────────────────
    st.divider()
    st.subheader("📊 Response Metadata")
    show_metadata(data)
