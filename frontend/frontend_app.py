import streamlit as st
import requests
import pandas as pd
import time
import os

API_BASE = os.environ.get("API_BASE", "http://localhost:9000/api/v1")
_SERVER_BASE = API_BASE.replace("/api/v1", "")

st.set_page_config(
    page_title="Qwen Question Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Qwen Question Generator")

if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_health_check" not in st.session_state:
    st.session_state.last_health_check = 0
if "pending_job_id" not in st.session_state:
    st.session_state.pending_job_id = None
if "pending_poll_url" not in st.session_state:
    st.session_state.pending_poll_url = None
if "job_start_ts" not in st.session_state:
    st.session_state.job_start_ts = None

is_generating = st.session_state.pending_job_id is not None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Server Controls")

    _health_cooldown = 10
    _now = time.time()
    _since_last = _now - st.session_state.last_health_check
    _health_disabled = _since_last < _health_cooldown
    _health_label = (
        f"🔍 Check Server Health ({int(_health_cooldown - _since_last)}s)"
        if _since_last < _health_cooldown
        else "🔍 Check Server Health"
    )

    if st.button(_health_label, use_container_width=True,
                 disabled=_health_disabled or is_generating, key="health_btn"):
        st.session_state.last_health_check = time.time()
        try:
            r = requests.get(f"{_SERVER_BASE}/health", timeout=5)
            h = r.json()
            if h.get("status") == "healthy":
                st.success("Server is healthy ✅")
                col_a, col_b = st.columns(2)
                with col_a:
                    model_ok = h.get("model_loaded", False)
                    st.metric("Model", "Loaded ✅" if model_ok else "Not loaded ❌")
                with col_b:
                    mem = h.get("memory_gb", 0)
                    st.metric("GPU Memory", f"{round(mem, 2)} GB" if mem and mem > 0 else "N/A")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.metric("Active Jobs", h.get("active_jobs", 0))
                with col_d:
                    st.metric("Jobs in Store", h.get("total_jobs_in_store", 0))
                qs = h.get("queue_sizes", {})
                stuck = {k: v for k, v in qs.items() if v > 0}
                if stuck:
                    st.warning(f"⚠️ Stuck queue items: {stuck}")
                else:
                    st.caption("All queues empty ✅")
            else:
                st.warning("Server responded but may not be fully ready.")
        except Exception as e:
            st.error(f"Cannot reach server at `{_SERVER_BASE}`: {e}")

    st.divider()

    st.subheader("🗑️ Cache")
    use_cache = st.toggle("Use Cache", value=True,
                          help="Uncheck to bypass cache and always regenerate fresh questions.")

    if st.button("🧹 Clear Cache", use_container_width=True, type="secondary", disabled=is_generating):
        try:
            r = requests.post(f"{API_BASE}/clear-cache", timeout=10)
            st.success(f"✅ {r.json().get('message', 'Cache cleared')}")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

    st.divider()

    if st.button("📊 View Stats", use_container_width=True, key="stats_btn", disabled=is_generating):
        try:
            r = requests.get(f"{_SERVER_BASE}/stats", timeout=5)
            s = r.json()
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Total Requests", s.get("total_requests", 0))
                st.metric("Cache Hit Rate", f"{s.get('cache_hit_rate_percent', 0)}%")
            with col_s2:
                st.metric("Errors", s.get("errors", 0))
                st.metric("Avg Batch Size", s.get("avg_batch_size", 0))
            by_ep = s.get("requests_by_endpoint", {})
            if by_ep:
                st.caption("Requests by endpoint: " + ", ".join(f"{k}: {v}" for k, v in by_ep.items()))
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
# INPUTS
# ─────────────────────────────────────────────
needs_topic_fields  = endpoint in ("generate-mcq", "generate-subjective",
                                   "generate-coding", "generate-sql", "generate-aiml")
needs_topics_fields = endpoint == "generate-topics"
needs_language      = endpoint == "generate-coding"
needs_db_type       = endpoint == "generate-sql"
needs_audience      = endpoint in ("generate-mcq", "generate-subjective")

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
        job_designation  = st.text_input("Job Designation", "Software Developer")
        skills           = st.text_input("Skills (comma separated)", "Python, FastAPI, SQL")
    with col2b:
        experience_min = st.number_input("Min Experience (yrs)", 0, 30, 1)
        experience_max = st.number_input("Max Experience (yrs)", 0, 30, 3)

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
        job_role = st.text_input("Job Role", "Software Engineer")
    with col_r2:
        experience_years = st.selectbox(
            "Experience Level",
            ["0-1 (Fresher)", "1-3 (Junior)", "3-5 (Mid-level)", "5-8 (Senior)", "8+ (Staff/Principal)"],
            index=2,
        )

st.divider()

# ─────────────────────────────────────────────
# PAYLOAD BUILDER
# ─────────────────────────────────────────────
def build_payload() -> dict:
    base = {"num_questions": int(num_questions), "use_cache": use_cache}
    if endpoint == "generate-topics":
        base.update({
            "assessment_title": assessment_title,
            "job_designation":  job_designation,
            "skills":           [s.strip() for s in skills.split(",") if s.strip()],
            "experience_min":   int(experience_min),
            "experience_max":   int(experience_max),
            "num_topics":       int(num_questions),
        })
    elif endpoint == "generate-mcq":
        base.update({"topic": topic, "difficulty": difficulty, "target_audience": target_audience})
    elif endpoint == "generate-subjective":
        base.update({"topic": topic, "difficulty": difficulty, "target_audience": target_audience})
    elif endpoint == "generate-coding":
        base.update({"topic": topic, "difficulty": difficulty, "language": language,
                     "job_role": job_role, "experience_years": experience_years})
    elif endpoint == "generate-sql":
        base.update({"topic": topic, "difficulty": difficulty, "database_type": database_type,
                     "job_role": job_role, "experience_years": experience_years})
    elif endpoint == "generate-aiml":
        base.update({"topic": topic, "difficulty": difficulty})
    return base

# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def show_metadata(data: dict):
    cols = st.columns(3)
    with cols[0]:
        st.metric("⏱ Generation Time", f"{round(data.get('generation_time_seconds', 0), 2)}s")
    with cols[1]:
        st.metric("📦 Batch Size", data.get("batch_size", 1))
    with cols[2]:
        st.metric("🗃 Cache Hit", "Yes" if data.get("cache_hit") else "No")


def render_mcq_question(q: dict, index: int):
    st.markdown(f"#### Question {index}")
    st.write(q.get("question", "*(no question text)*"))
    for opt in q.get("options", []):
        label, text = opt.get("label", "?"), opt.get("text", "")
        if opt.get("isCorrect"):
            st.markdown(f"**{label}. ✅ {text}**")
        else:
            st.write(f"{label}. {text}")
    if q.get("explanation"):
        with st.expander("💡 Explanation"):
            st.info(q["explanation"])
    c1, c2 = st.columns(2)
    with c1: st.caption(f"Difficulty: `{q.get('difficulty', '—')}`")
    with c2: st.caption(f"Bloom Level: `{q.get('bloomLevel', '—')}`")


def render_subjective_question(q: dict, index: int):
    st.markdown(f"#### Question {index}")
    st.write(q.get("question", "*(no question text)*"))
    if q.get("expectedAnswer"):
        with st.expander("📝 Expected Answer"):
            st.success(q["expectedAnswer"])
    if q.get("gradingCriteria"):
        with st.expander("📏 Grading Criteria"):
            for c in q["gradingCriteria"]: st.write(f"• {c}")
    c1, c2 = st.columns(2)
    with c1: st.caption(f"Difficulty: `{q.get('difficulty', '—')}`")
    with c2: st.caption(f"Bloom Level: `{q.get('bloomLevel', '—')}`")


def render_coding_problem(p: dict, index: int):
    st.markdown(f"#### Problem {index}")
    st.markdown("**Problem Statement**")
    st.write(p.get("problemStatement", ""))
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Input Format**"); st.write(p.get("inputFormat", ""))
    with col_b:
        st.markdown("**Output Format**"); st.write(p.get("outputFormat", ""))
    if p.get("constraints"):
        with st.expander("📐 Constraints"):
            for c in p["constraints"]: st.write(f"• {c}")
    if p.get("examples"):
        with st.expander(f"📖 Examples ({len(p['examples'])})"):
            for ex in p["examples"]:
                st.write(f"**Input:** `{ex.get('input', '')}`")
                st.write(f"**Output:** `{ex.get('output', '')}`")
                if "explanation" in ex: st.caption(ex["explanation"])
                st.divider()
    if p.get("starterCode"):
        lang_map = {"Python": "python", "JavaScript": "javascript", "Java": "java",
                    "C++": "cpp", "Go": "go", "TypeScript": "typescript"}
        with st.expander("🖊 Starter Code"):
            st.code(p["starterCode"], language=lang_map.get(p.get("language", "Python"), "python"))
    if p.get("testCases"):
        with st.expander(f"🧪 Test Cases ({len(p['testCases'])})"):
            try: st.dataframe(pd.DataFrame(p["testCases"]), use_container_width=True)
            except: st.json(p["testCases"])
    if p.get("expectedComplexity"):
        st.caption(f"Expected Complexity: `{p['expectedComplexity']}`")
    if p.get("hints"):
        with st.expander("💡 Hints"):
            for idx, h in enumerate(p["hints"], 1): st.write(f"Hint {idx}: {h}")
    st.caption(f"Difficulty: `{p.get('difficulty', '—')}`")


def render_sql_problem(p: dict, index: int):
    st.markdown(f"#### SQL Problem {index}")
    st.write(p.get("problemStatement", ""))
    if p.get("schema"):
        with st.expander("🗂 Schema"):
            for table in p["schema"].get("tables", []):
                st.markdown(f"**Table: `{table.get('name', '?')}`**")
                try: st.dataframe(pd.DataFrame(table.get("columns", [])), use_container_width=True)
                except: st.json(table.get("columns", []))
    if p.get("expectedQuery"):
        with st.expander("✅ Expected Query"):
            st.code(p["expectedQuery"], language="sql")
    if p.get("explanation"):
        with st.expander("💡 Explanation"):
            st.info(p["explanation"])
    if p.get("alternativeApproach"):
        with st.expander("⚠️ Alternative Approach & Why It's Worse"):
            st.write(p["alternativeApproach"])
    if p.get("concepts_tested"):
        st.caption(f"Concepts tested: {', '.join(p['concepts_tested'])}")
    st.caption(f"Difficulty: `{p.get('difficulty', '—')}`")


def render_aiml_problem(p: dict, index: int):
    st.markdown(f"#### AI/ML Problem {index}")
    st.write(p.get("problemStatement", ""))
    if p.get("dataset"):
        dataset = p["dataset"]
        with st.expander("📊 Dataset Details"):
            st.write(f"**Description:** {dataset.get('description', '')}")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Target:** `{dataset.get('target', '—')}`")
                st.write(f"**Target Type:** `{dataset.get('target_type', '—')}`")
                st.write(f"**Size:** `{dataset.get('size', '—')}`")
            with c2:
                if dataset.get("class_distribution"):
                    st.write("**Class Distribution:**")
                    st.json(dataset["class_distribution"])
            if dataset.get("features"):
                st.write(f"**Features:** {', '.join(f'`{f}`' for f in dataset['features'])}")
            if dataset.get("feature_types"):
                st.write("**Feature Types:**")
                st.json(dataset["feature_types"])
            data_rows = dataset.get("data", [])
            if data_rows:
                st.write(f"**Sample Data ({len(data_rows)} rows):**")
                try:
                    df = pd.DataFrame(data_rows)
                    st.dataframe(df.head(20), use_container_width=True)
                    if len(data_rows) > 20:
                        st.caption(f"Showing first 20 of {len(data_rows)} rows.")
                except: st.json(data_rows[:5])
    if p.get("expectedApproach"):
        with st.expander("🧪 Expected Approach"): st.write(p["expectedApproach"])
    if p.get("evaluationCriteria"):
        with st.expander("📏 Evaluation Criteria"):
            for c in p["evaluationCriteria"]: st.write(f"• {c}")
    st.caption(f"Difficulty: `{p.get('difficulty', '—')}`")


def render_topic(t: dict, index: int):
    label = t.get("label", t) if isinstance(t, dict) else str(t)
    qtype = t.get("questionType", "") if isinstance(t, dict) else ""
    diff  = t.get("difficulty", "")   if isinstance(t, dict) else ""
    judge = t.get("canUseJudge0", None) if isinstance(t, dict) else None
    cols = st.columns([3, 2, 2, 1])
    with cols[0]: st.write(f"**{index}.** {label}")
    with cols[1]: st.caption(f"Type: `{qtype}`" if qtype else "")
    with cols[2]: st.caption(f"Difficulty: `{diff}`" if diff else "")
    with cols[3]:
        if judge is not None: st.caption("Judge0 ✅" if judge else "")


# ─────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────
if st.button("🚀 Generate", type="primary", use_container_width=True, disabled=is_generating):
    payload = build_payload()
    url = f"{API_BASE}/{endpoint}"

    with st.expander("📨 Request Payload", expanded=False):
        st.json(payload)

    try:
        resp = requests.post(url, json=payload, timeout=30)

        if resp.status_code != 200:
            st.error(f"❌ Server returned HTTP {resp.status_code}")
            try:
                err_body = resp.json()
                detail = err_body.get("detail")
                if isinstance(detail, list):
                    for item in detail:
                        field = " → ".join(str(x) for x in item.get("loc", []))
                        st.error(f"Validation error on `{field}`: {item.get('msg', '')}")
                elif detail:
                    st.error(str(detail))
                else:
                    st.json(err_body)
            except Exception:
                st.text(resp.text[:500])
            st.stop()

        data = resp.json()

        # Job-based response — store job_id and start polling
        if "job_id" in data:
            st.session_state.pending_job_id = data["job_id"]
            st.session_state.pending_poll_url = f"{API_BASE}/job/{data['job_id']}"
            st.session_state.job_start_ts = time.time()
            st.session_state.last_response = None
            st.rerun()
        else:
            # Direct response (no job queue)
            st.session_state.last_response = data
            st.session_state.pending_job_id = None
            st.rerun()

    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to server at `{_SERVER_BASE}`. Is it running?")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ─────────────────────────────────────────────
# JOB POLLING
# ─────────────────────────────────────────────
if st.session_state.pending_job_id:
    elapsed = int(time.time() - (st.session_state.job_start_ts or time.time()))

    _messages = [
        "🧠 Thinking deeply...",
        "✍️ Crafting questions...",
        "🔍 Validating output...",
        "⚙️ Almost there...",
        "📦 Packaging results...",
    ]
    _msg = _messages[(elapsed // 5) % len(_messages)]
    _progress = min(0.9, elapsed / 120)
    st.progress(_progress, text=f"{_msg}  •  {elapsed}s elapsed")

    try:
        poll_resp = requests.get(st.session_state.pending_poll_url, timeout=60)
        poll_data = poll_resp.json()

        if poll_data.get("status") == "complete":
            st.progress(1.0, text="✅ Done!")
            result = poll_data.get("result") or poll_data
            st.session_state.last_response = result
            st.session_state.pending_job_id = None
            st.session_state.pending_poll_url = None
            st.session_state.job_start_ts = None
            st.rerun()

        elif poll_data.get("status") in ("failed", "error"):
            st.error(f"❌ Generation failed: {poll_data.get('error', 'Unknown error')}")
            st.session_state.pending_job_id = None
            st.session_state.pending_poll_url = None
            st.session_state.job_start_ts = None

        else:
            time.sleep(3)
            st.rerun()

    except requests.exceptions.Timeout:
        time.sleep(3)
        st.rerun()

    except Exception as e:
        st.warning(f"Polling hiccup, retrying… ({e})")
        time.sleep(3)
        st.rerun()


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
data = st.session_state.get("last_response")

if data:
    st.divider()
    st.header("📄 Generated Output")

    if "topics" in data:
        topics_list = data["topics"]
        st.subheader(f"📋 Generated Topics ({len(topics_list)})")
        for i, t in enumerate(topics_list, 1):
            render_topic(t, i)

    elif "questions" in data:
        questions_list = data["questions"]
        st.subheader(f"✅ Questions ({len(questions_list)})")
        for i, q in enumerate(questions_list, 1):
            with st.container(border=True):
                if "options" in q:
                    render_mcq_question(q, i)
                else:
                    render_subjective_question(q, i)

    elif "coding_problems" in data:
        problems = data["coding_problems"]
        st.subheader(f"💻 Coding Problems ({len(problems)})")
        for i, p in enumerate(problems, 1):
            with st.container(border=True):
                render_coding_problem(p, i)

    elif "sql_problems" in data:
        problems = data["sql_problems"]
        st.subheader(f"🗄️ SQL Problems ({len(problems)})")
        for i, p in enumerate(problems, 1):
            with st.container(border=True):
                render_sql_problem(p, i)

    elif "aiml_problems" in data:
        problems = data["aiml_problems"]
        st.subheader(f"🤖 AI/ML Problems ({len(problems)})")
        for i, p in enumerate(problems, 1):
            with st.container(border=True):
                render_aiml_problem(p, i)

    elif "dataset" in data and "problemStatement" in data:
        st.subheader("🤖 AI/ML Problem")
        with st.container(border=True):
            render_aiml_problem(data, 1)

    elif "options" in data and "question" in data:
        st.subheader("✅ MCQ Question")
        with st.container(border=True):
            render_mcq_question(data, 1)

    elif "expectedAnswer" in data and "question" in data:
        st.subheader("✍️ Subjective Question")
        with st.container(border=True):
            render_subjective_question(data, 1)

    elif "schema" in data and "problemStatement" in data:
        st.subheader("🗄️ SQL Problem")
        with st.container(border=True):
            render_sql_problem(data, 1)

    elif "inputFormat" in data and "problemStatement" in data:
        st.subheader("💻 Coding Problem")
        with st.container(border=True):
            render_coding_problem(data, 1)

    else:
        st.warning("Unrecognised response shape — showing raw JSON.")
        st.json(data)

    st.divider()
    st.subheader("📊 Response Metadata")
    show_metadata(data)

    st.divider()
    import json as _json
    st.download_button(
        label="⬇️ Download as JSON",
        data=_json.dumps(data, indent=2),
        file_name="generated_output.json",
        mime="application/json",
    )