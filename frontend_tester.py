import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Qwen Question Generator", layout="wide")

st.title("Qwen Question Generator Tester")

# ----------------------------
# Endpoint Selector
# ----------------------------

endpoint = st.selectbox(
    "Choose Endpoint",
    [
        "generate-topics",
        "generate-mcq",
        "generate-subjective",
        "generate-coding",
        "generate-sql",
        "generate-aiml"
    ]
)

st.divider()

# ----------------------------
# Inputs
# ----------------------------

st.subheader("Assessment Details")

col1, col2 = st.columns(2)

with col1:
    assessment_title = st.text_input("Assessment Title", "Python Assessment")
    job_designation = st.text_input("Job Designation", "Software Developer")
    skills = st.text_input("Skills (comma separated)", "Python")

with col2:
    experience_min = st.number_input("Minimum Experience", 0, 20, 1)
    experience_max = st.number_input("Maximum Experience", 0, 20, 3)
    target_audience = st.text_input("Target Audience", "Developers")

topic = st.text_input("Topic", "Python")

difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

num_questions = st.number_input("Number of Questions", 1, 20, 3)

st.divider()

# ----------------------------
# Generate Button
# ----------------------------

if st.button("Generate"):

    payload = {
        "assessment_title": assessment_title,
        "job_designation": job_designation,
        "skills": [s.strip() for s in skills.split(",")],
        "experience_min": experience_min,
        "experience_max": experience_max,
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "target_audience": target_audience
    }

    url = f"{API_BASE}/{endpoint}"

    st.write("### Request Payload")
    st.json(payload)

    try:

        with st.spinner("Generating..."):

            response = requests.post(url, json=payload, timeout=120)

        st.success(f"Response Status: {response.status_code}")

        data = response.json()

        st.divider()
        st.header("Generated Output")

        # --------------------------------
        # MCQ (multiple)
        # --------------------------------

        if "questions" in data:

            for i, q in enumerate(data["questions"], 1):

                st.subheader(f"Question {i}")
                st.write(q.get("question", ""))

                if "options" in q:
                    for opt in q["options"]:
                        st.write(f"{opt['label']}. {opt['text']}")

                if any(opt.get("isCorrect") for opt in q.get("options", [])):
                    correct = [o["label"] for o in q["options"] if o["isCorrect"]]
                    st.success(f"Correct Answer: {', '.join(correct)}")

                if "explanation" in q:
                    st.info(q["explanation"])

                st.divider()

        # --------------------------------
        # MCQ (single object)
        # --------------------------------

        elif "question" in data and "options" in data:

            st.subheader("Question")

            st.write(data["question"])

            for opt in data["options"]:
                st.write(f"{opt['label']}. {opt['text']}")

            correct = [o["label"] for o in data["options"] if o["isCorrect"]]

            if correct:
                st.success(f"Correct Answer: {', '.join(correct)}")

            if "explanation" in data:
                st.info(data["explanation"])

        # --------------------------------
        # Topics
        # --------------------------------

        elif "topics" in data:

            st.subheader("Generated Topics")

            for i, topic in enumerate(data["topics"], 1):
                st.write(f"{i}. {topic}")

        # --------------------------------
        # Subjective
        # --------------------------------

        elif "expectedAnswer" in data:

            st.subheader("Subjective Question")

            st.write(data.get("question", ""))

            st.markdown("### Expected Answer")

            st.info(data["expectedAnswer"])

        # --------------------------------
        # Coding Problem
        # --------------------------------

        elif "problemStatement" in data:

            st.subheader("Coding Problem")

            st.markdown("### Problem Statement")
            st.write(data["problemStatement"])

            st.markdown("### Input Format")
            st.write(data.get("inputFormat", ""))

            st.markdown("### Output Format")
            st.write(data.get("outputFormat", ""))

            if "constraints" in data:

                st.markdown("### Constraints")

                for c in data["constraints"]:
                    st.write(f"• {c}")

            if "examples" in data:

                st.markdown("### Examples")

                for ex in data["examples"]:

                    st.write("Input:", ex.get("input", ""))
                    st.write("Output:", ex.get("output", ""))

                    if "explanation" in ex:
                        st.info(ex["explanation"])

                    st.divider()

            if "starterCode" in data:

                st.markdown("### Starter Code")
                st.code(data["starterCode"], language="python")

            if "testCases" in data:

                st.markdown("### Test Cases")
                st.dataframe(data["testCases"])

        # --------------------------------
        # SQL
        # --------------------------------

        elif "schema" in data and "query" in data:

            st.subheader("SQL Question")

            st.markdown("### Schema")

            st.code(data["schema"], language="sql")

            st.markdown("### Query")

            st.code(data["query"], language="sql")

        # --------------------------------
        # AIML Dataset
        # --------------------------------

        elif "dataset" in data:

            st.subheader("Generated Dataset")

            st.dataframe(data["dataset"])

        # --------------------------------
        # Metadata
        # --------------------------------

        if "generation_time_seconds" in data:

            st.caption(
                f"Generated in {round(data['generation_time_seconds'],2)} seconds"
            )

        # --------------------------------
        # Fallback
        # --------------------------------

        else:

            st.subheader("Raw Response")
            st.json(data)

    except Exception as e:

        st.error(f"Error: {str(e)}")
