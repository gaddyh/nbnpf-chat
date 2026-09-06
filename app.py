from dotenv import load_dotenv
import streamlit as st

from src.repository import DrugRepository
from src.query_parser import QueryParser
from src.answer_generator import AnswerGenerator


load_dotenv()

st.set_page_config(
    page_title="NbN P&F Chat",
    page_icon="🧠",
)

repo = DrugRepository("data/nbnpf_drugs.json")
parser = QueryParser()
generator = AnswerGenerator()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_drug" not in st.session_state:
    st.session_state.current_drug = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "last_drug" not in st.session_state:
    st.session_state.last_drug = None

if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = None


st.title("NbN Patients & Families")

with st.sidebar:
    st.subheader("Grounding")
    if st.session_state.last_drug:
        drug = st.session_state.last_drug
        st.write(f"**Detected drug:** {drug['name']}")
        if drug.get("brand_names"):
            st.write(f"**Brand names:** {', '.join(drug['brand_names'])}")
        query = st.session_state.last_query
        st.write(f"**Intent:** {query.intent or '—'}")
        evidence = st.session_state.last_evidence
        if evidence:
            st.write(f"**Source:** {evidence.source}")
            st.write(f"**Drug ID:** {evidence.drug_id}")
            st.write(f"**Field:** {evidence.field}")
        else:
            st.write("**Source:** —")
            st.write("**Drug ID:** —")
    else:
        st.caption("Ask a question to see the detected drug and intent.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("evidence"):
            ev = message["evidence"]
            with st.expander("View source text"):
                if ev.text:
                    st.caption(
                        f"Drug: {ev.drug_name}  |  Field: {ev.field}  |  "
                        f"Source: {ev.source}  |  Drug ID: {ev.drug_id}"
                    )
                    st.text(ev.text)
                else:
                    st.caption(
                        f"Drug: {ev.drug_name}  |  Field: {ev.field}  |  "
                        f"Source: {ev.source}  |  Drug ID: {ev.drug_id}"
                    )
                    st.text("(no information available for this topic)")


prompt = st.chat_input(
    "Ask a question about your medication..."
)

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    query = parser.parse(
        prompt,
        current_drug=st.session_state.current_drug,
    )

    drug = repo.resolve_drug(query.drug_name)

    if drug:
        st.session_state.current_drug = drug["name"]

    evidence = repo.get_evidence(
        drug=drug,
        intent=query.intent,
    )

    answer = generator.generate(
        question=prompt,
        drug=drug,
        evidence=evidence,
    )

    st.session_state.last_query = query
    st.session_state.last_drug = drug
    st.session_state.last_evidence = evidence

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "evidence": evidence,
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.rerun()
