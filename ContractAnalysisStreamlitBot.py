import uuid
import os
import asyncio
from textwrap import dedent
from agno.agent import Agent
from agno.team import Team
from agno.models.mistral import MistralChat
import streamlit as st
from agno.db.postgres import PostgresDb
from agno.knowledge.reader.pdf_reader import PDFReader

mistral_api_key = os.getenv("MISTRAL_API_KEY")
SESSION_ID = str(uuid.uuid4())
USER_ID = str(uuid.uuid4())


st.set_page_config(
    page_title="Contract Scanning multi-agent system",
    layout="centered",
)
st.title("Contract scanning multi-agent system")
st.write("Upload a contract and get structure, legal and negotiation insights.")

uploaded_contract =st.file_uploader(
    "Upload your contract", type=["pdf"]
)
all_content = ""

if uploaded_contract is not None:
    reader = PDFReader()

    docs = reader.read(uploaded_contract)
    all_content = "\n".join(doc.content for doc in docs)

    if not all_content.strip():
        st.error("There is no content in the contract!")
        st.stop()

    st.success("Contract loaded successfully")


def get_docs():
    chunk_size=2000
    chunks=[]
    for i in range(0,len(all_content),chunk_size):
        chunks.append({"content":all_content[i:i+chunk_size],
                       "meta_data":{"source":uploaded_contract.name}
                       })
    return chunks


contractdb=PostgresDb(db_url="postgresql://postgres:12345@localhost:5432/contractDb")

contract_agent=Agent(
    session_id=SESSION_ID,
    user_id=USER_ID,
    name="Contract structure agent",
    model=MistralChat(id="mistral-large-latest",api_key=mistral_api_key),
    tools=[get_docs],
    instructions=dedent(""" 
        Behave as a specialist in contract organization and document structuring.  
        Carefully review the entire contract to assess its overall structure and logical flow.  
        Identify missing, unclear, repetitive, or disordered sections and clauses.  
        Return your findings in structured bullet points, highlighting issues clearly.  
        If necessary, produce a clean, well-organized markdown outline suggesting an improved contract structure.  
        Focus strictly on clarity, readability, and logical sequencing without providing legal advice.
            """)
    ,
    db=contractdb,
    markdown=True,
)
legalFramework_agent=Agent(
    session_id=SESSION_ID,
    user_id=USER_ID,
    name="Legal framework agent",
    model=MistralChat(id="mistral-large-latest",api_key=mistral_api_key),
    tools=[get_docs],
    instructions=dedent("""
        Act as a legal framework analyst with a focus on identifying potential risks, ambiguities, and key legal principles within the contract.  
        Carefully examine each section and clause to detect unclear, inconsistent, or risky language.  
        For every observation or identified risk, quote the exact clause, sentence, or paragraph from the contract that supports it.  
        Clearly indicate the section title, heading, or paragraph number for context.  
        Present your findings in a structured, factual, and concise manner, ensuring each point is directly supported by the contract text.  
        Avoid giving general legal advice; focus strictly on what is explicitly written in the contract.

                            """),
    db=contractdb,
    markdown=True,
)
negotiating_agent=Agent(
    session_id=SESSION_ID,
    user_id=USER_ID,
    name="Negotiating agent",
    model=MistralChat(id="mistral-large-latest",api_key=mistral_api_key),
    tools=[get_docs],
    instructions=dedent("""
        Act as a contract negotiation strategist focused on identifying negotiable clauses and potential imbalances between the parties.  
        Carefully review the contract to locate provisions that may be unfair, overly restrictive, or commonly subject to negotiation.  
        For every observation, quote the exact clause, sentence, or paragraph from the contract that supports your point.  
        Explain briefly why each quoted clause may be negotiable or require adjustment.  
        Propose a clear and concrete alternative wording or counter-proposal for each identified clause.  
        Present your findings in a structured, practical, and concise format, grounded strictly in the contract’s language.
                        """),
    db=contractdb,
    markdown=True,
)


team_manager = Team(
    session_id=SESSION_ID,
    user_id=USER_ID,
    id="team_manager",
    name="Team manager",
    members=[contract_agent, legalFramework_agent, negotiating_agent],
    instructions=dedent("""
        Act as the primary consolidation and summarization agent responsible for merging outputs from all other agents.  
        Combine their findings into a single, coherent, and well-structured final report.  
        Preserve all quoted contract clauses exactly as they appear, without modification or omission.  
        Remove duplicated, overlapping, or conflicting points while keeping the most clear and relevant version.  
        Organize the report in a clean, logical structure that is easy to read and follow.  
        Ensure that every observation remains traceable to a specific quoted clause from the contract.

    """),
    show_members_responses=True,
)

async def analyze_contract():
    return await team_manager.arun(
        "Analyze this contract using all agents and produce a consolidated and structured report."
    )

if uploaded_contract is not None:
    if st.button("Analyze Contract"):
        with st.spinner("Running analysis..."):
            result = asyncio.run(analyze_contract())

        st.subheader("Final Report")
        st.markdown(result.content)
