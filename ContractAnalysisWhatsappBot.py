import os
import uuid
import logging
from io import BytesIO
from textwrap import dedent
from dotenv import load_dotenv
import asyncio

from fastapi import FastAPI, Request
from pydantic import BaseModel

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.db.postgres import PostgresDb


load_dotenv()
logging.basicConfig(level=logging.INFO)

WHATSAPP_VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not WHATSAPP_VERIFY_TOKEN or not MISTRAL_API_KEY:
    raise RuntimeError("Missing WHATSAPP_VERIFY_TOKEN or MISTRAL_API_KEY")

reader = PDFReader()
contract_db = PostgresDb(db_url="postgresql://postgres:12345@localhost:5432/contractDb")

app = FastAPI()


class WhatsAppMessage(BaseModel):
    from_: str
    body: str
    file_bytes: bytes = None
    file_name: str = None


def create_agents(pdf_content: str, pdf_name: str) -> Team:
    """Create the 3 agents + team manager for contract analysis"""
    def get_docs():
        chunk_size = 2000
        return [
            {"content": pdf_content[i:i + chunk_size], "meta_data": {"source": pdf_name}}
            for i in range(0, len(pdf_content), chunk_size)
        ]

    model = OpenAIChat(id="gpt-4", api_key=MISTRAL_API_KEY)

    contract_agent = Agent(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Contract structure agent",
        model=model,
        tools=[get_docs],
        markdown=True,
        instructions=dedent("""
            Behave as a specialist in contract organization and document structuring.  
            Carefully review the entire contract to assess its overall structure and logical flow.  
            Identify missing, unclear, repetitive, or disordered sections and clauses.  
            Return your findings in structured bullet points, highlighting issues clearly.  
            If necessary, produce a clean, well-organized markdown outline suggesting an improved contract structure.  
            Focus strictly on clarity, readability, and logical sequencing without providing legal advice.
            """),
        db=contract_db
    )

    legal_agent = Agent(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Legal framework agent",
        model=model,
        tools=[get_docs],
        markdown=True,
        instructions=dedent("""
            Act as a legal framework analyst with a focus on identifying potential risks, ambiguities, and key legal principles within the contract.  
            Carefully examine each section and clause to detect unclear, inconsistent, or risky language.  
            For every observation or identified risk, quote the exact clause, sentence, or paragraph from the contract that supports it.  
            Clearly indicate the section title, heading, or paragraph number for context.  
            Present your findings in a structured, factual, and concise manner, ensuring each point is directly supported by the contract text.  
            Avoid giving general legal advice; focus strictly on what is explicitly written in the contract.

            """),
        db=contract_db
    )

    negotiation_agent = Agent(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Negotiation agent",
        model=model,
        tools=[get_docs],
        markdown=True,
        instructions=dedent("""
            Act as a contract negotiation strategist focused on identifying negotiable clauses and potential imbalances between the parties.  
            Carefully review the contract to locate provisions that may be unfair, overly restrictive, or commonly subject to negotiation.  
            For every observation, quote the exact clause, sentence, or paragraph from the contract that supports your point.  
            Explain briefly why each quoted clause may be negotiable or require adjustment.  
            Propose a clear and concrete alternative wording or counter-proposal for each identified clause.  
            Present your findings in a structured, practical, and concise format, grounded strictly in the contract’s language.
                    
        """),
        db=contract_db
    )

    return Team(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Team Manager",
        members=[contract_agent, legal_agent, negotiation_agent],
        show_members_responses=True,
        instructions=dedent("""
            Act as the primary consolidation and summarization agent responsible for merging outputs from all other agents.  
            Combine their findings into a single, coherent, and well-structured final report.  
            Preserve all quoted contract clauses exactly as they appear, without modification or omission.  
            Remove duplicated, overlapping, or conflicting points while keeping the most clear and relevant version.  
            Organize the report in a clean, logical structure that is easy to read and follow.  
            Ensure that every observation remains traceable to a specific quoted clause from the contract.

        """)
    )


@app.get("/webhook")
async def verify_webhook(mode: str, verify_token: str, challenge: str):
    """Verification endpoint for WhatsApp"""
    if mode == "subscribe" and verify_token == WHATSAPP_VERIFY_TOKEN:
        return int(challenge)
    return {"error": "Verification failed"}

@app.post("/webhook")
async def receive_message(msg: Request):
    """Receive incoming WhatsApp messages"""
    payload = await msg.json()
    logging.info("Incoming WhatsApp payload: %s", payload)
    body = payload.get("body", "")
    file_name = payload.get("file_name")
    file_bytes = payload.get("file_bytes")

    if file_bytes and file_name and file_name.lower().endswith(".pdf"):
        try:
            pdf_stream = BytesIO(bytes(file_bytes))
            docs = reader.read(pdf_stream)
            all_content = "\n".join(d.content for d in docs)

            if not all_content.strip():
                return {"reply": "Could not extract text from PDF!"}

            team = create_agents(all_content, file_name)
            result = await team.arun("Analyze this contract and produce a consolidated report.")

           
            response_chunks = [result.content[i:i + 4000] for i in range(0, len(result.content), 4000)]
            return {"reply": response_chunks}

        except Exception as e:
            logging.exception("Error analyzing pdf")
            return {"reply": f"Failed to analyze contract: {e}"}

    else:
        return {"reply": "Please send a PDF contract for analysis."}
