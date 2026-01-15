import uuid
import logging
from dotenv import load_dotenv
from textwrap import dedent
import os

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    ContextTypes,
    filters,
)

from agno.agent import Agent
from agno.team import Team
from agno.models.mistral import MistralChat
from agno.knowledge.reader.pdf_reader import PDFReader


load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not TELEGRAM_BOT_TOKEN or not MISTRAL_API_KEY:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or MISTRAL_API_KEY")

reader = PDFReader()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------
# Agents
# --------------------------------------------------

def create_agents(pdf_content: str, pdf_name: str) -> Team:
    def get_docs():
        chunk_size = 2000
        return [
            {
                "content": pdf_content[i:i + chunk_size],
                "meta_data": {"source": pdf_name},
            }
            for i in range(0, len(pdf_content), chunk_size)
        ]

    model = MistralChat(
        id="mistral-large-latest",
        api_key=MISTRAL_API_KEY,
    )

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

        """),
    )

async def analyze_contract(pdf_text: str, filename: str):
    team = create_agents(pdf_text, filename)
    return await team.arun(
        "Analyze this contract and return a consolidated report."
    )

# --------------------------------------------------
# Telegram Handlers
# --------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("TEXT RECEIVED: %s", update.message.text)
    await update.message.reply_text(
        "📄 Please upload a PDF contract for analysis."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("DOCUMENT RECEIVED")

    doc = update.message.document
    logging.info("Filename: %s", doc.file_name)

    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("❌ Please upload a PDF file.")
        return

    await update.message.reply_text("⏳ Analyzing your contract...")


# --------------------------------------------------
# App Startup
# --------------------------------------------------

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logging.info("🤖 Telegram Contract Analysis Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()