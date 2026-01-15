import uuid
import logging
from dotenv import load_dotenv
from textwrap import dedent
import os
import asyncio
from io import BytesIO

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not TELEGRAM_BOT_TOKEN or not MISTRAL_API_KEY:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or MISTRAL_API_KEY")

reader = PDFReader()


async def analyze_contract_pipeline(pdf_text: str, pdf_name: str):
    """
    Analyze a contract in a pipeline:
    1. Contract structure -> returns structured content
    2. Legal analysis -> uses structured content
    3. Negotiation analysis -> uses structured + legal content
    Finally merged by team manager.
    """

    model = MistralChat(id="mistral-large-latest", api_key=MISTRAL_API_KEY)

   
    structure_agent = Agent(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Contract Structure Agent",
        model=model,
        tools=[],  
        markdown=True,
        instructions=dedent(f"""
            Behave as a specialist in contract organization and document structuring.  
            Carefully review the entire contract to assess its overall structure and logical flow.  
            Identify missing, unclear, repetitive, or disordered sections and clauses.  
            Return your findings in structured bullet points, highlighting issues clearly.  
            If necessary, produce a clean, well-organized markdown outline suggesting an improved contract structure.  
            Focus strictly on clarity, readability, and logical sequencing without providing legal advice.
        """)
    )
    structured_result = await structure_agent.arun(pdf_text)

    legal_agent = Agent(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Legal Framework Agent",
        model=model,
        tools=[],
        markdown=True,
        instructions=dedent(f"""
           Act as a legal framework analyst with a focus on identifying potential risks, ambiguities, and key legal principles within the contract.  
            Carefully examine each section and clause to detect unclear, inconsistent, or risky language.  
            For every observation or identified risk, quote the exact clause, sentence, or paragraph from the contract that supports it.  
            Clearly indicate the section title, heading, or paragraph number for context.  
            Present your findings in a structured, factual, and concise manner, ensuring each point is directly supported by the contract text.  
            Avoid giving general legal advice; focus strictly on what is explicitly written in the contract

            Structured Content:
            {structured_result.content}
        """)
    )
    legal_result = await legal_agent.arun(structured_result.content)

    negotiation_agent = Agent(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Negotiation Agent",
        model=model,
        tools=[],
        markdown=True,
        instructions=dedent(f"""
            Act as a contract negotiation strategist focused on identifying negotiable clauses and potential imbalances between the parties.  
            Carefully review the contract to locate provisions that may be unfair, overly restrictive, or commonly subject to negotiation.  
            For every observation, quote the exact clause, sentence, or paragraph from the contract that supports your point.  
            Explain briefly why each quoted clause may be negotiable or require adjustment.  
            Propose a clear and concrete alternative wording or counter-proposal for each identified clause.  
            Present your findings in a structured, practical, and concise format, grounded strictly in the contract’s language.
                  
            Structured Content:
            {structured_result.content}

            Legal Analysis:
            {legal_result.content}
        """)
    )
    negotiation_result = await negotiation_agent.arun(
        structured_result.content + "\n" + legal_result.content
    )

    team_manager = Team(
        session_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        name="Team Manager",
        members=[],
        show_members_responses=True,
        instructions=dedent("""
            Merge structured outline, legal analysis, and negotiation points
            into a single clean report. Preserve quoted clauses and remove duplicates.
        """)
    )

    combined_text = "\n\n".join([
        structured_result.content,
        legal_result.content,
        negotiation_result.content
    ])
    final_result = await team_manager.arun(f"Merge the following outputs:\n{combined_text}")
    return final_result


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("TEXT RECEIVED: %s", update.message.text)
    await update.message.reply_text("Please upload a PDF contract for analysis.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("DOCUMENT RECEIVED")
    doc = update.message.document
    logging.info("Filename: %s", doc.file_name)

    if not doc.file_name.lower().endswith(".pdf"):
        await update.message.reply_text("Please upload a PDF file.")
        return

    await update.message.reply_text("Downloading and reading PDF...")

    pdf_file = await doc.get_file()
    pdf_bytes = await pdf_file.download_as_bytearray()
    pdf_stream = BytesIO(pdf_bytes)

    docs = reader.read(pdf_stream)
    all_content = "\n".join(d.content for d in docs)

    await update.message.reply_text("Running pipeline analysis...")

    try:
        result = await analyze_contract_pipeline(all_content, doc.file_name)

        for i in range(0, len(result.content), 4000):
            await update.message.reply_text(result.content[i:i+4000])
    except Exception as e:
        logging.exception("Error analyzing")
        await update.message.reply_text(f"Failed to analyze pdf: {e}")


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logging.info("🤖 Telegram Contract Analysis Bot started (pipeline)")
    app.run_polling()

if __name__ == "__main__":
    main()
