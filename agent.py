"""
NirmanAI - Agentic AI Procurement Assistant
===============================================
A conversational AI assistant for supply chain managers.
Integrates with NirmanAI's prediction models and optimization engine.
Falls back gracefully if no Gemini API key is present.
"""

import os
import json
import uuid
import re
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

# In-memory conversation store for hackathon purposes
CONVERSATIONS = {}

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
use_gemini = False
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        use_gemini = True
    except ImportError:
        pass

def get_or_create_conversation(conv_id: str = None) -> str:
    """Initialize or retrieve a conversation history."""
    if not conv_id or conv_id not in CONVERSATIONS:
        conv_id = f"conv-{uuid.uuid4().hex[:8]}"
        CONVERSATIONS[conv_id] = [
            {"role": "system", "content": "You are KAYA Jarvis, an elite AI procurement assistant for NirmanAI."}
        ]
    return conv_id

# Proper casing map (title() breaks abbreviations like OPC, TMT, HDPE, AAC)
MATERIAL_CASE_MAP = {
    "tmt steel": "TMT Steel", "opc cement": "OPC Cement", "river sand": "River Sand",
    "coarse aggregate": "Coarse Aggregate", "fly ash bricks": "Fly Ash Bricks",
    "aac blocks": "AAC Blocks", "structural steel": "Structural Steel",
    "electrical cable": "Electrical Cable", "hdpe pipes": "HDPE Pipes",
    "vitrified tiles": "Vitrified Tiles", "plywood": "Plywood", "paint": "Paint",
}
STATE_CASE_MAP = {
    "maharashtra": "Maharashtra", "tamil nadu": "Tamil Nadu", "karnataka": "Karnataka",
    "gujarat": "Gujarat", "rajasthan": "Rajasthan", "uttar pradesh": "Uttar Pradesh",
    "bihar": "Bihar", "west bengal": "West Bengal", "madhya pradesh": "Madhya Pradesh",
    "telangana": "Telangana", "kerala": "Kerala", "punjab": "Punjab",
    "odisha": "Odisha", "jharkhand": "Jharkhand", "haryana": "Haryana",
}

def _extract_material_and_state(message: str) -> tuple:
    """Basic NLP extraction for demo fallback if LLM is unavailable."""
    msg = message.lower()
    found_mat = next((MATERIAL_CASE_MAP[m] for m in MATERIAL_CASE_MAP if m in msg), "TMT Steel")
    found_state = next((STATE_CASE_MAP[s] for s in STATE_CASE_MAP if s in msg), "Bihar")
    return found_mat, found_state

def process_agent_message(message: str, conv_id: str = None, context: dict = None) -> Dict[str, Any]:
    """
    Process a user message through the agentic workflow.
    Attempts to use Gemini if API key is present, otherwise uses a smart fallback
    that still calls our actual backend tools (Supplier DB, Simulator, etc).
    """
    conv_id = get_or_create_conversation(conv_id)
    CONVERSATIONS[conv_id].append({"role": "user", "content": message})
    
    context = context or {}
    msg_lower = message.lower()
    
    llm_response_text = ""
    if use_gemini:
        try:
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction='''You are KAYA Jarvis, an authoritative, highly experienced supply chain logistics consultant for NirmanAI - an AI platform that predicts delivery delays, estimates material wastage, and optimizes procurement for Indian construction projects.

You have access to these tools (call them by describing what you need):
- Monte Carlo Simulation Engine: Simulates a project's entire supply chain 10,000 times to find failure modes
- Prescriptive Optimizer: Uses mathematical optimization to find the best supplier allocation strategy
- Supplier Database: Contains 65+ verified Indian construction material suppliers with reliability scores
- Weather Intelligence: Real-time monsoon and weather risk data for all Indian states

Rules:
- Always speak confidently and professionally, like a senior consultant billing Rs 5 lakhs/day
- Reference specific numbers, percentages, and data points - never be vague
- When recommending actions, be specific: name suppliers, give timelines, quantify savings
- Keep responses concise (3-5 paragraphs max) but information-dense
- Use markdown formatting for readability'''
            )
            response = model.generate_content(message)
            llm_response_text = response.text
        except Exception as e:
            llm_response_text = ""

    response_text = ""
    tool_used = None
    tool_data = None
    
    # Extract entities. Prefer context state unless explicitly mentioned in prompt.
    mat_msg, state_msg = _extract_material_and_state(message)
    state_keywords = ["maharashtra", "tamil nadu", "karnataka", "gujarat", "rajasthan", "uttar pradesh", "bihar", "west bengal", "madhya pradesh", "kerala"]
    mat_keywords = ["steel", "cement", "sand", "aggregate", "bricks", "paint", "plywood", "tiles", "pipes", "cable"]
    
    state = state_msg if any(s in msg_lower for s in state_keywords) else context.get("state", "Bihar")
    mat = mat_msg if any(m in msg_lower for m in mat_keywords) else "TMT Steel"
    project_type = context.get("project_type", "Residential Apartment")
    month = context.get("month", 7)
    
    # -- AGENTIC TOOL ROUTING (Fallback Logic) --
    
    if "optimize" in msg_lower or "strategy" in msg_lower or "mitigate" in msg_lower or "split" in msg_lower:
        # Route to Prescriptive Optimization
        try:
            from simulation_engine import optimize_procurement
            opt_res = optimize_procurement([{"material_type": mat, "quantity": 100}], state, month)
            strat = opt_res["procurement_strategy"][0]
            
            backend_text = (f"I've run the optimization engine for your **{mat}** order in **{state}**.\n\n"
                             f"**Optimal Procurement Strategy:**\n"
                             f"- Primary Supplier: {strat['primary_supplier']} ({strat['primary_allocation_pct']}% of order)\n"
                             f"- Backup Supplier: {strat['backup_supplier']} ({strat['backup_allocation_pct']}% of order)\n\n"
                             f"**Impact:** This split reduces your delay risk by {strat['estimated_delay_reduction_pct']}%.\n"
                             f"{opt_res['recommendation']}")
            tool_used = "Prescriptive_Optimizer"
            tool_data = opt_res
            response_text = llm_response_text + "\n\n---\n\n**Backend Data:**\n" + backend_text if llm_response_text else backend_text
        except Exception as e:
            response_text = f"I tried to run the optimization solver, but encountered an error: {str(e)}"
            if llm_response_text:
                response_text = llm_response_text + "\n\n---\n\n**Backend Status:**\n" + response_text
            
    elif "simulate" in msg_lower or "monte carlo" in msg_lower or "project" in msg_lower:
        # Route to Simulation Engine
        try:
            from simulation_engine import run_simulation
            sim_res = run_simulation(project_type, state, month, 1000)
            tl = sim_res["project_timeline"]
            summary = sim_res["executive_summary"]
            
            backend_text = (f"I've run the Monte Carlo simulation for your **{project_type}** in **{state}**.\n\n"
                             f"**Simulation Results (1,000 runs)**\n"
                             f"- On-time probability: {tl['on_time_probability_pct']}%\n"
                             f"- Most likely duration: {tl['most_likely_days']} days\n"
                             f"- Risk Level: {summary['risk_level']}\n\n"
                             f"**Agent Recommendation:**\n"
                             f"{summary['headline']} "
                             f"I highly recommend securing backup suppliers for your critical path materials.")
            tool_used = "Monte_Carlo_Simulator"
            tool_data = sim_res
            response_text = llm_response_text + "\n\n---\n\n**Backend Data:**\n" + backend_text if llm_response_text else backend_text
        except Exception as e:
            response_text = f"I tried to run the simulation engine, but encountered an error: {str(e)}"
            if llm_response_text:
                response_text = llm_response_text + "\n\n---\n\n**Backend Status:**\n" + response_text
            
    elif "supplier" in msg_lower or "alternate" in msg_lower or "buy" in msg_lower:
        # Route to Supplier DB
        try:
            from suppliers_db import find_suppliers
            suppliers = find_suppliers(mat, state)
            if not suppliers:
                suppliers = find_suppliers(mat) # Fallback to any state
                
            if suppliers:
                top_sup = suppliers[0]
                backend_text = (f"I found {len(suppliers)} suppliers for **{mat}**.\n\n"
                                 f"**Top Recommendation (Optimized for {state}):**\n"
                                 f"**{top_sup['name']}** ({top_sup['tier']})\n"
                                 f"- Reliability Score: {top_sup['reliability_score']:.2f}\n"
                                 f"- Lead Time: {top_sup['avg_lead_days']} days\n"
                                 f"- Location: {top_sup['state']}\n\n"
                                 f"Would you like me to draft an RFQ (Request for Quotation) to them?")
                tool_used = "Supplier_Database"
                tool_data = {"suppliers_found": len(suppliers), "top_match": top_sup}
                response_text = llm_response_text + "\n\n---\n\n**Backend Data:**\n" + backend_text if llm_response_text else backend_text
            else:
                backend_text = f"I couldn't find any suppliers for {mat} in our database. Try a different material or check the Procurement Planner tab."
                tool_used = "Supplier_Database"
                tool_data = {"suppliers_found": 0, "material": mat}
                response_text = llm_response_text + "\n\n---\n\n**Backend Data:**\n" + backend_text if llm_response_text else backend_text
        except Exception as e:
            response_text = f"I tried to query the supplier database, but encountered an error: {str(e)}"
            if llm_response_text:
                response_text = llm_response_text + "\n\n---\n\n**Backend Status:**\n" + response_text

    elif "report" in msg_lower or "pdf" in msg_lower:
        # Route to Report Generator
        try:
            from report_generator import generate_pdf_report
            rid = f"RPT-{uuid.uuid4().hex[:8].upper()}"
            report_path = generate_pdf_report(
                report_id=rid,
                project_name=context.get("project_name", "NirmanAI Project"),
                project_type=project_type,
                state=state,
                current_month=month,
                output_dir="reports/generated",
            )
            backend_text = f"I have generated the risk report **{rid}**. You can download it from the **Export PDF Report** tab, or find it at: `{report_path}`"
            tool_used = "Report_Generator"
            tool_data = {"report_path": report_path, "report_id": rid}
            response_text = llm_response_text + "\n\n---\n\n**Backend Data:**\n" + backend_text if llm_response_text else backend_text
        except Exception as e:
            response_text = f"I tried to generate the report, but encountered an error: {str(e)}"
            if llm_response_text:
                response_text = llm_response_text + "\n\n---\n\n**Backend Status:**\n" + response_text

    else:
        # General chat
        backend_text = ("I am KAYA Jarvis, your AI Procurement Assistant. I can help you with:\n"
                         "1. **Simulating project timelines** (type 'simulate my project')\n"
                         "2. **Finding alternate suppliers** (type 'find cement suppliers')\n"
                         "3. **Optimizing procurement** (type 'generate an optimized strategy')\n"
                         "4. **Generating reports** (type 'generate a report')\n\n"
                         "How can I assist you today?")
        if llm_response_text:
            response_text = llm_response_text
        else:
            response_text = backend_text
                         
    CONVERSATIONS[conv_id].append({"role": "assistant", "content": response_text})
    
    return {
        "conversation_id": conv_id,
        "message": response_text,
        "tool_used": tool_used,
        "tool_data": tool_data,
        "timestamp": uuid.uuid4().hex[:8]
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  NirmanAI Agent Test")
    print("="*50)
    
    print("\n[User]: simulate my project risk")
    res = process_agent_message("simulate my project risk")
    print(f"\n[Agent, via {res['tool_used']}]:\n{res['message']}")
    
    print("\n" + "-"*50)
    print("\n[User]: find me alternate suppliers for OPC cement in bihar")
    res = process_agent_message("find me alternate suppliers for OPC cement in bihar")
    print(f"\n[Agent, via {res['tool_used']}]:\n{res['message']}")
