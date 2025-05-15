**🚀 Data Sense AI Project Challenge 1 (May 9-15'2025)**

Welcome to the PolicyPilot: AI Agentic Application Challenge, where your task is to build an intelligent agent-based system that can answer complex internal company policy questions using real company documents and data.

This challenge tests your ability to:

Build multi-agent LLM orchestration

Implement RAG (Retrieval-Augmented Generation)

Do tool routing and document grounding

Handle traceability, follow-up reasoning, and structured outputs

*🧠 Problem Statement*

AlturaTech receives dozens of internal queries daily—from engineers, HR staff, and managers. These questions are buried across:

PDF documents (HR Handbook, Security Protocol, Sales Playbook)

A live HTML page (Engineering SOPs)


*You must build an AI Agentic System that can:*

Accept a user question

Route it to the correct document(s) via routing logic or classifier

Perform multi-hop retrieval if needed


Generate a final structured answer with:

Final answer

Source(s) used

Trace of document chunks or sections

Confidence score (optional)


*📁 Documents Provided*


PDF	Employee Lifecycle & Benefits Handbook	HR_Handbook.pdf

PDF	Confidential Security Standards	Security_Protocol.pdf

PDF	Enterprise Sales Playbook	Sales_Playbook.pdf

Web Page	Engineering SOPs (scrapable HTML)	https://datasense78.github.io/engineeringsop/


⚙️ Project Requirements


Agents you must build:

🧱 Router Agent – identifies relevant doc(s) from user query

📚 Retriever Agent – performs vector or keyword-based chunk retrieval

🧠 Reasoning Agent – handles follow-ups, conflicting answers, or combines multiple document sections

Compliance Agent – redacts risky content or adds policy-safe footnotes
