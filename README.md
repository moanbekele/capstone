# Bilingual AI-Enhanced Call Center Platform
**SWEN 2023: Software Engineering Capstone Project**  
*Department of Software Engineering, BITS College*

---

## 📌 Project Overview
Many Ethiopian small-and-medium enterprises (SMEs) receive high daily call volumes (14 to 100+ calls/day) handled by only 1 or 2 staff members. Field interviews and call analysis show that **only ~10% of callers intend to buy**, while ~90% ask routine questions about prices, locations, and working hours. This causes long wait times, missed sales, and overloaded staff.

This project delivers an **automated, bilingual (Amharic & English) call center platform** that answers customer calls, resolves routine inquiries using a business knowledge base, identifies genuine buyers, and connects them directly to staff.

---

## 🎯 Key System Capabilities
* **Automatic Call Answering:** Answers incoming calls immediately in Amharic and English without hold times or dropped calls.
* **Knowledge-Based Inquiries:** Accurately answers questions about products, prices, availability, and business policies.
* **Buyer Intent Classification:** Automatically separates callers into *Genuine Buyers* and *General Inquiries*.
* **Smart Call Routing & Callback Queue:** Transfers buyers to available sales staff in real-time, or logs their details to an automated callback list with a call summary if staff are busy.
* **Omnichannel Support:** Extends the same answering and classification intelligence to a website chat widget and WhatsApp.
* **Staff Web Dashboard:** Provides staff with a unified interface to view live calls, transcripts, summaries, and track leads through to conversion.

---

## 📸 Screenshots & Project Evidence
Project artifacts, prototype demonstrations, and Agile tracking reports are located in the [`screenshots/`](screenshots/) directory:
* **Voice Agent Prototype (`screenshots/pipecat-ui.jpg`):** Working voice agent demonstration showing real-time audio interaction, speech recognition, and event handling.
* **Jira Agile Tracking (`screenshots/jira-reports-full.png`):** Project management overview tracking 200 work items across sprint cycles.
* **Status & Workload Breakdown (`screenshots/jira-status-assignee.png`):** Distribution of tasks, testing, reviews, and team assignments.
* **Completion Trends (`screenshots/jira-completion-trends.png`):** Lead time and task resolution trends across sprint milestones.

---

## 📁 Repository Structure
```text
├── voicepipe/                 # Real-time voice agent and service integrations
├── screenshots/               # Working prototype screenshots & Jira reports
├── bot.py                     # Main voice agent application
├── pipecat_groq_bot.py        # WebRTC voice agent runner
├── requirements.txt           # Project dependencies
├── .env.example               # Environment configuration template
└── README.md                  # Project overview documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10+
* Working microphone and headphones
* Groq or OpenAI API Key

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/moanbekele/capstone.git
cd capstone

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and set your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Running the Application
```bash
# Run local voice agent
python bot.py

# Or run the WebRTC test agent
python pipecat_groq_bot.py
```
