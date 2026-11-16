# Bilingual AI-Enhanced Call Center Platform
**SWEN 2023: Software Engineering Capstone Project**  
*Department of Software Engineering, BITS College*

---

## 👥 Project Team & Supervision
* **Students:** 
  * Moan Bekele (UGR/SWE/0064/22)
  * Betemariam Abenet
  * Dawit Teshome
  * Amanuel Behailu
* **Project Advisors:** Meskerem Abebe & Dr. Tesfaye Biru
* **Academic Year:** 2026 / 2027

---

## 📌 Project Overview
Many Ethiopian small-and-medium enterprises (SMEs) face high incoming call volumes (14 to 100+ calls daily) handled by only 1 or 2 staff members. Data analysis and partner interviews reveal that **only ~10% of callers intend to buy**, while ~90% ask routine questions about prices, locations, and working hours. This leads to congested phone lines, lost sales, and staff burnout.

This project delivers an **automated, bilingual (Amharic & English) AI-powered call center platform** that:
1. **Answers 100% of incoming calls** automatically without wait times.
2. **Handles routine inquiries** in Amharic and English using a Retrieval-Augmented Generation (RAG) knowledge base.
3. **Classifies buyer intent** in real-time (*Genuine Buyer* vs. *General Inquiry*).
4. **Routes genuine buyers** directly to available sales staff via live call transfer or queues them to a callback list with automated call transcripts and summaries.
5. **Provides a web dashboard** for staff to monitor calls, view analytics, and follow up on leads.

---

## 🏗️ System Architecture & Voice Pipeline (`voicepipe`)

At the core of the voice agent is **`voicepipe`**, an event-driven, low-latency streaming pipeline engineered for real-time natural phone conversations and instant human interruption.

```
Caller Audio ─► WebRTC/SIP Transport ─► VAD ─► Speech-to-Text (ASR)
                                                       │
                                                       ▼
Audio Output ◄─ Text-to-Speech (TTS) ◄─ RAG & LLM (Groq/OpenAI) + Buyer Classifier
```

### Key Engineering Features:
* **Natural Interruption Handling:** When a caller speaks while the bot is talking, the Voice Activity Detector (VAD) instantly emits a `StartInterruptionFrame`, cancelling pending LLM tokens and flushing queued audio playback immediately.
* **Low Latency Sentence Streaming:** Responses are streamed token-by-token from the LLM, synthesised into speech per sentence chunk, reducing time-to-first-audio to under 2 seconds.
* **Pluggable AI Services:** Decoupled architecture supporting both **Groq** (ultra-fast Whisper + LLM inference) and **OpenAI** (GPT-4o / Whisper / TTS) providers.
* **Bilingual Support:** Designed for both Amharic and English voice and text interactions.

---

## 📁 Repository Structure

```text
├── voicepipe/                 # Core real-time streaming framework
│   ├── services/              # AI Service wrappers (Groq, OpenAI, STT, LLM, TTS)
│   │   ├── base.py            # Base abstract classes for pipeline services
│   │   ├── groq_service.py    # Ultra-low latency Groq STT/LLM implementation
│   │   └── openai_service.py  # OpenAI GPT-4o & Whisper service implementation
│   ├── transports/            # Audio transport layer (Microphone/Speaker, WebRTC, SIP)
│   │   ├── base.py            # Base transport interfaces and interruption logic
│   │   └── local.py           # Local microphone & speaker transport (PyAudio/SoundDevice)
│   ├── audio.py               # Audio conversion and resampling utilities
│   ├── frames.py              # Pipeline frame definitions (Audio, Text, Control, System)
│   ├── llm_context.py         # Conversation state, memory, and turn aggregation
│   ├── pipeline.py            # Pipeline runner, tasks, and asynchronous queue management
│   ├── processor.py           # Base FrameProcessor building block
│   └── vad.py                 # Silero/Energy Voice Activity Detection
├── screenshots/               # Project evidence, Jira dashboards & prototype UI
│   ├── pipecat-ui.jpg         # Live WebRTC voice bot running in playground
│   ├── jira-reports-full.png  # Jira sprint analytics & 200 work items overview
│   ├── jira-status-assignee.png # Jira status breakdown & team distribution
│   └── jira-completion-trends.png # Jira lead time and completion trends
├── bot.py                     # Main voice agent runner (Local audio / microphone)
├── pipecat_groq_bot.py        # WebRTC / RTVI playground agent runner
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* Working microphone and headphones (headphones prevent microphone audio loopback)
* Groq API Key ([console.groq.com](https://console.groq.com/keys)) or OpenAI API Key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/moanbekele/capstone.git
cd capstone

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy the template environment file and set your API keys:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Running the Agent
Run the standalone local voice bot:
```bash
python bot.py
```
Or run the WebRTC playground agent:
```bash
python pipecat_groq_bot.py
```

---

## 📊 Agile Project Management (Jira Tracking)
Development follows **Agile (Scrum)** in two-week sprints:
* **Total Tracked Work Items:** 200 items across Capstone I and Capstone II
* **Work Item Distribution:** Tasks (83%), Assets & Epics (17%)
* **Artifacts & Metrics:** Tracked via Jira under project `LUCY-AI` (see [screenshots/](screenshots/)).

---

## 📄 References & Standards
* **Requirements Specification:** IEEE Std 830-1998 & ISO/IEC/IEEE 29148:2018
* **Quality Model:** ISO/IEC 25010:2023 Product Quality Characteristics
* **System Design Description:** IEEE Std 1016-2009 & UML 2.5.1
* **Retrieval-Augmented Generation:** Lewis et al., NeurIPS 2020
