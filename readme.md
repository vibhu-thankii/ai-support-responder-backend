# Responder AI - AI-Powered Customer Support Responder

<p align="center">
  <img src="https://i.imgur.com/gK98h83.png" alt="Responder AI Dashboard" width="100%">
</p>

<p align="center">
  A full-stack, AI-powered SaaS application designed to streamline customer support by generating intelligent draft responses to user queries.
  <br />
  <a href="https://ai-support-responder-frontend-fbk6ikkdh-vibhu-thankis-projects.vercel.app/"><strong>View Live Demo »</strong></a>
  <br />
  <br />
  <a href="https://github.com/vibhu-thankii/ai-support-responder-frontend">Frontend Repo</a>
  ·
  <a href="https://github.com/vibhu-thankii/ai-support-responder-backend">Backend Repo</a>
</p>

---

## About The Project

In today's fast-paced digital world, businesses often struggle with the sheer volume of customer inquiries. Manually responding to each query is time-consuming, costly, and prone to inconsistency.

**Responder AI** tackles this challenge head-on. It's a web application where a business can see incoming customer queries and, with a single click, generate a well-crafted, context-aware draft response. The AI uses a knowledge base to find the most relevant information, which a support agent can then review, edit, and send in a fraction of the time.

This project is a demonstration of building a modern, full-stack application that solves a tangible business problem, showcasing skills in both frontend and backend development, as well as deployment and product thinking.

### Key Features:

* **Interactive Dashboard:** A clean, modern, and fully responsive UI for managing customer queries.
* **AI-Powered Drafts:** Generates context-aware responses by searching a knowledge base for keywords.
* **Confidence Score:** Each AI draft comes with a confidence score, helping agents gauge the relevance of the suggestion.
* **Light & Dark Mode:** A professional, polished UI with theme toggling.
* **Full-Stack Architecture:** Decoupled frontend and backend for scalability and maintainability.

---

## Built With

This project leverages a modern and powerful tech stack:

* **Frontend:**
    * [Next.js](https://nextjs.org/) (React Framework)
    * [TypeScript](https://www.typescriptlang.org/)
    * [Tailwind CSS](https://tailwindcss.com/)
    * [Shadcn/ui](https://ui.shadcn.com/) (Component Library)
* **Backend:**
    * [Python 3](https://www.python.org/)
    * [FastAPI](https://fastapi.tiangolo.com/) (High-performance API Framework)
    * [Pydantic](https://pydantic.dev/) (Data Validation)
* **Deployment:**
    * Frontend hosted on [Vercel](https://vercel.com/)
    * Backend hosted on [Render](https://render.com/)

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have the following installed:
* Node.js (v18 or later)
* npm
* Python 3.10 or later

### Installation & Setup

#### 1. Backend (`ai-support-responder-backend`)

```bash
# Clone the backend repository
git clone [https://github.com/vibhu-thankii/ai-support-responder-backend.git](https://github.com/vibhu-thankii/ai-support-responder-backend.git)

# Navigate to the project directory
cd ai-support-responder-backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the backend server
uvicorn main:app --reload
