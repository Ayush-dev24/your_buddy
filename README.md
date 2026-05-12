# Setup Instructions

## 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key

## 2. Install Dependencies
Run the following command in your terminal from the root folder:
```bash
pip install -r requirements.txt
```

## 3. Environment Setup
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

## 4. Run the Backend
From the root directory, start the FastAPI server:
```bash
uvicorn backend.main:app --reload --port 8000
```
This will start the backend at `http://localhost:8000`.

## 5. Run the Frontend
From the `frontend` folder, start a local static server:
```bash
cd frontend
python -m http.server 3000
```
Then visit `http://localhost:3000` in your browser.

> Make sure the backend runs on `http://localhost:8000` and the frontend runs on `http://localhost:3000`.

## Features
- **Upload PDF:** Drag and drop your PDF file to extract text and generate embeddings.
- **Q&A Mode:** Ask questions and get answers based on the PDF content.
- **Generate Quiz:** Ask the AI to generate multiple-choice questions from the text.
- **Simplify:** Get a simplified, ELI5 explanation of complex topics.
- **Agent Task:** Give complex, multi-step instructions (e.g., summarize then list keywords).
