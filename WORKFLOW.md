# BinaryBridge RAG Workflow

## 1. Open the project folder

Run all commands from:

```powershell
cd c:\Users\TS6205_ASHOK\Projects\Training Projects\BinaryBridge-RAG-chatbot
```

## 2. Run the RAG pipeline

This tests document loading, chunking, retrieval, and answer generation:

```powershell
python RAG.py
```

What happens:

1. The script loads the markdown files from `Documents/`
2. It cleans and chunks the text
3. It builds a local retrieval index
4. It answers 3 sample questions

## 3. Run the evaluation script

This evaluates the pipeline on the golden dataset:

```powershell
python RAGAS_evaluation_script.py
```

What happens:

1. The script loads `golden_question_answer_pairs.csv`
2. It asks all questions through `ask_question()` from `RAG.py`
3. It scores the answers
4. It writes a report file like `evaluation_report_Student.md`

## 4. Output files

After running the evaluation, you should get:

- `evaluation_report_<your-name>.md`

## 5. Optional Gemini mode

The project works without Gemini.

If you want to enable Gemini generation, add this to `.env`:

```env
GEMINI_API_KEY=your_key_here
RAG_USE_GEMINI=1
```

Then run:

```powershell
python RAG.py
```

## 6. If advanced RAGAS packages are installed

If you install the optional packages from `requirements.txt`, the evaluation script can use the advanced RAGAS path. If those packages are not installed, the script automatically falls back to the built-in lightweight evaluation so it still runs.

## 7. Verified commands

These commands were tested successfully:

```powershell
python RAG.py
python RAGAS_evaluation_script.py
```
