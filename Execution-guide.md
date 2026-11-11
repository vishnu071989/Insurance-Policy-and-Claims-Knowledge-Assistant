# Running the Insurance Knowledge Assistant

## Part A: AWS prerequisites (one-time)

1. **Get AWS credentials.** Install the AWS CLI and run `aws configure` (access key, secret, region such as `us-east-1`). The code reads credentials from this standard setup, so no keys go in the code.
2. **Enable Bedrock models.** In the AWS Console, open Bedrock → Model access in your region. Request access to **Titan Text Embeddings V2** and one **Claude** model. Embeddings turn text into vectors for search, and Claude writes the answers.
3. **Copy the Claude model ID.** In Bedrock → Model catalog, copy the model ID or inference-profile ID. You need it in step 7.
4. **Check IAM permission.** Your user or role needs `bedrock:InvokeModel`. Without it, calls fail with AccessDenied.

## Part B: Setup

5. **Unzip and enter the folder.**
   ```
   unzip insurance-assistant.zip && cd insurance-assistant
   ```
6. **Create a virtual environment and install dependencies** (Python 3.10+).
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
7. **Create `.env`.** Run `cp .env.example .env`, then edit it:
   ```
   AWS_REGION=us-east-1
   LLM_MODEL_ID=<model ID from step 3>
   ```
8. **Check the logic offline (optional).**
   ```
   python tests/test_offline.py
   ```
   This should print `OK`. It confirms the access control and evidence handling work, but it does not touch AWS.
9. **Build the index.**
   ```
   python ingest.py
   ```
   This reads `corpus/`, splits each document into chunks by section, and sends each chunk to Titan for embedding. It saves the result to `index/`. Expect a line like `Indexed N chunks from 8 documents`. If it fails here, the cause is almost always credentials or model access (steps 1–4). Re-run this command whenever you change the corpus.

## Part C: Run it

10. **Start the web app.**
    ```
    streamlit run app.py
    ```
    Open the URL it prints, normally http://localhost:8501.
    - The sidebar role selector is a simulated sign-in. Each role sees different documents.
    - A question is embedded, and the closest passages the role is allowed to see are retrieved. Claude then answers only from those passages and cites them as `[S1]`, `[S2]`.
    - The **Sources** expander shows the exact passages used.
11. **Run the six demo scenarios.** Use any role unless noted.
    1. Coverage: "Is a burst pipe covered under the homeowner policy?"
    2. Exclusions: "What does the homeowner policy exclude?"
    3. Claim documents: "Which documents are required to file a claim?"
    4. Claims process: "What is the claims processing procedure?"
    5. Follow-up: ask "Is sewer backup covered?", then "What is the limit?" The assistant rewrites the follow-up into a standalone question using the chat history.
    6. Insufficient evidence: "What is the policy on cyber insurance for small businesses?" It should refuse to answer rather than guess.

    **Extra checks:**
    - Conflict: ask "How long to report a loss?" The 14-day and 30-day sources should be flagged as conflicting.
    - Access control: ask "What are the red flags for SIU referral?" as `customer_support` (no answer), then as `claims_handler` (answered).

## Part D: MCP server

12. **Run the server.**
    ```
    MCP_USER_ROLE=claims_handler python mcp_server.py
    ```
    On Windows, set the variable first with `set MCP_USER_ROLE=claims_handler`. It runs over stdio, so a terminal alone will just wait silently. An MCP client launches it. The role is fixed by this environment variable and not by tool arguments, so a client cannot give itself more access.
13. **Connect a client.** In Claude Desktop (or another MCP client), add to its MCP config:
    ```json
    {"mcpServers":{"insurance-kb":{"command":"python","args":["/absolute/path/mcp_server.py"],"env":{"MCP_USER_ROLE":"claims_handler"}}}}
    ```
    If the client's Python lacks the packages, point `command` at `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows). The client then shows two tools: `search_documents` (raw passages) and `ask_question` (cited answer).

## Part E: Evaluation and submission

14. **Run the evaluation.**
    ```
    python eval/run_eval.py
    ```
    This asks the 14 test questions, each answerable one twice, and writes `eval/report.md`. It measures:
    - retrieval hit rate
    - answer accuracy (keyword check)
    - citation grounding
    - consistency between the two runs
    - restricted-document leaks and handling of unanswerable questions
    - error handling

    If too many answerable questions come back as "insufficient evidence", lower `MIN_SCORE` in `.env` (for example 0.25). If unrelated questions get answered, raise it, then re-run.
15. **Finish the deliverables.**
    - Paste the metrics from `eval/report.md` into `docs/REPORT.md`.
    - Paste the same metrics and the architecture diagram into `docs/PRESENTATION.md`. Render the Mermaid diagram at mermaid.live.
    - Record the demonstration from step 11.

## Common errors

| Error | Cause |
|---|---|
| `AccessDeniedException` | Model access not enabled, or IAM permission missing |
| `LLM_MODEL_ID is not set` | Step 7 not done |
| `Index not found` | Run `python ingest.py` first |
| `ValidationException` about model ID | Your region needs an inference-profile ID (for example one starting with `us.`) |