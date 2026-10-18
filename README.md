# IT policy questions

The assistant searches the single course policy document and uses Qwen3-0.6B to select an excerpt that answers the question. A second model call reviews the selection against the retrieved passages. The phone-number question is a test of missing evidence, not an invitation to guess contact details.

## Setup

Use Python 3.12 and Ollama. Install requirements.txt, then download qwen3:0.6b and nomic-embed-text with Ollama. Run Ollama before the commands below.

```powershell
python rag.py
python agent.py "What exact phone number should I call for the University IT Service Desk?" --record runs/phone.json
```

rag.py builds a Chroma index. It uses the original policy_loader.py without changing its code or the source document. The assignment PDF calls the loader load_policy.py; policy_loader.py is the filename in the supplied repository.

The index records the source hash and embedding-model digest. Interrupted builds can resume. Queries reject incomplete indexes and altered source text instead of quietly answering from stale documents.

## Answer review

rag_tool.py exposes a LangChain retrieval tool. It returns two matching policy sections. The agent asks Qwen to extract an answer-bearing excerpt and its policy ID. The excerpt becomes the answer without another free-form rewrite. Qwen then reviews whether the requested fact is present and whether the proposed answer agrees with the passages. A rejected draft receives one revision and a second review.

Local checks verify that the citation was retrieved, the quote occurs in that passage, and the displayed answer matches the quote. These checks do not establish that an authentic quote answers the question; the model review handles that judgement and can still make mistakes.

When the model returns empty evidence and the reviewer confirms that the requested fact is absent, the program displays only I don't know. and clears any citation. This is a generic empty-evidence response, not a phone-number keyword rule or an answer lookup. Service failures and invalid model output raise an error.

Records contain the actual model responses, retrieved text, review decisions and any revision. Offline tests check program behaviour with controlled responses. verify_live.py separately runs the actual model and includes negative controls.

```powershell
python -m unittest discover -s tests -v
python check_data.py
python verify_live.py --record runs/live.json
```

The code requests GPU offloading and two CPU threads. Use the optional run_local.ps1 wrapper for monitored runs and automatic cleanup of its temporary Ollama service. Passing a small test set is not a guarantee for every question.

## Recorded checks

The local run on 18 October 2026 indexed all 1196 original policy sections. Twenty-six offline tests passed. The live checks cover the missing phone number, 24-hour print retention, Campus-Guest and the 15-minute account-lock period. They also inject an unjustified unknown answer and an unsupported network name. Validation records preserve the actual retrieved passages, model outputs, review results and GPU loading information.

Development exposed several small-model failures. It confused review decisions, combined correct evidence with an unknown answer, and approved an altered network name. The final workflow separates evidence selection from display formatting and adds an exact-quote check. The model reviewer alone is not a reliable safety check. Earlier failed runs are retained alongside the final validation records.

This machine used Ollama's Vulkan backend on an RTX 3060 Laptop GPU, with both models fully offloaded and CPU threads limited to two. Backend and device selection depend on the machine; ordinary Ollama commands can use the default backend on a compatible system.

The submission for this part is this repository link. The three fine-tuning TXT files are in [SCSE_26_GEX3](https://github.com/Fgq2354/SCSE_26_GEX3). Uploading to GitHub does not submit either part to the course portal.
