# Well Log & Seismic Q&A Agent

A LangGraph-powered Python agent that answers **natural-language questions** about oil & gas well log (LAS) and seismic (SEG-Y) data by reading and calculating directly from files — no ML training, no SQL database.

---

## Architecture Overview

```
User question
     │
     ▼
FastAPI /ask  ──►  LangGraph ReAct Agent  ──►  Claude 3.5 Sonnet
                          │
              ┌───────────┴───────────────────────┐
              │                                   │
       Well Log Tools                      Seismic Tools
    (lasio → LAS files)              (segyio → SEG-Y files)
         data/wells/                      data/seismic/
```

The agent always:
1. **Discovers** files first (`list_wells` / `list_seismic_surveys`).
2. **Matches** the user's well/survey name to the right file.
3. **Reads** actual data — never fabricates numbers.
4. **Logs** every tool call to stdout for demo transparency.

---

## Quick Start

### 1. Install dependencies

```bash
cd well_seismic_agent
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Generate synthetic sample data

```bash
python generate_synthetic_data.py
```

This creates:
- `data/wells/Well-Alpha.las` — shale/sand sequence, 1500–3000 m
- `data/wells/Well-Beta.las`  — carbonate/shale sequence, 2500–4000 m
- `data/seismic/Survey-Apex.segy` — 3-D volume, inlines 100–120, xlines 50–70

### 4. Run the tests

```bash
pytest tests/ -v
```

Expected: **≥ 5 passing tests** covering all 7 tools.

### 5. Start the API server

```bash
uvicorn api:app --reload --port 8000
```

---

## API Reference

### `POST /ask`

Submit a natural-language question.

**Request body:**
```json
{ "question": "What is the average GR in Well-Alpha between 2000 and 2500 m?" }
```

**Response:**
```json
{
  "question": "...",
  "answer": "The average Gamma Ray in Well-Alpha between 2000 and 2500 m is 34.7 GAPI ...",
  "elapsed_seconds": 3.14
}
```

### `GET /tools`

List all registered agent tools and their descriptions.

### `GET /docs`

Interactive Swagger UI.

---

## Example Questions to Try

### Well-log questions

```bash
# 1. Average porosity in the reservoir zone
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average neutron porosity in Well-Alpha between 2000 and 2500 meters?"}'

# 2. Resistivity anomaly detection (reservoir vs shale)
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Flag all intervals in Well-Beta where resistivity exceeds 500 Ohm·m between 2500 and 4000 m."}'
```

### Seismic questions

```bash
# 3. Survey geometry
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the inline and crossline range of Survey-Apex, and how many samples per trace?"}'

# 4. Amplitude statistics for a specific trace
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the amplitude range at inline 100, crossline 50 in Survey-Apex between 500 and 800 ms?"}'
```

### Combined well + seismic questions

```bash
# 5. Compare porosity zone with seismic amplitude at the same position
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "In Well-Alpha the sandy zone is around 2000–2500 m. Compare the average porosity there with the seismic amplitude at inline 110, crossline 60 in Survey-Apex between 200 and 400 ms. Could the two be related?"}'

# 6. Full characterisation of a combined zone
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the log character of Well-Alpha between 2000 and 2200 m (GR, NPHI, RHOB, RT) and then tell me the seismic RMS amplitude at inline 105, crossline 55 in Survey-Apex between 0 and 600 ms."}'
```

---

## Synthetic Data Details

| File | Depth range | Key features |
|------|-------------|--------------|
| `Well-Alpha.las` | 1500–3000 m | Sandy reservoir (low GR ~20–50 GAPI, high RT ~40–180 Ω·m) at 2000–2500 m; shale above & below |
| `Well-Beta.las`  | 2500–4000 m | Carbonate host rock (GR 15–40 GAPI, RT 80–900 Ω·m); shale interbed at 3000–3200 m |
| `Survey-Apex.segy` | Inline 100–120, XL 50–70, 0–1000 ms | Chirp + noise; amplitude increases with crossline number |

---

## Project Structure

```
well_seismic_agent/
├── data/
│   ├── wells/
│   │   ├── Well-Alpha.las
│   │   └── Well-Beta.las
│   └── seismic/
│       └── Survey-Apex.segy
├── tools.py                  # 7 data-reading functions + LangChain wrappers
├── agent.py                  # LangGraph ReAct agent
├── api.py                    # FastAPI /ask endpoint
├── generate_synthetic_data.py
├── tests/
│   ├── test_well_tools.py
│   └── test_seismic_tools.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | — | Your Anthropic API key |
| `DATA_DIR` | No | `./data` | Override the data directory path |

---

## Troubleshooting

**`FileNotFoundError: Wells directory not found`**  
→ Run `python generate_synthetic_data.py` first.

**`EnvironmentError: ANTHROPIC_API_KEY not set`**  
→ Copy `.env.example` to `.env` and add your key.

**`ValueError: Curve 'XYZ' not found`**  
→ Call `list_available_curves` first to see what curves exist.

**Seismic inline/crossline out of range**  
→ Survey-Apex covers inlines 100–120 and crosslines 50–70 only.
