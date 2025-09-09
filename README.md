# catp-dspy-gepa
# CATP-LLM + GEPA (Shareable Layer)

This repo contains my **GEPA integration**, a **simple optimizer path**, setup notes, and **example results**.  
It assumes you already have the **CATP-LLM** repo locally (not included here).

---

## Quick Start

1. **Create and activate a venv, then install requirements**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

   Set your API key if using OpenAI:
   ```bash
   export OPENAI_API_KEY=sk-...
   '''
2. **Run GEPA**
   ```bash
   python scripts/run_gepa_optimizer.py \
  --train_plan_pool <PATH_TO_CATP>/src/catpllm/data/training_data/seq_plan_pool.pkl \
  --iterations 5 --candidates 3 --num_samples 3 --visualize
  '''
   
