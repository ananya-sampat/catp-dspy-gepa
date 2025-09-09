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
   ```

   Set your API key if using OpenAI:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
2. **Run GEPA**
   ```bash
   python scripts/run_gepa_optimizer.py\
   --train_plan_pool <PATH_TO_CATP>/src/catpllm/data/training_data/seq_plan_pool.pkl\
   --iterations 5 --candidates 3 --num_samples 3 --visualize
   ```
3. **Outputs land in:**
   - ```results/gepa_results/optimization_results.json```
   - ```results/gepa_results/optimization_performance.png (if --visualize)```
If GEPA has DSPy version friction, use the **Simple Optimizer** path described in docs/SIMPLE_OPTIMIZER_GUIDE.md

# Status from my run
 - Recorded scores were 0.0 across iterations and prompts didn’t meaningfully update.
 - Probably means the eval metric or score plumbing needs attention (check _eval_metric and the fields returned by the policy).

# Docs
 - ```docs/GEPA_INTEGRATION_REPORT.md``` – architecture & integration notes
 - ```docs/OPENAI_GEPA_SETUP.md``` – OpenAI/DSPy setup
 - ```docs/SIMPLE_OPTIMIZER_GUIDE.md``` – simple optimizer usage

Code Included
 - ```scripts/run_gepa_optimizer.py```
 - ```src/catpllm/optimizers/```: ```gepa_optimizer.py```, ```prompt_templates.py```, ```extract_samples.py```, optional wrappers

Use your existing local CATP-LLM checkout; this repo is just the GEPA layer + docs + results.
