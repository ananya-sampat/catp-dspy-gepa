# Simple Template Optimizer for CATP-LLM

This guide explains how to use the simplified template optimizer for CATP-LLM, which offers a more reliable alternative to the GEPA optimizer by using direct LLM calls for prompt optimization without depending on DSPy's more complex optimization machinery.

## Overview

The SimpleOptimizer works by:

1. Directly using the LLM to optimize prompt templates
2. Working with three key templates:
   - `system_prompt`: Sets the overall context for planning
   - `task_instruction`: Provides specific instructions for a task
   - `reflection_prompt`: Evaluates and reflects on plans for improvement
3. Using a straightforward iterative improvement process without relying on DSPy

This approach avoids many of the compatibility issues that can arise with DSPy's GEPA implementation and provides a more reliable way to optimize templates, especially when working with newer versions of the OpenAI API.

## Usage

### Command Line Arguments

To use the simple optimizer, add the `--use_simple_optimizer` flag to your command:

```bash
python run_gepa_optimizer.py \
  --train_plan_pool src/catpllm/data/training_data/seq_plan_pool.pkl \
  --dspy_model gpt-4-turbo \
  --iterations 3 \
  --num_samples 2 \
  --timeout 180 \
  --visualize \
  --use_simple_optimizer
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--use_simple_optimizer` | Flag to use the simplified optimizer instead of GEPA |
| `--dspy_model` | LLM model to use for optimization (e.g., gpt-4-turbo) |
| `--iterations` | Number of optimization iterations (default: 5) |
| `--timeout` | Maximum seconds for optimization (default: 300s/5min) |
| `--num_samples` | Number of sample entries to use for optimization |

## How It Works

1. **Template Initialization**: Starts with default templates for system prompt, task instructions, and reflection
2. **Example Preparation**: Generates examples using the current templates and evaluates their performance
3. **Template Optimization**: For each template, sends examples and the current template to the LLM for improvement
4. **Iterative Improvement**: Repeats the process over multiple iterations
5. **Result Collection**: Tracks performance history and saves the optimized templates

## Results

The optimizer produces:

1. **Optimized Templates**: Improved versions of each template
2. **Performance History**: Record of score improvements over iterations
3. **Visualization**: Chart showing performance trends (if `--visualize` is specified)
4. **JSON Results**: Complete results saved to file for further analysis

## Comparison with GEPA

| Feature | Simple Optimizer | GEPA Optimizer |
|---------|------------------|---------------|
| Reliability | Higher - less dependent on DSPy API | May have compatibility issues with DSPy versions |
| Speed | Faster - direct LLM calls | Slower - more complex optimization process |
| Complexity | Simpler - straightforward template optimization | More complex - uses evolutionary algorithms |
| Customization | Basic - focuses on three key templates | Advanced - can work with many template aspects |
| Dependencies | Minimal - mainly OpenAI API | Extensive - requires DSPy compatibility |

## When to Use

Use the simple optimizer when:

1. You encounter compatibility issues with the GEPA optimizer
2. You need a quick optimization solution that works reliably
3. You're working with newer versions of the OpenAI API
4. You want a more straightforward optimization process

## Example Results

The optimized templates typically contain improvements like:

- More specific and clear instructions
- Better guidance on balancing performance and cost
- More effective prompt structure and framing
- Improved reflection and analysis components

These improvements can lead to more effective planning by the CATP-LLM system.