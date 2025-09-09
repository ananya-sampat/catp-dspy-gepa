# Setting Up OpenAI Integration for GEPA Optimizer

This document explains how to set up the OpenAI integration for the GEPA optimizer in CATP-LLM.

## Prerequisites

1. **OpenAI API Key**: You need an API key from OpenAI to use their models
2. **Python Environment**: Python 3.7+ with pip installed
3. **CATP-LLM Repository**: Cloned repository with required dependencies

## Environment Setup

### 1. Install Required Packages

#### Option A: Using the latest OpenAI API (>=1.0.0, recommended)

```bash
# Install latest OpenAI API package
pip install openai>=1.0.0

# Install DSPy package
pip install dspy-ai>=2.3.0

# Install other requirements
pip install -r requirements.txt
```

#### Option B: Using the older OpenAI API (0.28.x)

```bash
# Install older OpenAI API package
pip install openai==0.28.1

# Install DSPy package
# Note: Try an older version if you encounter compatibility issues
pip install dspy-ai==2.3.4

# Install other requirements
pip install -r requirements.txt
```

The code has been updated to work with both the new (>=1.0.0) and legacy (<1.0.0) OpenAI APIs.

### 2. Configure OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
# Linux/macOS
export OPENAI_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY='your-api-key-here'
```

### 3. Test the OpenAI Integration

Run the test script to verify that the OpenAI integration is working correctly:

```bash
python test_openai_integration.py
```

If all tests pass, you're ready to use OpenAI with the GEPA optimizer.

## Using the GEPA Optimizer with OpenAI

Run the GEPA optimizer with OpenAI integration:

```bash
python run_gepa_optimizer.py \
  --train_plan_pool src/catpllm/data/training_data/seq_plan_pool.pkl \
  --dspy_model gpt-4-turbo \
  --iterations 5 \
  --candidates 3 \
  --num_samples 3 \
  --visualize
```

### Command-Line Arguments

- `--train_plan_pool`: Path to the plan pool data (required)
- `--dspy_model`: OpenAI model to use (e.g., gpt-4-turbo, gpt-3.5-turbo)
- `--iterations`: Number of optimization iterations (default: 5)
- `--candidates`: Number of candidates per iteration (default: 3)
- `--num_samples`: Number of samples to use from the test dataset (default: 3)
- `--task_ids`: Specific task IDs to use (comma-separated, overrides num_samples)
- `--visualize`: Generate visualization of optimization performance

## Testing the Custom Wrapper

A test script is provided to verify that the custom wrapper works correctly:

```bash
python test_custom_wrapper.py
```

This script tests:
1. Direct calls to the wrapper
2. Integration with DSPy
3. The convenience configuration function

If all tests pass, the custom wrapper is working correctly, and you can proceed with running the GEPA optimizer.

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
   - Check that your OPENAI_API_KEY environment variable is set correctly
   - Verify that the API key is valid and has sufficient credits

2. **ImportError: No module named 'openai'**:
   - Install the OpenAI package: `pip install openai`

3. **ImportError: No module named 'dspy'**:
   - Install the DSPy package: `pip install dspy-ai`

4. **ImportError: No module named 'dspy.openai'**:
   - This issue is now resolved by the custom wrapper
   - The implementation no longer depends on DSPy's native OpenAI integration
   - The custom wrapper provides a compatible interface regardless of DSPy version

5. **"You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0"**:
   - This issue is now resolved by the custom wrapper
   - The wrapper automatically detects your OpenAI API version and uses the appropriate calls
   - It works with both new (>=1.0.0) and legacy (<1.0.0) OpenAI APIs

6. **"Could not find OpenAI in any DSPy modules"**:
   - This is expected and not an error - it just means the custom wrapper is being used
   - The implementation now uses the custom wrapper instead of DSPy's native OpenAI integration

7. **OpenAI API Rate Limits**:
   - If you hit rate limits, try reducing the number of iterations or candidates
   - Consider using a paid API plan for higher rate limits

### Testing Individual Components

You can test specific components of the OpenAI integration:

```bash
# Test direct OpenAI API access
python test_openai_integration.py --direct

# Test DSPy's OpenAI integration
python test_openai_integration.py --dspy

# Test GEPA with OpenAI integration
python test_openai_integration.py --gepa
```

## Custom DSPy OpenAI Wrapper

To ensure maximum compatibility with both OpenAI API versions and DSPy, a custom wrapper has been implemented:

```python
from src.catpllm.optimizers.dspy_openai_wrapper import OpenAIWrapper, configure_dspy_with_wrapper

# Create the wrapper
wrapper = OpenAIWrapper(
    model="gpt-4-turbo",
    temperature=0.7,
    max_tokens=1024
)

# Configure DSPy with the wrapper
dspy.settings.configure(lm=wrapper)

# Or use the convenience function
wrapper = configure_dspy_with_wrapper(model="gpt-4-turbo")
```

The custom wrapper:

1. Supports both new (>=1.0.0) and legacy (<1.0.0) OpenAI APIs
2. Works as a drop-in replacement for DSPy's OpenAI integration
3. Provides a DSPy-compatible interface for use with GEPA
4. Includes graceful error handling and fallbacks
5. Automatically detects the OpenAI API version

## Fallback Mechanism

The implementation includes fallback mechanisms if OpenAI integration fails:

1. First tries to use the custom OpenAI wrapper
   - Automatically detects if using new (>=1.0.0) or legacy (<1.0.0) OpenAI API
   - Uses appropriate API calls for each version
2. Falls back to a mock language model for testing without API access

This ensures that the code can still run even if the OpenAI API is unavailable or if API keys are not set up correctly.

## Cost Considerations

Using OpenAI's GPT-4 and GPT-3.5 models with the GEPA optimizer will incur API usage costs. The optimizer makes multiple calls to the API during optimization, especially with:

- Higher values for `--iterations`
- Higher values for `--candidates`
- More samples (higher `--num_samples` or more `--task_ids`)

To minimize costs while testing:
1. Use fewer iterations and candidates
2. Use a smaller number of samples
3. Consider using gpt-3.5-turbo instead of gpt-4 for initial testing

## Credits

This integration uses:
- OpenAI's API: https://platform.openai.com/
- DSPy framework: https://github.com/stanfordnlp/dspy
- CATP-LLM: Cost-Aware Tool Planning framework for LLMs