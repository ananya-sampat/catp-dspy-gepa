#!/usr/bin/env python
"""
Script to run GEPA optimization on CATP-LLM prompts using a sample of the test dataset.
"""

import os
import json
import pickle
import argparse
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from munch import munchify

import dspy
from dspy.teleprompt import GEPA

from src.config import GlobalToolConfig, GlobalPathConfig, GlobalMetricsConfig
from src.catpllm.data.plan_dataset import PlanDataset
from src.catpllm.model import OfflineRLPolicy, TokenEncoder
from src.catpllm.model.llm import peft_model
from src.catpllm.utils.llm_utils import load_llm
from src.catpllm.utils.utils import set_random_seed, save_model, load_model

# Import both optimizers so user can choose which one to use
from src.catpllm.optimizers.gepa_optimizer import GEPAOptimizer
from src.catpllm.optimizers.simple_optimizer import SimpleOptimizer
from src.catpllm.optimizers.prompt_templates import CATPOptimizablePrompts
from src.catpllm.optimizers.extract_samples import extract_sample_entries
from src.catpllm.optimizers.dspy_openai_wrapper import OpenAIWrapper, configure_dspy_with_wrapper


def setup_lm(model_name):
    """Set up the DSPy language model to use OpenAI."""
    try:
        # First, try DSPy's native OpenAI integration
        try:
            # Try different DSPy OpenAI integrations
            openai_module_found = False
            
            # Option 1: Try dspy.openai
            try:
                from dspy.openai import OpenAI
                openai_module_found = True
                print("Found OpenAI in dspy.openai module")
                
                # Create OpenAI LM instance
                lm = OpenAI(
                    model=model_name,            # Use specified model name (e.g., "gpt-4")
                    api_key=None,                # None means use environment variable
                    temperature=0.7,             # Add some creativity
                    max_tokens=1024,             # Adequate response length
                    top_p=0.9,                   # Sample from top probability mass
                    seed=42                      # For reproducibility
                )
                
                # Configure DSPy with our OpenAI LM
                dspy.settings.configure(lm=lm)
                print(f"Successfully configured DSPy with native OpenAI integration: {model_name}")
                return lm
            except (ImportError, AttributeError) as e:
                print(f"Could not use dspy.openai: {e}")
                pass
                
            # Option 2: Try dspy.primitives.openai
            if not openai_module_found:
                try:
                    from dspy.primitives.openai import OpenAI
                    openai_module_found = True
                    print("Found OpenAI in dspy.primitives.openai module")
                    
                    # Create OpenAI LM instance
                    lm = OpenAI(
                        model=model_name,
                        api_key=None,
                        temperature=0.7,
                        max_tokens=1024,
                        top_p=0.9,
                        seed=42
                    )
                    
                    # Configure DSPy with our OpenAI LM
                    dspy.settings.configure(lm=lm)
                    print(f"Successfully configured DSPy with OpenAI from primitives module: {model_name}")
                    return lm
                except (ImportError, AttributeError) as e:
                    print(f"Could not use dspy.primitives.openai: {e}")
                    pass
            
            print("Could not find native DSPy OpenAI integration, using custom wrapper")
            
            # Use our custom wrapper
            lm = configure_dspy_with_wrapper(
                model=model_name,
                api_key=None,
                temperature=0.7,
                max_tokens=1024
            )
            
            print(f"Successfully configured DSPy with custom OpenAI wrapper: {model_name}")
            return lm
            
        except Exception as e:
            print(f"Error with OpenAI LM setup: {e}")
            print("Falling back to custom OpenAI wrapper")
            
            # Use our custom wrapper as a fallback
            lm = configure_dspy_with_wrapper(
                model=model_name,
                api_key=None,
                temperature=0.7,
                max_tokens=1024
            )
            
            return lm
            
    except Exception as e:
        print(f"Warning: Could not configure OpenAI LM: {e}")
        print("Falling back to mock language model")
        
        # Create a simple mock language model for testing when OpenAI setup fails
        class MockLanguageModel:
            def __init__(self, model_name):
                self.model_name = model_name
                
            def __call__(self, prompt, **kwargs):
                # For testing purposes, return a realistic mock response
                return {
                    "response": (
                        "Based on my analysis, the prompt could be improved by: "
                        "1. Adding more specific instructions about balancing performance and cost. "
                        "2. Clarifying the expected format of the plan. "
                        "3. Providing examples of good plans."
                    )
                }
        
        # Create a mock language model instance
        lm = MockLanguageModel(model_name)
        
        # Configure DSPy with our mock language model
        try:
            dspy.settings.configure(lm=lm)
        except Exception as e2:
            print(f"Warning: Could not configure DSPy settings: {e2}")
            
        return lm


def run_gepa_optimizer(args):
    """
    Run the GEPA optimizer on CATP-LLM.
    
    Args:
        args: Command-line arguments
    """
    # 1. Set random seed
    set_random_seed(args.seed)
    
    # 2. Load the plan dataset
    plan_pools = pickle.load(open(args.train_plan_pool, 'rb'))
    plan_dataset = PlanDataset(plan_pools, args.alpha, args.gamma, args.scale, args.context_len)
    dataset_info = munchify(plan_dataset.dataset_info)
    print("Dataset information loaded:")
    print(f"  Max return: {dataset_info.max_return}")
    print(f"  Min return: {dataset_info.min_return}")
    
    # 3. Load the CATP-LLM model
    # 3.1 Load LLM
    # Use HuggingFace cache dir from GlobalPathConfig for model downloads
    model_path = args.llm
    print(f"Loading LLM model: {model_path}")
    try:
        llm, tokenizer, llm_config = load_llm(args.llm, model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to use default model path")
        # Try loading directly with the model name as path (for HF hub download)
        llm, tokenizer, llm_config = load_llm(args.llm, args.llm)
    llm = llm.to(args.llm_device)
    if args.rank != -1:
        llm = peft_model(llm, args.llm, args.rank)
    
    # 3.2 Create token encoder
    llm_embed_dim = llm.get_embed_dim()
    encoder = TokenEncoder(
        GlobalToolConfig.max_num_tokens, 
        llm_embed_dim, 
        device=args.llm_device
    ).to(args.llm_device)
    
    # 3.3 Create policy
    num_tool_tokens = len(GlobalToolConfig.tool_token_vocabulary.keys()) + 2
    num_dependency_tokens = len(GlobalToolConfig.dependency_token_vocabulary.keys()) + 2
    policy = OfflineRLPolicy(
        encoder, 
        tokenizer, 
        llm, 
        llm_embed_dim, 
        num_tool_tokens, 
        num_dependency_tokens,
        GlobalToolConfig.max_num_tokens, 
        device=args.llm_device, 
        max_window_size=args.context_len
    )
    
    # 4. Load model if specified
    if args.load_model_dir:
        if os.path.exists(args.load_model_dir):
            policy = load_model(args, policy, args.load_model_dir)
            print('Loaded model from:', args.load_model_dir)
        else:
            print('Model directory does not exist, skipping loading model from:', args.load_model_dir)
    
    # 5. Extract sample entries from test dataset
    if args.task_ids:
        task_ids = [int(tid) for tid in args.task_ids.split(',')]
        print(f"Using provided task IDs: {task_ids}")
    else:
        dataset_path = os.path.join(GlobalPathConfig.data_path, "test_task_samples.json")
        task_ids = extract_sample_entries(
            dataset_path, 
            num_samples=args.num_samples,
            seed=args.seed
        )
        print(f"Extracted task IDs: {task_ids}")
    
    # 6. Setup DSPy model
    dspy_lm = setup_lm(args.dspy_model)
    
    # 7. Create and run the optimizer
    if args.use_simple_optimizer:
        print(f"Using simple optimizer with {args.iterations} iterations")
        optimizer = SimpleOptimizer(
            policy=policy,
            dataset_info=dataset_info,
            num_iterations=args.iterations,
            candidates_per_iteration=args.candidates,
            model_name=args.dspy_model,
            device=args.llm_device
        )
    else:
        print(f"Using GEPA optimizer with {args.iterations} iterations")
        optimizer = GEPAOptimizer(
            policy=policy,
            dataset_info=dataset_info,
            num_iterations=args.iterations,
            candidates_per_iteration=args.candidates,
            device=args.llm_device
        )
    
    # Create results directory
    results_dir = os.path.join(GlobalPathConfig.result_path, "gepa_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Run optimization with timeout
    results = optimizer.optimize(
        task_ids=task_ids,
        save_path=os.path.join(results_dir, "optimization_results.json"),
        timeout_seconds=args.timeout
    )
    
    # 8. Visualize results
    if args.visualize:
        history = results["performance_history"]
        
        # Extract performance data
        iterations = list(range(len(history)))
        scores = [h.get("score", 0) for h in history]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, 'o-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Performance Score')
        plt.title('GEPA Optimization Performance')
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, "optimization_performance.png"))
        print(f"Saved performance visualization to {results_dir}/optimization_performance.png")
    
    # 9. Report final results
    best_templates = results["optimized_templates"]
    print("\nBest optimized prompts:")
    for name, template in best_templates.items():
        print(f"\n--- {name} ---")
        print(template[:100] + "..." if len(template) > 100 else template)
    
    print(f"\nComplete results saved to {results_dir}/optimization_results.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GEPA optimization on CATP-LLM")
    
    # CATP-LLM settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--llm", type=str, default="facebook/opt-350m", 
                      help="Name of the LLM (format: [org/]llm-size, e.g., facebook/opt-350m)")
    parser.add_argument("--llm_device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help="Device for LLM")
    parser.add_argument("--train_plan_pool", type=str, required=True, 
                      help="Path to the plan pool data")
    parser.add_argument("--alpha", type=float, default=0.5,
                      help="Weight parameter to balance performance scores and execution costs")
    parser.add_argument("--gamma", type=float, default=1.0,
                      help="Reward discount factor")
    parser.add_argument("--scale", type=float, default=1.0,
                      help="Factor to scale the reward")
    parser.add_argument("--context_len", type=int, default=1,
                      help="Context length for the policy")
    parser.add_argument("--rank", type=int, default=-1,
                      help="Rank for PEFT model (-1 to disable)")
    parser.add_argument("--load_model_dir", type=str,
                      help="Directory to load model from")
    
    # GEPA optimization settings
    parser.add_argument("--dspy_model", type=str, default="gpt-3.5-turbo",
                      help="Model name for DSPy")
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of GEPA optimization iterations")
    parser.add_argument("--candidates", type=int, default=3,
                      help="Number of candidates per iteration")
    parser.add_argument("--num_samples", type=int, default=3,
                      help="Number of sample entries to use from the test dataset")
    parser.add_argument("--task_ids", type=str, 
                      help="Comma-separated list of task IDs to use (overrides --num_samples)")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualization of optimization performance")
    parser.add_argument("--timeout", type=int, default=300,
                      help="Timeout in seconds for optimization (default: 300s/5min)")
    parser.add_argument("--use_simple_optimizer", action="store_true",
                      help="Use the simple optimizer instead of GEPA")
    
    args = parser.parse_args()
    
    run_gepa_optimizer(args)


if __name__ == "__main__":
    main()