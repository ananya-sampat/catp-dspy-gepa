"""
Optimizable prompt templates for CATP-LLM using DSPy's GEPA optimizer.

These templates extend the default CATP prompts with parameters that can be
optimized by the GEPA optimizer.
"""

import dspy
from typing import Dict, Any, List, Optional

# Base prompt templates from CATP
from src.catpllm.model.prompt import INSTRUCTION_PROMPT, TOOL_PROMPT2_P1, TOOL_PROMPT2_P2, TASK_PROMPT

# Create a simple template class since dspy.template.OptimizableTemplate might not exist
class OptimizableTemplate:
    """A simple template class that can be optimized by GEPA."""
    def __init__(self, name, template, instructions=None):
        self.name = name
        self.template = template
        self.instructions = instructions
        
    def __call__(self):
        return self.template


class CATPOptimizablePrompts:
    """
    Class containing optimizable prompt templates for CATP-LLM.
    """
    
    def __init__(self):
        """Initialize the optimizable prompt templates with default values."""
        # Create optimizable instruction prompt
        self.instruction_prompt = OptimizableTemplate(
            name="instruction_prompt",
            template=INSTRUCTION_PROMPT,
            instructions="""
            Improve this prompt to help the model generate better tool plans.
            Focus on emphasizing the balance between performance and cost.
            Ensure the prompt maintains clarity about the model's role as a planning agent.
            """
        )
        
        # Create optimizable tool prompt part 1
        self.tool_prompt_p1 = OptimizableTemplate(
            name="tool_prompt_p1",
            template=TOOL_PROMPT2_P1,
            instructions="""
            Improve this prompt section that introduces tools and their costs.
            Focus on making the relationship between tool functionality and cost clearer.
            """
        )
        
        # Create optimizable tool prompt part 2
        self.tool_prompt_p2 = OptimizableTemplate(
            name="tool_prompt_p2",
            template=TOOL_PROMPT2_P2,
            instructions="""
            Improve this prompt section that transitions to tool cost features.
            Make sure it maintains clarity about how costs relate to tools.
            """
        )
        
        # Create optimizable task prompt
        self.task_prompt = OptimizableTemplate(
            name="task_prompt",
            template=TASK_PROMPT,
            instructions="""
            Improve this prompt section that provides task specifications and input attributes.
            Ensure it clearly asks for a tool plan that balances performance and cost.
            Focus on encouraging efficient use of tools.
            """
        )
        
    def get_full_prompt(self, 
                        tool_features: str, 
                        cost_features: str,
                        task_spec: str,
                        input_attrs: str) -> str:
        """
        Get the full prompt with all components filled in.
        
        Args:
            tool_features: String describing tool features
            cost_features: String describing cost features
            task_spec: String describing task specifications
            input_attrs: String describing task input attributes
            
        Returns:
            Complete prompt string with all components
        """
        # Fill in the tool features and cost features
        tool_section = f"{self.tool_prompt_p1()}{tool_features}{self.tool_prompt_p2()}{cost_features}"
        
        # Replace placeholders in task prompt
        task_section = (self.task_prompt()
                        .replace("[Task Specification]", task_spec)
                        .replace("[Task Input Attributes]", input_attrs))
        
        # Combine all prompt components
        return f"{self.instruction_prompt()}{tool_section}{task_section}"
    
    def get_all_templates(self) -> Dict[str, OptimizableTemplate]:
        """Get all optimizable templates."""
        return {
            "instruction_prompt": self.instruction_prompt,
            "tool_prompt_p1": self.tool_prompt_p1,
            "tool_prompt_p2": self.tool_prompt_p2,
            "task_prompt": self.task_prompt,
        }
    
    def update_templates(self, templates: Dict[str, str]) -> None:
        """
        Update templates with optimized versions.
        
        Args:
            templates: Dictionary mapping template names to new template strings
        """
        for name, template in templates.items():
            if name == "instruction_prompt" and template:
                self.instruction_prompt.template = template
            elif name == "tool_prompt_p1" and template:
                self.tool_prompt_p1.template = template
            elif name == "tool_prompt_p2" and template:
                self.tool_prompt_p2.template = template
            elif name == "task_prompt" and template:
                self.task_prompt.template = template