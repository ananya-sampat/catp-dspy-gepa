"""
Custom OpenAI wrapper for DSPy to ensure compatibility with different OpenAI API versions.
This wrapper can be used when DSPy's native OpenAI integration is not available.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union

# Check if OpenAI is available
try:
    import openai
    OPENAI_AVAILABLE = True
    # Check if we're using the new OpenAI API (>=1.0.0)
    USING_NEW_OPENAI_API = hasattr(openai, 'OpenAI')
except ImportError:
    OPENAI_AVAILABLE = False
    USING_NEW_OPENAI_API = False
    print("Warning: OpenAI library not found, using mock implementation")

# Check if DSPy is available
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("Warning: DSPy library not found")


class OpenAIWrapper:
    """
    A custom OpenAI wrapper that provides compatibility with DSPy and supports
    both new (>=1.0.0) and legacy (<1.0.0) OpenAI APIs.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI wrapper.
        
        Args:
            model: The name of the OpenAI model to use
            api_key: Your OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
            temperature: Controls randomness of the output (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility
            **kwargs: Additional parameters to pass to the OpenAI API
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.kwargs = kwargs
        
        # Set up OpenAI client if available
        if OPENAI_AVAILABLE:
            if USING_NEW_OPENAI_API:
                self.client = openai.OpenAI(api_key=self.api_key)
                print(f"Initialized custom OpenAIWrapper using new OpenAI API (>=1.0.0) with model: {model}")
            else:
                if self.api_key:
                    openai.api_key = self.api_key
                print(f"Initialized custom OpenAIWrapper using legacy OpenAI API (<1.0.0) with model: {model}")
        else:
            print("OpenAI library not available, using mock implementation")
    
    def __call__(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Call the OpenAI API with a prompt, in a format compatible with DSPy.
        
        Args:
            prompt: The prompt to send to the API
            temperature: Override the default temperature if provided
            max_tokens: Override the default max_tokens if provided
            **kwargs: Additional parameters to pass to the OpenAI API
        
        Returns:
            Dictionary with a "response" key containing the text response
        """
        # Use instance defaults if not overridden
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Combine instance kwargs with call kwargs, with call kwargs taking precedence
        call_kwargs = {**self.kwargs, **kwargs}
        
        if not OPENAI_AVAILABLE:
            # Mock implementation for when OpenAI is not available
            print(f"Mock call to {self.model} with prompt: {prompt[:50]}...")
            return {"response": "This is a mock response from OpenAIWrapper because the OpenAI library is not available."}
        
        try:
            if USING_NEW_OPENAI_API:
                # New OpenAI API (>=1.0.0)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    **call_kwargs
                )
                return {"response": response.choices[0].message.content}
            else:
                # Legacy OpenAI API (<1.0.0)
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.top_p,
                    **call_kwargs
                )
                return {"response": response.choices[0].message.content}
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            error_msg = f"Error: {str(e)}"
            # Return a valid response format even in case of errors
            return {"response": error_msg}
    
    def get_response(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call the OpenAI API and return just the text response.
        
        Args:
            prompt: The prompt to send to the API
            temperature: Override the default temperature if provided
            max_tokens: Override the default max_tokens if provided
            **kwargs: Additional parameters to pass to the OpenAI API
        
        Returns:
            String containing the text response
        """
        result = self.__call__(prompt, temperature, max_tokens, **kwargs)
        return result.get("response", "Error: No response received")


# Configure DSPy to use our wrapper if DSPy is available
def configure_dspy_with_wrapper(
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> OpenAIWrapper:
    """
    Configure DSPy to use our custom OpenAI wrapper.
    
    Args:
        model: The name of the OpenAI model to use
        api_key: Your OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
        temperature: Controls randomness of the output (0.0-1.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The OpenAIWrapper instance that was created and configured with DSPy
    """
    if not DSPY_AVAILABLE:
        print("DSPy not available, cannot configure")
        wrapper = OpenAIWrapper(model, api_key, temperature, max_tokens)
        return wrapper
    
    wrapper = OpenAIWrapper(model, api_key, temperature, max_tokens)
    
    try:
        dspy.settings.configure(lm=wrapper)
        print(f"Successfully configured DSPy with custom OpenAI wrapper using model: {model}")
    except Exception as e:
        print(f"Error configuring DSPy with custom wrapper: {e}")
    
    return wrapper