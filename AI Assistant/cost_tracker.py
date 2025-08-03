"""
Cost and Token Usage Tracker for OpenAI API calls
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostTracker:
    """Track OpenAI API costs and token usage"""
    
    # OpenAI pricing (as of 2025) - adjust these if prices change
    PRICING = {
        'gpt-4': {
            'input': 0.00003,   # $0.03 per 1K tokens
            'output': 0.00006,  # $0.06 per 1K tokens
        },
        'gpt-4-turbo': {
            'input': 0.00001,   # $0.01 per 1K tokens
            'output': 0.00003,  # $0.03 per 1K tokens
        },
        'gpt-3.5-turbo': {
            'input': 0.0000005, # $0.0005 per 1K tokens
            'output': 0.0000015, # $0.0015 per 1K tokens
        }
    }
    
    def __init__(self):
        self.session_stats = {
            'total_calls': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'calls_by_model': {},
            'start_time': datetime.now()
        }
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a single API call"""
        # Normalize model name for pricing lookup
        model_key = model.lower()
        if 'gpt-4-turbo' in model_key:
            pricing = self.PRICING['gpt-4-turbo']
        elif 'gpt-4' in model_key:
            pricing = self.PRICING['gpt-4']
        elif 'gpt-3.5' in model_key:
            pricing = self.PRICING['gpt-3.5-turbo']
        else:
            # Default to gpt-4 pricing if unknown
            pricing = self.PRICING['gpt-4']
            logger.warning(f"Unknown model {model}, using gpt-4 pricing")
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def log_api_call(self, model: str, input_tokens: int, output_tokens: int, 
                     context: str = "", duration: Optional[float] = None):
        """Log an API call with cost information"""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Update session stats
        self.session_stats['total_calls'] += 1
        self.session_stats['total_input_tokens'] += input_tokens
        self.session_stats['total_output_tokens'] += output_tokens
        self.session_stats['total_cost'] += cost
        
        # Track by model
        if model not in self.session_stats['calls_by_model']:
            self.session_stats['calls_by_model'][model] = {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0
            }
        
        model_stats = self.session_stats['calls_by_model'][model]
        model_stats['calls'] += 1
        model_stats['input_tokens'] += input_tokens
        model_stats['output_tokens'] += output_tokens
        model_stats['cost'] += cost
        
        # Log the call details
        duration_str = f" ({duration:.2f}s)" if duration else ""
        logger.info(f"[COST DEBUG] {context}")
        logger.info(f"[COST DEBUG] Model: {model}, Input: {input_tokens} tokens, "
                   f"Output: {output_tokens} tokens, Cost: ${cost:.6f}{duration_str}")
        logger.info(f"[COST DEBUG] Session Total: {self.session_stats['total_calls']} calls, "
                   f"${self.session_stats['total_cost']:.6f}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        session_duration = datetime.now() - self.session_stats['start_time']
        
        return {
            'session_duration': str(session_duration),
            'total_calls': self.session_stats['total_calls'],
            'total_tokens': self.session_stats['total_input_tokens'] + self.session_stats['total_output_tokens'],
            'total_input_tokens': self.session_stats['total_input_tokens'],
            'total_output_tokens': self.session_stats['total_output_tokens'],
            'total_cost': round(self.session_stats['total_cost'], 6),
            'avg_cost_per_call': round(self.session_stats['total_cost'] / max(1, self.session_stats['total_calls']), 6),
            'calls_by_model': self.session_stats['calls_by_model']
        }
    
    def print_session_summary(self):
        """Print a formatted session summary"""
        summary = self.get_session_summary()
        
        print("\n" + "="*60)
        print("📊 OPENAI API COST SUMMARY")
        print("="*60)
        print(f"Session Duration: {summary['session_duration']}")
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  - Input Tokens: {summary['total_input_tokens']:,}")
        print(f"  - Output Tokens: {summary['total_output_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost']:.6f}")
        print(f"Average Cost per Call: ${summary['avg_cost_per_call']:.6f}")
        
        if summary['calls_by_model']:
            print("\nBy Model:")
            for model, stats in summary['calls_by_model'].items():
                print(f"  {model}:")
                print(f"    Calls: {stats['calls']}")
                print(f"    Tokens: {stats['input_tokens'] + stats['output_tokens']:,}")
                print(f"    Cost: ${stats['cost']:.6f}")
        
        print("="*60)

# Global instance
cost_tracker = CostTracker()

def track_langchain_response(response, model: str, context: str = "", duration: Optional[float] = None):
    """
    Track a LangChain response for cost calculation
    
    Args:
        response: The response object from LangChain
        model: The model name used
        context: Description of what this call was for
        duration: Optional duration of the call in seconds
    """
    try:
        # LangChain stores usage info in response.response_metadata
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            cost_tracker.log_api_call(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context=context,
                duration=duration
            )
        else:
            logger.warning(f"[COST DEBUG] No token usage found in response for: {context}")
    except Exception as e:
        logger.error(f"[COST DEBUG] Error tracking response: {e}")

def get_session_summary():
    """Get the current session cost summary"""
    return cost_tracker.get_session_summary()

def print_session_summary():
    """Print the current session cost summary"""
    cost_tracker.print_session_summary()
