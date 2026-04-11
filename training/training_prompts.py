"""
Simple, straightforward prompt configuration for training.
Each function returns exactly the prompt string that will be used.
Easy to inspect and understand what each prompt will look like.

To test the prompts without training:
    python training_prompts.py
    
Or use VSCode debugger:
    Select "Test Training Prompts" configuration and press F5
"""

import json
from typing import Dict, Any, Optional


# ============================================================================
# RESPONSE TEMPLATES - These are the answer formats the model should produce
# ============================================================================

def format_ra_answer(ra_dict: Dict, cot_content: Optional[str], eos_token: str, no_eos = False) -> str:
    """Format RA answer with optional chain-of-thought.
    
    Args:
        ra_dict: Relational algebra dictionary to format as JSON
        cot_content: Optional thinking/reasoning content to include
        eos_token: End-of-sequence token for the model (e.g., '<|endoftext|>')
    
    Returns:
        Formatted string with <think> and <answer> tags containing the RA JSON
    """
    answer = ""
    
    # Add CoT if provided
    if cot_content:
        answer += f"<think>\n{cot_content}\n</think>\n"
    
    # Add RA in JSON format
    ra_json = json.dumps(ra_dict, indent=2)
    answer += f"<answer>\n```json\n{ra_json}\n```\n</answer>"
    if not no_eos:
        answer += eos_token

    return answer


def format_sql_answer(sql_query: str, cot_content: Optional[str], eos_token: str, no_eos = False) -> str:
    """Format SQL answer with optional chain-of-thought.
    
    Args:
        sql_query: SQL query string to format
        cot_content: Optional thinking/reasoning content to include
        eos_token: End-of-sequence token for the model
    
    Returns:
        Formatted string with <think> and <answer> tags containing the SQL
    """
    answer = ""
    
    # Add CoT if provided
    if cot_content:
        answer += f"<think>\n{cot_content}\n</think>\n"
    
    # Add SQL query
    if no_eos:
        answer += f"<answer>\n```sql\n{sql_query}\n```\n</answer>"
    else:
        answer += f"<answer>\n```sql\n{sql_query}\n```\n</answer>{eos_token}"

    return answer


def format_ra_sql_answer(ra_dict: Dict, sql_query: str, cot_content: Optional[str], eos_token: str, no_eos = False) -> str:
    """Format combined RA+SQL answer where RA is in thinking section.
    
    This is the verbose version with full explanations about the RA tree construction
    and translation to SQL. The RA tree appears in the <think> section as part of
    the reasoning process, while the final SQL appears in the <answer> section.
    
    Args:
        ra_dict: Relational algebra dictionary to include in thinking
        sql_query: Final SQL query for the answer section
        cot_content: Optional additional reasoning content
        eos_token: End-of-sequence token for the model
    
    Returns:
        Formatted string with RA in <think> and SQL in <answer>
    """
    
    # Build thinking section with RA
    think_content = ""
    
    # Add CoT content if provided
    if cot_content:
        think_content += cot_content + "\n\n"
    
    # Add RA explanation and tree
    think_content += "To solve this problem, I need to first construct a relational algebra operator tree that captures the logical operations needed.\n\n"
    # think_content += "Let me identify the required tables and operations:\n"
    # think_content += "Based on the question, I need to work with the relevant tables and apply appropriate operations.\n\n"
    think_content += "Here's the relational algebra operator tree:\n"
    
    ra_json = json.dumps(ra_dict, indent=2)
    think_content += f"```json\n{ra_json}\n```\n\n"
    
    think_content += "Now I'll translate this relational algebra into an equivalent SQL query. "
    think_content += "The operator tree shows me the logical flow of operations that need to be performed."
    
    # Combine everything
    answer = f"<think>\n{think_content}\n</think>\n"
    if no_eos:
        answer += f"<answer>\n```sql\n{sql_query}\n```\n</answer>"
    else:
        answer += f"<answer>\n```sql\n{sql_query}\n```\n</answer>{eos_token}"

    return answer


def format_ra_sql_answer_simple(ra_dict: Dict, sql_query: str, eos_token: str) -> str:
    """Simpler version of RA+SQL answer without verbose explanation.
    
    Minimal version that still includes RA in thinking but with less explanatory text.
    Use this when you want a more concise format.
    
    Args:
        ra_dict: Relational algebra dictionary
        sql_query: Final SQL query
        eos_token: End-of-sequence token
    
    Returns:
        Compact formatted string with RA and SQL
    """
    
    think_content = "To solve this problem, I'll first construct a relational algebra operator tree.\n\n"
    think_content += "Here's the relational algebra operator tree:\n"
    
    ra_json = json.dumps(ra_dict, indent=2)
    think_content += f"```json\n{ra_json}\n```\n\n"
    think_content += "Now I'll translate this relational algebra into SQL."
    
    answer = f"<think>\n{think_content}\n</think>\n"
    answer += f"<answer>\n```sql\n{sql_query}\n```\n</answer>{eos_token}"
    
    return answer


# ============================================================================
# EXTRACTION HELPERS - Extract content from model responses
# ============================================================================

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text, trying code blocks first then raw JSON.
    
    Attempts extraction in this order:
    1. JSON within ```json code blocks
    2. Raw JSON objects found in text (looks for balanced braces)
    
    Args:
        text: Text that may contain JSON
    
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    import re
    
    # Try to extract from ```json blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def extract_sql_from_text(text: str) -> Optional[str]:
    """Extract SQL from text, looking for ```sql blocks.
    
    Args:
        text: Text that may contain SQL in code blocks
    
    Returns:
        SQL query string if found, None otherwise
    """
    import re
    
    sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    
    return None


def extract_thinking_content(text: str) -> Optional[str]:
    """Extract content between <think> tags."""
    import re
    
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    
    return None


def extract_answer_content(text: str) -> Optional[str]:
    """Extract content between <answer> tags."""
    import re
    
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    return None


# ============================================================================
# DATASET-SPECIFIC SQL FIELD EXTRACTION
# ============================================================================

def get_sql_field_from_item(item: Dict, dataset_name: str) -> str:
    """Get SQL field from dataset item based on dataset conventions.
    
    Different datasets use different field names for SQL:
    - BIRD: 'SQL'
    - Spider variants: 'query' or 'SQL'
    - Others: 'sql' or 'SQL'
    
    Args:
        item: Dataset item dictionary
        dataset_name: Name of the dataset ('bird', 'spider', etc.)
    
    Returns:
        SQL query string from the appropriate field
    """
    if dataset_name == 'bird':
        return item['SQL']
    elif dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic']:
        return item.get('query', item.get('SQL', ''))
    else:
        return item.get('sql', item.get('SQL', ''))

