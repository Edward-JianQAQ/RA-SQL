"""
Prompt Template Manager for SQL-R1 Project

This module provides centralized management of prompt templates for various tasks:
- SQL generation
- Relational Algebra (RA) generation
- Combined RA+SQL generation
- Special prompts for reasoning tasks
"""

import os
from pathlib import Path
from typing import Dict, Optional

class PromptManager:
    """Manages loading and formatting of prompt templates."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            template_dir: Path to the prompt templates directory.
                         If None, uses 'prompt_templates' in the current directory.
        """
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'prompt_templates')
        self.template_dir = Path(template_dir)
        self._cache: Dict[str, str] = {}
    
    def _load_template(self, template_path: str) -> str:
        """
        Load a template from file with caching.
        
        Args:
            template_path: Relative path to the template file from template_dir
        
        Returns:
            The template content as a string
        """
        if template_path not in self._cache:
            full_path = self.template_dir / template_path
            if not full_path.exists():
                raise FileNotFoundError(f"Template not found: {full_path}")
            with open(full_path, 'r', encoding='utf-8') as f:
                self._cache[template_path] = f.read()
        return self._cache[template_path]
    
    # NEW: safely format templates that contain JSON braces
    def _safe_format(self, template: str, mapping: Dict[str, str], allowed_keys: tuple) -> str:
        """
        Escape all braces in template, then unescape only allowed placeholders,
        and finally apply str.format with given mapping.
        This prevents KeyError from JSON examples like { "name": "FILTER", ... }.
        """
        escaped = template.replace("{", "{{").replace("}", "}}")
        for k in allowed_keys:
            escaped = escaped.replace(f"{{{{{k}}}}}", f"{{{k}}}")
        return escaped.format(**mapping)
    
    def get_sql_prompt(self, cot: bool = False) -> str:
        """
        Get SQL generation prompt template.
        
        Args:
            cot: Whether to use Chain-of-Thought prompt
        
        Returns:
            The SQL prompt template
        """
        if cot:
            return self._load_template('sql/cot_prompt.txt')
        return self._load_template('sql/base_prompt.txt')
    
    def get_ra_prompt(self, cot: bool = False, include_head: bool = True) -> str:
        """
        Get Relational Algebra generation prompt template.
        
        Args:
            cot: Whether to use Chain-of-Thought prompt
            include_head: Whether to include the head section with operator definitions
        
        Returns:
            The RA prompt template
        """
        prompt = ""
        if include_head:
            prompt = self._load_template('ra/base_prompt_head.txt') + '\n'
            # prompt = self._load_template('ra/base_prompt_head_v1.txt') + '\n'
        
        if cot:
            prompt += self._load_template('ra/cot_prompt.txt')
            # prompt += self._load_template('ra/cot_prompt_v1.txt')
        else:
            prompt += self._load_template('ra/base_prompt.txt')
        
        return prompt
    
    def get_ra_sql_prompt(self, cot: bool = False) -> str:
        """
        Get combined RA+SQL generation prompt template.
        
        Args:
            cot: Whether to use Chain-of-Thought prompt
        
        Returns:
            The RA+SQL prompt template
        """
        if cot:
            return self._load_template('ra_sql/cot_prompt.txt')
        # return self._load_template('ra_sql/base_prompt.txt')
        return self._load_template('ra_sql/base_prompt_v1.txt')
    
    def get_build_relalg_prompt(self) -> str:
        """
        Get the special build_relalg prompt template.
        
        Returns:
            The build_relalg prompt template
        """
        return self._load_template('special/build_relalg_prompt.txt')
    
    def format_sql_prompt(
        self,
        db_engine: str,
        db_details: str,
        question: str,
        cot: bool = False
    ) -> str:
        """
        Format SQL prompt with provided values.
        
        Args:
            db_engine: Database engine (e.g., "SQLite")
            db_details: Database schema details
            question: Natural language question
            cot: Whether to use Chain-of-Thought prompt
        
        Returns:
            Formatted SQL prompt
        """
        template = self.get_sql_prompt(cot=cot)
        # CHANGED: use safe formatter
        # return template.format(
        #     db_engine=db_engine,
        #     db_details=db_details,
        #     question=question

        return self._safe_format(
            template,
            {"db_engine": db_engine, "db_details": db_details, "question": question},
            ("db_engine", "db_details", "question"),
        )
    
    def format_ra_prompt(
        self,
        db_details: str,
        question: str,
        cot: bool = False,
        include_head: bool = True
    ) -> str:
        """
        Format RA prompt with provided values.
        
        Args:
            db_details: Database schema details
            question: Natural language question
            cot: Whether to use Chain-of-Thought prompt
            include_head: Whether to include the head section
        
        Returns:
            Formatted RA prompt
        """
        template = self.get_ra_prompt(cot=cot, include_head=include_head)
        # CHANGED: use safe formatter
        return template.format(
            db_details=db_details,
            question=question)

        # return self._safe_format(
        #     template,
        #     {"db_details": db_details, "question": question},
        #     ("db_details", "question"),
        # )
    
    def format_ra_sql_prompt(
        self,
        db_details: str,
        question: str,
        cot: bool = False
    ) -> str:
        """
        Format RA+SQL prompt with provided values.
        
        Args:
            db_details: Database schema details
            question: Natural language question
            cot: Whether to use Chain-of-Thought prompt
        
        Returns:
            Formatted RA+SQL prompt
        """
        template = self.get_ra_sql_prompt(cot=cot)
        # CHANGED: use safe formatter
        # return template.format(
        #     db_details=db_details,
        #     question=question)

        return self._safe_format(
            template,
            {"db_details": db_details, "question": question},
            ("db_details", "question"),
        )
    
    def format_build_relalg_prompt(
        self,
        schema: str,
        nl_question: str,
        gt_ra: str,
        hint: Optional[str] = None
    ) -> str:
        """
        Format build_relalg prompt with provided values.
        
        Args:
            schema: Database schema
            nl_question: Natural language question
            gt_ra: Ground truth relational algebra
            hint: Optional hint text
        
        Returns:
            Formatted build_relalg prompt
        """
        template = self.get_build_relalg_prompt()
        return template.format(
            schema=schema,
            nl_question=nl_question,
            gt_ra=gt_ra,
            hint=hint if hint else "None"
        )


# Global instance for convenient access
_default_manager = None

def get_prompt_manager(template_dir: Optional[str] = None) -> PromptManager:
    """
    Get a prompt manager instance.
    
    Args:
        template_dir: Optional path to template directory
    
    Returns:
        PromptManager instance
    """
    global _default_manager
    if _default_manager is None or template_dir is not None:
        _default_manager = PromptManager(template_dir)
    return _default_manager