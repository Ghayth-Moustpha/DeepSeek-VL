"""Prompt builder utilities for DeepDrive-VL research experiments.

Centralizes prompt templates and few-shot assembly for consistent prompts
across experiments and papers.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PromptTemplate:
    instruction: str
    context: Optional[str] = None
    examples: List[str] = field(default_factory=list)

    def build(self) -> str:
        parts = [self.instruction.strip()]
        if self.context:
            parts.append("Context:\n" + self.context.strip())
        if self.examples:
            parts.append("Examples:\n" + "\n---\n".join(e.strip() for e in self.examples))
        return "\n\n".join(parts)


class PromptBuilder:
    """Helper to construct prompts from template parts and parameters."""

    def __init__(self, base_instruction: Optional[str] = None):
        self.base_instruction = base_instruction or "Answer the query based on the image and context. Be concise."

    def create(self, instruction: Optional[str] = None, context: Optional[str] = None, examples: Optional[List[str]] = None) -> PromptTemplate:
        instr = instruction or self.base_instruction
        return PromptTemplate(instruction=instr, context=context, examples=examples or [])

    def build(self, instruction: Optional[str] = None, context: Optional[str] = None, examples: Optional[List[str]] = None) -> str:
        tpl = self.create(instruction=instruction, context=context, examples=examples)
        return tpl.build()
