"""
Parser interface + registry.

On CPU (default): CPUParser (pymupdf4llm + pdfplumber)
On GPU:           DoclingParser
Hosted:           LlamaParseParser
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from app.schema import Page


class Parser(ABC):
    name: str = "base"

    @abstractmethod
    def parse(self, pages: list[Page]) -> list[Page]:
        """Populate .blocks on each page in-place."""


def build_parser(name: str, **kwargs) -> Parser:
    if name == "cpu":
        from parsing.cpu_parser import CPUParser
        return CPUParser(**kwargs)
    if name == "docling":
        from parsing.docling_parser import DoclingParser
        return DoclingParser()
    if name == "llamaparse":
        from parsing.llamaparse_parser import LlamaParseParser
        return LlamaParseParser()
    raise ValueError(f"Unknown parser: {name}")
