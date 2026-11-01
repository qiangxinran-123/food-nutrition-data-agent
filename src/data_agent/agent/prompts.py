"""Prompt loading with content hashes for audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

from data_agent.io import sha256_file


@dataclass(frozen=True)
class PromptSet:
    system_version: str
    system_text: str
    system_sha256: str
    task_file: str
    task_text: str
    task_sha256: str

    def render_system(self, **values: object) -> str:
        return Template(self.system_text).safe_substitute(values)


def load_prompts(prompt_dir: str | Path, *, version: str = "v1", task_file: str = "task_v1.md") -> PromptSet:
    directory = Path(prompt_dir)
    system_path = directory / f"system_{version}.md"
    task_path = directory / task_file
    if not system_path.is_file():
        raise FileNotFoundError(f"system prompt not found: {system_path}")
    if not task_path.is_file():
        raise FileNotFoundError(f"task prompt not found: {task_path}")
    return PromptSet(
        system_version=version,
        system_text=system_path.read_text(encoding="utf-8"),
        system_sha256=sha256_file(system_path),
        task_file=task_path.name,
        task_text=task_path.read_text(encoding="utf-8"),
        task_sha256=sha256_file(task_path),
    )


__all__ = ["PromptSet", "load_prompts"]
