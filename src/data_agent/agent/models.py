"""Versioned model and prompt configuration for the food-nutrition agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Pricing(BaseModel):
    """Published price snapshot used only to report cost, never to guess usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    currency: str = "USD"
    unit: str = "per_1m_tokens"
    prompt_cache_hit: float = Field(default=0.0, ge=0)
    prompt_cache_miss: float = Field(default=0.0, ge=0)
    completion: float = Field(default=0.0, ge=0)
    basis: str = "peak"
    source: str = ""


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    thinking: Literal["enabled", "disabled"] = "disabled"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: int = Field(default=120, ge=1)
    max_retries: int = Field(default=2, ge=0, le=5)
    documented_version: str = ""
    pricing: Pricing | None = None
    notes: str = ""

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ModelKeyMissing(
                f"environment variable {self.api_key_env} is not set; "
                "export it before running the agent"
            )
        return value

    def public_dict(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "thinking": self.thinking,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "documented_version": self.documented_version,
            "pricing": self.pricing.model_dump() if self.pricing else None,
            "notes": self.notes,
        }


class ModelKeyMissing(RuntimeError):
    pass


class ModelCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    default_profile: str
    profiles: dict[str, ModelProfile]

    @model_validator(mode="after")
    def validate_default(self) -> "ModelCatalog":
        if self.default_profile not in self.profiles:
            raise ValueError("default_profile is not defined in profiles")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ModelCatalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def profile(self, name: str | None = None) -> ModelProfile:
        return self.profiles[name or self.default_profile]


def project_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    config_dir = root / "config" / "agent"
    return {
        "root": root,
        "models": config_dir / "models.json",
        "prompts": config_dir / "prompts",
        "profiles": root / "config" / "profiles",
        "data": root / "data" / "processed",
        "outputs": root / "outputs" / "agent_runs",
    }


__all__ = ["ModelCatalog", "ModelKeyMissing", "ModelProfile", "Pricing", "project_paths"]
