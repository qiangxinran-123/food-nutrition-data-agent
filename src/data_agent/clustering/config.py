from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClusterConfig(BaseModel):
    """Recorded intent only; no preprocessing or algorithm is executed."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dataset_id: str = Field(min_length=1)
    dataset_version: Literal["1", "2"]
    data_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    profile_path: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    row_id_column: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    sample_filter: dict[str, Any] = Field(default_factory=dict)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    algorithm: str = Field(min_length=1)
    algorithm_params: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")

    @model_validator(mode="after")
    def validate_features(self):
        if any(not name.strip() for name in self.features) or len(set(self.features)) != len(self.features):
            raise ValueError("feature names must be nonblank and unique")
        if self.row_id_column in self.features:
            raise ValueError("row identifier cannot be a feature")
        return self
