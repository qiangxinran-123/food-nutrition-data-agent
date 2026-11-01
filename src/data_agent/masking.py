from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas.api.types as ptypes

from .contracts import ColumnSpec, SchemaProfile


@dataclass(frozen=True)
class MissingValueMask:
    total: pd.Series
    null: pd.Series
    empty_string: pd.Series
    declared_token: pd.Series
    token_counts: dict[str, int]


def as_bool_mask(mask: pd.Series, index: pd.Index) -> pd.Series:
    return mask.reindex(index, fill_value=False).fillna(False).astype(bool)


def missing_policy_from_schema(schema: SchemaProfile) -> dict[str, bool]:
    policy = schema.metadata.get("missing_policy", {})
    return {
        "treat_empty_string_as_missing": bool(
            policy.get("treat_empty_string_as_missing", True)
        ),
        "trim_whitespace": bool(policy.get("trim_whitespace", True)),
        "case_sensitive": bool(policy.get("case_sensitive", False)),
    }


def build_missing_mask(
    series: pd.Series,
    spec: ColumnSpec,
    *,
    treat_empty_string_as_missing: bool = True,
    trim_whitespace: bool = True,
    case_sensitive: bool = False,
) -> MissingValueMask:
    index = series.index
    null_mask = as_bool_mask(series.isna(), index)

    empty_mask = pd.Series(False, index=index, dtype=bool)
    if treat_empty_string_as_missing and (
        ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series)
    ):
        strings = series.astype("string")
        if trim_whitespace:
            strings = strings.str.strip()
        empty_mask = as_bool_mask(strings.eq(""), index)

    token_mask = pd.Series(False, index=index, dtype=bool)
    token_counts: dict[str, int] = {}
    if spec.missing_tokens:
        strings = series.astype("string")
        if trim_whitespace:
            strings = strings.str.strip()
        if not case_sensitive:
            strings = strings.str.casefold()
        for token in spec.missing_tokens:
            comparable = token.strip()
            if not case_sensitive:
                comparable = comparable.casefold()
            current = as_bool_mask(strings.eq(comparable), index)
            token_counts[token] = int(current.sum())
            token_mask = token_mask | current

    return MissingValueMask(
        total=as_bool_mask(null_mask | empty_mask | token_mask, index),
        null=null_mask,
        empty_string=empty_mask,
        declared_token=token_mask,
        token_counts=token_counts,
    )


__all__ = [
    "MissingValueMask",
    "as_bool_mask",
    "build_missing_mask",
    "missing_policy_from_schema",
]
