"""JSON schemas for agent outputs. Passed to the CLI (`--json-schema`) and re-validated in Python."""
from __future__ import annotations

META = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis", "mutations", "candidates"],
    "properties": {
        "analysis": {"type": "string", "maxLength": 2000},
        "mutations": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "name"],
                "properties": {
                    "op": {"enum": ["add_dimension", "add_values", "prune_values", "prune_dimension"]},
                    "name": {"type": "string"},
                    "spec": {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["categorical", "integer", "float"]},
                            "values": {"type": "array"},
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "description": {"type": "string"},
                        },
                        "required": ["type", "description"],
                    },
                    "values": {"type": "array"},
                    "reason": {"type": "string"},
                },
            },
        },
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["next_coordinate", "hypothesis"],
                "properties": {
                    "next_coordinate": {"type": "object"},
                    "hypothesis": {"type": "string", "maxLength": 1000},
                },
            },
        },
    },
}

WORKER = {
    "type": "object",
    "additionalProperties": False,
    "required": ["model_py", "design_notes"],
    "properties": {
        "model_py": {"type": "string", "minLength": 200, "maxLength": 60000},
        "design_notes": {"type": "string", "maxLength": 2500},
    },
}

JUDGE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "feedback", "next_steps", "strengths", "weaknesses"],
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "feedback": {"type": "string", "minLength": 20, "maxLength": 2500},
        "next_steps": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string", "maxLength": 400}},
        "strengths": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 300}},
        "weaknesses": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 300}},
    },
}

DOC_DRAFT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["iteration_section", "abstract", "results_narrative"],
    "properties": {
        "iteration_section": {"type": "string", "minLength": 200, "maxLength": 8000},
        "abstract": {"type": "string", "minLength": 200, "maxLength": 3000},
        "results_narrative": {"type": "string", "minLength": 100, "maxLength": 5000},
    },
}

DOC_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["formatted_markdown"],
    "properties": {"formatted_markdown": {"type": "string", "minLength": 100, "maxLength": 10000}},
}

DOC_FINAL = {
    "type": "object",
    "additionalProperties": False,
    "required": ["abstract", "results_narrative", "discussion", "conclusion"],
    "properties": {
        "abstract": {"type": "string", "minLength": 200, "maxLength": 3000},
        "results_narrative": {"type": "string", "minLength": 100, "maxLength": 6000},
        "discussion": {"type": "string", "minLength": 300, "maxLength": 9000},
        "conclusion": {"type": "string", "minLength": 150, "maxLength": 4000},
    },
}

BY_NAME = {
    ("meta", "primary"): META,
    ("worker", "primary"): WORKER,
    ("judge", "primary"): JUDGE,
    ("doc", "draft"): DOC_DRAFT,
    ("doc", "format"): DOC_FORMAT,
    ("doc", "final"): DOC_FINAL,
}
