#!/usr/bin/env python3
"""Utilities for the Mythos-style non-FL author sheet baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


CATEGORIES = ("structure", "creativity", "development", "language_use")
BAD_OPENINGS = (
    "as an ai",
    "i can't",
    "i cannot",
    "sure,",
    "certainly,",
    "here is",
    "here's",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def safe_id(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    return text or hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]


def get_field(rec: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name and rec.get(name) not in (None, ""):
            return rec.get(name)
    return default


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def truncate_chars(text: str, max_chars: int) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0]


def word_count(text: str) -> int:
    return len(str(text).split())


def split_prefix_continuation(text: str, prefix_words: int = 96, min_target_words: int = 32) -> tuple[str, str]:
    words = str(text).split()
    if len(words) <= prefix_words + min_target_words:
        cut = max(1, min(prefix_words, len(words) // 2))
    else:
        cut = prefix_words
    return " ".join(words[:cut]).strip(), " ".join(words[cut:]).strip()


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.I | re.S)


def _strip_reasoning_and_fences(s: str) -> str:
    s = _THINK_BLOCK_RE.sub("", s).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I).strip()
        s = re.sub(r"```\s*$", "", s).strip()
    s = re.sub(r"^(?:Output|Result|Response|JSON)\s*[:\-]\s*", "", s, flags=re.I)
    return s


def _decode_top_level_objects(s: str) -> list[Any]:
    """Walk through ``s`` and return every balanced JSON object/array via raw_decode."""
    objects: list[Any] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i] not in "{[":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(s, idx=i)
            objects.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return objects


def _deep_merge_dicts(objs: list[Any]) -> dict:
    """Merge a list of dicts; concatenate same-key lists, last-wins for scalars."""
    out: dict = {}
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            if k in out and isinstance(out[k], list) and isinstance(v, list):
                out[k] = out[k] + v
            elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


def parse_json_object(text: str) -> Any:
    """Robust JSON extraction tolerant to small-model output noise.

    Handles: ``<think>`` reasoning blocks, ```` ``` ```` code fences, leading prefixes
    like ``Output:``, multiple consecutive top-level JSON blocks, and trailing
    prose after the JSON. Multiple blocks are merged via ``_deep_merge_dicts``.
    """
    s = _strip_reasoning_and_fences(str(text).strip())
    if not s:
        raise json.JSONDecodeError("empty model output", s, 0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    objects = _decode_top_level_objects(s)
    if not objects:
        raise json.JSONDecodeError("no JSON object found in model output", s, 0)
    if len(objects) == 1:
        return objects[0]
    return _deep_merge_dicts(objects)


_CATEGORY_ALIASES = {
    "structure": "structure",
    "plot": "structure",
    "discourse": "structure",
    "organization": "structure",
    "structural": "structure",
    "creativity": "creativity",
    "creative": "creativity",
    "imagery": "creativity",
    "framing": "creativity",
    "development": "development",
    "character_development": "development",
    "argument_development": "development",
    "emotion": "development",
    "emotional_development": "development",
    "language_use": "language_use",
    "language": "language_use",
    "languageuse": "language_use",
    "diction": "language_use",
    "style": "language_use",
    "syntax": "language_use",
    "rhythm": "language_use",
}


def coerce_to_categories(
    parsed: Any,
    max_items_per_cat: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize an arbitrary parsed JSON value into the 4-category schema.

    Tolerates wrapped containers (``categories`` / ``sheet`` / ``writing_sheet``),
    case/punctuation variants of category names, and alternate field names
    (``description`` for ``claim``, ``quote`` for ``evidence``). Items missing a
    claim or evidence are dropped.
    """
    out: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    if not isinstance(parsed, dict):
        return out
    wrapper_keys = ("categories", "sheet", "writing_sheet", "author_writing_sheet", "result", "data")
    while isinstance(parsed, dict):
        inner = next((parsed.get(key) for key in wrapper_keys if isinstance(parsed.get(key), dict)), None)
        if inner is None:
            break
        parsed = inner
    for k, v in parsed.items():
        norm_key = re.sub(r"[^a-z]+", "_", str(k).lower()).strip("_")
        canonical = _CATEGORY_ALIASES.get(norm_key)
        if not canonical or not isinstance(v, list):
            continue
        for item in v:
            if not isinstance(item, dict):
                continue
            claim = item.get("claim") or item.get("description") or item.get("rule")
            evidence = item.get("evidence") or item.get("quote") or item.get("example")
            sid = item.get("source_example_id") or item.get("source_id") or item.get("source")
            if not claim or not evidence:
                continue
            out[canonical].append(
                {
                    "claim": str(claim).strip(),
                    "evidence": str(evidence).strip(),
                    "source_example_id": str(sid or "").strip(),
                }
            )
    if max_items_per_cat is not None:
        for cat in CATEGORIES:
            out[cat] = out[cat][:max_items_per_cat]
    return out


def normalize_category_keys(parsed: Any) -> dict[str, list]:
    """Map a parsed JSON dict's top-level keys to canonical category names.

    Returns ``{cat: [...]}`` for the four categories regardless of input casing,
    aliasing (``language``/``Structure``), or wrapper keys.
    """
    out: dict[str, list] = {cat: [] for cat in CATEGORIES}
    if not isinstance(parsed, dict):
        return out
    wrapper_keys = ("categories", "rules", "sheet", "writing_sheet", "author_writing_sheet", "result", "data")
    while isinstance(parsed, dict):
        inner = next((parsed.get(key) for key in wrapper_keys if isinstance(parsed.get(key), dict)), None)
        if inner is None:
            break
        parsed = inner
    for k, v in parsed.items():
        norm_key = re.sub(r"[^a-z]+", "_", str(k).lower()).strip("_")
        canonical = _CATEGORY_ALIASES.get(norm_key)
        if canonical and isinstance(v, list):
            out[canonical] = v
    return out


def merge_candidate_claims_python(
    candidates: dict[str, list[dict[str, Any]]],
    max_final_per_category: int,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic Python-side dedup: keep first item per (claim, evidence) key, truncate."""
    out: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    for cat in CATEGORIES:
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for item in candidates.get(cat, []) or []:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "") or "").strip()
            evidence = str(item.get("evidence", "") or "").strip()
            if not claim or not evidence:
                continue
            key = (
                normalize_ws(claim.lower())[:160],
                normalize_ws(evidence.lower())[:160],
            )
            if key in seen:
                continue
            seen[key] = item
        out[cat] = list(seen.values())[:max_final_per_category]
    return out


def cache_key(messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    payload = {"messages": messages, "config": config}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMBackend:
    """Small cached chat-generation facade.

    Backends:
      * ``openai_compatible``: POSTs to /chat/completions.
      * ``transformers``: local HuggingFace causal LM.
      * ``mock``: deterministic smoke-test backend; not a fair baseline.
    """

    def __init__(
        self,
        backend: str,
        model_name: str,
        cache_dir: str | Path = ".cache/mythos_sheet",
        base_url: str | None = None,
        api_key: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        cfg = {
            "backend": self.backend,
            "model": self.model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        key = cache_key(messages, cfg)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["text"]

        if self.backend == "openai_compatible":
            text = self._generate_openai_compatible(messages, temperature, top_p, max_tokens, seed)
        elif self.backend == "transformers":
            text = self._generate_transformers(messages, temperature, top_p, max_tokens, seed)
        elif self.backend == "mock":
            text = self._generate_mock(messages)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"text": text, "created_at": time.time(), "config": cfg}, f, ensure_ascii=False, indent=2)
        return text

    def _generate_openai_compatible(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("openai_compatible backend requires OPENAI_BASE_URL or --base_url")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed: {exc.code} {detail}") from exc
        return body["choices"][0]["message"]["content"].strip()

    def _load_transformers(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        self._model.eval()

    def _generate_transformers(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        import torch

        self._load_transformers()
        assert self._model is not None and self._tokenizer is not None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **kwargs)
        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _generate_mock(self, messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if "Candidate claims:" in prompt:
            try:
                return json.dumps(parse_json_object(prompt.split("Candidate claims:", 1)[1]))
            except Exception:
                pass
        if "Profile excerpts:" in prompt and "structure" in prompt:
            source_match = re.search(r"\[source_example_id=([^\]]+)\]\n(.+?)(?:\n\n\[source_example_id=|\Z)", prompt, flags=re.S)
            source_id = source_match.group(1) if source_match else ""
            source_text = normalize_ws(source_match.group(2)) if source_match else ""
            evidence = " ".join(source_text.split()[:8])
            return json.dumps(
                {
                    cat: [
                        {
                            "claim": "The author uses concrete, direct phrasing.",
                            "evidence": evidence,
                            "source_example_id": source_id,
                        }
                    ]
                    for cat in CATEGORIES
                }
            )
        if "persona" in prompt.lower():
            return (
                "You tend to write in a direct, concrete voice with compact movement from scene to reaction. "
                "You keep the continuation grounded in the immediate situation, using plain diction, casual transitions, "
                "and small reflective turns. You favor reusable stylistic cues over elaborate exposition."
            )
        prefix_match = re.search(r"Held-out prefix:\n(.+?)\n\nContinuation:", prompt, flags=re.S)
        prefix = prefix_match.group(1).strip() if prefix_match else ""
        if prefix_match:
            tail = " ".join(prefix.split()[-18:])
            return f"{tail} The thought keeps moving forward in the same plain, personal rhythm, with one more concrete detail."
        if "rules" in prompt.lower():
            return json.dumps({cat: ["Continue from the given prefix using direct, concrete prose."] for cat in CATEGORIES})
        return "The continuation moves forward in a plain, concrete voice with a small reflective turn."


EVIDENCE_TIERS = ("exact", "whitespace", "case", "punct", "ngram5")


def _strip_punct(t: str) -> str:
    return normalize_ws(re.sub(r"[\W_]+", " ", str(t)).lower())


def evidence_match_tier(evidence: str, source_text: str) -> str | None:
    """Return the loosest tier at which ``evidence`` matches ``source_text``, else None.

    Tiers (strict to loose): ``exact`` → ``whitespace`` → ``case`` → ``punct`` → ``ngram5``.
    ``ngram5`` requires any 5 consecutive content words from the evidence to appear
    in the source after lowercasing and punctuation stripping.
    """
    if not evidence or not source_text:
        return None
    if evidence in source_text:
        return "exact"
    e_n, s_n = normalize_ws(evidence), normalize_ws(source_text)
    if e_n and e_n in s_n:
        return "whitespace"
    e_l, s_l = e_n.lower(), s_n.lower()
    if e_l and e_l in s_l:
        return "case"
    e_p, s_p = _strip_punct(evidence), _strip_punct(source_text)
    if e_p and e_p in s_p:
        return "punct"
    e_words = e_p.split()
    if len(e_words) >= 5:
        for i in range(len(e_words) - 4):
            ngram = " ".join(e_words[i : i + 5])
            if ngram in s_p:
                return "ngram5"
    return None


def validate_evidence(
    sheet: dict[str, Any],
    profile_lookup: dict[str, str],
    keep_invalid: bool = False,
    strict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate sheet evidence with tiered substring matching.

    ``sheet`` may be either a full sheet with a ``categories`` key or a bare
    categories dict; the returned object preserves the input shape.

    By default any non-None tier counts as valid, which forgives the small-model
    habit of altering punctuation, capitalization, or whitespace when copying
    evidence. Pass ``strict=True`` to require an exact substring.

    Each retained item gains ``evidence_valid`` and ``evidence_match_tier`` fields.
    The stats dict carries per-tier counts and a cross-source rescue counter for
    cases where the model misattributes ``source_example_id``.
    """
    tier_counts = {tier: 0 for tier in EVIDENCE_TIERS}
    stats: dict[str, Any] = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "tier_counts": tier_counts,
        "source_attribution_corrected": 0,
    }
    wrapped = "categories" in sheet
    categories = sheet.get("categories", sheet)
    cleaned = {cat: [] for cat in CATEGORIES}
    for cat in CATEGORIES:
        for raw_item in categories.get(cat, []) or []:
            item = dict(raw_item)
            stats["total"] += 1
            evidence = str(item.get("evidence", "") or "")
            sid = str(item.get("source_example_id", "") or "")
            source_text = profile_lookup.get(sid, "")
            tier = evidence_match_tier(evidence, source_text)
            if not tier:
                # Cross-source rescue: small models sometimes misattribute the source_example_id.
                for alt_sid, alt_text in profile_lookup.items():
                    if alt_sid == sid:
                        continue
                    alt_tier = evidence_match_tier(evidence, alt_text)
                    if alt_tier:
                        tier = alt_tier
                        item["source_example_id"] = alt_sid
                        item["source_attribution_corrected"] = True
                        stats["source_attribution_corrected"] += 1
                        break
            valid = (tier == "exact") if strict else (tier is not None)
            item["evidence_valid"] = valid
            item["evidence_match_tier"] = tier
            if tier:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if valid:
                stats["valid"] += 1
                cleaned[cat].append(item)
            else:
                stats["invalid"] += 1
                if keep_invalid:
                    cleaned[cat].append(item)
    if wrapped:
        result = dict(sheet)
        result["categories"] = cleaned
        return result, stats
    return cleaned, stats


def tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


class SimpleBM25:
    def __init__(self, docs: list[dict[str, Any]], text_key: str = "text") -> None:
        self.docs = docs
        self.text_key = text_key
        self.tokens = [tokenize_for_retrieval(str(doc.get(text_key, ""))) for doc in docs]
        self.avgdl = sum(len(toks) for toks in self.tokens) / max(len(self.tokens), 1)
        df = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        n = max(len(self.tokens), 1)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def topk(self, query: str, k: int = 1, exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
        exclude_ids = exclude_ids or set()
        q_terms = tokenize_for_retrieval(query)
        scored: list[tuple[float, int]] = []
        for idx, toks in enumerate(self.tokens):
            doc = self.docs[idx]
            doc_id = str(doc.get("example_id") or doc.get("doc_id") or "")
            if doc_id in exclude_ids:
                continue
            tf = Counter(toks)
            dl = len(toks) or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                denom = tf[term] + 1.5 * (1 - 0.75 + 0.75 * dl / max(self.avgdl, 1e-6))
                score += self.idf.get(term, 0.0) * (tf[term] * 2.5 / denom)
            scored.append((score, idx))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self.docs[idx] for score, idx in scored[:k] if score > 0]


def postprocess_generation(text: str, prefix: str = "") -> str:
    text = str(text).strip()
    text = re.sub(r"^\s*(Continuation|Output)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*(Sure|Certainly),?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*Here(?: is|'s)\b[^:\n]*:?\s*", "", text, flags=re.I)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    norm_prefix = normalize_ws(prefix)
    norm_text = normalize_ws(text)
    if norm_prefix and norm_text.startswith(norm_prefix):
        text = norm_text[len(norm_prefix) :].strip()
    return text.strip()


def has_bad_opening(text: str) -> bool:
    lower = text.lstrip().lower()
    return any(lower.startswith(opening) for opening in BAD_OPENINGS)
