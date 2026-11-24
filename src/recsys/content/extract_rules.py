from __future__ import annotations

from ..text_utils import fold


def gazetteer_extract(
    title: str, text: str, known_names: list[str], max_words: int = 4
) -> list[str]:
    surfaces: dict[str, str] = {}
    for name in known_names:
        surfaces.setdefault(fold(name), name)
    ordered = dict(
        sorted(surfaces.items(), key=lambda kv: kv[0].count(" "), reverse=True)
    )

    words = fold(f"{title} {title} {text}").split()
    found: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(words):
        match = None
        for size in range(min(max_words, len(words) - i), 0, -1):
            surface = " ".join(words[i : i + size])
            canonical = ordered.get(surface)
            if canonical:
                match = (size, canonical)
                break
        if match:
            size, canonical = match
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
            i += size
        else:
            i += 1
    return found
