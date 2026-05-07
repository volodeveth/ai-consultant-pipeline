import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: int
    section: str
    text: str
    tokens: list[str]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def load_and_index(path: Path) -> list[Chunk]:
    content = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    chunk_id = 0

    for section_block in re.split(r"\n(?=## )", content):
        lines = section_block.strip().splitlines()
        if not lines:
            continue
        header = re.match(r"^##\s+(.+)", lines[0])
        if not header:
            continue
        section_name = header.group(1).strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        for sentence in _split_sentences(body):
            chunks.append(
                Chunk(
                    id=chunk_id,
                    section=section_name,
                    text=sentence,
                    tokens=_tokenize(sentence),
                )
            )
            chunk_id += 1

    return chunks
