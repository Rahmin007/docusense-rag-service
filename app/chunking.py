from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    result = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            result.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return result


def _merge_pieces(pieces: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        proposed_len = current_len + (sep_len if current else 0) + len(piece)
        if current and proposed_len > chunk_size:
            chunk = separator.join(current).strip()
            if chunk:
                chunks.append(chunk)

            # Retain the smallest possible suffix of the previous chunk
            # that stays within the configured overlap.
            suffix: list[str] = []
            suffix_len = 0
            for previous in reversed(current):
                extra = len(previous) + (sep_len if suffix else 0)
                if suffix_len + extra <= overlap:
                    suffix.append(previous)
                    suffix_len += extra
                else:
                    break
            suffix.reverse()
            current = suffix
            current_len = len(separator.join(current)) if current else 0

            # If even the overlap plus the new piece is too large, keep the
            # overlap and let the oversized piece be split by the caller.
            proposed_len = current_len + (sep_len if current else 0) + len(piece)
            if proposed_len > chunk_size and len(piece) > chunk_size:
                if current:
                    chunks.append(separator.join(current).strip())
                current = []
                current_len = 0

        current.append(piece)
        current_len = len(separator.join(current))

    if current:
        final = separator.join(current).strip()
        if final and (not chunks or final != chunks[-1]):
            chunks.append(final)
    return chunks


def recursive_split(text: str, chunk_size: int, overlap: int, separators: list[str] | None = None) -> list[str]:
    """Deterministic recursive character splitter with fixed overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    separator_index = None
    for i, separator in enumerate(separators):
        if separator == "" or separator in text:
            separator_index = i
            break
    if separator_index is None:
        return _hard_split(text, chunk_size, overlap)

    separator = separators[separator_index]
    if separator == "":
        return _hard_split(text, chunk_size, overlap)

    raw_parts = [part.strip() for part in text.split(separator) if part.strip()]
    pieces: list[str] = []
    next_separators = separators[separator_index + 1 :]
    for part in raw_parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        elif next_separators:
            pieces.extend(recursive_split(part, chunk_size, overlap, next_separators))
        else:
            pieces.extend(_hard_split(part, chunk_size, overlap))

    merged = _merge_pieces(pieces, separator, chunk_size, overlap)
    # Safety check: no chunk may exceed the contract.
    result: list[str] = []
    for item in merged:
        if len(item) <= chunk_size:
            result.append(item)
        else:
            result.extend(_hard_split(item, chunk_size, overlap))
    return result


def chunk_documents(documents: list[tuple[str, str]], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 1
    for source_file, text in sorted(documents, key=lambda item: item[0].lower()):
        for piece in recursive_split(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{counter:03d}",
                    text=piece,
                    source_file=source_file,
                )
            )
            counter += 1
    return chunks
