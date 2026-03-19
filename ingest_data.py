#!/usr/bin/env python3
"""
Ingest Nepali song lyrics from data/sabda into the embedding API.

Folder structure:
  artist_name/song_name.ext             (direct song files)
  artist_name/album_name/song_name.ext  (songs inside album subfolder)

Supported file types: .txt, .lrc, .tra, .Lyric, or no extension.
LRC files are cleaned of timestamps, metadata headers, and translation lines.
"""

from __future__ import annotations

import os
import re
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SABDA_DIR = PROJECT_ROOT / "data" / "sabda"
API_BASE_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8000")

SKIP_FILES = {"README.org"}

# LRC metadata header pattern: [key:value]
LRC_METADATA_RE = re.compile(
    r"^\s*\["
    r"(?:ti|ar|al|au|la|by|re|ve|youtube|offset|length|tsl-en)"
    r"[:\s].*\]\s*$",
    re.IGNORECASE,
)

# LRC timestamp pattern: [mm:ss.xx] or [mm:ss]
LRC_TIMESTAMP_RE = re.compile(r"\[\d{1,3}[:.]\d{2}(?:[:.]\d{1,3})?\]\s*")

# Lines that are purely a timestamp with no text after
LRC_EMPTY_TIMESTAMP_RE = re.compile(r"^\s*\[\d{1,3}[:.]\d{2}(?:[:.]\d{1,3})?\]\s*$")

# Translation lines: [tsl-en] ...
LRC_TRANSLATION_LINE_RE = re.compile(r"^\s*\[tsl-\w+\]", re.IGNORECASE)

# synced by / credit lines
LRC_CREDIT_RE = re.compile(r"^\s*\[\d{1,3}[:.]\d{2}.*\]\s*synced\s+by", re.IGNORECASE)


MAX_WORDS_PER_CHUNK = 50


def chunk_lyrics(lyrics: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """Split cleaned lyrics into chunks of at most max_words words each."""
    words = lyrics.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def clean_lyrics(raw_text: str, file_ext: str) -> str:
    """Clean raw lyrics text, stripping LRC timestamps, metadata, and translations."""
    lines = raw_text.splitlines()
    cleaned = []

    for line in lines:
        # Skip LRC metadata headers
        if LRC_METADATA_RE.match(line):
            continue
        # Skip pure translation lines
        if LRC_TRANSLATION_LINE_RE.match(line):
            continue
        # Skip credit/sync lines
        if LRC_CREDIT_RE.match(line):
            continue
        # Skip lines that are only a timestamp with no lyrics
        if LRC_EMPTY_TIMESTAMP_RE.match(line):
            continue
        # Strip inline timestamps
        line = LRC_TIMESTAMP_RE.sub("", line)
        # Strip youtube metadata lines like [youtube: ...]
        if re.match(r"^\s*\[youtube\s*:", line, re.IGNORECASE):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned).strip()
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_song_title(filepath: Path) -> str:
    """Extract song title from filename by removing the extension."""
    return filepath.stem if filepath.suffix else filepath.name


def is_song_file(path: Path) -> bool:
    """Check if a path looks like a song lyrics file (not a directory, not a skip file)."""
    if not path.is_file():
        return False
    if path.name in SKIP_FILES:
        return False
    return True


def discover_songs(sabda_dir: Path) -> list[dict]:
    """
    Walk the sabda directory and discover all songs.

    Returns a list of dicts with keys: artist, title, album (optional), filepath.
    """
    songs = []

    for artist_dir in sorted(sabda_dir.iterdir()):
        if not artist_dir.is_dir():
            continue
        # Skip hidden directories (.git, etc.)
        if artist_dir.name.startswith("."):
            continue

        artist_name = artist_dir.name

        for entry in sorted(artist_dir.iterdir()):
            if entry.is_file() and entry.name not in SKIP_FILES:
                # Direct song: artist/song_name.ext
                songs.append({
                    "artist": artist_name,
                    "title": extract_song_title(entry),
                    "album": None,
                    "filepath": entry,
                })
            elif entry.is_dir():
                # Album subfolder: artist/album/song_name.ext
                album_name = entry.name
                for song_file in sorted(entry.iterdir()):
                    if is_song_file(song_file):
                        songs.append({
                            "artist": artist_name,
                            "title": extract_song_title(song_file),
                            "album": album_name,
                            "filepath": song_file,
                        })

    return songs


def ingest_song(session: requests.Session, song: dict, ingest_url: str) -> int:
    """Read lyrics from file, chunk them, and POST each chunk to the ingest API.

    Returns the number of successfully ingested chunks.
    """
    filepath: Path = song["filepath"]

    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = filepath.read_text(encoding="latin-1")
        except Exception as exc:
            logger.error("Failed to read %s: %s", filepath, exc)
            return 0

    lyrics = clean_lyrics(raw_text, filepath.suffix)

    if not lyrics.strip():
        logger.warning("Empty lyrics after cleaning: %s", filepath)
        return 0

    chunks = chunk_lyrics(lyrics)
    if not chunks:
        logger.warning("No chunks produced for: %s", filepath)
        return 0

    ingested = 0
    for chunk in chunks:
        payload = {
            "title": song["title"],
            "artist": song["artist"],
            "lyrics": chunk,
        }
        try:
            resp = session.post(ingest_url, json=payload, timeout=120)
            resp.raise_for_status()
            ingested += 1
        except requests.RequestException as exc:
            logger.error(
                "API error for %s - %s (chunk): %s",
                song["artist"],
                song["title"],
                exc,
            )

    return ingested


def main():
    parser = argparse.ArgumentParser(description="Ingest sabda lyrics into embedding API")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SABDA_DIR,
        help="Path to the sabda data directory",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help="Base URL of the embedding API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List songs without sending to API",
    )
    args = parser.parse_args()

    ingest_url = f"{args.api_url}/ingest"

    sabda_dir = args.data_dir
    if not sabda_dir.is_dir():
        logger.error("Data directory not found: %s", sabda_dir)
        sys.exit(1)

    logger.info("Discovering songs in %s", sabda_dir)
    songs = discover_songs(sabda_dir)
    logger.info("Found %d songs", len(songs))

    if args.dry_run:
        for i, song in enumerate(songs, 1):
            album_info = f" [{song['album']}]" if song["album"] else ""
            logger.info(
                "  %3d. %s - %s%s  (%s)",
                i,
                song["artist"],
                song["title"],
                album_info,
                song["filepath"].suffix or "no-ext",
            )
        logger.info("Dry run complete. %d songs found.", len(songs))
        return

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "X-API-Key": os.getenv("API_KEY", ""),
    })

    songs_ok = 0
    songs_fail = 0
    total_chunks = 0

    for i, song in enumerate(songs, 1):
        album_info = f" [{song['album']}]" if song["album"] else ""
        logger.info(
            "[%d/%d] Ingesting: %s - %s%s",
            i,
            len(songs),
            song["artist"],
            song["title"],
            album_info,
        )

        chunks_ingested = ingest_song(session, song, ingest_url)
        if chunks_ingested > 0:
            logger.info("  ✓ %d chunks ingested", chunks_ingested)
            songs_ok += 1
            total_chunks += chunks_ingested
        else:
            songs_fail += 1

    logger.info(
        "Done! %d songs succeeded (%d total chunks), %d songs failed out of %d total.",
        songs_ok,
        total_chunks,
        songs_fail,
        len(songs),
    )


if __name__ == "__main__":
    main()
