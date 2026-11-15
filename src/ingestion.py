"""Discover and fetch audio files from local, cloud and Google Drive sources.

Every source is exposed through the same small interface:

    source = open_source("gs://chirp-public-bucket/soundscapes/powdermill/")
    for item in source.list_audio():
        local_path = source.fetch(item)   # always a readable local file

Local files are read in place. Cloud objects are downloaded lazily, one at a
time, into a cache directory, so a large bucket is never copied up front.
Google Drive folders are downloaded once (``gdown`` has no listing API for
public folders) and then treated as a local folder.

Cloud and Drive support are optional: ``fsspec`` + ``s3fs``/``gcsfs`` and
``gdown`` are imported only when such a source is used.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Protocol

AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac")

DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "bioacoustic_cache"

_DRIVE_FOLDER_RE = re.compile(r"drive\.google\.com/.*?/folders/([A-Za-z0-9_-]+)")
_DRIVE_FILE_RE = re.compile(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([A-Za-z0-9_-]+)")
_DRIVE_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{25,}$")


class IngestionError(RuntimeError):
    """Raised when a source cannot be opened, listed or fetched."""


class SourceType(str, Enum):
    """Supported input source types."""

    LOCAL = "local"
    CLOUD = "cloud"
    GDRIVE = "gdrive"
    AUTO = "auto"


@dataclass(frozen=True)
class AudioItem:
    """One audio file discovered in a source.

    Attributes:
        filename: Base name of the file, e.g. ``Recording_2_Segment_01.wav``.
        uri: Full location in the original source (local path or ``s3://`` URI).
        relative_path: Path relative to the source root, using ``/`` separators.
            This is what the tutorial's database stores as a recording filename.
    """

    filename: str
    uri: str
    relative_path: str


class AudioSource(Protocol):
    """Interface shared by all audio sources."""

    root: str

    def list_audio(self) -> list[AudioItem]:
        """Return every ``.wav``/``.flac`` file under the source, sorted."""
        ...

    def fetch(self, item: AudioItem) -> Path:
        """Return a local path from which ``item`` can be read."""
        ...


def is_audio_file(name: str) -> bool:
    """Return True if ``name`` has a supported audio extension (case-insensitive)."""
    return name.lower().endswith(AUDIO_EXTENSIONS)


# --------------------------------------------------------------------------- #
# Local
# --------------------------------------------------------------------------- #
class LocalSource:
    """Audio files in a local directory (scanned recursively) or a single file."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        path = Path(root).expanduser()
        if not path.exists():
            raise IngestionError(f"Local path does not exist: {path}")
        self.root = str(path.resolve())

    def list_audio(self) -> list[AudioItem]:
        root = Path(self.root)
        if root.is_file():
            if not is_audio_file(root.name):
                raise IngestionError(f"Not a .wav/.flac file: {root}")
            return [AudioItem(root.name, str(root), root.name)]
        # os.walk rather than Path.rglob: rglob skips symlinked folders before
        # Python 3.13, and field data is often linked in from mounted drives.
        items = []
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for name in files:
                if is_audio_file(name) and not name.startswith("._"):
                    p = Path(dirpath) / name
                    items.append(AudioItem(name, str(p), p.relative_to(root).as_posix()))
        return sorted(items, key=lambda i: i.relative_path)

    def fetch(self, item: AudioItem) -> Path:
        return Path(item.uri)


# --------------------------------------------------------------------------- #
# Cloud object storage (S3 / GCS) via fsspec
# --------------------------------------------------------------------------- #
class CloudSource:
    """Audio objects under an ``s3://``, ``gs://`` or ``gcs://`` prefix.

    Credentials follow the usual provider conventions (``AWS_PROFILE``,
    ``GOOGLE_APPLICATION_CREDENTIALS``, ...). Public buckets are tried
    anonymously if an authenticated listing fails.
    """

    SCHEMES: tuple[str, ...] = ("s3", "gs", "gcs")

    def __init__(self, uri: str, cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> None:
        try:
            import fsspec  # noqa: F401  (imported lazily: optional dependency)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise IngestionError(
                "Cloud sources need fsspec plus s3fs (S3) or gcsfs (GCS): "
                "pip install fsspec s3fs gcsfs"
            ) from exc
        scheme = uri.split("://", 1)[0].lower()
        if scheme not in self.SCHEMES:
            raise IngestionError(f"Unsupported cloud URI scheme {scheme!r} in {uri!r}")
        self.scheme = scheme
        self.root = uri.rstrip("/")
        self.cache_dir = Path(cache_dir) / "cloud"
        self._fs = None

    def _filesystem(self, anon: bool = False):  # noqa: ANN202 - fsspec type is dynamic
        import fsspec

        try:
            return fsspec.filesystem(self.scheme, anon=anon) if anon else fsspec.filesystem(self.scheme)
        except ImportError as exc:
            pkg = "s3fs" if self.scheme == "s3" else "gcsfs"
            raise IngestionError(f"{self.scheme}:// URIs need the {pkg} package: pip install {pkg}") from exc

    def list_audio(self) -> list[AudioItem]:
        prefix = self.root.split("://", 1)[1]
        last_error: Exception | None = None
        for anon in (False, True):
            try:
                fs = self._filesystem(anon=anon)
                paths = fs.find(prefix)
                self._fs = fs
                break
            except IngestionError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                last_error = exc
        else:
            raise IngestionError(f"Could not list {self.root}: {last_error}") from last_error

        items = []
        for p in paths:
            if not is_audio_file(p):
                continue
            rel = p[len(prefix):].lstrip("/") or p.rsplit("/", 1)[-1]
            items.append(AudioItem(p.rsplit("/", 1)[-1], f"{self.scheme}://{p}", rel))
        return sorted(items, key=lambda i: i.relative_path)

    def fetch(self, item: AudioItem) -> Path:
        digest = hashlib.sha1(item.uri.encode()).hexdigest()[:12]
        dest = self.cache_dir / digest / item.filename
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        fs = self._fs or self._filesystem()
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            fs.get(item.uri.split("://", 1)[1], str(tmp))
            tmp.replace(dest)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise IngestionError(f"Failed to download {item.uri}: {exc}") from exc
        return dest


# --------------------------------------------------------------------------- #
# Google Drive via gdown
# --------------------------------------------------------------------------- #
def parse_drive_id(link_or_id: str) -> tuple[str, bool]:
    """Extract a Drive ID from a share link or bare ID.

    Returns:
        ``(drive_id, is_folder)``. Bare IDs are assumed to be folders.

    Raises:
        IngestionError: If no Drive ID can be found.
    """
    text = link_or_id.strip()
    if m := _DRIVE_FOLDER_RE.search(text):
        return m.group(1), True
    if m := _DRIVE_FILE_RE.search(text):
        return m.group(1), False
    if _DRIVE_BARE_ID_RE.match(text):
        return text, True
    raise IngestionError(f"Not a recognisable Google Drive link or ID: {link_or_id!r}")


class GoogleDriveSource:
    """A publicly shared ("anyone with the link") Google Drive folder or file.

    The content is downloaded once into the cache and then read locally. Note
    that ``gdown`` lists at most 50 files per Drive folder; for larger
    collections use nested sub-folders or a cloud bucket instead.
    """

    def __init__(self, link_or_id: str, cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> None:
        self.drive_id, self.is_folder = parse_drive_id(link_or_id)
        self.root = link_or_id.strip()
        self.cache_dir = Path(cache_dir) / "gdrive" / self.drive_id
        self._local: LocalSource | None = None

    def _download(self) -> LocalSource:
        if self._local is not None:
            return self._local
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise IngestionError("Google Drive sources need gdown: pip install gdown") from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.is_folder:
                result = gdown.download_folder(
                    id=self.drive_id, output=str(self.cache_dir), quiet=True, remaining_ok=True
                )
            else:
                result = gdown.download(id=self.drive_id, output=str(self.cache_dir) + os.sep, quiet=True)
        except Exception as exc:  # noqa: BLE001 - gdown raises many exception types
            raise IngestionError(
                f"Google Drive download failed for {self.root!r}: {exc}. "
                "Check the item is shared as 'Anyone with the link'."
            ) from exc
        if not result:
            raise IngestionError(
                f"Nothing was downloaded from {self.root!r}. Check the link and its sharing settings."
            )
        self._local = LocalSource(self.cache_dir)
        return self._local

    def list_audio(self) -> list[AudioItem]:
        return self._download().list_audio()

    def fetch(self, item: AudioItem) -> Path:
        return Path(item.uri)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def detect_source_type(uri: str) -> SourceType:
    """Infer the source type of ``uri``.

    Rules: ``s3://``/``gs://``/``gcs://`` -> cloud; a ``drive.google.com`` link
    -> Drive; an existing local path -> local; a long bare ID -> Drive;
    anything else -> local (which then fails with a clear "does not exist").
    """
    text = uri.strip()
    scheme = text.split("://", 1)[0].lower() if "://" in text else ""
    if scheme in CloudSource.SCHEMES:
        return SourceType.CLOUD
    if "drive.google.com" in text:
        return SourceType.GDRIVE
    if Path(text).expanduser().exists():
        return SourceType.LOCAL
    if _DRIVE_BARE_ID_RE.match(text):
        return SourceType.GDRIVE
    return SourceType.LOCAL


def open_source(
    uri: str,
    source_type: SourceType | str = SourceType.AUTO,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
) -> AudioSource:
    """Open ``uri`` as an :class:`AudioSource`.

    Args:
        uri: Local path, ``s3://``/``gs://`` URI, or Google Drive link/ID.
        source_type: One of :class:`SourceType` (or its string value). ``auto``
            infers the type with :func:`detect_source_type`.
        cache_dir: Where remote files are downloaded to.

    Raises:
        IngestionError: If the source is invalid or its backend is unavailable.
    """
    if not uri or not uri.strip():
        raise IngestionError("No input path or URI given.")
    stype = SourceType(source_type)
    if stype is SourceType.AUTO:
        stype = detect_source_type(uri)
    if stype is SourceType.LOCAL:
        return LocalSource(uri.strip())
    if stype is SourceType.CLOUD:
        return CloudSource(uri.strip(), cache_dir=cache_dir)
    return GoogleDriveSource(uri.strip(), cache_dir=cache_dir)


def iter_fetched(source: AudioSource, items: list[AudioItem]) -> Iterator[tuple[AudioItem, Path | Exception]]:
    """Yield ``(item, local_path)`` pairs, or ``(item, exception)`` if a fetch fails.

    A failed download never stops the batch; the caller records it and moves on.
    """
    for item in items:
        try:
            yield item, source.fetch(item)
        except Exception as exc:  # noqa: BLE001
            yield item, exc
