from __future__ import annotations

# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import errno
import logging
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)


def replace_file(path: str | Path, write: Callable[[str], object]) -> None:
    """Replace a cache file atomically using a uniquely named neighboring temporary file.

    Permission errors preserve the existing cache and log a skipped update.

    Args:
        path: Cache destination whose parent directory already exists.
        write: Callback that creates and populates the supplied temporary path.
            Its return value is ignored.

    Raises:
        Exception: The write or replacement fails for a reason other than a
            permission error. Temporary files are removed before propagation.

    """
    path = Path(path)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")

    try:
        write(str(temp))
        os.replace(temp, path)
    except PermissionError as exc:
        temp.unlink(missing_ok=True)
        logger.warning("Kept the cached %s; it is in use (%s)", path.name, exc)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


@contextmanager
def locked(path: str | Path, wait: bool = True) -> Iterator[bool]:
    """Coordinate writers across threads and processes with an adjacent lock file.

    Args:
        path: Resource name used to derive its .lock file.
        wait: Whether to wait for another writer to release the lock.

    Yields:
        True while the lock is held. A nonblocking attempt can yield False,
        in which case the caller must skip the protected work.

    Raises:
        OSError: The lock file cannot be opened or the platform lock operation fails.
    """
    with open(f"{path}.lock", "a+b") as handle:
        hold = _hold_windows if os.name == "nt" else _hold_posix
        with hold(handle, wait) as acquired:
            yield acquired


@contextmanager
def _hold_posix(handle: BinaryIO, wait: bool = True) -> Iterator[bool]:
    """Hold an advisory file lock until the caller leaves the context."""
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _hold_windows(handle: BinaryIO, wait: bool = True) -> Iterator[bool]:
    """Hold a Windows byte-range lock until the caller leaves the context."""
    import msvcrt

    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            break
        except OSError as exc:
            if not wait and exc.errno in (errno.EACCES, errno.EDEADLOCK):
                yield False
                return

            if exc.errno != errno.EDEADLOCK:
                raise
    try:
        yield True
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
