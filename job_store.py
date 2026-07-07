import threading
import time
import uuid
from typing import Any, Dict, Optional



_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 3600


def create_job() -> str:
	job_id = uuid.uuid4().hex
	with _JOBS_LOCK:
		_purge_stale_locked()
		_JOBS[job_id] = {
			"status": "pending",
			"result": None,
			"error": None,
			"created_at": time.time(),
		}
	return job_id


def set_job_result(job_id: str, result: dict) -> None:
	with _JOBS_LOCK:
		job = _JOBS.get(job_id)
		if job is not None:
			job["status"] = "done"
			job["result"] = result


def set_job_error(job_id: str, message: str) -> None:
	with _JOBS_LOCK:
		job = _JOBS.get(job_id)
		if job is not None:
			job["status"] = "error"
			job["error"] = message


def get_job(job_id: str) -> Optional[dict]:
	with _JOBS_LOCK:
		job = _JOBS.get(job_id)
		return dict(job) if job is not None else None


def _purge_stale_locked() -> None:
	"""Drops jobs older than the TTL. Caller must hold _JOBS_LOCK."""
	cutoff = time.time() - _JOB_TTL_SECONDS
	stale = [jid for jid, job in _JOBS.items() if job["created_at"] < cutoff]
	for jid in stale:
		del _JOBS[jid]
