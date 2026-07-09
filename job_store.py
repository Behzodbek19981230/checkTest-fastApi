import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from typing import Optional

# Jobs must be visible across all gunicorn worker processes (each is a
# separate OS process with its own memory), so an in-process dict silently
# drops jobs created by a different worker than the one polling for them.
# SQLite on shared disk is the simplest thing that actually works here.
_DB_PATH = os.path.join(tempfile.gettempdir(), "checktest_jobs.db")
_JOB_TTL_SECONDS = 3600

_init_lock = threading.Lock()
_initialized = False


def _get_conn() -> sqlite3.Connection:
	conn = sqlite3.connect(_DB_PATH, timeout=10)
	conn.execute("PRAGMA journal_mode=WAL")
	conn.execute("PRAGMA busy_timeout=10000")
	return conn


def _ensure_schema() -> None:
	global _initialized
	if _initialized:
		return
	with _init_lock:
		if _initialized:
			return
		conn = _get_conn()
		try:
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS jobs (
					job_id TEXT PRIMARY KEY,
					status TEXT NOT NULL,
					result TEXT,
					error TEXT,
					created_at REAL NOT NULL
				)
				"""
			)
			conn.commit()
		finally:
			conn.close()
		_initialized = True


def create_job() -> str:
	_ensure_schema()
	job_id = uuid.uuid4().hex
	conn = _get_conn()
	try:
		_purge_stale(conn)
		conn.execute(
			"INSERT INTO jobs (job_id, status, result, error, created_at) VALUES (?, 'pending', NULL, NULL, ?)",
			(job_id, time.time()),
		)
		conn.commit()
	finally:
		conn.close()
	return job_id


def set_job_result(job_id: str, result: dict) -> None:
	_ensure_schema()
	conn = _get_conn()
	try:
		conn.execute(
			"UPDATE jobs SET status = 'done', result = ? WHERE job_id = ?",
			(json.dumps(result), job_id),
		)
		conn.commit()
	finally:
		conn.close()


def set_job_error(job_id: str, message: str) -> None:
	_ensure_schema()
	conn = _get_conn()
	try:
		conn.execute(
			"UPDATE jobs SET status = 'error', error = ? WHERE job_id = ?",
			(message, job_id),
		)
		conn.commit()
	finally:
		conn.close()


def get_job(job_id: str) -> Optional[dict]:
	_ensure_schema()
	conn = _get_conn()
	try:
		row = conn.execute(
			"SELECT status, result, error FROM jobs WHERE job_id = ?", (job_id,)
		).fetchone()
	finally:
		conn.close()
	if row is None:
		return None
	status, result_json, error = row
	return {
		"status": status,
		"result": json.loads(result_json) if result_json else None,
		"error": error,
	}


def _purge_stale(conn: sqlite3.Connection) -> None:
	cutoff = time.time() - _JOB_TTL_SECONDS
	conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
