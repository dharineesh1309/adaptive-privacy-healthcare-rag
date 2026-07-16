"""Dependency-free local HTTP interface for the adaptive privacy pipeline."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .chatdoctor import prepare_chatdoctor_benchmark
from .config import PROJECT_ROOT, load_config
from .io import load_json
from .pipeline import AdaptivePrivacyPipeline
from .session_pipeline import SessionAwarePrivacyPipeline
from .tier_builder import TierBuilder


WEB_ROOT = PROJECT_ROOT / "web_demo"
ALLOWED_ROLES = {"public", "guest", "patient", "customer", "doctor", "admin"}
ALLOWED_MODES = {"adaptive", "fixed_low", "fixed_high"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class WebDemoApplication:
    def __init__(self, pipeline: AdaptivePrivacyPipeline, dataset_name: str):
        self.pipeline = pipeline
        self.session_pipeline = SessionAwarePrivacyPipeline(pipeline)
        self.dataset_name = dataset_name

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset": self.dataset_name,
            "documents": len(self.pipeline.documents),
            "roles": sorted(ALLOWED_ROLES),
            "modes": ["adaptive", "fixed_low", "fixed_high"],
            "session_aware": True,
        }

    def create_session(self) -> dict[str, Any]:
        return self.session_pipeline.create_session()

    def reset_session(self, session_id: str) -> dict[str, Any]:
        return self.session_pipeline.reset_session(session_id)

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query", "")).strip()
        role = str(payload.get("role", "public")).strip().lower()
        mode = str(payload.get("mode", "adaptive")).strip().lower()
        if not query:
            raise ValueError("Query is required.")
        if len(query) > 5000:
            raise ValueError("Query must be 5,000 characters or fewer.")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported mode: {mode}")

        session_aware = bool(payload.get("session_aware", False))
        session_id = str(payload.get("session_id", "")).strip()
        role_verified = bool(payload.get("role_verified", False))
        if session_aware:
            if not session_id:
                raise ValueError("A session ID is required for session-aware mode.")
            response = self.session_pipeline.process(
                session_id=session_id,
                query=query,
                role=role,
                mode=mode,
                role_verified=role_verified,
            )
        else:
            result = self.pipeline.process(
                query=query,
                role=role,
                mode=mode,
                role_verified=role_verified,
            )
            response = result.to_dict()
        response["dataset"] = self.dataset_name
        return response


def create_application(dataset: str = "synthetic") -> WebDemoApplication:
    config = load_config()
    if dataset == "synthetic":
        raw_documents = load_json(
            PROJECT_ROOT / "examples" / "adaptive_privacy" / "documents.json"
        )
        dataset_name = "Controlled synthetic demonstration"
    elif dataset == "chatdoctor":
        cache_path = PROJECT_ROOT / "data" / "external" / "chatdoctor_sample.json"
        if not cache_path.exists():
            raise FileNotFoundError(
                "ChatDoctor cache not found. Run "
                "'python scripts/run_chatdoctor_evaluation.py' first."
            )
        cached = load_json(cache_path)
        raw_documents, _ = prepare_chatdoctor_benchmark(
            cached["rows"], cached.get("source", {})
        )
        dataset_name = "ChatDoctor sample with controlled synthetic PII"
    else:
        raise ValueError("dataset must be 'synthetic' or 'chatdoctor'")

    documents = TierBuilder().build_many(raw_documents)
    audit_path = PROJECT_ROOT / "results" / "adaptive_privacy" / "web_audit.jsonl"
    pipeline = AdaptivePrivacyPipeline(
        documents,
        config=config,
        audit_path=str(audit_path),
    )
    return WebDemoApplication(pipeline, dataset_name)


def create_server(
    application: WebDemoApplication,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    class RequestHandler(BaseHTTPRequestHandler):
        app = application

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self._send_json(self.app.health())
                return
            static = STATIC_FILES.get(self.path)
            if static is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            filename, content_type = static
            path = WEB_ROOT / filename
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            if self.path not in {"/api/query", "/api/session"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 20_000:
                    raise ValueError("Invalid request size.")
                payload = (
                    json.loads(self.rfile.read(length).decode("utf-8"))
                    if length
                    else {}
                )
                if not isinstance(payload, dict):
                    raise ValueError("JSON object expected.")
                if self.path == "/api/session":
                    if payload.get("action") == "reset":
                        response = self.app.reset_session(str(payload.get("session_id", "")))
                    else:
                        response = self.app.create_session()
                else:
                    response = self.app.query(payload)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response)

        def _send_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid logging queries or other request content to the console.
            return

    return ThreadingHTTPServer((host, port), RequestHandler)
