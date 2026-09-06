#!/usr/bin/env python3
import argparse
import asyncio
import copy
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

LOG = logging.getLogger("pd_proxy")


class MissingKVMetadata(RuntimeError):
    pass


PREFILL_KV_FIELDS = (
    "remote_engine_id",
    "remote_request_id",
    "remote_host",
    "remote_port",
    "tp_size",
    "pp_size",
)


class RoundRobinPool:
    def __init__(self, items: list[Any]):
        if not items:
            raise ValueError("round-robin pool cannot be empty")
        self._items = list(items)
        self._index = 0

    def next(self) -> Any:
        item = self._items[self._index]
        self._index = (self._index + 1) % len(self._items)
        return item


class LeastInFlightPool:
    """Choose the least-loaded endpoint, with round-robin tie breaking."""

    def __init__(
        self, items: list["Endpoint"], max_inflight_per_endpoint: int = 1
    ):
        if not items:
            raise ValueError("least-in-flight pool cannot be empty")
        if max_inflight_per_endpoint < 1:
            raise ValueError("max_inflight_per_endpoint must be positive")
        self._items = list(items)
        self._max_inflight_per_endpoint = max_inflight_per_endpoint
        self._cursor = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> "Endpoint":
        async with self._condition:
            while True:
                minimum = min(item.inflight for item in self._items)
                if minimum < self._max_inflight_per_endpoint:
                    for offset in range(len(self._items)):
                        index = (self._cursor + offset) % len(self._items)
                        item = self._items[index]
                        if item.inflight == minimum:
                            item.inflight += 1
                            self._cursor = (index + 1) % len(self._items)
                            return item
                await self._condition.wait()

    async def release(self, item: "Endpoint") -> None:
        async with self._condition:
            if item.inflight <= 0:
                raise RuntimeError(f"invalid in-flight count for {item.address}")
            item.inflight -= 1
            self._condition.notify(1)


def make_prefill_payload(
    request_payload: dict[str, Any], transfer_id: str
) -> dict[str, Any]:
    payload = copy.deepcopy(request_payload)
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": transfer_id,
    }
    return payload


def extract_prefill_kv_metadata(
    prefill_response_payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = prefill_response_payload.get("kv_transfer_params")
    if not isinstance(metadata, dict):
        raise MissingKVMetadata("prefill response has no kv_transfer_params object")
    missing = [field for field in PREFILL_KV_FIELDS if metadata.get(field) is None]
    if missing:
        raise MissingKVMetadata(
            "prefill response is missing NIXL push metadata: " + ", ".join(missing)
        )
    if metadata.get("transfer_mode") not in (None, "push"):
        raise MissingKVMetadata(
            f"unexpected KV transfer mode: {metadata.get('transfer_mode')!r}"
        )
    return metadata


def build_decode_payload(
    request_payload: dict[str, Any],
    prefill_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(request_payload)
    # NixlPushConnector returns the authoritative P coordinates and request ID.
    # Forward those values instead of fabricating legacy bootstrap metadata.
    params = {
        field: prefill_metadata[field]
        for field in PREFILL_KV_FIELDS
    }
    params.update(
        do_remote_decode=False,
        do_remote_prefill=True,
        remote_num_tokens=prefill_metadata.get("remote_num_tokens", 0),
        transfer_mode="push",
    )
    payload["kv_transfer_params"] = params
    return payload


@dataclass
class Endpoint:
    address: str
    host: str
    client: Any
    inflight: int = 0


def parse_endpoint(address: str) -> tuple[str, int]:
    host, separator, port_text = address.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid endpoint: {address}")
    port = int(port_text)
    if port < 1 or port > 65535:
        raise ValueError(f"invalid endpoint port: {address}")
    return host, port


def forwarded_headers(request: "Request") -> dict[str, str]:
    headers: dict[str, str] = {}
    for name in ("authorization", "x-request-id"):
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


def label_prometheus_samples(text: str, endpoint: str) -> str:
    """Add the source D endpoint as a label to every Prometheus sample."""
    escaped = endpoint.replace("\\", "\\\\").replace('"', '\\"')
    output: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            output.append(line)
            continue
        sample, separator, value = line.partition(" ")
        if not separator:
            output.append(line)
            continue
        if "{" in sample and sample.endswith("}"):
            sample = sample[:-1] + f',decode_endpoint="{escaped}"' + "}"
        else:
            sample += f'{{decode_endpoint="{escaped}"}}'
        output.append(f"{sample} {value}")
    return "\n".join(output) + "\n"


def create_app(
    prefill_addresses: list[str],
    decode_addresses: list[str],
    max_decode_inflight: int = 1,
) -> "FastAPI":
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        prefills: list[Endpoint] = []
        for address in prefill_addresses:
            host, _ = parse_endpoint(address)
            prefills.append(
                Endpoint(
                    address=address,
                    host=host,
                    client=httpx.AsyncClient(
                        base_url=f"http://{address}",
                        timeout=None,
                        limits=httpx.Limits(
                            max_connections=None,
                            max_keepalive_connections=None,
                        ),
                    ),
                )
            )
        decodes: list[Endpoint] = []
        for address in decode_addresses:
            decode_host, _ = parse_endpoint(address)
            decodes.append(Endpoint(
                address=address,
                host=decode_host,
                client=httpx.AsyncClient(
                    base_url=f"http://{address}",
                    timeout=None,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
            ))
        app.state.prefills = prefills
        app.state.prefill_pool = RoundRobinPool(prefills)
        app.state.decodes = decodes
        app.state.decode_pool = LeastInFlightPool(
            decodes, max_inflight_per_endpoint=max_decode_inflight
        )
        LOG.info("proxy ready: prefills=%s decodes=%s", prefill_addresses, decode_addresses)
        yield
        await asyncio.gather(
            *(endpoint.client.aclose() for endpoint in prefills),
            *(endpoint.client.aclose() for endpoint in decodes),
        )

    app = FastAPI(title="GLM-5.2 multi-P/1D proxy", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        endpoints = [*request.app.state.prefills, *request.app.state.decodes]

        async def probe(endpoint: Endpoint) -> dict[str, Any]:
            try:
                response = await endpoint.client.get("/v1/models", timeout=5.0)
                return {
                    "address": endpoint.address,
                    "ready": response.status_code == 200,
                    "status_code": response.status_code,
                    "inflight": endpoint.inflight,
                }
            except Exception as error:
                return {
                    "address": endpoint.address,
                    "ready": False,
                    "error": str(error),
                }

        results = await asyncio.gather(*(probe(endpoint) for endpoint in endpoints))
        prefill_count = len(request.app.state.prefills)
        prefill_results = results[:prefill_count]
        decode_results = results[prefill_count:]
        ready = all(item["ready"] for item in results)
        return JSONResponse(
            {
                "status": "ok" if ready else "degraded",
                "prefill_count": len(prefill_results),
                "decode_count": len(decode_results),
                "prefills": prefill_results,
                "decodes": decode_results,
            },
            status_code=200 if ready else 503,
        )

    @app.get("/v1/models")
    async def models(request: Request):
        decode: Endpoint = request.app.state.decodes[0]
        response = await decode.client.get(
            "/v1/models", headers=forwarded_headers(request)
        )
        return JSONResponse(response.json(), status_code=response.status_code)

    @app.get("/metrics")
    async def metrics(request: Request):
        decodes: list[Endpoint] = request.app.state.decodes

        async def scrape(decode: Endpoint):
            try:
                response = await decode.client.get(
                    "/metrics", headers=forwarded_headers(request), timeout=10.0
                )
                response.raise_for_status()
                return decode, response.text, None
            except Exception as error:
                return decode, "", error

        results = await asyncio.gather(*(scrape(decode) for decode in decodes))
        seen_metadata: set[tuple[str, str]] = set()
        parts: list[str] = []
        failures = 0
        for decode, text, error in results:
            if error is not None:
                failures += 1
                parts.append(
                    f'# scrape_error decode_endpoint="{decode.address}" '
                    f'message="{str(error)}"\n'
                )
                continue
            labeled = label_prometheus_samples(text, decode.address)
            filtered: list[str] = []
            for line in labeled.splitlines():
                if line.startswith("# HELP ") or line.startswith("# TYPE "):
                    fields = line.split(None, 3)
                    key = (fields[1], fields[2])
                    if key in seen_metadata:
                        continue
                    seen_metadata.add(key)
                filtered.append(line)
            parts.append("\n".join(filtered) + "\n")
        for decode in decodes:
            parts.append(
                'pd_proxy_decode_inflight{decode_endpoint="'
                f'{decode.address}"}} {decode.inflight}\n'
            )
        parts.append(f"pd_proxy_decode_scrape_failures {failures}\n")
        return Response(
            content="".join(parts),
            status_code=200 if failures == 0 else 503,
            media_type="text/plain; version=0.0.4",
        )

    async def proxy_generate(request: Request, path: str):
        original_payload = await request.json()
        # Reserve decode capacity before prefill. Generating remote KV first and
        # then queueing behind a busy decoder can let the producer's KV lease
        # expire before D registers the transfer.
        decode_pool: LeastInFlightPool = request.app.state.decode_pool
        decode = await decode_pool.acquire()
        producer: Endpoint = request.app.state.prefill_pool.next()
        headers = forwarded_headers(request)
        transfer_id = f"xfer-{uuid.uuid4()}"
        # vLLM appends a per-engine suffix to this common API request ID.
        # NixlPushConnector strips that suffix when pairing P's completed
        # blocks with D's registration, so both legs must share this header.
        headers.setdefault("x-request-id", transfer_id)
        LOG.info("request path=%s selected_prefill=%s", path, producer.address)

        handed_to_stream = False
        upstream_response = None
        try:
            prefill_response = await producer.client.post(
                path,
                json=make_prefill_payload(original_payload, transfer_id),
                headers=headers,
            )
            if prefill_response.status_code >= 400:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"prefill {producer.address} returned "
                            f"HTTP {prefill_response.status_code}",
                            "body": prefill_response.text,
                        }
                    },
                    status_code=502,
                    headers={"x-prefill-instance": producer.address},
                )
            try:
                prefill_metadata = extract_prefill_kv_metadata(prefill_response.json())
            except (ValueError, MissingKVMetadata) as error:
                return JSONResponse(
                    {
                        "error": {
                            "message": "prefill returned invalid NIXL push metadata",
                            "body": str(error),
                        }
                    },
                    status_code=502,
                    headers={"x-prefill-instance": producer.address},
                )
            LOG.info(
                "prefill KV ready request=%s engine=%s host=%s port=%s tp=%s pp=%s",
                prefill_metadata["remote_request_id"],
                prefill_metadata["remote_engine_id"],
                prefill_metadata["remote_host"],
                prefill_metadata["remote_port"],
                prefill_metadata["tp_size"],
                prefill_metadata["pp_size"],
            )
            decode_payload = build_decode_payload(original_payload, prefill_metadata)

            LOG.info(
                "selected_decode=%s inflight=%s request=%s",
                decode.address,
                decode.inflight,
                prefill_metadata["remote_request_id"],
            )
            upstream_request = decode.client.build_request(
                "POST", path, json=decode_payload, headers=headers
            )
            upstream_response = await decode.client.send(upstream_request, stream=True)

            async def relay():
                try:
                    async for chunk in upstream_response.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_response.aclose()
                    await decode_pool.release(decode)

            response_headers = {"x-prefill-instance": producer.address}
            content_type = upstream_response.headers.get("content-type")
            if content_type:
                response_headers["content-type"] = content_type
            request_id = upstream_response.headers.get("x-request-id")
            if request_id:
                response_headers["x-request-id"] = request_id
            handed_to_stream = True
            return StreamingResponse(
                relay(),
                status_code=upstream_response.status_code,
                headers=response_headers,
            )
        finally:
            if not handed_to_stream:
                if upstream_response is not None:
                    await upstream_response.aclose()
                await decode_pool.release(decode)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await proxy_generate(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy_generate(request, "/v1/completions")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM multi-P/1D NIXL proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill", nargs="+", required=True)
    parser.add_argument("--decode", nargs="+", required=True)
    parser.add_argument("--max-decode-inflight", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    uvicorn.run(
        create_app(
            args.prefill,
            args.decode,
            args.max_decode_inflight,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
