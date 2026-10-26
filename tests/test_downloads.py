"""Download integrity and cache recovery without external network access."""
from __future__ import annotations

import hashlib
import http.client
import io
import re
import tarfile

import pytest

from cell_property_prediction import assets


class Response(io.BytesIO):
    def __init__(self, payload: bytes, *, content_length: int | None = None, status: int = 200):
        super().__init__(payload)
        self.status = status
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def getcode(self) -> int:
        return self.status


@pytest.fixture
def delays(monkeypatch):
    values = []
    monkeypatch.setattr(assets.time, "sleep", values.append)
    return values


@pytest.fixture
def pinned(monkeypatch):
    payload = b"verified fixture payload"
    checksum = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(assets, "_FIGSHARE_FILES", {"42715213": (len(payload), checksum)})
    return payload, checksum


def install_responses(monkeypatch, responses):
    """Record requests and supply a new response (or failure) for each attempt."""
    pending = iter(responses)
    requests = []

    def open_response(request, *, timeout):
        requests.append((request.full_url, timeout))
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_response)
    return requests


def test_nonempty_generic_cache_is_reused(tmp_path, monkeypatch, delays):
    destination = tmp_path / "cached.bin"
    destination.write_bytes(b"already complete")
    requests = install_responses(monkeypatch, [])

    record = assets.download("https://example.org/cached.bin", destination)

    assert requests == []
    assert delays == []
    assert record["bytes"] == len(b"already complete")
    assert record["sha256"] == hashlib.sha256(b"already complete").hexdigest()


def test_zero_byte_cache_is_replaced(tmp_path, monkeypatch, delays):
    destination = tmp_path / "cached.bin"
    destination.touch()
    requests = install_responses(monkeypatch, [Response(b"complete", content_length=8)])

    record = assets.download("https://example.org/cached.bin", destination)

    assert len(requests) == 1
    assert destination.read_bytes() == b"complete"
    assert record["bytes"] == 8
    assert not destination.with_name("cached.bin.download").exists()


@pytest.mark.parametrize("kind", ["empty", "empty_no_length", "short", "status"])
def test_invalid_response_is_retried_and_never_promoted(tmp_path, monkeypatch, delays, kind):
    destination = tmp_path / "archive.bin"
    factories = {
        "empty": lambda: Response(b"", content_length=0),
        "empty_no_length": lambda: Response(b""),
        "short": lambda: Response(b"partial", content_length=100),
        "status": lambda: Response(b"unexpected response", status=204),
    }
    requests = install_responses(monkeypatch, [factories[kind]() for _ in range(3)])

    with pytest.raises(RuntimeError, match=re.escape(destination.name)):
        assets.download("https://example.org/archive.bin", destination)

    assert len(requests) == 3
    assert delays == [1, 2]
    assert not destination.exists()
    assert not destination.with_name("archive.bin.download").exists()


@pytest.mark.parametrize("failure", [OSError("interrupted"), http.client.IncompleteRead(b"partial", 100)])
def test_stream_failure_restarts_download(tmp_path, monkeypatch, delays, failure):
    class InterruptedResponse(Response):
        def read(self, size=-1):
            if self.tell():
                raise failure
            return super().read(size)

    destination = tmp_path / "archive.bin"
    requests = install_responses(monkeypatch, [InterruptedResponse(b"partial"), Response(b"complete")])

    record = assets.download("https://example.org/archive.bin", destination)

    assert len(requests) == 2
    assert delays == [1]
    assert destination.read_bytes() == b"complete"
    assert record["sha256"] == hashlib.sha256(b"complete").hexdigest()
    assert not destination.with_name("archive.bin.download").exists()


@pytest.mark.parametrize("url", [
    "https://figshare.com/ndownloader/files/42715213",
    "https://ndownloader.figshare.com/files/42715213",
])
def test_figshare_uses_canonical_url_and_published_integrity(tmp_path, monkeypatch, delays, pinned, url):
    payload, checksum = pinned
    destination = tmp_path / "protein_embeddings.tar.gz"
    requests = install_responses(monkeypatch, [Response(payload, content_length=len(payload))])

    record = assets.download(url, destination)

    assert [request[0] for request in requests] == ["https://ndownloader.figshare.com/files/42715213"]
    assert destination.read_bytes() == payload
    assert record["url"] == url
    assert record["download_url"] == "https://ndownloader.figshare.com/files/42715213"
    assert record["expected_bytes"] == len(payload)
    assert record["published_md5"] == checksum


def test_valid_figshare_cache_is_reused(tmp_path, monkeypatch, delays, pinned):
    payload, _ = pinned
    destination = tmp_path / "protein_embeddings.tar.gz"
    destination.write_bytes(payload)
    requests = install_responses(monkeypatch, [])

    assets.download("https://figshare.com/ndownloader/files/42715213", destination)

    assert requests == []
    assert delays == []


@pytest.mark.parametrize("same_size", [False, True])
def test_corrupt_figshare_cache_is_replaced(tmp_path, monkeypatch, delays, pinned, same_size):
    payload, _ = pinned
    destination = tmp_path / "protein_embeddings.tar.gz"
    destination.write_bytes(b"x" * len(payload) if same_size else b"short")
    requests = install_responses(monkeypatch, [Response(payload)])

    assets.download("https://figshare.com/ndownloader/files/42715213", destination)

    assert len(requests) == 1
    assert destination.read_bytes() == payload


def test_wrong_figshare_checksum_is_retried(tmp_path, monkeypatch, delays, pinned):
    payload, _ = pinned
    destination = tmp_path / "protein_embeddings.tar.gz"
    requests = install_responses(monkeypatch, [Response(b"x" * len(payload)), Response(payload)])

    assets.download("https://figshare.com/ndownloader/files/42715213", destination)

    assert len(requests) == 2
    assert delays == [1]
    assert destination.read_bytes() == payload


def test_failed_replacement_preserves_cache_and_cleans_temporary_file(tmp_path, monkeypatch, delays, pinned):
    destination = tmp_path / "protein_embeddings.tar.gz"
    destination.write_bytes(b"previous incomplete cache")
    requests = install_responses(monkeypatch, [Response(b"short") for _ in range(3)])

    with pytest.raises(RuntimeError, match=re.escape(destination.name)):
        assets.download("https://figshare.com/ndownloader/files/42715213", destination)

    assert len(requests) == 3
    assert delays == [1, 2]
    assert destination.read_bytes() == b"previous incomplete cache"
    assert not destination.with_name("protein_embeddings.tar.gz.download").exists()


def test_uce_prefetch_recovers_all_zero_byte_resources(tmp_path, monkeypatch, delays):
    from cell_property_prediction.embeddings import prefetch_uce

    archive_buffer = io.BytesIO()
    protein_payloads = {f"species_{index}.pt": f"protein {index}".encode() for index in range(8)}
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for name, payload in protein_payloads.items():
            member = tarfile.TarInfo(f"protein_embeddings/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    payloads = {
        "42706558": b"species,chromosome\nmouse,1\n",
        "42706555": b"synthetic offsets",
        "42706585": b"synthetic tokens",
        "42715213": archive_buffer.getvalue(),
        "43423236": b"synthetic checkpoint",
    }
    monkeypatch.setattr(assets, "_FIGSHARE_FILES", {
        file_id: (len(payload), hashlib.md5(payload).hexdigest())
        for file_id, payload in payloads.items()
    })
    config = {"resources_dir": "resources", "checkpoint": "checkpoints/model.torch"}
    resources = tmp_path / "resources"
    resources.mkdir()
    for name in ("species_chrom.csv", "species_offsets.pkl", "all_tokens.torch", "protein_embeddings.tar.gz"):
        (resources / name).touch()
    checkpoint = tmp_path / config["checkpoint"]
    checkpoint.parent.mkdir()
    checkpoint.touch()
    requests = []

    def open_response(request, *, timeout):
        requests.append(request.full_url)
        file_id = request.full_url.rsplit("/", 1)[-1]
        payload = payloads[file_id]
        return Response(payload, content_length=len(payload))

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_response)

    record = prefetch_uce(config, root=tmp_path)

    assert len(requests) == 5
    assert set(requests) == {f"https://ndownloader.figshare.com/files/{file_id}" for file_id in payloads}
    assert len(record["files"]) == 5
    assert record["layers"] == 33
    assert {item["name"]: item["sha256"] for item in record["protein_files"]} == {
        name: hashlib.sha256(payload).hexdigest() for name, payload in protein_payloads.items()
    }
    assert all((resources / "protein_embeddings" / name).read_bytes() == payload
               for name, payload in protein_payloads.items())
    assets.verify_resource_record(record)

    assert prefetch_uce(config, root=tmp_path) == record
    assert len(requests) == 5
    assert delays == []