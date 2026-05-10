import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from _prefill_hook import PrefillConfig
from prefix_cache import PrefixCache, hash_prefix
from server import build_app


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "<|im_end|>": [1],
            "<|im_start|>": [2],
            "system": [3],
        }
        return mapping.get(text, [99])


def make_cache(tmp_path: Path, *, cap: int = 2, full_cap: int = 2,
               kv_k_type: str = "q8_0", fa_window: int = 2048) -> PrefixCache:
    async def await_reply(prefix: str, timeout: float = 10.0) -> str:
        return prefix

    cache = PrefixCache(
        daemon_stdin=SimpleNamespace(write=lambda *_: None, flush=lambda: None),
        await_reply=await_reply,
        daemon_lock=asyncio.Lock(),
        tokenizer=FakeTokenizer(),
        kv_k_type=kv_k_type,
        fa_window=fa_window,
        cap=cap,
    )
    cache.init_full_cache(full_cap, cache_dir=str(tmp_path))
    return cache


def write_meta(cache: PrefixCache, key: bytes, *, cur_ids_len: int,
               last_used_ns: int, kv_k_type: str | None = None,
               fa_window: int | None = None) -> None:
    meta = {
        "version": cache.FULL_META_VERSION,
        "key_hex": key.hex(),
        "kv_k_type": kv_k_type or cache.kv_k_type,
        "fa_window": cache.fa_window if fa_window is None else fa_window,
        "cur_ids_len": cur_ids_len,
        "last_used_ns": last_used_ns,
    }
    cache._full_meta_path(key).write_text(json.dumps(meta), encoding="utf-8")


def test_rehydrate_full_cache_restores_valid_entries(tmp_path):
    prompt_ids = [11, 22, 33]
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"cached")

    cache = make_cache(tmp_path)
    cache.confirm_full_snap(cache._full_slot_base, prompt_ids, source_path, 7)

    restored_cache = make_cache(tmp_path)
    replay = AsyncMock(return_value=True)

    restored = asyncio.run(restored_cache.rehydrate_full_cache(replay))

    key = hash_prefix(prompt_ids, restored_cache.kv_k_type, restored_cache.fa_window)
    assert restored == 1
    replay.assert_awaited_once_with(
        restored_cache._full_slot_base,
        str(restored_cache._full_bin_path(key)),
        7,
    )
    assert restored_cache.full_entries[key] == (
        restored_cache._full_slot_base,
        str(restored_cache._full_bin_path(key)),
        7,
    )


def test_rehydrate_full_cache_skips_stale_metadata(tmp_path):
    cache = make_cache(tmp_path)
    key = b"\x01" * 16
    cache._full_bin_path(key).write_bytes(b"stale")
    write_meta(cache, key, cur_ids_len=5, last_used_ns=1, kv_k_type="q4_0")

    replay = AsyncMock(return_value=True)
    restored = asyncio.run(cache.rehydrate_full_cache(replay))

    assert restored == 0
    replay.assert_not_called()
    assert cache.full_entries == {}


def test_rehydrate_full_cache_keeps_most_recent_entries_within_cap(tmp_path):
    cache = make_cache(tmp_path, full_cap=2)
    keys = [bytes([i]) * 16 for i in range(1, 4)]
    for idx, key in enumerate(keys, start=1):
        cache._full_bin_path(key).write_bytes(f"bin-{idx}".encode())
        write_meta(cache, key, cur_ids_len=idx, last_used_ns=idx)

    replay_calls = []

    async def replay(slot: int, cur_bin_path: str, cur_ids_len: int) -> bool:
        replay_calls.append((slot, Path(cur_bin_path).name, cur_ids_len))
        return True

    restored = asyncio.run(cache.rehydrate_full_cache(replay))

    assert restored == 2
    assert replay_calls == [
        (cache._full_slot_base, f"{keys[1].hex()}.bin", 2),
        (cache._full_slot_base + 1, f"{keys[2].hex()}.bin", 3),
    ]
    assert list(cache.full_entries.keys()) == [keys[1], keys[2]]


def test_server_startup_rehydrates_full_cache(tmp_path):
    tokenizer = FakeTokenizer()
    prefill_cfg = PrefillConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.05,
        drafter_gguf=Path("drafter.gguf"),
        drafter_tokenizer_id="Qwen/Qwen3-0.6B",
    )

    with patch("server.subprocess.Popen") as mock_popen, \
            patch("server.PrefixCache.startup_sync", new_callable=AsyncMock) as mock_sync, \
            patch("server.PrefixCache.rehydrate_full_cache",
                  new_callable=AsyncMock, return_value=1) as mock_rehydrate:
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.stdout.readline.return_value = b""

        app = build_app(
            target=Path("target.gguf"),
            draft=Path("draft.safetensors"),
            bin_path=Path("test_dflash"),
            budget=22,
            max_ctx=4096,
            tokenizer=tokenizer,
            stop_ids={2},
            prefill_cfg=prefill_cfg,
            drafter_tokenizer=tokenizer,
            prefill_cache_slots=2,
        )

        with TestClient(app):
            pass

    mock_sync.assert_awaited_once()
    mock_rehydrate.assert_awaited_once()
