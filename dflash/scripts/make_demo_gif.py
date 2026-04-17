"""
Render a side-by-side comparison GIF: AR target (slow) vs Luce DFlash (fast).

Runs the two binaries sequentially (each needs 16 GB VRAM, so they can't
coexist on a single 24 GB 3090), records timestamped token streams, then
plays both back in sync against a shared wall clock.

    pip install pillow
    python3 scripts/make_demo_gif.py \\
        --prompt "Expand (x+1)^3" \\
        --n-gen 128 \\
        --out demo.gif

Runs both `test_generate` (AR) and `test_dflash` in parallel, captures each
committed token with a wall-clock timestamp, then renders a GIF showing the
two terminal panes filling in real time with a live tok/s counter overlay.

The visual layout mirrors https://github.com/Luce-Org screenshots:
left pane "Autoregressive" — right pane "Luce DFlash" — wall-clock timer up
top — tok/s badge per pane.

Known limits: model reloads inside each child process (~10 s before first
token). The GIF skips the prefill and starts at the first streamed token.
"""
import argparse
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"
DRAFT_SEARCH_ROOT = Path.home() / ".cache/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots"
TEST_DFLASH   = ROOT / "build" / "test_dflash"
TEST_GENERATE = ROOT / "build" / "test_generate"

# Terminal-ish rendering constants
W, H      = 1200, 760
PAD       = 24
PANE_W    = (W - PAD * 3) // 2
PANE_H    = H - PAD * 2 - 40
FG        = (230, 230, 235)
DIM       = (120, 120, 130)
BG        = (14, 14, 18)
PANE_BG   = (24, 24, 30)
ACCENT_L  = (255, 95, 95)     # red-coral = slow AR
ACCENT_R  = (255, 209, 0)     # Luce yellow = DFlash
CHAR_W    = 9
LINE_H    = 18
FPS       = 20


MONO_CANDIDATES = [
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]
# Sans-serif with plain oval "0" (no dot/slash) — used for the tok/s badge.
SANS_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def resolve_font(size: int, sans: bool = False) -> ImageFont.FreeTypeFont:
    for p in (SANS_CANDIDATES if sans else MONO_CANDIDATES):
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def resolve_draft() -> Path:
    for st in DRAFT_SEARCH_ROOT.rglob("model.safetensors"):
        return st
    raise FileNotFoundError("draft weights not found")


def tokenize_prompt(text: str, tokenizer, out_path: Path):
    ids = tokenizer.encode(text, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))


class StreamReader(threading.Thread):
    """Capture int32 tokens from an fd + timestamp them."""
    def __init__(self, fd: int, tokenizer, out_q: queue.Queue, label: str):
        super().__init__(daemon=True)
        self.fd = fd
        self.tok = tokenizer
        self.q = out_q
        self.label = label

    def run(self):
        while True:
            try:
                b = os.read(self.fd, 4)
            except OSError:
                break
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            self.q.put((self.label, time.time(), tok_id, self.tok.decode([tok_id])))
        os.close(self.fd)
        self.q.put((self.label, time.time(), None, None))  # sentinel


def spawn_ar(prompt_bin: Path, n_gen: int, tokenizer, out_q: queue.Queue) -> subprocess.Popen:
    out_bin = Path("/tmp") / f"ar_{os.getpid()}.bin"
    r, w = os.pipe()
    cmd = [str(TEST_GENERATE), str(TARGET), str(prompt_bin),
           str(n_gen), str(out_bin), f"--stream-fd={w}"]
    proc = subprocess.Popen(cmd, pass_fds=(w,),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    os.close(w)
    StreamReader(r, tokenizer, out_q, "AR").start()
    return proc


def spawn_dflash(prompt_bin: Path, n_gen: int, tokenizer, out_q: queue.Queue,
                 budget: int, use_kv_q4: bool) -> subprocess.Popen:
    draft = resolve_draft()
    out_bin = Path("/tmp") / f"df_{os.getpid()}.bin"
    r, w = os.pipe()
    env = {**os.environ}
    if use_kv_q4:
        env["DFLASH27B_KV_Q4"] = "1"
    cmd = [str(TEST_DFLASH), str(TARGET), str(draft), str(prompt_bin),
           str(n_gen), str(out_bin),
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--stream-fd={w}"]
    proc = subprocess.Popen(cmd, pass_fds=(w,), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(w)
    StreamReader(r, tokenizer, out_q, "DFlash").start()
    return proc


def wrap_text(text: str, max_cols: int) -> list[str]:
    """Wrap by columns; keep newlines. Very naive."""
    out_lines = []
    for line in text.split("\n"):
        if not line:
            out_lines.append("")
            continue
        while len(line) > max_cols:
            out_lines.append(line[:max_cols])
            line = line[max_cols:]
        out_lines.append(line)
    return out_lines


def render_frame(ar_text: str, df_text: str, elapsed: float,
                 ar_tok_count: int, df_tok_count: int,
                 ar_badge_tps: float | None, df_badge_tps: float | None,
                 font: ImageFont.FreeTypeFont, font_big: ImageFont.FreeTypeFont,
                 badge_font: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Top timer
    draw.text((PAD, PAD // 2), f"{elapsed:5.2f}s",
              font=font_big, fill=FG)

    # Panes
    pane_y_top = PAD + 34
    for idx, (title, colour, text_body, tok_count, tps) in enumerate([
        ("Autoregressive  Qwen3.5-27B", ACCENT_L, ar_text, ar_tok_count, ar_badge_tps),
        ("Luce DFlash  (DDTree b22)",    ACCENT_R, df_text, df_tok_count, df_badge_tps),
    ]):
        x = PAD + idx * (PANE_W + PAD)
        # Outer glow: 2-px colored border + fill
        draw.rounded_rectangle((x - 2, pane_y_top - 2,
                                x + PANE_W + 2, pane_y_top + PANE_H + 2),
                               radius=10, fill=colour)
        draw.rounded_rectangle((x, pane_y_top, x + PANE_W, pane_y_top + PANE_H),
                               radius=8, fill=PANE_BG)
        # Title text sits directly on the pane background — no tint strip.
        draw.text((x + 12, pane_y_top + 8), title, font=font, fill=colour)
        draw.text((x + PANE_W - 120, pane_y_top + 8),
                  f"[{tok_count:3d} tok]", font=font, fill=DIM)
        # Body
        cols = (PANE_W - 24) // CHAR_W
        rows = (PANE_H - 36) // LINE_H
        lines = wrap_text(text_body, cols)[-rows:]
        for i, line in enumerate(lines):
            draw.text((x + 12, pane_y_top + 30 + i * LINE_H),
                      line, font=font, fill=FG)
        # tok/s badge: text only, colored fill + thin white outline, no box.
        if tps is not None:
            badge = f"{tps:.1f} TOK/S"
            tw = badge_font.getlength(badge)
            bx = x + (PANE_W - tw) // 2
            by = pane_y_top + PANE_H - 120
            draw.text((bx, by), badge, font=badge_font, fill=colour,
                      stroke_width=2, stroke_fill=(255, 255, 255))

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--out", type=str, default="demo.gif")
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--kv-q4", action="store_true")
    ap.add_argument("--max-duration", type=float, default=12.0,
                    help="Stop rendering after this many wall-clock seconds")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",
                                        trust_remote_code=True)

    prompt_bin = Path("/tmp") / f"demo_prompt_{os.getpid()}.bin"
    tokenize_prompt(args.prompt, tok, prompt_bin)

    def collect(label: str, spawn_fn) -> list[tuple[float, int, str]]:
        """Run a generator, return list of (dt_from_first_tok, tok_id, decoded_str)."""
        q: queue.Queue = queue.Queue()
        print(f"[demo] running {label}…", flush=True)
        proc = spawn_fn(prompt_bin, args.n_gen, tok, q) \
            if label == "AR" else \
            spawn_fn(prompt_bin, args.n_gen, tok, q, args.budget, args.kv_q4)
        events, first_t = [], None
        while True:
            lbl, ts, tid, s = q.get()
            if tid is None: break
            if first_t is None: first_t = ts
            events.append((ts - first_t, tid, s))
        proc.wait()
        print(f"[demo]   {label}: {len(events)} tokens over "
              f"{events[-1][0]:.2f}s = {len(events)/max(0.01, events[-1][0]):.1f} tok/s",
              flush=True)
        return events

    ar_events = collect("AR", spawn_ar)
    df_events = collect("DFlash", spawn_dflash)

    # Play both back against shared wall clock
    ar_text = df_text = ""
    ar_count = df_count = 0
    ar_idx = df_idx = 0
    ar_final_tps = len(ar_events) / max(0.01, ar_events[-1][0])
    df_final_tps = len(df_events) / max(0.01, df_events[-1][0])

    end_time = max(ar_events[-1][0], df_events[-1][0]) + 0.3
    total_render = min(args.max_duration, end_time)
    frames: list[Image.Image] = []
    frame_dt = 1.0 / FPS

    font = resolve_font(14)
    font_big = resolve_font(22)
    badge_font = resolve_font(52, sans=True)

    t = 0.0
    while t <= total_render:
        # Consume AR events up to t
        while ar_idx < len(ar_events) and ar_events[ar_idx][0] <= t:
            _, _, s = ar_events[ar_idx]
            ar_text += s; ar_count += 1; ar_idx += 1
        while df_idx < len(df_events) and df_events[df_idx][0] <= t:
            _, _, s = df_events[df_idx]
            df_text += s; df_count += 1; df_idx += 1

        ar_tps = ar_count / max(0.01, min(t, ar_events[-1][0])) if ar_count else None
        df_tps = df_count / max(0.01, min(t, df_events[-1][0])) if df_count else None

        frames.append(render_frame(
            ar_text, df_text, t,
            ar_count, df_count,
            ar_final_tps if ar_count > 0 else None,
            df_final_tps if df_count > 0 else None,
            font, font_big, badge_font,
        ))
        t += frame_dt

    # Hold last frame ~1.5s
    for _ in range(int(1.5 * FPS)):
        frames.append(frames[-1])

    print(f"[demo] rendered {len(frames)} frames, saving {args.out}", flush=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"[demo] ok — AR {ar_count} tok ({ar_final_tps:.1f} tok/s),  "
          f"DFlash {df_count} tok ({df_final_tps:.1f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
