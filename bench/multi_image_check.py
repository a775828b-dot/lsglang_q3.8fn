"""Multi-image requests against an OpenAI-compatible server: does it answer, how long, and peak VRAM.

Usage: python multi_image_check.py --port 18080 --model Qwen3.8-Flash-Next --images 1,4,8,16
Each request carries N synthetic 1920x1080 JPEG screenshots (gradient, boxes, a caption with a number)
and asks for the numbers in order, so a correct answer shows every image was read.
"""
import argparse, base64, io, json, random, subprocess, threading, time, urllib.request
from PIL import Image, ImageDraw, ImageFont

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, required=True)
P.add_argument('--model', required=True)
P.add_argument('--images', default='1,4,8,16')
P.add_argument('--size', default='1920x1080')
P.add_argument('--output', type=int, default=256)
P.add_argument('--out')
A = P.parse_args()
W, H = map(int, A.size.split('x'))
BASE = f'http://127.0.0.1:{A.port}'


def picture(i, rng):
    img = Image.new('RGB', (W, H))
    d = ImageDraw.Draw(img)
    c0, c1 = [tuple(rng.randint(0, 255) for _ in range(3)) for _ in range(2)]
    for y in range(0, H, 4):
        t = y / H
        d.rectangle([0, y, W, y + 4], fill=tuple(int(a + (b - a) * t) for a, b in zip(c0, c1)))
    for _ in range(12):
        x, y = rng.randint(0, W - 200), rng.randint(0, H - 200)
        d.rectangle([x, y, x + rng.randint(40, 400), y + rng.randint(40, 300)],
                    outline=(255, 255, 255), fill=tuple(rng.randint(0, 255) for _ in range(3)))
    number = rng.randint(100, 999)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 160)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([60, 60, 1000, 300], fill=(255, 255, 255))
    d.text((80, 80), f'No. {number}', fill=(0, 0, 0), font=font)
    buf = io.BytesIO(); img.save(buf, 'JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(), number


def vram():
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                       text=True).split()[0])


def run(n):
    rng = random.Random(1000 + n)
    pics = [picture(i, rng) for i in range(n)]
    content = [{'type': 'image_url', 'image_url': {'url': url}} for url, _ in pics]
    content.append({'type': 'text', 'text': f'There are {n} images. Each shows "No. <number>" in a white box. '
                                            'List the numbers in image order, comma separated, nothing else.'})
    body = dict(model=A.model, messages=[{'role': 'user', 'content': content}], max_tokens=A.output, temperature=0,
                stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': False})
    peak, stop = [vram()], threading.Event()
    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], vram()); time.sleep(0.1)
    threading.Thread(target=poll, daemon=True).start()
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time(); first = None; out = []; usage = {}; error = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                if not raw.startswith(b'data:') or raw[5:].strip() == b'[DONE]':
                    continue
                v = json.loads(raw[5:])
                usage = v.get('usage') or usage
                for c in v.get('choices', []):
                    piece = (c.get('delta', {}).get('content') or '') + (c.get('delta', {}).get('reasoning_content') or '')
                    if piece:
                        first = first or time.time(); out.append(piece)
    except Exception as exc:
        error = repr(exc)
    stop.set()
    answer = ''.join(out)
    want = [str(num) for _, num in pics]
    found = sum(1 for w in want if w in answer)
    return dict(images=n, prompt_tokens=usage.get('prompt_tokens'), ttft_s=first and round(first - t0, 2),
                numbers_found=f'{found}/{n}', answer_head=answer[:120], vram_peak_mib=peak[0], error=error)


rows = []
for n in [int(v) for v in A.images.split(',')]:
    rows.append(run(n))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
if A.out:
    json.dump(rows, open(A.out, 'w'), indent=1)
