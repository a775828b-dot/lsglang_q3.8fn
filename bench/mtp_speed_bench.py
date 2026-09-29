"""Full-model MTP speed probe: 4 domains x 2K input / 512 output (temperature 0,
ignore_eos, cache flushed) plus one cold 32K prefill for TTFT. verify_step_ms =
ms/token x accept length, i.e. the cost of one speculative verify step."""
import argparse, json, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer
p = argparse.ArgumentParser(); p.add_argument('--label', required=True); p.add_argument('--output', type=Path, required=True)
p.add_argument('--base-url', default='http://127.0.0.1:18080'); p.add_argument('--repeats', type=int, default=2); p.add_argument('--model', required=True); a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
def post(path, body):
    req = urllib.request.Request(a.base_url+path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r: return r.read()
tok = AutoTokenizer.from_pretrained(a.model)
filler = tok.encode('Archive background: the following pages describe a small library and its catalogue, which is updated every week.\n', add_special_tokens=False)
tasks = [('chinese', '请用中文详细解释操作系统如何调度进程、管理虚拟内存和处理文件缓存，给出具体例子及常见误区。'),
         ('english', 'Explain in detail how a community library can design a practical programme for teaching adults to read, including scheduling, feedback and evaluation.'),
         ('python', 'Implement a robust Python asynchronous task scheduler with bounded concurrency, retries, cancellation, shutdown and complete tests. Explain the design.'),
         ('math', 'Explain step by step how to derive the sum of the first n integers, the sum of their squares, and the relationship to induction. Include worked numerical examples and verification.')]
def ids_for(question, size_total):
    prompt = tok.apply_chat_template([{'role': 'system', 'content': 'Answer the final question accurately and in detail.'},
        {'role': 'user', 'content': 'Background material:\n<BODY>\nFinal question:\n'+question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    left, right = prompt.split('<BODY>'); pre = tok.encode(left, add_special_tokens=False); suf = tok.encode(right, add_special_tokens=False)
    size = size_total-len(pre)-len(suf); return pre+(filler*((size+len(filler)-1)//len(filler)))[:size]+suf
def generate(ids, new_tokens):
    post('/flush_cache', {})
    req = urllib.request.Request(a.base_url+'/generate', data=json.dumps(dict(input_ids=ids,
        sampling_params=dict(temperature=0, max_new_tokens=new_tokens, ignore_eos=True), stream=True)).encode(), headers={'Content-Type': 'application/json'})
    begin = time.perf_counter(); first = None; ft = None; last = {}; text = ''
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b'data:'): continue
            raw = raw[5:].strip()
            if raw == b'[DONE]': break
            v = json.loads(raw); last = v.get('meta_info', last)
            if v.get('text'):
                if first is None: first = time.perf_counter(); ft = int(last['completion_tokens'])
                text = v['text']
    end = time.perf_counter()
    return begin, first, ft, end, last, text
rows = []
for rep in range(a.repeats):
    for name, q in tasks:
        b, f0, ft, e, last, text = generate(ids_for(q, 2048), 512)
        ms = (e-f0)*1000/(512-ft); acc = last.get('spec_accept_length') or 1.0
        rows.append(dict(kind='decode2k', domain=name, rep=rep, ttft_s=f0-b, decode_tok_s=1000/ms, ms_per_token=ms,
                         accept_length=acc, verify_step_ms=ms*acc, text_head=text[:120]))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
b, f0, ft, e, last, text = generate(ids_for(tasks[1][1], 32768), 8)
rows.append(dict(kind='cold32k_ttft', ttft_s=f0-b)); print(json.dumps(rows[-1]), flush=True)
d = [r for r in rows if r['kind'] == 'decode2k']
summary = dict(label=a.label, decode_tok_s=sum(r['decode_tok_s'] for r in d)/len(d), verify_step_ms=sum(r['verify_step_ms'] for r in d)/len(d),
               accept_length=sum(r['accept_length'] for r in d)/len(d), cold32k_ttft_s=rows[-1]['ttft_s'])
(a.output/f'{a.label}.json').write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1))
print(json.dumps(summary), flush=True)
