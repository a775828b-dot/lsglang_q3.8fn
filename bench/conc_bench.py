#!/usr/bin/env python3
"""Concurrent decode speed: C streaming /generate requests started at the same moment.

For each concurrency C in --levels: --rounds rounds; a round starts C requests together (different short prompts,
chat template, thinking on), --max-new tokens each with ignore_eos. Per request: TTFT and decode tok/s
= (tokens - 1) / (last - first token time). Per round: aggregate = all generated tokens / (last token of the round
- first token of the round). With --max-running-requests 1 the second request waits for the first, so the
aggregate stays at the single-request speed.

Usage: python conc_bench.py --port 18080 --tokenizer /path/to/model [--levels 1,2] [--rounds 4] [--max-new 512]
       [--temperature 1.0] [--out result.json]
"""
import argparse
import http.client
import json
import statistics as st
import threading
import time

from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18080)
P.add_argument('--tokenizer', required=True, help='model directory with the tokenizer')
P.add_argument('--levels', default='1,2')
P.add_argument('--rounds', type=int, default=4)
P.add_argument('--max-new', type=int, default=512)
P.add_argument('--temperature', type=float, default=1.0)
P.add_argument('--out')
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
PROMPTS = [
    '请解释为什么天空是蓝色的，以及日落时天空为什么会变红。分点说明，语言通俗。',
    '用 Python 写一个异步网页下载器，要求有并发限制、失败重试和断点续传，并用中文解释关键设计。',
    'Write a short technical blog post comparing Rust and Go for building network services, with code examples.',
    '一个水池有进水管和出水管，进水管单独 6 小时注满，出水管单独 9 小时放完。两管同时开，多少小时注满？给出详细推理，再推广到一般情况。',
    'Implement an LRU cache in TypeScript with O(1) get/put, generics, and unit tests using vitest. Explain the design briefly.',
    '设计一个支持多租户的 API 网关限流方案：说明令牌桶与滑动窗口的取舍，给出 Redis Lua 脚本实现，并讨论分布式一致性问题。',
    'Explain the CAP theorem with concrete examples of databases in each category, then discuss PACELC.',
    '帮我写一封给客户的道歉邮件：我们的服务昨晚中断了 3 小时，说明原因（数据库主从切换失败）、补偿方案和改进措施。',
]
PARAMS = dict(temperature=A.temperature, top_p=0.95) if A.temperature > 0 else dict(temperature=0.0)


def one(text, res, i):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, tokenize=False)
    body = json.dumps({'text': prompt, 'stream': True,
                       'sampling_params': dict(PARAMS, max_new_tokens=A.max_new, ignore_eos=True)}).encode()
    conn = http.client.HTTPConnection('127.0.0.1', A.port, timeout=3600)
    t0 = time.time()
    conn.request('POST', '/generate', body=body, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    first = last = None
    meta, buf = {}, b''
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            event, buf = buf.split(b'\n\n', 1)
            if not event.startswith(b'data: ') or event == b'data: [DONE]':
                continue
            now = time.time()
            first = first or now
            last = now
            meta = json.loads(event[6:]).get('meta_info') or meta
    n = meta.get('completion_tokens', 0)
    v = meta.get('spec_verify_ct') or 0
    res[i] = dict(tokens=n, ttft=first - t0, first=first, last=last,
                  decode=(n - 1) / (last - first) if last and last > first else 0.0, accept=n / v if v else None)


summary, k = {}, 0
for c in [int(x) for x in A.levels.split(',')]:
    rows = []
    for r in range(A.rounds):
        res = [None] * c
        th = [threading.Thread(target=one, args=(PROMPTS[(k + i) % len(PROMPTS)], res, i)) for i in range(c)]
        k += c
        for t in th:
            t.start()
        for t in th:
            t.join()
        agg = sum(x['tokens'] for x in res) / (max(x['last'] for x in res) - min(x['first'] for x in res))
        rows.append(dict(round=r, aggregate=agg, requests=res))
        print(f'C={c} round {r}: per-request decode ' + ' / '.join(f"{x['decode']:.1f}" for x in res) +
              f' tok/s, aggregate {agg:.1f} tok/s, TTFT ' + ' / '.join(f"{x['ttft']:.2f}" for x in res) + ' s', flush=True)
    acc = [x['accept'] for row in rows for x in row['requests'] if x['accept']]
    summary[c] = dict(per_request=round(st.mean(x['decode'] for row in rows for x in row['requests']), 1),
                      aggregate=round(st.mean(row['aggregate'] for row in rows), 1),
                      accept=round(st.mean(acc), 2) if acc else None,
                      ttft=round(st.mean(x['ttft'] for row in rows for x in row['requests']), 2), rows=rows)
    print(f'== C={c}: per-request {summary[c]["per_request"]} tok/s, aggregate {summary[c]["aggregate"]} tok/s, '
          f'accept {summary[c]["accept"]}, TTFT {summary[c]["ttft"]} s', flush=True)
if A.out:
    json.dump(dict(max_new=A.max_new, params=PARAMS, summary=summary), open(A.out, 'w'), ensure_ascii=False, indent=1)
