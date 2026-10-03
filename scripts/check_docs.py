#!/usr/bin/env python3
"""Check local Markdown links, fences, snippet syntax and bilingual coverage.

This does not execute tutorial snippets or validate external integrations.
Run from any working directory: python3 scripts/check_docs.py
"""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')
FENCE = re.compile(r'^\s*(`{3,}|~{3,})(.*)$')


def parse_markdown(path):
    prose, snippets, errors = [], [], []
    opening = None
    body = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if opening is None:
            if match:
                opening = (match[1], match[2].strip().split(" ")[0], line_no)
                body = []
            else:
                prose.append((line_no, line))
        elif match and match[1][0] == opening[0][0] and len(match[1]) >= len(opening[0]) and not match[2].strip():
            snippets.append((opening[1].lower(), "\n".join(body) + "\n", opening[2]))
            opening = None
        else:
            body.append(line)
    if opening:
        errors.append(f"{path.relative_to(ROOT)}:{opening[2]} unclosed code fence")
    return prose, snippets, errors


def anchors(prose):
    result, counts = set(), Counter()
    for _, line in prose:
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        title = re.sub(r'<[^>]*>', '', match[1])
        title = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', title)
        slug = ''.join(c for c in title.lower() if c.isalnum() or c in '_- ').replace(' ', '-')
        number = counts[slug]
        counts[slug] += 1
        result.add(slug if number == 0 else f"{slug}-{number}")
    return result


def main():
    paths = sorted(ROOT.rglob('*.md'))
    parsed = {p: parse_markdown(p) for p in paths if '.git' not in p.parts}
    errors, warnings, external = [], [], set()
    local_count = python_count = shell_count = 0
    for path, (prose, snippets, parse_errors) in parsed.items():
        rel = path.relative_to(ROOT)
        errors.extend(parse_errors)
        for line_no, line in prose:
            for target in LINK.findall(line):
                target = target.strip('<>')
                parts = urlsplit(target)
                if parts.scheme in ('http', 'https'):
                    external.add(target)
                    continue
                if parts.scheme or target.startswith('//'):
                    continue
                local_count += 1
                dest = (path.parent / unquote(parts.path)).resolve() if parts.path else path
                if not dest.is_relative_to(ROOT) or not dest.exists():
                    errors.append(f'{rel}:{line_no} missing/escaping local link: {target}')
                elif parts.fragment and dest.suffix == '.md':
                    if unquote(parts.fragment) not in anchors(parsed[dest][0]):
                        errors.append(f'{rel}:{line_no} missing heading anchor: {target}')
        for language, code, line_no in snippets:
            try:
                if language in ('python', 'py'):
                    ast.parse(code)
                    python_count += 1
                elif language in ('bash', 'sh'):
                    checked = subprocess.run(['bash', '-n'], input=code, text=True, capture_output=True)
                    if checked.returncode:
                        raise ValueError(checked.stderr.strip())
                    shell_count += 1
                elif language == 'json':
                    json.loads(code)
                elif language in ('xml', 'svg'):
                    ET.fromstring(code)
            except (SyntaxError, ValueError, ET.ParseError) as exc:
                errors.append(f'{rel}:{line_no} {language} syntax: {exc}')
    pairs = [(ROOT / 'README.md', ROOT / 'README.en.md'), (ROOT / 'VALIDATION.md', ROOT / 'VALIDATION.en.md')]
    pairs += [(p, ROOT / 'docs/en' / p.name) for p in sorted((ROOT / 'docs').glob('*.md'))]
    pairs += [(ROOT / 'examples/README.md', ROOT / 'examples/README.en.md')]
    for zh, en in pairs:
        if zh not in parsed or en not in parsed:
            errors.append(f'missing bilingual pair: {zh.relative_to(ROOT)} / {en.relative_to(ROOT)}')
            continue
        zh_text = zh.read_text(encoding='utf-8')
        en_text = en.read_text(encoding='utf-8')
        def local_destinations(path):
            result = set()
            for _, line in parsed[path][0]:
                for target in LINK.findall(line):
                    parts = urlsplit(target.strip('<>'))
                    if not parts.scheme and parts.path:
                        result.add((path.parent / unquote(parts.path)).resolve())
            return result
        if en not in local_destinations(zh) or zh not in local_destinations(en):
            errors.append(f'missing language navigation: {zh.relative_to(ROOT)}')
        zh_counts = Counter(lang for lang, _, _ in parsed[zh][1])
        en_counts = Counter(lang for lang, _, _ in parsed[en][1])
        if zh_counts != en_counts:
            errors.append(f'code-fence language/count mismatch: {zh.relative_to(ROOT)} {dict(zh_counts)} vs {dict(en_counts)}')
        urls = lambda text: {x.rstrip('.,') for x in re.findall(r'https?://[^\s)<>"`]+', text)}
        if urls(zh_text) != urls(en_text):
            warnings.append(f'external-source differences: {zh.relative_to(ROOT)}')
    for svg in (ROOT / 'assets').glob('*.svg'):
        try:
            ET.parse(svg)
        except ET.ParseError as exc:
            errors.append(f'{svg.relative_to(ROOT)} invalid SVG XML: {exc}')
    for line in errors:
        print('ERROR:', line)
    for line in warnings:
        print('REVIEW:', line)
    print(f'{len(parsed)} Markdown files; {len(pairs)} bilingual pairs; {local_count} local links; {len(external)} external link targets inventoried.')
    print(f'Syntax only: {python_count} Python and {shell_count} shell blocks; SVG XML parsed. External URLs were not requested by this script.')
    print(f'{len(errors)} errors; {len(warnings)} source-parity items for manual review.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
