#!/bin/bash
# post.sh: create the sub-issues, then the umbrella, in tenstorrent/tt-metal. Needs gh auth. Run from this folder.
set -e
REPO=tenstorrent/tt-metal
PAGE="${PAGE:-(internal page link)}"
# 1. create each sub-issue with placeholders, remember numbers
: > numbers.tsv
while IFS=$'\t' read -r key title; do
  url=$(gh issue create -R $REPO --title "$title" --body "(being filled in)")
  echo -e "$key\t${url##*/}" >> numbers.tsv
done < titles.tsv
# 2. a placeholder umbrella to get its number
uurl=$(gh issue create -R $REPO --title "Wormhole LLK perf measurements: every cause of unstable, bistable or layout-dependent results" --body "(being filled in)")
U=${uurl##*/}
# 3. fill bodies with real numbers
fill() { python3 - "$1" "$U" "$PAGE" <<'PY'
import sys
fn,U,page=sys.argv[1:4]
s=open(fn).read()
nums=dict(l.strip().split('\t') for l in open('numbers.tsv'))
s=s.replace('{{UMBRELLA}}','#'+U).replace('{{PAGE}}',page)
for k,v in nums.items(): s=s.replace('{{'+k.upper()+'}}','#'+v)
print(s)
PY
}
while IFS=$'\t' read -r key num; do fill $key.md > /tmp/body_$key.md; gh issue edit -R $REPO $num --body-file /tmp/body_$key.md; done < numbers.tsv
fill umbrella.md > /tmp/body_umbrella.md; gh issue edit -R $REPO $U --body-file /tmp/body_umbrella.md
echo "umbrella #$U"; cat numbers.tsv
