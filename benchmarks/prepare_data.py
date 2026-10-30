"""Prepare the author's four DeepMind extrapolation files and optional mirror check.

Streams the official gzip archive, storing only the four selected files.
An existing hash-verified extraction is reused. No GPU or model download.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz'
MIRROR='https://modelscope.ai/api/v1/models/Spartan1925/Qwen3.5-4B-Qwenputer/repo/files?Revision=master&Recursive=true'
NAMES=('mul_big','div_big','mul_div_multiple_longer','add_or_sub_big')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(8*1024*1024): h.update(data)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/qwenputer_math_data')
    p.add_argument('--model',type=Path,help='hash existing checkpoint against ModelScope SHA256; no download')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    wanted={f'mathematics_dataset-v1.0/extrapolate/arithmetic__{n}.txt' for n in NAMES}
    manifest=args.output/'download_manifest.json'
    prior=json.loads(manifest.read_text()) if manifest.exists() else {}
    cached=set(prior.get('files',{}))==wanted and all(
        (args.output/'dm'/n).exists() and digest(args.output/'dm'/n)==f['sha256']
        for n,f in prior.get('files',{}).items())
    if not cached:
        record={'url':URL,'files':{}}
        with urllib.request.urlopen(URL,timeout=90) as response:
            record['headers']={k:v for k,v in response.headers.items() if k.lower() in ('etag','content-length','x-goog-hash')}
            with tarfile.open(fileobj=response,mode='r|gz') as archive:
                for member in archive:
                    name=member.name.lstrip('./')
                    if name not in wanted: continue
                    data=archive.extractfile(member).read()
                    target=args.output/'dm'/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(data)
                    record['files'][name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
                    print(name,len(data),flush=True)
                    if set(record['files'])==wanted: break
        assert set(record['files'])==wanted
        manifest.write_text(json.dumps(record,indent=2)+'\n')
    else:
        print('Four cached official test files match their recorded SHA256 hashes.',flush=True)
    if args.model:
        with urllib.request.urlopen(MIRROR,timeout=30) as response: files=json.load(response)['Data']['Files']
        checks=[]
        for f in files:
            name=f['Path']
            if not (name.endswith('.safetensors') or name in ('config.json','chipformer.json','tokenizer.json','tokenizer_config.json','chat_template.jinja')): continue
            actual=digest(args.model/name)
            match=actual==f['Sha256']
            checks.append({'path':name,'sha256':actual,'modelscope_sha256':f['Sha256'],'match':match})
            print(name,match,flush=True)
        assert checks and all(v['match'] for v in checks),'Checkpoint differs from the ModelScope mirror'
        (args.output/'modelscope_identity.json').write_text(json.dumps({'modelscope_repo':'Spartan1925/Qwen3.5-4B-Qwenputer','checks':checks},indent=2)+'\n')

if __name__=='__main__': main()
