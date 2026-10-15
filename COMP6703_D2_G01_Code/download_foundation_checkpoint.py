"""Attempt to obtain a public EEG foundation-model checkpoint.

The script is intentionally separate from the offline baseline: it never falls
back to fabricated weights. Network access is required. Candidate repositories
are documented from the EEGPT / LaBraM model families and every attempt is
recorded in `foundation_download_status.json`.
"""
from pathlib import Path
import json, os, urllib.request

CANDIDATES = [
    'BINEAI/EEGPT',
    'LaBraM/LaBraM-base',
]
def main():
    status={'network_available':False,'candidates':CANDIDATES,'attempts':[]}
    for repo in CANDIDATES:
        url='https://huggingface.co/api/models/'+repo
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                raw=r.read().decode('utf-8'); status['network_available']=True
                meta=json.loads(raw); status['attempts'].append({'repo':repo,'ok':True,'sha':meta.get('sha'),'siblings':[x.get('rfilename') for x in meta.get('siblings',[])][:30]})
        except Exception as e:
            status['attempts'].append({'repo':repo,'ok':False,'error':repr(e)})
    Path('foundation_download_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
if __name__=='__main__': main()
