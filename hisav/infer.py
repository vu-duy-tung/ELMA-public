"""Full-recording inference with exact tail trimming; metrics require references."""
from pathlib import Path
import json,time
import numpy as np,torch,soundfile as sf
from .data import SaveData,CampusData,crop
from .utils import sha

def si(y,r):
 y=y.astype(np.float64);r=r.astype(np.float64);y-=y.mean();r-=r.mean();e=np.dot(r,r)
 if e<1e-12:return None
 p=r*np.dot(y,r)/e;return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
def run(args):
 from .model import StreamingEngine
 from .paths import BUNDLE
 torch.set_num_threads(2)
 campus=getattr(args,'data_format','save')=='campus'
 if campus:
  # Match the retained campus inference precision settings.
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=(CampusData if campus else SaveData)(args.data,args.split)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 engine=StreamingEngine(device=args.device);results=[];begin=time.time()
 ids=list(range(len(ds.rows)))[:args.limit or None][args.rank::args.world]
 for i in ids:
  engine.reset();s=ds.scene(i);L=s['wave'].shape[-1];targets=len(s['mouth']);parts=[];choices=[]
  with torch.inference_mode():
   for pos in range(0,L,6400):
    wave=torch.from_numpy(crop(s['wave'],pos-6400,16000)[None]).to(args.device)
    mouth=torch.from_numpy(np.stack([crop(x,(pos-6400)//640,25,0) for x in s['mouth']])).to(args.device)
    valid=torch.from_numpy(np.stack([crop(x,(pos-6400)//640,25,0) for x in s['valid']])).to(args.device)
    y=engine.step(wave,mouth,valid);a=y['waveform'].cpu().numpy()
    assert a.shape==(targets,6400) and np.isfinite(a).all()
    parts.append(a[:,:min(6400,L-pos)]);choices.append(y['choice'].cpu().tolist())
  y=np.concatenate(parts,axis=1);folder=out/f'{i:06d}';folder.mkdir(exist_ok=True);files=[]
  for t in range(targets):
   p=folder/('target.wav' if campus else f'target_{"ab"[t]}.wav')
   sf.write(p,y[t],16000,subtype='FLOAT');files.append(dict(path=str(p),sha256=sha(p)))
  scores=[None]*targets if s['reference'] is None else [si(y[t],s['reference'][t]) for t in range(targets)]
  r=dict(scene=i,source_uid=ds.rows[i]['uid'],samples=L,si_snr=scores,choices=choices,files=files)
  if campus:r.update(script_id=ds.rows[i]['script_id'],reference_audio_available=False,input_sha256=ds.rows[i]['sha256'])
  (folder/'complete.json').write_text(json.dumps(r,allow_nan=False));results.append(r);print('SCENE',i,'complete',flush=True)
 values=[v for r in results for v in r['si_snr'] if v is not None]
 summary=dict(scenes=len(results),targets=sum(len(r['files']) for r in results),si_snr=float(np.mean(values)) if values else None,seconds=time.time()-begin,window=[.4,.4,.2],bundle_sha=sha(BUNDLE/'PROVENANCE.json'),manifest_sha=sha(Path(args.data)/'manifests'/f'{args.split}.jsonl'))
 if campus:summary.update(data_format='campus',reference_audio_available=False,metric_note='SI-SNR is unavailable without clean target references; no ASR is run.')
 (out/f'COMPLETE_rank{args.rank}.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,allow_nan=False))
