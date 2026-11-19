import json, tempfile, unittest, subprocess, sys, types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from hisav.data import CampusData
from hisav.utils import sha
from hisav.infer import run

class CampusContracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name);(self.root/'manifests').mkdir();self.rows=[]
  self.bundle=self.root/'bundle';self.bundle.mkdir();(self.bundle/'PROVENANCE.json').write_text('{"files":{}}')
  for i,n in enumerate((7000,300)):
   d=self.root/f'recording{i}';d.mkdir();x=np.tile(np.linspace(-.2,.2,n,dtype=np.float32)[:,None],(1,6))
   sf.write(d/'audio.wav',x,16000,subtype='FLOAT');(d/'video.mp4').write_bytes(b'video-hash-contract-fixture')
   frames=(n+639)//640
   np.savez(d/'mouth.npz',data=np.ones((frames,96,96),dtype=np.uint8),valid=np.ones(frames,dtype=bool))
   row=dict(schema_version='elma-campus-v1',uid=f'recording{i}',script_id=i+1,sample_rate=16000,channels=6,mouth_fps=25,samples=n,mouth_frames=frames,reference_audio_available=False,sha256={})
   for k,name in [('audio','audio.wav'),('video','video.mp4'),('mouth','mouth.npz')]:row[k+'_path']=f'recording{i}/{name}';row['sha256'][k]=sha(d/name)
   self.rows.append(row)
  self.manifest=self.root/'manifests/tt.jsonl';self.write_manifest()
 def write_manifest(self):self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
 def args(self):return SimpleNamespace(data=str(self.root),output=str(self.root/'output'),data_format='campus',split='tt',rank=0,world=1,limit=0,device='cpu')
 def test_one_reader_without_references(self):
  ds=CampusData(self.root);self.assertEqual(ds.check(),2);s=ds.scene(0)
  self.assertEqual(s['wave'].shape,(6,7000));self.assertEqual(len(s['mouth']),1);self.assertIsNone(s['reference'])
 def test_manifest_rejects_geometry_and_path_errors(self):
  for key,value in [('channels',1),('mouth_fps',30),('audio_path','../escape.wav')]:
   with self.subTest(key=key):
    old=self.rows[0][key];self.rows[0][key]=value;self.write_manifest()
    with self.assertRaises(ValueError):CampusData(self.root)
    self.rows[0][key]=old
 def test_payload_hash_required(self):
  with (self.root/'recording0/audio.wav').open('ab') as f:f.write(b'changed')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):CampusData(self.root).scene(0)
 def test_real_audio_channels_checked(self):
  p=self.root/'recording0/audio.wav';sf.write(p,np.zeros(7000),16000);self.rows[0]['sha256']['audio']=sha(p);self.write_manifest()
  with self.assertRaisesRegex(ValueError,'six-channel'):CampusData(self.root).scene(0)
 def test_mouth_mask_and_lengths_checked(self):
  p=self.root/'recording0/mouth.npz';np.savez(p,data=np.zeros((11,96,96),np.uint8),valid=np.ones(10,bool));self.rows[0]['sha256']['mouth']=sha(p);self.write_manifest()
  with self.assertRaisesRegex(ValueError,'validity mask'):CampusData(self.root).scene(0)
 def test_cli_check_needs_only_test_manifest_and_no_model(self):
  proc=subprocess.run([sys.executable,'-m','hisav','infer','--data',str(self.root),'--data-format','campus','--bundle',str(self.bundle),'--output',str(self.root/'unused'),'--check'],text=True,capture_output=True)
  self.assertEqual(proc.returncode,0,proc.stderr);result=json.loads(proc.stdout)
  self.assertEqual(result['counts'],{'tt':2});self.assertFalse(result['model_initialized']);self.assertFalse((self.root/'unused').exists())
 def test_streaming_reset_start_tail_and_no_reference_metric(self):
  observed=[];outer=self
  class Engine:
   def __init__(self,device):self.resets=0
   def reset(self):self.resets+=1;self.first=True;observed.append(('reset',self.resets))
   def step(self,wave,mouth,valid):
    outer.assertEqual(tuple(wave.shape),(1,6,16000));outer.assertEqual(tuple(mouth.shape),(1,25,96,96))
    if self.first:outer.assertTrue((wave[:,:,:6400]==0).all());outer.assertFalse(valid[:,:10].any());self.first=False
    observed.append(('step',self.resets))
    return {'waveform':wave[:,0,6400:12800],'choice':torch.zeros(1,dtype=torch.long)}
  with patch.dict(sys.modules,{'hisav.model':types.SimpleNamespace(StreamingEngine=Engine)}),patch('hisav.paths.BUNDLE',self.bundle),patch('hisav.infer.si',side_effect=AssertionError('No clean references')):
   run(self.args())
  self.assertEqual(observed,[('reset',1),('step',1),('step',1),('reset',2),('step',2)])
  for i,r in enumerate(self.rows):
   audio,_=sf.read(self.root/r['audio_path'],dtype='float32',always_2d=True);y,sr=sf.read(self.root/f'output/{i:06d}/target.wav',dtype='float32')
   np.testing.assert_array_equal(y,audio[:,0]);self.assertEqual(len(y),r['samples']);self.assertEqual(sr,16000)
  result=json.loads((self.root/'output/COMPLETE_rank0.json').read_text());self.assertEqual(result['targets'],2);self.assertIsNone(result['si_snr'])
 def test_save_still_emits_two_targets_and_reference_metrics(self):
  x=np.sin(np.arange(7000,dtype=np.float32)*.03);scene={'wave':np.tile(x[None],(6,1)),'mouth':[np.zeros((11,96,96),np.uint8)]*2,'valid':[np.ones(11,bool)]*2,'reference':np.stack([x,-x])}
  class Engine:
   def __init__(self,device):pass
   def reset(self):pass
   def step(self,w,m,v):return {'waveform':torch.cat([w[:,0,6400:12800],-w[:,0,6400:12800]]),'choice':torch.tensor([0,1])}
  ds=SimpleNamespace(rows=[{'uid':'save_fixture'}],scene=lambda i:scene)
  args=self.args();args.data_format='save'
  with patch.dict(sys.modules,{'hisav.model':types.SimpleNamespace(StreamingEngine=Engine)}),patch('hisav.paths.BUNDLE',self.bundle),patch('hisav.infer.SaveData',return_value=ds):run(args)
  result=json.loads((self.root/'output/COMPLETE_rank0.json').read_text());self.assertEqual(result['targets'],2);self.assertTrue(np.isfinite(result['si_snr']))
  self.assertTrue((self.root/'output/000000/target_a.wav').exists());self.assertTrue((self.root/'output/000000/target_b.wav').exists())

if __name__=='__main__':unittest.main()
