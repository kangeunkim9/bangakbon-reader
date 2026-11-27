# 8차 학습 셀(Colab). 7차 셀을 실행한 뒤 이어서 돌린다.
#@title ⑪ 8차 학습: 7차에서 이어서, AI허브 TS_1까지 넣어 학습 → 금향졍긔·AI허브 소설 시험 → 웹용 내보내기
import os, json, random, time, shutil, torch, numpy as np, torch.nn as nn
from torch.utils.data import DataLoader
ROOT='/content/drive/MyDrive/방각본_판독기'
AH8=[json.loads(l) for l in open('/content/aihub_lines/lines.jsonl',encoding='utf-8')]
AH8=[(f"/content/aihub_lines/{r['img']}",r['text'],r['book']) for r in AH8 if len(G(r['text']))>=3 and os.path.exists(f"/content/aihub_lines/{r['img']}")]
AH8tr=[(p,t) for p,t,b in AH8 if not hold(b)]; AH8te=[(p,t) for p,t,b in AH8 if hold(b)]
print('AI허브 학습',len(AH8tr),'행 / 소설 시험',len(AH8te),'행 / 경판 학습',len(KTR),'행',flush=True)
cs8=Charset([t for _,t in KTR+KVA]+[t for _,t,_ in AH8]); print('자모',len(cs7b.chars),'→',len(cs8.chars),'종',flush=True)
random.seed(0); torch.manual_seed(0)
m8=CRNN(len(cs8)).to(DEV); sd=m7b.state_dict(); own=m8.state_dict()
for k,v in sd.items():
    if not k.startswith('fc.'): own[k]=v
for c,i in cs7b.idx.items():
    if c in cs8.idx: own['fc.weight'][cs8.idx[c]]=sd['fc.weight'][i]; own['fc.bias'][cs8.idx[c]]=sd['fc.bias'][i]
own['fc.weight'][0]=sd['fc.weight'][0]; own['fc.bias'][0]=sd['fc.bias'][0]; m8.load_state_dict(own)
EP=20; NAH=min(len(AH8tr),25000); BS=16; LR=5e-4
steps=EP*((len(KTR)+NAH)//BS)
opt=torch.optim.AdamW(m8.parameters(),lr=LR,weight_decay=1e-4); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1)
ctc=nn.CTCLoss(blank=0,zero_infinity=True); best=9; log8=[]
SAVE8='/content/gyeongpan_model_v8.pt'
for ep in range(1,EP+1):
    tr=KTR+random.sample(AH8tr,NAH); dl=DataLoader(LineDS(tr,cs8,True),batch_size=BS,shuffle=True,num_workers=2,collate_fn=collate,drop_last=True)
    m8.train(); t0=time.time(); tot=0; n=0
    for x,tg in dl:
        x=x.to(DEV); y=m8(x).log_softmax(-1).permute(1,0,2)
        loss=ctc(y,torch.tensor(sum(tg,[])),torch.full((len(tg),),y.shape[0],dtype=torch.long),torch.tensor([len(t) for t in tg]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m8.parameters(),5); opt.step()
        if sch.last_epoch+1<steps: sch.step()
        tot+=loss.item(); n+=1
    c=cer(predict(m8,cs8,[p for p,_ in KVA],DEV),[t for _,t in KVA]); log8.append((ep,tot/n,c))
    if c<best:
        best=c; torch.save({'state':m8.state_dict(),'chars':cs8.chars,'H':H},SAVE8); shutil.copy(SAVE8,ROOT+'/1_모델/gyeongpan_model_v8_진행중.pt')
    print(f'epoch {ep:2d}  loss {tot/n:.3f}  경판 검증 {c*100:.1f}% (최고 {best*100:.1f}%)  {time.time()-t0:.0f}s',flush=True)
m8b,cs8b=load_model(SAVE8,DEV)
r8={'gh_v7':cer(predict(m7b,cs7b,tp2,DEV),tg2),'gh_v8':cer(predict(m8b,cs8b,tp2,DEV),tg2)}
sm=random.Random(1).sample(AH8te,min(800,len(AH8te)))
r8['novel_v7']=cer(predict(m7b,cs7b,[p for p,_ in sm],DEV),[t for _,t in sm]); r8['novel_v8']=cer(predict(m8b,cs8b,[p for p,_ in sm],DEV),[t for _,t in sm])
print('[처음 보는 작품] 금향졍긔 뒤쪽 절반  7차 %.1f%% → 8차 %.1f%%'%(r8['gh_v7']*100,r8['gh_v8']*100))
print('[AI허브 소설]  7차 %.1f%% → 8차 %.1f%%'%(r8['novel_v7']*100,r8['novel_v8']*100))
torch.save({'state':m8b.state_dict(),'chars':cs8b.chars,'H':H,'arch':'CRNN'},ROOT+'/1_모델/gyeongpan_model_v8.pt')
json.dump({'result':r8,'log':log8,'n_kyeong':len(KTR),'n_aihub':len(AH8tr)},open(ROOT+'/기록/result_v8.json','w'),ensure_ascii=False)
# 웹용 내보내기 (9MB 조각)
import onnxruntime as ort
OUTW=ROOT+'/1_모델/web_v8'; os.makedirs(OUTW,exist_ok=True)
mm=m8b.cpu().eval(); kw=dict(input_names=['x'],output_names=['y'],dynamic_axes={'x':{3:'w'},'y':{1:'t'}},opset_version=17)
try: torch.onnx.export(mm,torch.zeros(1,1,64,800),'/content/m8.onnx',dynamo=False,**kw)
except TypeError: torch.onnx.export(mm,torch.zeros(1,1,64,800),'/content/m8.onnx',**kw)
d=open('/content/m8.onnx','rb').read(); P=9_000_000; n=0
for i in range(0,len(d),P): open(f'{OUTW}/model_v8_{n}.wasm','wb').write(d[i:i+P]); n+=1
json.dump({'chars':cs8b.chars,'H':64,'MAXW':1600},open(OUTW+'/charset.json','w'),ensure_ascii=False)
print('8차 웹용 조각',n,'개 저장 끝')
