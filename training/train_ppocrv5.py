# PP-OCRv5 파인튜닝 시험 셀(Colab).
#@title ㉜ PP-OCRv5 시험: PaddleOCR의 PP-OCRv5 server 인식 모델을 옛한글 자모로 다시 가르쳐 10차와 비교
# 10차와 같은 학습 자료를 쓰고 같은 시험(금향졍긔 권하 뒤쪽 절반 등)으로 비교한다. 사이트에는 아직 올리지 않는다.
# PP-OCRv5: PaddleOCR(Apache-2.0), https://github.com/PaddlePaddle/PaddleOCR
# 학습은 PaddleOCR 공식 학습 도구(tools/train.py)로 한다. 덧붙임도 PaddleOCR 것(RecAug)을 쓴다.
# 판독 출력은 CTC라서 지금 사이트의 언어 모델 보정을 그대로 붙일 수 있다.
import subprocess, sys, os
PADDLEOCR_COMMIT = 'dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf'
PP = '/content/PaddleOCR'
if not os.path.isdir(PP):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/PaddlePaddle/PaddleOCR', PP], check=True)
    subprocess.run(['git', '-C', PP, 'checkout', '-q', PADDLEOCR_COMMIT], check=True)
try:
    import paddle
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'paddlepaddle-gpu', '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', PP + '/requirements.txt'], check=False)
import requests
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
src10 = requests.get(RAW + 'train_v10.py', timeout=60).text
exec(src10[:src10.index('# ---------------- 4)')], globals())     # 10차와 같은 학습 자료

import yaml, copy, multiprocessing as mp
CH = list(load_model(ROOT + '/gyeongpan_model_v10.pt', 'cpu')[1].chars)   # 10차와 같은 자모 222종
CHS = set(CH); MAXL = 100; PW = 1280        # 행 하나에 자모 100개까지, 높이 48에 너비 1280까지
lab = lambda t: ''.join(c for c in nfd(t) if c in CHS)

# ---------------- 1) 학습 자료를 PaddleOCR 형식으로: 눕힌 행 그림 + 목록 파일 ----------------
D = '/content/ppd'; os.makedirs(D + '/img', exist_ok=True)
open(D + '/dict.txt', 'w', encoding='utf-8').write('\n'.join(CH) + '\n')
def _prep(args):
    src, dst = args
    if not os.path.exists(dst):
        im = Image.open(src).convert('L').rotate(90, expand=True)
        ImageOps.autocontrast(im, cutoff=1).save(dst)
    return dst
def to_list(pairs, tag):
    items = [(p, t) for p, t in pairs if 1 <= len(lab(t)) <= MAXL]
    jobs = [(p, f'{D}/img/{tag}_{i}.png') for i, (p, _) in enumerate(items)]
    with mp.Pool(8) as pool: pool.map(_prep, jobs, chunksize=256)
    return [f'img/{tag}_{i}.png\t{lab(t)}' for i, (_, t) in enumerate(items)], len(pairs) - len(items)
L_k, a = to_list(KTR, 'k'); L_i, b = to_list(INTR, 'i'); L_a, c = to_list(AHtr, 'a'); L_p, d = to_list(PS_tr, 'p'); L_v, _ = to_list(KVA, 'v')
print(f'자모 {MAXL}개를 넘어 뺀 행: 경판 {a}, 보낸 면 {b}, 목판본 {c}, 필사본 {d}', flush=True)
# 10차는 에폭마다 경판·보낸 면(x3)·필사본 전부 + 목판본 2만 5천 행을 8에폭 본다(목판본은 에폭마다 새로 뽑음).
# 여기서는 목판본 전부 + 나머지를 4번씩 넣은 목록으로 2에폭 돌려 본 행 수와 비율을 비슷하게 맞춘다.
TR = (L_k + L_i * 3 + L_p) * 4 + L_a
random.Random(0).shuffle(TR)
open(D + '/train.txt', 'w', encoding='utf-8').write('\n'.join(TR) + '\n')
open(D + '/val.txt', 'w', encoding='utf-8').write('\n'.join(L_v) + '\n')
print('학습 목록', len(TR), '행', flush=True)

# ---------------- 2) 설정: PP-OCRv5 server 인식 모델 + 중국어 등 학습 가중치에서 출발 ----------------
BS = 32; EP = 2
cfg = yaml.safe_load(open(PP + '/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml'))
OUT = ROOT + '/ppocrv5_out'
cfg['Global'].update(epoch_num=EP, use_gpu=True, distributed=False, character_dict_path=D + '/dict.txt', max_text_length=MAXL,
                     use_space_char=False, save_model_dir=OUT, save_epoch_step=1, print_batch_step=200, use_amp=True, amp_level='O1',
                     eval_batch_step=[0, len(TR) // BS // 2], cal_metric_during_train=False, d2s_train_image_shape=[3, 48, PW],
                     pretrained_model='https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams')
cfg['Optimizer']['lr'].update(learning_rate=2e-4, warmup_epoch=0.1)
cfg['Architecture']['Head']['head_list'][1]['NRTRHead']['max_text_length'] = MAXL
keep = {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']}}
cfg['Train']['dataset'] = {'name': 'SimpleDataSet', 'data_dir': D, 'label_file_list': [D + '/train.txt'],
    'transforms': [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}, {'RecAug': None},
                   {'MultiLabelEncode': {'gtc_encode': 'NRTRLabelEncode'}}, {'RecResizeImg': {'image_shape': [3, 48, PW]}}, keep]}
cfg['Train'].pop('sampler', None); cfg['Train']['loader'].update(batch_size_per_card=BS, num_workers=4)
cfg['Eval']['dataset'].update(data_dir=D, label_file_list=[D + '/val.txt'])
cfg['Eval']['dataset']['transforms'][2]['RecResizeImg']['image_shape'] = [3, 48, PW]
cfg['Eval']['loader'].update(batch_size_per_card=64, num_workers=2)
yaml.safe_dump(cfg, open(D + '/cfg.yml', 'w'), allow_unicode=True)

# ---------------- 3) 학습 (PaddleOCR 공식 도구, 끊기면 latest에서 이어서) ----------------
import gc
for _k in ('sysm', 'svm', 'm10', 'R10', 'opt'): globals().pop(_k, None)   # 앞 셀이 남긴 GPU 메모리 비우기
gc.collect(); torch.cuda.empty_cache()
resume = ['-o', f'Global.checkpoints={OUT}/latest'] if os.path.exists(OUT + '/latest.pdparams') else []
p = subprocess.Popen([sys.executable, 'tools/train.py', '-c', D + '/cfg.yml'] + resume, cwd=PP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if any(k in line for k in ('epoch:', 'best metric', 'Error', 'Traceback', 'save model', 'load pretrain')):
        print(line.rstrip()[:300], flush=True)
assert p.wait() == 0, '학습이 중간에 멈췄습니다'

# ---------------- 4) 시험: 10차와 나란히 ----------------
import paddle
sys.path.insert(0, PP)
from ppocr.modeling.architectures import build_model as pp_build
arch = copy.deepcopy(cfg['Architecture']); nc = len(CH) + 1
arch['Head']['out_channels_list'] = {'CTCLabelDecode': nc, 'NRTRLabelDecode': nc + 3}
ppm = pp_build(arch)
WGT = OUT + '/best_accuracy.pdparams' if os.path.exists(OUT + '/best_accuracy.pdparams') else OUT + '/latest.pdparams'
ppm.set_state_dict(paddle.load(WGT)); ppm.eval()
def prep_pp(src):
    im = (Image.open(src) if isinstance(src, str) else src).convert('L').rotate(90, expand=True)
    im = ImageOps.autocontrast(im, cutoff=1); w, h = im.size
    nw = min(PW, max(16, int(math.ceil(48 * w / h))))
    return np.asarray(im.resize((nw, 48), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
def read_pp(srcs, bs=32):
    out = []
    for i in range(0, len(srcs), bs):
        arrs = [prep_pp(s) for s in srcs[i:i + bs]]; W = max(a.shape[1] for a in arrs)
        x = np.zeros((len(arrs), 3, 48, W), np.float32)            # PaddleOCR처럼 남는 자리는 0
        for j, a in enumerate(arrs): x[j, :, :, :a.shape[1]] = a
        with paddle.no_grad():
            pr = ppm(paddle.to_tensor(x)); pr = pr['ctc'] if isinstance(pr, dict) else pr
        for r in pr.numpy().argmax(-1):
            s, prev = [], 0
            for k in r:
                if k != prev and k != 0: s.append(CH[k - 1])
                prev = k
            out.append(nfc(''.join(s)))
    return out
PSV = random.Random(2).sample(PS_te, min(600, len(PS_te)))
m10, cs10 = load_model(ROOT + '/gyeongpan_model_v10.pt', DEV)
R = {}
R['val_v10'] = cer(predict(m10, cs10, [p for p, _ in KVA], DEV), [t for _, t in KVA]); R['val_pp'] = cer(read_pp([p for p, _ in KVA]), [t for _, t in KVA])
R['pilsa_v10'] = cer(predict(m10, cs10, [p for p, _ in PSV], DEV), [t for _, t in PSV]); R['pilsa_pp'] = cer(read_pp([p for p, _ in PSV]), [t for _, t in PSV])
sm = random.Random(1).sample(AHte, min(800, len(AHte)))
R['simcheong_v10'] = cer(predict(m10, cs10, [p for p, _ in sm], DEV), [t for _, t in sm]); R['simcheong_pp'] = cer(read_pp([p for p, _ in sm]), [t for _, t in sm])
from rapidfuzz.distance import Levenshtein as Lv
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(requests.get(RAW + 'bulk_ocr.py', timeout=60).text, B)
KPG = json.load(gzip.open(ROOT + '/private/kohico_pages.json.gz', 'rt', encoding='utf-8'))
CL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
toks = lambda s: CL.findall(unicodedata.normalize('NFC', s))
GP = [v for k, v in KPG.items() if k.startswith('금향정기_2')][0]
GT = [g for g, _ in fix_rep([(t, 0) for k in sorted(GP) if k not in ('31ᄀ', '31ᄂ') for t in toks(re.sub(r'\s+', '', GP[k]))])[0]]
pdf = B['get_commons']('File:CNTS-00109637790 금향졍긔 이종. 권하.pdf', '/content/geumhyang2.pdf'); doc = B['fitz'].open(pdf)
def geum(read_face):
    own_ = []; flat = []
    for i in range(len(doc)):
        no = i + 1
        if not (3 <= no <= 62 or no == 65): continue
        im = B['norm_width'](B['page_gray'](doc, i)); t, bt, cols_ = B['segment'](im)[:3]
        crops = [im.crop((max(0, a - 3), t, c + 3, bt)) for a, c in cols_]
        for s in read_face(crops):
            tk = toks(s); flat += tk; own_ += [no] * len(tk)
    err = collections.Counter(); nn_ = collections.Counter(no >= 36 for no in own_)
    for op in Lv.editops(flat, GT):
        p = min(op.src_pos, len(own_) - 1); err[own_[p] >= 36] += 1
    return err[True] / nn_[True]
R10 = B['Reader'](ROOT + '/gyeongpan_model_v10.pt')
R['geum_v10'] = geum(lambda crops: [s for s, _ in R10.read([B['prep_arr'](c) for c in crops], bs=1)])
R['geum_pp'] = geum(read_pp)
print(f"\n[금향졍긔 뒤쪽 절반]  10차 {R['geum_v10']*100:.2f}%  PP-OCRv5 {R['geum_pp']*100:.2f}%  (언어 모델 없이)")
print(f"[경판 검증]          10차 {R['val_v10']*100:.2f}%  PP-OCRv5 {R['val_pp']*100:.2f}%")
print(f"[필사본 시험 행]      10차 {R['pilsa_v10']*100:.2f}%  PP-OCRv5 {R['pilsa_pp']*100:.2f}%")
print(f"[AI허브 심청전_04]   10차 {R['simcheong_v10']*100:.2f}%  PP-OCRv5 {R['simcheong_pp']*100:.2f}%")
json.dump({'result': R, 'weights': WGT}, open(ROOT + '/result_ppocrv5_v1.json', 'w'), ensure_ascii=False)
