// 방각본 판독기 학습 자료 받는 창구 (Google Apps Script 웹 앱)
// 판독기에서 '학습용으로 보내기'를 누르면 면 사진(JPEG)과 고친 판독문(JSON)이
// 내 드라이브 방각본_판독기/7_보내준_면/받은날짜 폴더에 쌓인다.
// 배포: 배포 → 새 배포 → 웹 앱, 실행 사용자 '나', 액세스 '모든 사용자'

var ROOT = '방각본_판독기', BOX = '7_보내준_면';
var MAX_IMAGE = 15 * 1024 * 1024;   // 사진 한 장 15MB까지
var MAX_PER_DAY = 1000;             // 하루에 받는 면 수 상한

function folder_(parent, name) {
  var it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : parent.createFolder(name);
}

function out_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}

// 면별로 몇 번 받았는지 (판독기가 아직 덜 고쳐진 면을 고를 때 씀)
function counts_() { try { return JSON.parse(PropertiesService.getScriptProperties().getProperty('cnt') || '{}'); } catch (e) { return {}; } }
// 함께 고친 사람들: 이름별 보낸 면 수, 고친 행 수, 처음·마지막 보낸 날
// 이름을 비우고 보낸 면은 '' 아래에 모인다. 목록에서 숨길 이름은 스크립트 속성 'hide'에 쉼표로 적는다.
function who_() { try { return JSON.parse(PropertiesService.getScriptProperties().getProperty('who') || 'null'); } catch (e) { return null; } }
function addWho_(w, d) {
  var name = String(d.by || '').trim().slice(0, 40), day = String(d.at || new Date().toISOString()).slice(0, 10);
  var n = 0, L = d.lines || [];
  for (var i = 0; i < L.length; i++) if (L[i] && (L[i].added || L[i].text !== L[i].orig)) n++;
  var r = w[name] || (w[name] = {p: 0, l: 0, f: day, t: day});
  r.p += 1; r.l += n; if (day < r.f) r.f = day; if (day > r.t) r.t = day;
}
// 처음 한 번: 지금까지 받은 JSON을 훑어 목록을 만든다
function rebuildWho_() {
  var w = {}, root = folder_(folder_(DriveApp.getRootFolder(), ROOT), BOX), days = root.getFolders();
  while (days.hasNext()) {
    var fs = days.next().getFilesByType('application/json');
    while (fs.hasNext()) { try { addWho_(w, JSON.parse(fs.next().getBlob().getDataAsString())); } catch (e) {} }
  }
  PropertiesService.getScriptProperties().setProperty('who', JSON.stringify(w));
  return w;
}
function doGet() {
  var c = counts_(), total = 0; for (var k in c) total += c[k];
  var w = who_(); if (!w) w = rebuildWho_();
  var hide = String(PropertiesService.getScriptProperties().getProperty('hide') || '').split(','), list = [];
  for (var nm in w) {
    if (/^Claude 시험/.test(nm)) continue;   // 판독기를 만들며 보낸 시험 면
    var shown = nm && hide.indexOf(nm) < 0 ? nm : '';
    list.push({name: shown, pages: w[nm].p, lines: w[nm].l, first: w[nm].f, last: w[nm].t});
  }
  return out_({ok: true, service: '방각본 판독기 학습 자료 창구', total: total, counts: c, contributors: list});
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    var d = JSON.parse(e.postData.contents);
    if (!d || !d.image || !d.lines || !d.lines.length) return out_({ok: false, error: 'bad_request'});
    var bytes = Utilities.base64Decode(d.image);
    if (bytes.length > MAX_IMAGE) return out_({ok: false, error: 'too_large'});
    if (bytes[0] !== -1 || bytes[1] !== -40) return out_({ok: false, error: 'not_jpeg'});  // FF D8

    lock.waitLock(20000);
    var props = PropertiesService.getScriptProperties();
    var day = Utilities.formatDate(new Date(), 'Asia/Seoul', 'yyyy-MM-dd');
    var key = 'n_' + day, n = Number(props.getProperty(key) || 0);
    if (n >= MAX_PER_DAY) return out_({ok: false, error: 'daily_limit'});
    props.setProperty(key, String(n + 1));
    try {   // 면별 받은 수 (속성 하나에 9KB까지라 이름은 짧게)
      var ck = String(d.commons || d.src || '').replace(/^File:/, '').replace(/\.pdf$/i, '').slice(-24) + '|' + String(d.page || '');
      var c = counts_(); c[ck] = (c[ck] || 0) + 1; props.setProperty('cnt', JSON.stringify(c));
    } catch (x) {}
    try { var w = who_() || rebuildWho_(); addWho_(w, d); props.setProperty('who', JSON.stringify(w)); } catch (x) {}

    var root = folder_(folder_(DriveApp.getRootFolder(), ROOT), BOX);
    var dir = folder_(root, day);
    var id = Utilities.formatDate(new Date(), 'Asia/Seoul', 'HHmmss') + '_' + Utilities.getUuid().slice(0, 8);
    var img = dir.createFile(Utilities.newBlob(bytes, 'image/jpeg', id + '.jpg'));
    delete d.image;
    d.receivedAt = new Date().toISOString();
    d.imageFile = img.getName();
    dir.createFile(id + '.json', JSON.stringify(d, null, 1), 'application/json');
    return out_({ok: true, id: id});
  } catch (err) {
    return out_({ok: false, error: String(err)});
  } finally {
    try { lock.releaseLock(); } catch (x) {}
  }
}
