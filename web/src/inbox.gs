// 방각본 판독기 학습 자료 받는 창구 (Google Apps Script 웹 앱)
// 판독기에서 '학습용으로 보내기'를 누르면 면 사진(JPEG)과 고친 판독문(JSON)이
// 내 드라이브 방각본_판독기/학습자료/받은날짜 폴더에 쌓인다.
// 배포: 배포 → 새 배포 → 웹 앱, 실행 사용자 '나', 액세스 '모든 사용자'

var ROOT = '방각본_판독기', BOX = '학습자료';
var MAX_IMAGE = 15 * 1024 * 1024;   // 사진 한 장 15MB까지
var MAX_PER_DAY = 1000;             // 하루에 받는 면 수 상한

function folder_(parent, name) {
  var it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : parent.createFolder(name);
}

function out_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}

function doGet() {
  return out_({ok: true, service: '방각본 판독기 학습 자료 창구'});
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
