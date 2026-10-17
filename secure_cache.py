"""Local, Windows-user-bound encrypted cache. No network calls."""
import ctypes,hashlib,json,os,zlib
from pathlib import Path
from ctypes import wintypes

class Blob(ctypes.Structure):
    _fields_=[('cbData',wintypes.DWORD),('pbData',ctypes.POINTER(ctypes.c_ubyte))]

def crypt(data,protect):
    if os.name!='nt':raise RuntimeError('Windows DPAPI required for private cache')
    buffer=(ctypes.c_ubyte*len(data)).from_buffer_copy(data)
    source=Blob(len(data),buffer);target=Blob()
    library=ctypes.WinDLL('Crypt32.dll',use_last_error=True)
    call=library.CryptProtectData if protect else library.CryptUnprotectData
    call.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    call.restype=wintypes.BOOL
    # User scope: deliberately omit CRYPTPROTECT_LOCAL_MACHINE.
    if not call(ctypes.byref(source),None,None,None,None,1,ctypes.byref(target)):
        raise RuntimeError('Windows private cache encryption/decryption failed (code '+str(ctypes.get_last_error())+')')
    free=ctypes.WinDLL('Kernel32.dll').LocalFree
    free.argtypes=[ctypes.c_void_p];free.restype=ctypes.c_void_p
    try:return ctypes.string_at(target.pbData,target.cbData)
    finally:free(ctypes.cast(target.pbData,ctypes.c_void_p))

def signature(source):
    return hashlib.sha256(b'IntraRAGPoC-cache-v1:qwen3-embedding-local:'+Path(source,'manifest.json').read_bytes()+Path(__file__).with_name('app.py').read_bytes()).hexdigest()

def save(path,data,key):
    data=Path(data);path=Path(path)
    payload={'signature':key,'index':(data/'index.json').read_text('utf-8'),'report':(data/'ingestion_report.json').read_text('utf-8')}
    ciphertext=crypt(zlib.compress(json.dumps(payload,ensure_ascii=False).encode('utf-8')),True)
    temp=path.with_suffix('.new');temp.write_bytes(ciphertext);os.replace(temp,path)

def restore(path,data,key):
    path=Path(path);data=Path(data)
    if not path.is_file():return False
    try:payload=json.loads(zlib.decompress(crypt(path.read_bytes(),False)))
    except Exception:return False
    if payload.get('signature')!=key:return False
    # Validate structure before making any plaintext working files.
    try:index=json.loads(payload['index']);report=json.loads(payload['report'])
    except (KeyError,TypeError,ValueError):return False
    if not isinstance(index.get('documents'),dict) or not isinstance(index.get('chunks'),list) or not isinstance(report,list):return False
    data.mkdir(parents=True,exist_ok=True)
    (data/'index.json').write_text(payload['index'],'utf-8')
    (data/'ingestion_report.json').write_text(payload['report'],'utf-8')
    return True
