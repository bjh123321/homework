"""Fetch the public archive, verify its published SHA-256 and analyze numeric CSVs."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.request import Request, urlopen
import zipfile
from bearing_monitor import collect

REV="274226336036cd8deb2c40b4ba323af2d11a7290"
URL=f"https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/resolve/{REV}/XJTU-SY_Bearing_Datasets.zip"
SIZE=5447931446
DIGEST="3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10"

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()

def fetch(out):
    archive=out/"XJTU-SY_Bearing_Datasets.zip"
    part=out/"XJTU-SY_Bearing_Datasets.zip.part"
    if not archive.exists():
        for attempt in range(4):
            offset=part.stat().st_size if part.exists() else 0
            if offset==SIZE:
                break
            try:
                req=Request(URL,headers={"Range":f"bytes={offset}-"} if offset else {})
                with urlopen(req,timeout=90) as response:
                    if offset and response.status==206 and not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                        raise ValueError("镜像返回了不匹配的续传范围")
                    mode="ab" if offset and response.status==206 else "wb"
                    with part.open(mode) as dest:
                        shutil.copyfileobj(response,dest,1024**2)
                if part.stat().st_size==SIZE:
                    break
            except OSError:
                if attempt==3:
                    raise
                time.sleep(2**attempt)
        if not part.exists() or part.stat().st_size!=SIZE or digest(part)!=DIGEST:
            raise ValueError("下载不完整或校验失败；保留 .part 供检查")
        part.rename(archive)
    elif archive.stat().st_size!=SIZE or digest(archive)!=DIGEST:
        raise ValueError("已有压缩包与公布的 SHA-256 不符")
    return archive

def unpack(archive,dest):
    dest.mkdir(parents=True,exist_ok=True)
    seen=set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            parts=PurePosixPath(info.filename.replace(chr(92),"/")).parts
            if len(parts)<2:
                continue
            bearing,filename=parts[-2:]
            if not re.fullmatch(r"Bearing[1-3]_[1-5]",bearing,re.I) or not re.fullmatch(r"[1-9][0-9]*\.csv",filename,re.I):
                continue
            target=dest/bearing/filename
            if target in seen:
                raise ValueError("压缩包含重复轴承和记录号")
            seen.add(target)
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                import zlib
                with target.open("rb") as f:
                    if target.stat().st_size==info.file_size and zlib.crc32(f.read())==info.CRC:
                        continue
                raise ValueError("已解压文件与档案不一致："+str(target))
            tmp=target.with_suffix(".csv.part")
            with bundle.open(info) as src,tmp.open("wb") as dst:
                shutil.copyfileobj(src,dst)
            tmp.rename(target)
    if not seen:
        raise ValueError("包内未找到轴承 CSV，请使用发布方解压后的数据目录")
    return len(seen)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,default=Path(__file__).resolve().parent/"xjtu_real")
    a=parser.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    print("下载并校验公开档案（5.45 GB），可中断续传。",flush=True)
    archive=fetch(a.work)
    n=unpack(archive,a.work/"raw")
    print("已核验 CSV 数：",n,flush=True)
    print("纳入分析的记录数：",collect(a.work/"results",a.work/"raw"))

