#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen coherent layer-2 ratio-4 CSA attention fixture."""

import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO,"tests","fixtures","deepseek_v4","v5_csa_attention_real")
INDEX = os.path.join(REPO,"tests","fixtures","deepseek_v4","v5_csa_indexer_real")
REV="9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SEQLEN=8; HIDDEN=4096; Q_RANK=1024; HEAD=512


def raw(base,name):
    with open(os.path.join(base,name),"rb") as f: return f.read()

def u16(base,name):
    data=raw(base,name); return list(struct.unpack(f"<{len(data)//2}H",data))

def i32(base,name):
    data=raw(base,name); return list(struct.unpack(f"<{len(data)//4}i",data))

def f32s(base,name):
    data=raw(base,name); return list(struct.unpack(f"<{len(data)//4}f",data))

def bf16_to_f32(b):
    return struct.unpack("<f",struct.pack("<I",int(b)<<16))[0]

def f32_to_bf16(x):
    u=struct.unpack("<I",struct.pack("<f",float(x)))[0]
    if (u & 0x7F800000)==0x7F800000:
        if u & 0x007FFFFF: u|=0x00400000
        return (u>>16)&0xFFFF
    u=(u+0x7FFF+((u>>16)&1))&0xFFFFFFFF
    return (u>>16)&0xFFFF

def c_f32(xs): return (ctypes.c_float*len(xs))(*xs)
def c_u8(data): return (ctypes.c_uint8*len(data)).from_buffer_copy(data)
def c_u16(xs): return (ctypes.c_uint16*len(xs))(*xs)
def bits_from(arr,n): return [f32_to_bf16(arr[i]) for i in range(n)]

def exact(label,arr,want):
    got=bits_from(arr,len(want))
    if got==want: return
    first=next(i for i,(a,b) in enumerate(zip(got,want)) if a!=b)
    raise AssertionError(f"{label} mismatch at {first}: got 0x{got[first]:04x}, want 0x{want[first]:04x}")


def compile_lib(td):
    cc=os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc: return None
    out=os.path.join(td,"libcsaattention.so")
    cmd=[cc,"-std=c11","-O1","-Wall","-Wextra","-Werror","-shared","-fPIC",
         "-I",os.path.join(REPO,"src"),"-I",os.path.join(REPO,"src","quant"),
         os.path.join(REPO,"src","deepseek_v4_csa_ref.c"),
         os.path.join(REPO,"src","deepseek_v4_compressor_ref.c"),
         os.path.join(REPO,"src","deepseek_v4_attention_ref.c"),
         os.path.join(REPO,"src","quant","deepseek_v4_linear_ref.c"),
         os.path.join(REPO,"src","quant","fp4_e2m1.c"),
         os.path.join(REPO,"src","quant","fp8_e4m3.c"),"-lm","-o",out]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise AssertionError("CSA attention compile failed:\n"+p.stdout+p.stderr)
    return out


def main():
    prov_path=os.path.join(FIX,"provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: coherent real ratio-4 CSA attention fixture not frozen"); return 77
    with open(prov_path,encoding="utf-8") as f: prov=json.load(f)
    if prov.get("revision")!=REV or prov.get("layer")!=2 or prov.get("compress_ratio")!=4:
        raise AssertionError("CSA attention provenance drifted")

    with open(os.path.join(INDEX,"provenance.json"),encoding="utf-8") as f: idxprov=json.load(f)
    recipe=idxprov["input_recipe"]["active_tokens"]
    if [x["position"] for x in recipe]!=[3,7]: raise AssertionError("CSA input dependency drifted")
    x=[0.0]*(SEQLEN*HIDDEN)
    for item in recipe:
        pos=int(item["position"]); lane=int(item["lane"]); bits=int(item["bf16_hex"],16)
        x[pos*HIDDEN+lane]=bf16_to_f32(bits)
    qr_bits=u16(INDEX,"qr-pos7.bf16.bin")
    index_topk=i32(INDEX,"topk-row7.i32.bin")
    if index_topk not in ([8,9],[9,8]): raise AssertionError("CSA index top-k dependency drifted")

    wq_b=c_u8(raw(FIX,"main-wq-b-head0.fp8.bin")); wq_b_scale=c_u8(raw(FIX,"main-wq-b-scale-head0.e8m0.bin"))
    wkv=c_u8(raw(FIX,"main-wkv.fp8.bin")); wkv_scale=c_u8(raw(FIX,"main-wkv-scale.e8m0.bin"))
    kv_norm=c_f32([bf16_to_f32(b) for b in u16(FIX,"main-kv-norm.bf16.bin")])
    sink=f32s(FIX,"attn-sink-head0.f32.bin")[0]
    comp_wkv=c_u16(u16(FIX,"main-comp-wkv.bf16.bin")); comp_wgate=c_u16(u16(FIX,"main-comp-wgate.bf16.bin"))
    comp_ape=c_f32(f32s(FIX,"main-comp-ape.f32.bin")); comp_norm=c_f32([bf16_to_f32(b) for b in u16(FIX,"main-comp-norm.bf16.bin")])

    want_q=u16(FIX,"q-head0-pos7.bf16.bin")
    want_active=u16(FIX,"local-kv-active-3-7.bf16.bin")
    want_pool=u16(FIX,"main-comp-pooled.bf16.bin")
    want_norm=u16(FIX,"main-comp-post-norm.bf16.bin")
    want_rope=u16(FIX,"main-comp-post-rope.bf16.bin")
    want_comp=u16(FIX,"main-comp-kv.bf16.bin")
    want_pre=u16(FIX,"attn-pre-inverse.bf16.bin")
    want_post=u16(FIX,"attn-post-inverse.bf16.bin")
    want_topk=i32(FIX,"combined-topk-row7.i32.bin")
    if want_topk!=list(range(8))+index_topk: raise AssertionError("CSA combined top-k drifted")

    with tempfile.TemporaryDirectory(prefix="v5-csa-attn-") as td:
        libpath=compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler"); return 77
        lib=ctypes.CDLL(libpath)
        linear=lib.waste_ds_v4_fp8_linear_e8m0_ref
        linear.restype=ctypes.c_int
        linear.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_uint8),
                         ctypes.c_size_t,ctypes.POINTER(ctypes.c_float)]
        headnorm=lib.waste_ds_v4_head_rmsnorm_bf16_ref
        headnorm.restype=ctypes.c_int
        headnorm.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
        rms=lib.waste_ds_v4_rmsnorm_ref
        rms.restype=ctypes.c_int
        rms.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
        ropefn=lib.waste_ds_v4_compressed_rope_ref
        ropefn.restype=ctypes.c_int
        ropefn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float]
        inverse=lib.waste_ds_v4_compressed_inverse_rope_ref
        inverse.restype=ctypes.c_int; inverse.argtypes=ropefn.argtypes
        qat=lib.waste_ds_v4_fp8_sim_inplace_ref
        qat.restype=ctypes.c_int; qat.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_size_t]

        qr=c_f32([bf16_to_f32(b) for b in qr_bits])
        q0=(ctypes.c_float*HEAD)()
        if linear(qr,1,Q_RANK,wq_b,wq_b_scale,HEAD,q0)!=0: raise AssertionError("main head0 wq_b failed")
        q=(ctypes.c_float*HEAD)()
        if headnorm(q0,HEAD,ctypes.c_float(1e-6),q)!=0: raise AssertionError("main head q norm failed")
        if ropefn(q,HEAD,64,7,65536,ctypes.c_float(160000.0),ctypes.c_float(16.0),ctypes.c_float(32.0),ctypes.c_float(1.0))!=0:
            raise AssertionError("main head q compressed RoPE failed")
        exact("CSA main q",q,want_q)

        local=[0.0]*(SEQLEN*HEAD); active=[]
        for item in recipe:
            pos=int(item["position"])
            xv=c_f32(x[pos*HIDDEN:(pos+1)*HIDDEN])
            kv0=(ctypes.c_float*HEAD)()
            if linear(xv,1,HIDDEN,wkv,wkv_scale,HEAD,kv0)!=0: raise AssertionError(f"main wkv failed at {pos}")
            kv1=(ctypes.c_float*HEAD)()
            if rms(kv0,kv_norm,HEAD,ctypes.c_float(1e-6),kv1)!=0: raise AssertionError(f"main kv_norm failed at {pos}")
            if ropefn(kv1,HEAD,64,pos,65536,ctypes.c_float(160000.0),ctypes.c_float(16.0),ctypes.c_float(32.0),ctypes.c_float(1.0))!=0:
                raise AssertionError(f"main kv RoPE failed at {pos}")
            if qat(kv1,448,64)!=0: raise AssertionError(f"main kv QAT failed at {pos}")
            row=[kv1[i] for i in range(HEAD)]; local[pos*HEAD:(pos+1)*HEAD]=row; active.extend(row)
        exact("CSA active local KV",c_f32(active),want_active)

        compressor=lib.waste_ds_v4_csa_compressor_prefill_ref
        compressor.restype=ctypes.c_int
        compressor.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_uint16),
                             ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),
                             ctypes.c_size_t,ctypes.c_int,
                             ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
        pool=(ctypes.c_float*(2*HEAD))(); norm=(ctypes.c_float*(2*HEAD))(); rope=(ctypes.c_float*(2*HEAD))(); comp=(ctypes.c_float*(2*HEAD))()
        if compressor(c_f32(x),SEQLEN,comp_wkv,comp_wgate,comp_ape,comp_norm,HEAD,0,pool,norm,rope,comp)!=0:
            raise AssertionError("main ratio4 compressor failed")
        exact("CSA main comp pool",pool,want_pool); exact("CSA main comp norm",norm,want_norm)
        exact("CSA main comp rope",rope,want_rope); exact("CSA main comp kv",comp,want_comp)

        # Prove the local window row through the existing source-contract helper.
        win=lib.waste_ds_v4_window_indices_ref
        win.restype=ctypes.c_size_t
        win.argtypes=[ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t]
        matrix=(ctypes.c_int32*(SEQLEN*SEQLEN))()
        n=win(128,SEQLEN,0,matrix,SEQLEN*SEQLEN)
        if n!=SEQLEN*SEQLEN: raise AssertionError("CSA window index matrix size mismatch")
        local_idxs=list(matrix[(SEQLEN-1)*SEQLEN:SEQLEN*SEQLEN])
        if local_idxs!=list(range(SEQLEN)): raise AssertionError(f"CSA local row mismatch: {local_idxs}")
        combined=local_idxs+index_topk
        if combined!=want_topk: raise AssertionError("CSA combined selection mismatch")

        kv_all=local+[comp[i] for i in range(2*HEAD)]
        idxbuf=(ctypes.c_int32*len(combined))(*combined)
        sparse=lib.waste_ds_v4_sparse_attn_head_ref
        sparse.restype=ctypes.c_int
        sparse.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,ctypes.c_size_t,
                         ctypes.c_float,ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
        pre=(ctypes.c_float*HEAD)()
        if sparse(q,c_f32(kv_all),SEQLEN+2,idxbuf,len(combined),HEAD,ctypes.c_float(sink),ctypes.c_float(HEAD**-0.5),pre)!=0:
            raise AssertionError("CSA sparse attention failed")
        exact("CSA pre-inverse attention",pre,want_pre)
        post=(ctypes.c_float*HEAD)(*[pre[i] for i in range(HEAD)])
        if inverse(post,HEAD,64,7,65536,ctypes.c_float(160000.0),ctypes.c_float(16.0),ctypes.c_float(32.0),ctypes.c_float(1.0))!=0:
            raise AssertionError("CSA inverse compressed RoPE failed")
        exact("CSA post-inverse attention",post,want_post)

        # Load-bearing integration mutation: drop the two indexer-selected compressed entries.
        localbuf=(ctypes.c_int32*SEQLEN)(*list(range(SEQLEN)))
        mutated=(ctypes.c_float*HEAD)()
        if sparse(q,c_f32(kv_all),SEQLEN+2,localbuf,SEQLEN,HEAD,ctypes.c_float(sink),ctypes.c_float(HEAD**-0.5),mutated)!=0:
            raise AssertionError("CSA window-only mutation failed to execute")
        if bits_from(mutated,HEAD)==want_pre: raise AssertionError("dropping CSA compressed selections survived")

    print("PASS Gate E coherent ratio-4 CSA: real indexer top-k + real main overlap compressor + sparse attention + inverse YaRN")
    print("indexer topk:",index_topk,"post first8:"," ".join(f"0x{x:04x}" for x in want_post[:8]))
    return 0

if __name__=="__main__": raise SystemExit(main())
