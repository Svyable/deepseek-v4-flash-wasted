/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Standalone expected-value producer for the fixed Gate E ratio-0 fixture.
 * It intentionally includes/links no WASTE runtime or quantization helper.
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEQLEN 2u
#define HEADS 2u
#define HIDDEN 4096u
#define QRANK 1024u
#define KVDIM 512u
#define HDIM 512u
#define ROPE 64u
#define NOPE (HDIM-ROPE)
#define ACT 128u
#define KVSIM 64u
#define WINDOW 128u

static uint32_t f32_bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float bits_f32(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

static uint16_t bf16_bits(float x)
{
    uint32_t u = f32_bits(x);
    uint32_t exp = u & 0x7f800000u, frac = u & 0x007fffffu;
    if (exp == 0x7f800000u) {
        if (frac) u |= 0x00400000u;
        return (uint16_t)(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static float bf16(float x) { return bits_f32((uint32_t)bf16_bits(x) << 16); }
static float bf16_from_bits(uint16_t b) { return bits_f32((uint32_t)b << 16); }

static float e4m3(uint8_t b)
{
    unsigned sign = b >> 7, exp = (b >> 3) & 15u, mant = b & 7u;
    if ((b & 0x7fu) == 0x7fu) return NAN;
    float mag = exp == 0 ? ldexpf((float)mant / 8.0f, -6)
                         : ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -mag : mag;
}

static float e8m0(uint8_t b)
{
    if (b == 0xffu) return NAN;
    return ldexpf(1.0f, (int)b - 127);
}

static uint8_t e4m3_encode(float x)
{
    if (isnan(x)) return 0x7fu;
    if (x == 0.0f) return signbit(x) ? 0x80u : 0u;
    if (x > 448.0f) x = 448.0f;
    if (x < -448.0f) x = -448.0f;
    int neg = signbit(x) != 0;
    float best_err = INFINITY;
    uint8_t best = neg ? 0x80u : 0u;
    for (unsigned mag = 0; mag < 0x7fu; mag++) {
        uint8_t code = (uint8_t)(mag | (neg ? 0x80u : 0u));
        float err = fabsf(x - e4m3(code));
        if (err < best_err ||
            (err == best_err && !(code & 1u) && (best & 1u))) {
            best_err = err;
            best = code;
        }
    }
    return best;
}

static float act_scale(const float *x, size_t n)
{
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-4f) amax = 1e-4f;
    float ratio = amax * (1.0f / 448.0f);
    uint32_t u = f32_bits(ratio);
    int e = (int)((u >> 23) & 0xffu) - 127;
    if (u & 0x7fffffu) e++;
    return ldexpf(1.0f, e);
}

static int quant_row(const float *x, size_t k, uint8_t *q, float *scales)
{
    if (k % ACT) return -1;
    for (size_t base = 0; base < k; base += ACT) {
        float s = act_scale(x + base, ACT);
        scales[base / ACT] = s;
        for (size_t j = 0; j < ACT; j++) {
            float z = x[base + j] / s;
            if (z > 448.0f) z = 448.0f;
            if (z < -448.0f) z = -448.0f;
            q[base + j] = e4m3_encode(z);
        }
    }
    return 0;
}

static int fp8_linear(const float *x, size_t m, size_t k,
                      const uint8_t *w, const uint8_t *s,
                      size_t n, float *out)
{
    if (!x || !w || !s || !out || !m || !n || !k || k % ACT) return -1;
    size_t kb = k / ACT;
    uint8_t *qx = malloc(m * k);
    float *as = malloc(m * kb * sizeof(float));
    if (!qx || !as) { free(qx); free(as); return -1; }
    for (size_t row = 0; row < m; row++)
        if (quant_row(x + row*k, k, qx + row*k, as + row*kb)) {
            free(qx); free(as); return -1;
        }
    for (size_t row = 0; row < m; row++) {
        for (size_t r = 0; r < n; r++) {
            float acc = 0.0f;
            const uint8_t *wr = w + r*k;
            const uint8_t *sr = s + (r/128u)*kb;
            for (size_t b = 0; b < kb; b++) {
                float part = 0.0f;
                size_t base = b*ACT;
                for (size_t j = 0; j < ACT; j++)
                    part += e4m3(qx[row*k+base+j]) * e4m3(wr[base+j]);
                float ws = e8m0(sr[b]);
                if (!isfinite(ws)) { free(qx); free(as); return -1; }
                acc += part * as[row*kb+b] * ws;
            }
            out[row*n+r] = bf16(acc);
        }
    }
    free(qx); free(as); return 0;
}

static void learned_rms(const float *x, const uint16_t *wb, size_t n, float eps, float *out)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += x[i]*x[i];
    float inv = 1.0f / sqrtf(sum/(float)n + eps);
    for (size_t i = 0; i < n; i++)
        out[i] = bf16(x[i] * inv * bf16_from_bits(wb[i]));
}

static void head_norm(float *x, size_t n, float eps)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = bf16(x[i]);
        sum += bf16(v*v);
    }
    float mean = bf16(sum/(float)n);
    float shifted = bf16(mean + eps);
    float inv = bf16(1.0f / sqrtf(shifted));
    for (size_t i = 0; i < n; i++) x[i] = bf16(bf16(x[i]) * inv);
}

static void rope(float *x, size_t dim, size_t pos, float theta, int inverse)
{
    for (size_t pair = 0; pair < dim/2u; pair++) {
        float freq = 1.0f / powf(theta, (float)(2u*pair)/(float)dim);
        float angle = (float)pos * freq * (inverse ? -1.0f : 1.0f);
        float c = cosf(angle), s = sinf(angle);
        float re = x[2u*pair], im = x[2u*pair+1u];
        x[2u*pair] = bf16(re*c-im*s);
        x[2u*pair+1u] = bf16(re*s+im*c);
    }
}

static void fp8_sim(float *x, size_t n, size_t block)
{
    for (size_t base = 0; base < n; base += block) {
        float s = act_scale(x+base, block);
        for (size_t j = 0; j < block; j++) {
            float z = x[base+j]/s;
            if (z > 448.0f) z = 448.0f;
            if (z < -448.0f) z = -448.0f;
            x[base+j] = bf16(e4m3(e4m3_encode(z))*s);
        }
    }
}

static int sparse_head(const float *q, const float *kv, size_t n_kv,
                       const int32_t *idx, size_t topk, float sink, float *out)
{
    float *acc = calloc(HDIM, sizeof(float));
    if (!acc) return -1;
    float maxv = -INFINITY, denom = 0.0f;
    const float scale = 1.0f/sqrtf((float)HDIM);
    for (size_t base=0; base<topk; base+=64u) {
        size_t cnt=topk-base; if(cnt>64u)cnt=64u;
        float scores[64], next=maxv;
        for(size_t j=0;j<cnt;j++){
            if(idx[base+j]<0){scores[j]=-INFINITY;continue;}
            size_t p=(size_t)idx[base+j]; if(p>=n_kv){free(acc);return -1;}
            float dot=0.0f; const float *v=kv+p*HDIM;
            for(size_t k=0;k<HDIM;k++) dot += q[k]*v[k];
            scores[j]=dot*scale; if(scores[j]>next)next=scores[j];
        }
        float res=isfinite(maxv)?expf(maxv-next):0.0f;
        denom*=res; for(size_t k=0;k<HDIM;k++)acc[k]*=res;
        for(size_t j=0;j<cnt;j++){
            if(!isfinite(scores[j]))continue;
            float e=expf(scores[j]-next); denom+=e; float eb=bf16(e);
            const float *v=kv+(size_t)idx[base+j]*HDIM;
            for(size_t k=0;k<HDIM;k++) acc[k]+=eb*v[k];
        }
        maxv=next;
    }
    denom += expf(sink-maxv);
    if(!(denom>0.0f)||!isfinite(denom)){free(acc);return -1;}
    for(size_t k=0;k<HDIM;k++)out[k]=bf16(acc[k]/denom);
    free(acc); return 0;
}

static void *read_exact(const char *path,size_t n)
{
    FILE *f=fopen(path,"rb");
    if(!f){fprintf(stderr,"%s: %s\n",path,strerror(errno));return NULL;}
    void *p=malloc(n?n:1); if(!p){fclose(f);return NULL;}
    size_t got=fread(p,1,n,f); int extra=fgetc(f); fclose(f);
    if(got!=n||extra!=EOF){fprintf(stderr,"%s: expected %zu bytes, got %zu%s\n",path,n,got,extra==EOF?"":"+");free(p);return NULL;}
    return p;
}

static int write_u16(const char *path,const float *x,size_t n)
{
    FILE *f=fopen(path,"wb"); if(!f)return -1;
    for(size_t i=0;i<n;i++){uint16_t b=bf16_bits(x[i]);if(fwrite(&b,2,1,f)!=1){fclose(f);return -1;}}
    return fclose(f)==0?0:-1;
}

static char *path_join(const char *dir,const char *name)
{
    size_t n=strlen(dir)+strlen(name)+2u; char *p=malloc(n); if(!p)return NULL;
    snprintf(p,n,"%s/%s",dir,name); return p;
}

int main(int argc,char **argv)
{
    if(argc!=2){fprintf(stderr,"usage: %s FIXTURE_DIR\n",argv[0]);return 2;}
    const char *d=argv[1];
    const char *names[]={"input.bf16.bin","wq-a.fp8.bin","wq-a-scale.e8m0.bin","q-norm.bf16.bin","wq-b-head0-2.fp8.bin","wq-b-scale-head0-2.e8m0.bin","wkv.fp8.bin","wkv-scale.e8m0.bin","kv-norm.bf16.bin","attn-sink-head0-2.f32.bin"};
    char *p[10]={0}; for(int i=0;i<10;i++){p[i]=path_join(d,names[i]);if(!p[i])return 1;}
    uint16_t *inb=read_exact(p[0],SEQLEN*HIDDEN*2u);
    uint8_t *wqa=read_exact(p[1],QRANK*HIDDEN);
    uint8_t *wqas=read_exact(p[2],(QRANK/128u)*(HIDDEN/128u));
    uint16_t *qn=read_exact(p[3],QRANK*2u);
    uint8_t *wqb=read_exact(p[4],HEADS*HDIM*QRANK);
    uint8_t *wqbs=read_exact(p[5],((HEADS*HDIM)/128u)*(QRANK/128u));
    uint8_t *wkv=read_exact(p[6],KVDIM*HIDDEN);
    uint8_t *wkvs=read_exact(p[7],(KVDIM/128u)*(HIDDEN/128u));
    uint16_t *kn=read_exact(p[8],KVDIM*2u);
    float *sink=read_exact(p[9],HEADS*sizeof(float));
    for(int i=0;i<10;i++)free(p[i]);
    if(!inb||!wqa||!wqas||!qn||!wqb||!wqbs||!wkv||!wkvs||!kn||!sink)return 1;

    float *x=malloc(SEQLEN*HIDDEN*sizeof(float));
    float *qr=malloc(SEQLEN*QRANK*sizeof(float));
    float *qrn=malloc(SEQLEN*QRANK*sizeof(float));
    float *q=malloc(SEQLEN*HEADS*HDIM*sizeof(float));
    float *kv=malloc(SEQLEN*KVDIM*sizeof(float));
    float *kvn=malloc(SEQLEN*KVDIM*sizeof(float));
    float *o=malloc(SEQLEN*HEADS*HDIM*sizeof(float));
    if(!x||!qr||!qrn||!q||!kv||!kvn||!o)return 1;
    for(size_t i=0;i<SEQLEN*HIDDEN;i++)x[i]=bf16_from_bits(inb[i]);
    if(fp8_linear(x,SEQLEN,HIDDEN,wqa,wqas,QRANK,qr))return 1;
    for(size_t t=0;t<SEQLEN;t++)learned_rms(qr+t*QRANK,qn,QRANK,1e-6f,qrn+t*QRANK);
    if(fp8_linear(qrn,SEQLEN,QRANK,wqb,wqbs,HEADS*HDIM,q))return 1;
    for(size_t t=0;t<SEQLEN;t++)for(size_t h=0;h<HEADS;h++){
        float *qh=q+(t*HEADS+h)*HDIM; head_norm(qh,HDIM,1e-6f); rope(qh+NOPE,ROPE,t,10000.0f,0);
    }
    if(fp8_linear(x,SEQLEN,HIDDEN,wkv,wkvs,KVDIM,kv))return 1;
    for(size_t t=0;t<SEQLEN;t++){
        learned_rms(kv+t*KVDIM,kn,KVDIM,1e-6f,kvn+t*KVDIM);
        rope(kvn+t*KVDIM+NOPE,ROPE,t,10000.0f,0); fp8_sim(kvn+t*KVDIM,NOPE,KVSIM);
    }
    const int32_t idx0[2]={0,-1},idx1[2]={0,1};
    for(size_t h=0;h<HEADS;h++){
        if(sparse_head(q+h*HDIM,kvn,SEQLEN,idx0,2,sink[h],o+h*HDIM))return 1;
        if(sparse_head(q+(HEADS+h)*HDIM,kvn,SEQLEN,idx1,2,sink[h],o+(HEADS+h)*HDIM))return 1;
    }
    for(size_t t=0;t<SEQLEN;t++)for(size_t h=0;h<HEADS;h++)rope(o+(t*HEADS+h)*HDIM+NOPE,ROPE,t,10000.0f,1);

    char *qa8=path_join(d,"q-a-token0-first8.bf16.bin");
    char *qo=path_join(d,"q-after-rope.bf16.bin");
    char *ko=path_join(d,"kv-after-qat.bf16.bin");
    char *oo=path_join(d,"attn-after-inverse-rope.bf16.bin");
    if(!qa8||!qo||!ko||!oo)return 1;
    if(write_u16(qa8,qr,8)||write_u16(qo,q,SEQLEN*HEADS*HDIM)||write_u16(ko,kvn,SEQLEN*KVDIM)||write_u16(oo,o,SEQLEN*HEADS*HDIM))return 1;
    free(qa8);free(qo);free(ko);free(oo);
    free(inb);free(wqa);free(wqas);free(qn);free(wqb);free(wqbs);free(wkv);free(wkvs);free(kn);free(sink);
    free(x);free(qr);free(qrn);free(q);free(kv);free(kvn);free(o);
    return 0;
}
