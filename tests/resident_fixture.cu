// A scripted decoder checks the controller's actual feedback and cache positions.
// Arithmetic is never performed here: it runs in the resident C guest.
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

struct Fixture {
    int64_t *input,*next,*position;
    uint16_t* mask;
    uint8_t* bytes;
    int32_t *offsets,*printable,*script,*cursor,*errors;
    int count,length,prompt;
    cudaGraph_t prefill,decode;
};
static void ck(cudaError_t x) { if(x!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
__global__ void init(Fixture f) { *f.cursor=*f.errors=0; *f.next=f.script[0]; }
__global__ void step(Fixture f) {
    int i=(*f.cursor)++;
    if (i>=f.count || *f.input!=f.script[i] || *f.position!=f.prompt+i) ++*f.errors;
    for(int j=0;j<f.length;++j) if(f.mask[j]!=(j<=*f.position?0:0xff7f)) ++*f.errors;
    *f.next=i+1<f.count?f.script[i+1]:262;
}
static cudaGraph_t graph(Fixture f,void* fn) {
    cudaGraph_t g; ck(cudaGraphCreate(&g,0));
    void* args[]={&f};
    cudaKernelNodeParams p{}; p.func=fn; p.gridDim=p.blockDim=dim3(1); p.kernelParams=args;
    cudaGraphNode_t n; ck(cudaGraphAddKernelNode(&n,g,nullptr,0,&p)); return g;
}
template<class T> void alloc(T*& p,int n,const T* src=nullptr) {
    ck(cudaMalloc(&p,n*sizeof(T)));
    if(src) ck(cudaMemcpy(p,src,n*sizeof(T),cudaMemcpyHostToDevice));
}
extern "C" void* fixture_create(const uint8_t* bytes,int nb,const int32_t* offsets,int vocab,
                                 const int32_t* script,int count,int length,int prompt) {
    Fixture f{}; f.count=count; f.length=length; f.prompt=prompt;
    alloc(f.input,1); alloc(f.next,1); alloc(f.position,1); alloc(f.mask,length);
    alloc(f.bytes,nb,bytes); alloc(f.offsets,vocab+1,offsets);
    int32_t printable[260]; for(int i=0;i<260;++i) printable[i]=i;
    alloc(f.printable,260,printable); alloc(f.script,count,script); alloc(f.cursor,1); alloc(f.errors,1);
    f.prefill=graph(f,reinterpret_cast<void*>(init));
    f.decode=graph(f,reinterpret_cast<void*>(step));
    return new Fixture(f);
}
extern "C" uint64_t fixture_get(void* ptr,int field) {
    auto& f=*static_cast<Fixture*>(ptr);
    void* fields[]={f.input,f.next,f.position,f.mask,f.bytes,f.offsets,f.printable,f.prefill,f.decode};
    return reinterpret_cast<uint64_t>(fields[field]);
}
extern "C" int fixture_errors(void* ptr) {
    int n; ck(cudaMemcpy(&n,static_cast<Fixture*>(ptr)->errors,4,cudaMemcpyDeviceToHost)); return n;
}
extern "C" void fixture_close(void* ptr) {
    auto* f=static_cast<Fixture*>(ptr);
    cudaGraphDestroy(f->prefill); cudaGraphDestroy(f->decode);
    cudaFree(f->input); cudaFree(f->next); cudaFree(f->position); cudaFree(f->mask);
    cudaFree(f->bytes); cudaFree(f->offsets); cudaFree(f->printable); cudaFree(f->script);
    cudaFree(f->cursor); cudaFree(f->errors); delete f;
}
