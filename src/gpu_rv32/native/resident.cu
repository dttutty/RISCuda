#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "include/resident.h"
#include "include/rv32_core.cuh"
#include "include/bigint.cuh"

namespace {
using namespace resident;
thread_local std::string last_error;
void check(cudaError_t result) {
    if (result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Config {
    State* state;
    int64_t *input, *next, *position;
    uint16_t* mask;
    const uint8_t* bytes;
    const int32_t *offsets, *printable;
    uint32_t* code;
    uint8_t *initial, *ram;
    int vocab, cache_length, prompt_length, max_generated, max_total, max_calls;
    int eos, eos2, tco, tcc, words, ram_size, entry, source, output, error, budget;
};
__device__ uint64_t now() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));
    return value;
}
__global__ void initialize(Config c) { c.state->position=c.prompt_length; }

__device__ void gather(State* s,Tool& t) {
    int gt=0,n=0,i;
    bool ended=false;
    for (i=1;i<=ROWS;++i) {
        if (!s->rows[i][0]) { ended=true; break; }
        for (int k=0;k<16 && s->rows[i][k];++k) {
            int b=s->rows[i][k];
            if (gt<2) gt+=b=='>';
            else if (b=='<') { ended=true; break; }
            else if (b!='\n' && n<255) t.expression[n++]=b;
        }
        if (ended) break;
    }
    t.expression[n]=0; t.input_length=n;
    t.gather_error=gt<2?1:!ended?2:0;
}
// stop: 1 EOS, 2 total tokens, 3 generated tokens, 4 cache capacity,
//       5 tool-call capacity, 6 feedback/output capacity, 7 invalid model token.
__global__ void prepare(Config c,cudaGraphConditionalHandle enabled) {
    State* s=c.state;
    s->pending=0;
    if (!s->stop && s->total>=c.max_total) s->stop=2;
    int token=0;
    bool forced=s->queue_index<s->queue_length;
    if (!s->stop) {
        if (forced) { token=s->queue[s->queue_index++]; ++s->injected; }
        else if (s->generated>=c.max_generated) s->stop=3;
        else {
            token=static_cast<int>(*c.next);
            if (token<0 || token>=c.vocab) s->stop=7;
            else ++s->generated;
        }
    }
    if (!s->stop) {
        s->tokens[s->total]=token; s->forced[s->total]=forced; ++s->total;
        if (token==c.eos || token==c.eos2) s->stop=1;
        else if (!forced) {
            if (token==c.tco) {
                s->armed=1; s->cap=0;
                for (int i=0;i<=ROWS;++i) for (int j=0;j<16;++j) s->rows[i][j]=0;
            } else if (token==c.tcc) {
                s->armed=0;
                if (s->calls>=c.max_calls) s->stop=5;
                else {
                    Tool& t=s->tools[s->calls++];
                    gather(s,t); s->pending=1;
                }
            } else if (s->armed) {
                int row=++s->cap; if (row>ROWS) row=ROWS;
                int begin=c.offsets[token],end=c.offsets[token+1];
                for (int i=0;i<16;++i) s->rows[row][i]=begin+i<end?c.bytes[begin+i]:0;
            }
        }
    }
    if (!s->stop) { *c.input=token; *c.position=s->position; }
    cudaGraphSetConditional(enabled,s->stop?0:1);
}
__global__ void attention_mask(Config c) {
    for (int i=threadIdx.x;i<c.cache_length;i+=blockDim.x)
        c.mask[i]=i<=c.state->position?0:0xff7f;
}
__global__ void tool_gate(Config c,cudaGraphConditionalHandle enabled) {
    if (c.state->pending) c.state->tools[c.state->calls-1].started=now();
    cudaGraphSetConditional(enabled,c.state->pending);
}
__global__ void restore_ram(Config c) {
    if (c.state->tools[c.state->calls-1].gather_error) return;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<c.ram_size;i+=blockDim.x*gridDim.x)
        c.ram[i]=c.initial[i];
}
__device__ void output_text(Tool& t,const char* text) {
    int i=0;
    for (;text[i] && i<255;++i) t.output[i]=text[i];
    t.output[i]=0; t.output_length=i;
}
__device__ void enqueue(Config c,int byte) {
    State* s=c.state;
    if (s->queue_length==QUEUE) { s->stop=6; return; }
    s->queue[s->queue_length++]=c.printable[byte];
}
__global__ void calculate(Config c) {
    State* s=c.state;
    Tool& t=s->tools[s->calls-1];
    if (t.gather_error) output_text(t,t.gather_error==2?"error (too long: split it)\n":"error\n");
    else {
        for (int i=0;i<256;++i) c.ram[c.source+i]=i<t.input_length?t.expression[i]:0;
        rv32::Machine machine;
        rv32::initialize(machine,c.ram_size,c.entry);
        rv32::Result result;
        for (;;) {
            result=rv32::resume_memory(c.code,c.words,c.budget,c.ram,c.ram_size,machine,true);
            if (result.status!=6) break;
            if (machine.r[17]==0x200)
                machine.r[10]=rv32::big_mul(c.ram,c.ram_size,machine.r[10],machine.r[11],machine.r[12],machine.r[13],machine.r[14]);
            else if (machine.r[17]==0x201)
                machine.r[10]=rv32::decimal_row(c.ram,c.ram_size,machine.r[10],machine.r[11],machine.r[12],machine.r[13]);
            else { result.status=7; break; }
            ++t.bigint_calls;
        }
        t.status=result.status; t.steps=result.steps;
        t.error=*reinterpret_cast<const int32_t*>(c.ram+c.error);
        if (t.status || t.error) output_text(t,"error\n");
        else {
            int i=0;
            for (;i<255 && c.ram[c.output+i];++i) t.output[i]=c.ram[c.output+i];
            t.output[i]=0; t.output_length=i;
            if (c.ram[c.output+i]) { t.status=8; s->stop=6; }
        }
    }
    s->queue_index=s->queue_length=0;
    enqueue(c,10); enqueue(c,256); enqueue(c,10);
    for (int i=0;i<t.output_length;++i) enqueue(c,static_cast<uint8_t>(t.output[i]));
    const int suffix[]={257,10,258,10,10,259,10,10};
    for (int i:suffix) enqueue(c,i);
    t.elapsed_ns=now()-t.started;
    s->pending=0;
}
__global__ void advance(Config c) {
    ++c.state->forwards; ++c.state->position;
    if (!c.state->stop && c.state->total>=c.max_total) c.state->stop=2;
    if (!c.state->stop && c.state->position>=c.cache_length) c.state->stop=4;
}
__global__ void continue_loop(Config c,cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle,c.state->stop?0:1);
}
cudaGraphNode_t kernel(cudaGraph_t g,cudaGraphNode_t dep,void* fn,void** args,int blocks=1,int threads=1) {
    cudaKernelNodeParams p{};
    p.func=fn; p.gridDim=dim3(blocks); p.blockDim=dim3(threads); p.kernelParams=args;
    cudaGraphNode_t n;
    check(cudaGraphAddKernelNode(&n,g,dep?&dep:nullptr,dep?1:0,&p));
    return n;
}
cudaGraphNode_t conditional(cudaGraph_t g,cudaGraphNode_t dep,cudaGraphConditionalHandle handle,
                             cudaGraphConditionalNodeType type,cudaGraph_t& body) {
    cudaGraphNodeParams p{};
    p.type=cudaGraphNodeTypeConditional; p.conditional.handle=handle;
    p.conditional.type=type; p.conditional.size=1;
    cudaGraphNode_t node;
    check(cudaGraphAddNode(&node,g,dep?&dep:nullptr,dep?1:0,&p));
    body=p.conditional.phGraph_out[0]; return node;
}
cudaGraphNode_t generation_step(cudaGraph_t g,Config c,cudaGraph_t decode) {
    cudaGraphConditionalHandle enabled;
    check(cudaGraphConditionalHandleCreate(&enabled,g,0,cudaGraphCondAssignDefault));
    void* prep[]={&c,&enabled};
    auto p=kernel(g,nullptr,reinterpret_cast<void*>(prepare),prep);
    cudaGraph_t step;
    auto node=conditional(g,p,enabled,cudaGraphCondTypeIf,step);
    void* args[]={&c};
    auto mask=kernel(step,nullptr,reinterpret_cast<void*>(attention_mask),args,1,256);
    cudaGraphNode_t model;
    check(cudaGraphAddChildGraphNode(&model,step,&mask,1,decode));
    cudaGraphConditionalHandle call;
    check(cudaGraphConditionalHandleCreate(&call,step,0,cudaGraphCondAssignDefault));
    void* gate_args[]={&c,&call};
    auto gate=kernel(step,model,reinterpret_cast<void*>(tool_gate),gate_args);
    cudaGraph_t tool;
    auto tool_node=conditional(step,gate,call,cudaGraphCondTypeIf,tool);
    auto restore=kernel(tool,nullptr,reinterpret_cast<void*>(restore_ram),args,128,256);
    kernel(tool,restore,reinterpret_cast<void*>(calculate),args);
    kernel(step,tool_node,reinterpret_cast<void*>(advance),args);
    return node;
}
cudaGraphNode_t prefill(cudaGraph_t g,Config c,cudaGraph_t model) {
    cudaMemsetParams p{};
    p.dst=c.state; p.elementSize=4; p.width=sizeof(State)/4; p.height=1;
    cudaGraphNode_t clear;
    check(cudaGraphAddMemsetNode(&clear,g,nullptr,0,&p));
    void* args[]={&c};
    auto init=kernel(g,clear,reinterpret_cast<void*>(initialize),args);
    cudaGraphNode_t node;
    check(cudaGraphAddChildGraphNode(&node,g,&init,1,model)); return node;
}
struct Controller {
    Config c{};
    cudaGraph_t graph=nullptr,init_graph=nullptr,step_graph=nullptr;
    cudaGraphExec_t executable=nullptr,init_exec=nullptr,step_exec=nullptr;
    bool ran=false;
    ~Controller() {
        if (executable) cudaGraphExecDestroy(executable);
        if (init_exec) cudaGraphExecDestroy(init_exec);
        if (step_exec) cudaGraphExecDestroy(step_exec);
        if (graph) cudaGraphDestroy(graph);
        if (init_graph) cudaGraphDestroy(init_graph);
        if (step_graph) cudaGraphDestroy(step_graph);
        cudaFree(c.state); cudaFree(c.code); cudaFree(c.initial); cudaFree(c.ram);
    }
};
} // namespace

extern "C" const char* resident_error() { return last_error.c_str(); }
extern "C" size_t resident_state_size() { return sizeof(resident::State); }
extern "C" size_t resident_options_size() { return sizeof(resident::Options); }
extern "C" int resident_create(void** output,const resident::Options* o) {
    Controller* controller=nullptr;
    try {
        if (!output || !o || !o->prefill_graph || !o->decode_graph || !o->input || !o->next ||
            !o->position || !o->mask || !o->bytes || !o->offsets || !o->printable || !o->code || !o->initial ||
            o->prompt_length<1 || o->prompt_length>65536 || o->cache_length>73728 ||
            o->max_total<1 || o->max_total>TOKENS ||
            o->max_generated<1 || o->max_generated>4096 || o->max_calls<1 || o->max_calls>CALLS ||
            o->cache_length<o->prompt_length+o->max_total || o->words<1 || o->words>262144 ||
            o->ram_size<16384 || o->ram_size>4194304 || o->entry<0 || o->entry%4 || o->entry/4>=o->words ||
            o->source<0 || o->source>o->ram_size-256 || o->output<0 || o->output>o->ram_size-16384 ||
            o->error<0 || o->error%4 || o->error>o->ram_size-4 || o->budget<1)
            throw std::runtime_error("invalid resident graph/image configuration");
        int ids[]={o->eos,o->eos2,o->tco,o->tcc};
        for (int id:ids) if (id<0 || id>=o->vocab) throw std::runtime_error("invalid protocol token ID");
        controller=new Controller;
        Config& c=controller->c;
        c.input=static_cast<int64_t*>(o->input); c.next=static_cast<int64_t*>(o->next);
        c.position=static_cast<int64_t*>(o->position); c.mask=static_cast<uint16_t*>(o->mask);
        c.bytes=static_cast<const uint8_t*>(o->bytes); c.offsets=static_cast<const int32_t*>(o->offsets);
        c.printable=static_cast<const int32_t*>(o->printable);
        c.vocab=o->vocab; c.cache_length=o->cache_length; c.prompt_length=o->prompt_length;
        c.max_generated=o->max_generated; c.max_total=o->max_total; c.max_calls=o->max_calls;
        c.eos=o->eos; c.eos2=o->eos2; c.tco=o->tco; c.tcc=o->tcc;
        c.words=o->words; c.ram_size=o->ram_size; c.entry=o->entry;
        c.source=o->source; c.output=o->output; c.error=o->error; c.budget=o->budget;
        check(cudaMalloc(&c.state,sizeof(State))); check(cudaMalloc(&c.code,c.words*4));
        check(cudaMalloc(&c.initial,c.ram_size)); check(cudaMalloc(&c.ram,c.ram_size));
        check(cudaMemcpy(c.code,o->code,c.words*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(c.initial,o->initial,c.ram_size,cudaMemcpyHostToDevice));
        auto decode=reinterpret_cast<cudaGraph_t>(o->decode_graph);
        auto model_prefill=reinterpret_cast<cudaGraph_t>(o->prefill_graph);
        check(cudaGraphCreate(&controller->graph,0));
        auto initialized=prefill(controller->graph,c,model_prefill);
        cudaGraphConditionalHandle loop;
        check(cudaGraphConditionalHandleCreate(&loop,controller->graph,1,cudaGraphCondAssignDefault));
        cudaGraph_t body;
        conditional(controller->graph,initialized,loop,cudaGraphCondTypeWhile,body);
        auto step=generation_step(body,c,decode);
        void* args[]={&c,&loop};
        kernel(body,step,reinterpret_cast<void*>(continue_loop),args);
        check(cudaGraphInstantiate(&controller->executable,controller->graph,0));
        // Control variant: identical GPU step, host launches/checks one iteration at a time.
        check(cudaGraphCreate(&controller->init_graph,0));
        prefill(controller->init_graph,c,model_prefill);
        check(cudaGraphInstantiate(&controller->init_exec,controller->init_graph,0));
        check(cudaGraphCreate(&controller->step_graph,0));
        generation_step(controller->step_graph,c,decode);
        check(cudaGraphInstantiate(&controller->step_exec,controller->step_graph,0));
        *output=controller; return 0;
    } catch (const std::exception& e) { last_error=e.what(); delete controller; return 1; }
}
extern "C" int resident_run(void* ptr,uint64_t stream,int host_loop,int profile,uint64_t* launches) {
    try {
        auto* p=static_cast<Controller*>(ptr);
        if (!p || p->ran || !launches) throw std::runtime_error("resident controller already used or closed");
        p->ran=true; *launches=0;
        auto io=reinterpret_cast<cudaStream_t>(stream);
        if (profile) check(cudaProfilerStart());
        if (!host_loop) { check(cudaGraphLaunch(p->executable,io)); ++*launches; }
        else {
            check(cudaGraphLaunch(p->init_exec,io)); ++*launches;
            int stop=0;
            while (!stop) {
                check(cudaGraphLaunch(p->step_exec,io)); ++*launches;
                check(cudaMemcpyAsync(&stop,&p->c.state->stop,sizeof(stop),cudaMemcpyDeviceToHost,io));
                check(cudaStreamSynchronize(io));
            }
        }
        check(cudaStreamSynchronize(io));
        if (profile) check(cudaProfilerStop());
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
extern "C" int resident_read(void* ptr,void* output,size_t size) {
    try {
        auto* p=static_cast<Controller*>(ptr);
        if (!p || !p->ran || size!=sizeof(State)) throw std::runtime_error("invalid resident read");
        check(cudaMemcpy(output,p->c.state,size,cudaMemcpyDeviceToHost)); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
extern "C" void resident_close(void* ptr) { delete static_cast<Controller*>(ptr); }
