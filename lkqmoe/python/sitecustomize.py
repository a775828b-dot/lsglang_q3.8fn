"""lkqmoe startup hook for the Qwen3.8 Flash Next NVFP4 pipeline; everything is enabled by the launcher."""
import os
def _auto_main_cpus():
    # Every allowed CPU except the expert workers' (same rule as lkqmoe_create, current NUMA
    # topology) and the mailbox dispatcher's: survives NPS/thread-count changes unedited.
    import ctypes as C
    lib=C.CDLL(os.environ['LKQMOE_LIBRARY'])
    threads=int(os.environ.get('LKQMOE_THREADS') or os.environ.get('LK_THREADS','64'))
    out=(C.c_int*threads)()
    if lib.lkqmoe_worker_cpus(threads,out,threads)<0:
        lib.lkqmoe_last_error.restype=C.c_char_p
        raise RuntimeError('LKQMOE_MAIN_CPUS=auto: '+lib.lkqmoe_last_error().decode())
    busy=set(out)|{int(os.environ.get('LKQMOE_PROBE_DISPATCH_CPU','80'))}
    return os.sched_getaffinity(0)-busy
if os.environ.get('LKQMOE_MAIN_CPUS')=='auto':
    os.sched_setaffinity(0,_auto_main_cpus() or os.sched_getaffinity(0))
elif os.environ.get('LKQMOE_MAIN_CPUS'):
    # Keep Python/scheduler threads off the pinned CPU expert workers; the workers and the
    # mailbox dispatcher set their own affinity.
    cpus=set()
    for part in os.environ['LKQMOE_MAIN_CPUS'].split(','):
        lo,_,hi=part.partition('-');cpus.update(range(int(lo),int(hi or lo)+1))
    os.sched_setaffinity(0,cpus&os.sched_getaffinity(0) or os.sched_getaffinity(0))
if os.environ.get('LKQMOE_MODE') in ('shadow','replace','standalone'):
    from lkqmoe.integration import install
    install()
if os.environ.get('LKQMOE_MTP_QUANT') == '1':
    from lkqmoe.gpu.mtp_quant import install as install_mtp_quant
    install_mtp_quant()
if os.environ.get('LKQMOE_MTP_ROUTER_GUARD') == '1':
    from lkqmoe.gpu.router_dependency import install as install_router
    install_router()
