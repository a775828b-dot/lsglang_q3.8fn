"""lkqmoe startup hook for the Qwen3.8 Flash Next NVFP4 pipeline; everything is enabled by the launcher."""
import os
if os.environ.get('LKQMOE_MAIN_CPUS'):
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
