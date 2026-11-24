from crossfeed.audex.runtime import add_official_paths

add_official_paths()

from vllm.v1.core.sched.scheduler import Scheduler
from vllm_cfg_patch import apply_cfg_patches

apply_cfg_patches()


class AudexCFGScheduler(Scheduler):
    pass
