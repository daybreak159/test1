"""SWE uses mini-swe-agent; other domains retain their own adapters."""
from .mini_swe import MiniSWEAdapter


class SWEAdapter(MiniSWEAdapter):
    pass
