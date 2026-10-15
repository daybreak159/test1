from .base import DomainAdapter


class WebAdapter(DomainAdapter):
    domain = "web"

    def consumer_system(self):
        return (
            "Complete the website task using the current accessibility tree. Return "
            '{"action":"WebArena id-based action"} or {"final":"answer"}. '
            "Use only observed page elements. Verify changes from tool results."
        )
