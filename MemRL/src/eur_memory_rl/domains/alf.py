from .base import SourceThenQueryAdapter


class ALFAdapter(SourceThenQueryAdapter):
    domain = "alf"
    independent_queries = False  # The v3 current-query return shortcut is specific to readonly QA.

    def build_plan(self, plan_id, scope, sources, tasks, split):
        supports = {s.identity.sample_id for s in sources}
        targets = {t.identity.sample_id for t in tasks}
        if supports & targets:
            raise ValueError("ALF support experts and target instances must be disjoint")
        return super().build_plan(plan_id, scope, sources, tasks, split)

    def consumer_system(self):
        return (
            "Complete the household goal using textual ALFWorld commands. At each step choose "
            "one actual command from the current admissible_commands list and copy it exactly. "
            'Return JSON with exactly one key: {"action":"<chosen admissible command>"}. '
            "The value must be an actual environment command, never the placeholder 'command'. "
            'If stopping, return {"final":"<your summary>"}. Only observations establish progress.'
        )
