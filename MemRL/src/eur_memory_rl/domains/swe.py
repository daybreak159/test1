from .base import DomainAdapter


class SWEAdapter(DomainAdapter):
    domain = "swe"

    def consumer_system(self):
        return (
            "Resolve the issue in the repository. Run commands using "
            '{"action":{"command":"..."}}. Finish with {"final":"summary"}. '
            "Submission is not proof of resolution; the frozen grader evaluates the resulting patch."
        )
