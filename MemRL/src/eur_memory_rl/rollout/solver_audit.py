"""Apply identical execution logging and grade accounting to every Solver lane."""
from dataclasses import replace
import time


def install_solver_audit(adapters, record, stage, replay=None):
    def install(adapter, index):
        run_consumer, grade = adapter.run_consumer, adapter.grade

        def consumed(task, execution_id, seed, read, budget=None):
            started = time.monotonic()
            context = {"task_id": task.id, "execution_id": execution_id, "solver_worker": index}
            stage("consumer", **context)
            record("solver_dispatch", {**context, "phase": "start"})
            try:
                result = replay(task, execution_id, seed, read, budget) if replay else None
                if result is None:
                    result = run_consumer(task, execution_id, seed, read, budget)
                record("executions", {"execution": result, "task_identity": task.identity,
                                      "solver_worker": index})
            except Exception as exc:
                record("solver_dispatch", {**context, "phase": "failed", "error_type": type(exc).__name__,
                                           "seconds": time.monotonic() - started})
                raise
            record("solver_dispatch", {**context, "phase": "completed",
                                       "seconds": time.monotonic() - started})
            return result

        def graded(task, execution):
            result = grade(task, execution)
            if execution.finish_reason != "submitted":
                record("budget_grade_adjustments", {
                    "execution_id": execution.execution_id, "solver_worker": index,
                    "finish_reason": execution.finish_reason, "raw_official_grade": result,
                    "effective_quality": 0.0,
                })
                result = replace(result, quality=0.0,
                    metrics={**result.metrics, "raw_official_resolved": result.quality, "resolved": 0.0})
            record("grades", {"task_id": task.id, "execution_id": execution.execution_id,
                              "grade": result, "solver_worker": index})
            return result

        adapter.run_consumer, adapter.grade = consumed, graded

    for index, adapter in enumerate(adapters):
        install(adapter, index)
