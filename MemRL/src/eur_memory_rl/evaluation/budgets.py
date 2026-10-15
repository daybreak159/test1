from __future__ import annotations


def planned_costs(
    replicas=2, anchors_per_replica=1, I=2, J=2, K=2, L=2, B=1, tasks_per_stream=8, executable_judge=True
):
    anchors = replicas * anchors_per_replica
    return {
        "anchors": anchors,
        "local_consumer_executions": anchors * K * B * (I * J * L + 1),
        "local_E_generations": anchors * I,
        "local_U_generations": anchors * I * J,
        "local_R_slates": anchors * I * J * K * L,
        "local_R_forwards_max": anchors * I * J * K,
        "pair_judge_calls_max": anchors * I * J * K * L * B * 2 if executable_judge else 0,
        "main_full_executions": replicas * tasks_per_stream,
        "main_R_forwards_max_single_turn": replicas * tasks_per_stream,
        "main_E_generations": replicas * max(tasks_per_stream - 1, 0),
        "main_U_generations": replicas * max(tasks_per_stream - 1, 0),
        "scope": "rollout only; excludes teacher-forced recomputation/backward; invalid/empty nodes reduce actual calls",
    }
