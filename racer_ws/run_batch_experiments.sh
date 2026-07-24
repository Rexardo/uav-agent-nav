#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/home/qian/racer_ws}"
RUN_COUNT="${1:-20}"
FIRST_SEED="${2:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/results}"
SCENE="${SCENE:-dense_maze}"
COMMUNICATION_RANGE="${COMMUNICATION_RANGE:--1.0}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-180}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
POLL_SECONDS="${POLL_SECONDS:-2}"

if ! [[ "${RUN_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_COUNT must be a positive integer, got: ${RUN_COUNT}" >&2
  exit 2
fi
if ! [[ "${FIRST_SEED}" =~ ^[0-9]+$ ]]; then
  echo "FIRST_SEED must be a non-negative integer, got: ${FIRST_SEED}" >&2
  exit 2
fi
if ! [[ "${COMMUNICATION_RANGE}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "COMMUNICATION_RANGE must be numeric, got: ${COMMUNICATION_RANGE}" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source "${WORKSPACE}/devel/setup.bash"

if [[ "${COMMUNICATION_RANGE}" == -* ]]; then
  CR_DIRECTORY="CR_inf"
else
  cr_text="$(printf '%.3f' "${COMMUNICATION_RANGE}")"
  while [[ "${cr_text}" == *0 ]]; do cr_text="${cr_text%0}"; done
  cr_text="${cr_text%.}"
  CR_DIRECTORY="CR_${cr_text//./p}m"
fi

RESULT_DIRECTORY="${OUTPUT_ROOT}/${SCENE}/${CR_DIRECTORY}"
SUMMARY_FILE="${RESULT_DIRECTORY}/summary.csv"
LOG_DIRECTORY="${OUTPUT_ROOT}/batch_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIRECTORY}"

LAUNCH_PID=""

stop_launch() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
    fi
    if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      kill -KILL "${LAUNCH_PID}" 2>/dev/null || true
    fi
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  LAUNCH_PID=""
}

handle_interrupt() {
  echo
  echo "Batch experiment interrupted; shutting down the current simulation..."
  stop_launch
  exit 130
}

trap stop_launch EXIT
trap handle_interrupt INT TERM

summary_line_count() {
  if [[ -f "${SUMMARY_FILE}" ]]; then
    wc -l < "${SUMMARY_FILE}"
  else
    echo 0
  fi
}

read_state() {
  local drone_id="$1"
  timeout 3s rostopic echo -n 1 "/experiment/fsm_state_${drone_id}" 2>/dev/null \
    | awk '/^data:/{print $2; exit}' || true
}

wait_until_ready() {
  local start_time
  start_time="$(date +%s)"
  while true; do
    local ready=true
    for drone_id in 1 2 3 4; do
      if [[ "$(read_state "${drone_id}")" != "1" ]]; then
        ready=false
        break
      fi
    done
    if [[ "${ready}" == true ]]; then
      return 0
    fi
    if (( $(date +%s) - start_time >= STARTUP_TIMEOUT_SECONDS )); then
      return 1
    fi
    if [[ -n "${LAUNCH_PID}" ]] && ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_until_recorded() {
  local previous_lines="$1"
  local start_time
  start_time="$(date +%s)"
  while true; do
    if (( $(summary_line_count) > previous_lines )); then
      return 0
    fi
    if (( $(date +%s) - start_time >= RUN_TIMEOUT_SECONDS )); then
      return 1
    fi
    if [[ -n "${LAUNCH_PID}" ]] && ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
}

echo "Batch experiment started: runs=${RUN_COUNT}, first_seed=${FIRST_SEED}"
echo "Results: ${RESULT_DIRECTORY}"
echo "Logs: ${LOG_DIRECTORY}"

for ((run_index = 0; run_index < RUN_COUNT; ++run_index)); do
  seed=$((FIRST_SEED + run_index))
  run_number=$((run_index + 1))
  log_file="${LOG_DIRECTORY}/run_$(printf '%03d' "${run_number}")_seed_$(printf '%03d' "${seed}").log"
  previous_lines="$(summary_line_count)"

  echo "[${run_number}/${RUN_COUNT}] Starting metrics_seed=${seed}"
  roslaunch exploration_manager swarm_exploration_dense_maze.launch \
    record_metrics:=true \
    metrics_seed:="${seed}" \
    metrics_scene:="${SCENE}" \
    metrics_communication_range:="${COMMUNICATION_RANGE}" \
    metrics_output_root:="${OUTPUT_ROOT}" \
    >"${log_file}" 2>&1 &
  LAUNCH_PID=$!

  if ! wait_until_ready; then
    echo "[${run_number}/${RUN_COUNT}] Startup failed or timed out; see ${log_file}" >&2
    exit 1
  fi

  echo "[${run_number}/${RUN_COUNT}] All four UAVs are ready; triggering exploration"
  rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
    "{header: {frame_id: 'world'}, pose: {position: {x: 0, y: 0, z: 1}, orientation: {w: 1}}}" \
    >/dev/null

  if ! wait_until_recorded "${previous_lines}"; then
    echo "[${run_number}/${RUN_COUNT}] Run failed or timed out; see ${log_file}" >&2
    exit 1
  fi

  echo "[${run_number}/${RUN_COUNT}] Metrics saved; shutting down this simulation"
  stop_launch
  sleep 3
done

trap - EXIT INT TERM
echo "All ${RUN_COUNT} experiments completed successfully."
echo "Summary: ${SUMMARY_FILE}"
