#!/bin/bash
# Local evaluation script for SimLingo (without SLURM)
# This script evaluates a single route at a time

# =================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =================================================================
REPO_ROOT=${WORK_DIR}
CHECKPOINT_PATH="${REPO_ROOT}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
AGENT_FILE="${REPO_ROOT}/team_code/agent_simlingo.py"
ROUTE_DIR="${REPO_ROOT}/leaderboard/data/bench2drive_split"
OUTPUT_DIR="${REPO_ROOT}/eval_results/Bench2Drive/simlingo/bench2drive"

# Evaluation parameters
SEED=1  # Traffic manager seed
PORT=2000  # CARLA server port
TM_PORT=8000  # Traffic manager port

# Route selection:
# - Leave ROUTE_SELECTION empty to evaluate ALL routes
# - Set to single route like "bench2drive_01.xml" 
# - Set to range like "2-11" to evaluate routes 2 through 11
# - Set to list like "1,5,10,15" to evaluate specific routes
ROUTE_SELECTION="0-9"  # Examples: "", "bench2drive_01.xml", "2-11", "1,5,10"

# =================================================================
# SETUP ENVIRONMENT
# =================================================================

# echo current python path
echo $PYTHONPATH

# Create output directories
mkdir -p "${OUTPUT_DIR}/${SEED}/res"
mkdir -p "${OUTPUT_DIR}/${SEED}/viz"
mkdir -p "${OUTPUT_DIR}/${SEED}/logs"

# =================================================================
# CHECK PREREQUISITES
# =================================================================
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Please download the model from HuggingFace and place it in the correct location."
    exit 1
fi

if [ ! -d "${CARLA_ROOT}" ]; then
    echo "ERROR: CARLA not found at ${CARLA_ROOT}"
    echo "Please update CARLA_ROOT in this script."
    exit 1
fi

# =================================================================
# PARSE ROUTE SELECTION
# =================================================================
parse_routes() {
    local selection="$1"
    local routes=()
    
    if [ -z "${selection}" ]; then
        # Empty: evaluate all routes
        routes=($(ls ${ROUTE_DIR}/*.xml | xargs -n 1 basename))
    elif [[ "${selection}" == *.xml ]]; then
        # Single route file specified
        routes=("${selection}")
    elif [[ "${selection}" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Range format: "2-11"
        local start=$(echo ${selection} | cut -d'-' -f1)
        local end=$(echo ${selection} | cut -d'-' -f2)
        for i in $(seq ${start} ${end}); do
            # Use 2 digits for numbers < 100, 3 digits for >= 100
            if [ ${i} -lt 100 ]; then
                routes+=("bench2drive_$(printf "%02d" ${i}).xml")
            else
                routes+=("bench2drive_$(printf "%03d" ${i}).xml")
            fi
        done
    elif [[ "${selection}" =~ ^[0-9,]+$ ]]; then
        # Comma-separated list: "1,5,10,15"
        IFS=',' read -ra NUMS <<< "${selection}"
        for i in "${NUMS[@]}"; do
            # Use 2 digits for numbers < 100, 3 digits for >= 100
            if [ ${i} -lt 100 ]; then
                routes+=("bench2drive_$(printf "%02d" ${i}).xml")
            else
                routes+=("bench2drive_$(printf "%03d" ${i}).xml")
            fi
        done
    else
        echo "ERROR: Invalid ROUTE_SELECTION format: ${selection}"
        echo "Valid formats:"
        echo "  - Empty string for all routes"
        echo "  - 'bench2drive_01.xml' for single route"
        echo "  - '2-11' for range"
        echo "  - '1,5,10' for specific routes"
        exit 1
    fi
    
    echo "${routes[@]}"
}

# Parse route selection
ROUTES=($(parse_routes "${ROUTE_SELECTION}"))

# Validate that routes exist
VALID_ROUTES=()
for route in "${ROUTES[@]}"; do
    if [ -f "${ROUTE_DIR}/${route}" ]; then
        VALID_ROUTES+=("${route}")
    else
        echo "WARNING: Route file not found: ${route}"
    fi
done
ROUTES=("${VALID_ROUTES[@]}")

echo "=========================================="
echo "Starting Bench2Drive Evaluation"
echo "=========================================="
echo "Total routes to evaluate: ${#ROUTES[@]}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output directory: ${OUTPUT_DIR}/${SEED}"
echo "=========================================="
echo ""

COMPLETED=0
FAILED=0

for route_file in "${ROUTES[@]}"; do
    # Extract route ID
    route_id=$(echo ${route_file} | sed 's/bench2drive_//' | sed 's/.xml//')
    route_id=$(printf "%03d" ${route_id})
    
    route_path="${ROUTE_DIR}/${route_file}"
    result_file="${OUTPUT_DIR}/${SEED}/res/${route_id}_res.json"
    viz_path="${OUTPUT_DIR}/${SEED}/viz/${route_id}"
    log_file="${OUTPUT_DIR}/${SEED}/logs/${route_id}.log"
    
    # Skip if already completed
    if [ -f "${result_file}" ]; then
        echo "[SKIP] Route ${route_id} already completed"
        ((COMPLETED++))
        continue
    fi
    
    echo "=========================================="
    echo "[$(date)] Evaluating route ${route_id} (${route_file})"
    echo "=========================================="
    
    # Create viz directory
    mkdir -p "${viz_path}"
    
    # Set save path for agent
    export SAVE_PATH="${viz_path}/"

    echo "Running evaluation"
    
    # Run evaluation
    python -u ${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
        --routes="${route_path}" \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint="${result_file}" \
        --timeout=600 \
        --agent="${AGENT_FILE}" \
        --agent-config="${CHECKPOINT_PATH}" \
        --traffic-manager-seed=${SEED} \
        --port=${PORT} \
        --traffic-manager-port=${TM_PORT} \
        2>&1 | tee "${log_file}"
    
    # Check if evaluation succeeded
    if [ -f "${result_file}" ]; then
        echo "[SUCCESS] Route ${route_id} completed"
        ((COMPLETED++))
    else
        echo "[FAILED] Route ${route_id} failed"
        ((FAILED++))
    fi
    
    echo ""
done

# =================================================================
# SUMMARY
# =================================================================
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Completed: ${COMPLETED} / ${#ROUTES[@]}"
echo "Failed: ${FAILED}"
echo ""
echo "To get final metrics, run:"
echo "  python ${REPO_ROOT}/Bench2Drive/tools/merge_route_json.py"
echo "=========================================="

