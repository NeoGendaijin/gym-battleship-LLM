#!/bin/bash
# Run the complete Battleship LLM strategy distillation experiment

set -e  # Exit on error

# Define directories
EXPERT_DIR="expert"
TRAJ_DIR="trajectories"
LPML_DIR="lpml"
DISTILL_DIR="distill"
EVAL_DIR="eval"

# Create directories if they don't exist
mkdir -p $EXPERT_DIR $TRAJ_DIR $LPML_DIR $DISTILL_DIR $EVAL_DIR

# Define parameters
BOARD_SIZE=6
EXPERT_TIMESTEPS=50000  # Reduced for faster training
SEED=42
TRAJECTORIES=1000  # Sufficient for LPML/strategy distillation
MAX_LPML_TRAJECTORIES=300  # Annotation trajectories count
MAX_CONCURRENT=10  # Maximum number of concurrent API requests for async annotation
EPOCHS=10

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set."
    echo "The LPML annotation step will fail without an API key."
    echo "You can skip the LPML step if you only want to train the expert and collect trajectories."
fi

# Parse command line arguments
SKIP_EXPERT=false
SKIP_TRAJECTORIES=false
SKIP_LPML=false
SKIP_STRATEGY=false
SKIP_KL=false
SKIP_EVAL=false
TIMESTEPS=$EXPERT_TIMESTEPS  # Default value

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --skip-expert)
        SKIP_EXPERT=true
        shift
        ;;
        --skip-trajectories)
        SKIP_TRAJECTORIES=true
        shift
        ;;
        --skip-lpml)
        SKIP_LPML=true
        shift
        ;;
        --skip-strategy)
        SKIP_STRATEGY=true
        shift
        ;;
        --skip-kl)
        SKIP_KL=true
        shift
        ;;
        --skip-eval)
        SKIP_EVAL=true
        shift
        ;;
        --timesteps)
        TIMESTEPS="$2"
        shift
        shift
        ;;
        --board-size)
        BOARD_SIZE="$2"
        shift
        shift
        ;;
        --help)
        echo "Usage: ./run_experiment.sh [options]"
        echo "Options:"
        echo "  --skip-expert        Skip training expert model"
        echo "  --skip-trajectories  Skip collecting trajectories"
        echo "  --skip-lpml          Skip LPML annotation"
        echo "  --skip-strategy      Skip strategy-based distillation"
        echo "  --skip-kl            Skip KL-based distillation"
        echo "  --skip-eval          Skip evaluation"
        echo "  --timesteps N        Set the number of timesteps for expert training (default: $EXPERT_TIMESTEPS)"
        echo "  --board-size N       Set the board size (default: $BOARD_SIZE)"
        echo "  --help               Show this help message"
        exit 0
        ;;
        *)
        echo "Unknown option: $key"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
done

# Step 1: Train expert agent with MaskablePPO
EXPERT_MODEL="$EXPERT_DIR/maskable_expert.zip"
if [ "$SKIP_EXPERT" = false ]; then
    echo "=== Step 1: Training expert agent with MaskablePPO ==="
    python expert/train_basic.py \
        --total-timesteps $TIMESTEPS \
        --save-path $EXPERT_MODEL \
        --board-size $BOARD_SIZE \
        --seed $SEED
else
    echo "=== Skipping Step 1: Expert training ==="
fi

# Step 2: Collect trajectories
TRAJECTORY_FILE="$TRAJ_DIR/battleship.pkl"
if [ "$SKIP_TRAJECTORIES" = false ]; then
    echo "=== Step 2: Collecting trajectories ==="
    python expert/collect_trajectories.py \
        --model $EXPERT_MODEL \
        --episodes $TRAJECTORIES \
        --board-size $BOARD_SIZE \
        --out $TRAJECTORY_FILE \
        --seed $SEED \
        --use-maskable  # Add this flag since we're using MaskablePPO
else
    echo "=== Skipping Step 2: Trajectory collection ==="
fi

# Step 3: LPML annotation
LPML_FILE="$LPML_DIR/battleship.xml"
if [ "$SKIP_LPML" = false ]; then
    echo "=== Step 3: LPML annotation (Async) ==="
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable not set. Cannot perform LPML annotation."
        exit 1
    fi
    python lpml/async_annotate.py \
        --traj $TRAJECTORY_FILE \
        --out $LPML_FILE \
        --n_candidates 3 \
        --max_trajectories $MAX_LPML_TRAJECTORIES \
        --max_concurrent $MAX_CONCURRENT
else
    echo "=== Skipping Step 3: LPML annotation ==="
fi

# Step 4A: Train strategy-based student policy
STRATEGY_MODEL="$DISTILL_DIR/student_strategy.pth"
if [ "$SKIP_STRATEGY" = false ]; then
    echo "=== Step 4A: Training strategy-based student policy ==="
    python examples/train_student_policy.py \
        --lpml $LPML_FILE \
        --model-type strategy \
        --out $STRATEGY_MODEL \
        --grid-size $BOARD_SIZE \
        --epochs $EPOCHS
else
    echo "=== Skipping Step 4A: Strategy-based distillation ==="
fi

# Step 4B: Train KL-based student policy with improved implementation
KL_MODEL="$DISTILL_DIR/student_kl.pth"
if [ "$SKIP_KL" = false ]; then
    echo "=== Step 4B: Training KL-based student policy ==="
    python examples/train_student_policy.py \
        --teacher $EXPERT_MODEL \
        --model-type kl \
        --out $KL_MODEL \
        --grid-size $BOARD_SIZE \
        --epochs $EPOCHS
else
    echo "=== Skipping Step 4B: KL-based distillation ==="
fi

# Step 5: Evaluate models
if [ "$SKIP_EVAL" = false ]; then
    echo "=== Step 5: Evaluating models ==="
    
    # Evaluate expert model
    echo "Evaluating expert model..."
    python examples/evaluate_model.py --agent $EXPERT_MODEL --episodes 50
    
    # Evaluate strategy-based student model
    if [ -f $STRATEGY_MODEL ]; then
        echo "Evaluating strategy-based student model..."
        python examples/evaluate_model.py --agent $STRATEGY_MODEL --episodes 50
    fi
    
    # Evaluate KL-based student model
    if [ -f $KL_MODEL ]; then
        echo "Evaluating KL-based student model..."
        python examples/evaluate_model.py --agent $KL_MODEL --episodes 50
    fi
    
    # Transfer task evaluation (larger grid)
    echo "Transfer task evaluation (larger grid)..."
    LARGE_GRID_SIZE=15
    
    if [ -f $STRATEGY_MODEL ]; then
        echo "Evaluating strategy-based student model on larger grid..."
        python examples/evaluate_model.py \
            --agent $STRATEGY_MODEL \
            --episodes 50 \
            --grid-size $LARGE_GRID_SIZE
    fi
    
    if [ -f $KL_MODEL ]; then
        echo "Evaluating KL-based student model on larger grid..."
        python examples/evaluate_model.py \
            --agent $KL_MODEL \
            --episodes 50 \
            --grid-size $LARGE_GRID_SIZE
    fi
else
    echo "=== Skipping Step 5: Evaluation ==="
fi

echo "=== Experiment completed ==="
