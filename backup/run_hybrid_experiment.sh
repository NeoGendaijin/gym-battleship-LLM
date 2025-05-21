#!/bin/bash
# Run the complete Battleship experiment with hybrid model approach
# This pipeline trains and evaluates a hybrid agent that combines heuristic strategies with RL

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
BOARD_SIZE=6  # Board size for all experiments
EXPERT_TIMESTEPS=10000  # Training steps for expert
SEED=42
TRAJECTORIES=100  # Number of trajectories to collect
MAX_LPML_TRAJECTORIES=50  # How many trajectories to annotate with LPML
MAX_CONCURRENT=10  # Maximum number of concurrent API requests
EPOCHS=5  # Training epochs for student models
EVAL_EPISODES=50  # Number of episodes for evaluation
HEURISTIC_PROB=0.8  # Probability of using heuristic strategy in hybrid model

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
        EXPERT_TIMESTEPS="$2"
        shift
        shift
        ;;
        --board-size)
        BOARD_SIZE="$2"
        shift
        shift
        ;;
        --heuristic-prob)
        HEURISTIC_PROB="$2"
        shift
        shift
        ;;
        --trajectories)
        TRAJECTORIES="$2"
        shift
        shift
        ;;
        --max-lpml)
        MAX_LPML_TRAJECTORIES="$2"
        shift
        shift
        ;;
        --eval-episodes)
        EVAL_EPISODES="$2"
        shift
        shift
        ;;
        --help)
        echo "Usage: ./run_hybrid_experiment.sh [options]"
        echo "Options:"
        echo "  --skip-expert        Skip training hybrid expert model"
        echo "  --skip-trajectories  Skip collecting trajectories"
        echo "  --skip-lpml          Skip LPML annotation"
        echo "  --skip-strategy      Skip strategy-based distillation"
        echo "  --skip-kl            Skip KL-based distillation"
        echo "  --skip-eval          Skip evaluation"
        echo "  --timesteps N        Set the number of timesteps for expert training (default: $EXPERT_TIMESTEPS)"
        echo "  --board-size N       Set the board size (default: $BOARD_SIZE)"
        echo "  --heuristic-prob N   Set probability of using heuristic strategy in hybrid model (default: $HEURISTIC_PROB)"
        echo "  --trajectories N     Set number of trajectories to collect (default: $TRAJECTORIES)"
        echo "  --max-lpml N         Set max trajectories for LPML annotation (default: $MAX_LPML_TRAJECTORIES)"
        echo "  --eval-episodes N    Set number of episodes for evaluation (default: $EVAL_EPISODES)"
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

# Step 1: Train hybrid expert agent
HYBRID_MODEL="$EXPERT_DIR/hybrid_expert.zip"
if [ "$SKIP_EXPERT" = false ]; then
    echo "=== Step 1: Training hybrid expert agent ==="
    echo "Board size: $BOARD_SIZE, Timesteps: $EXPERT_TIMESTEPS, Heuristic probability: $HEURISTIC_PROB"
    python expert/train_basic.py \
        --total-timesteps $EXPERT_TIMESTEPS \
        --save-path $HYBRID_MODEL \
        --board-size $BOARD_SIZE \
        --seed $SEED \
        --heuristic-prob $HEURISTIC_PROB
else
    echo "=== Skipping Step 1: Hybrid expert training ==="
fi

# Step 2: Collect trajectories from hybrid model
HYBRID_TRAJECTORY_FILE="$TRAJ_DIR/hybrid_trajectories.pkl"
if [ "$SKIP_TRAJECTORIES" = false ]; then
    echo "=== Step 2: Collecting trajectories from hybrid model ==="
    python expert/collect_trajectories.py \
        --model $HYBRID_MODEL \
        --episodes $TRAJECTORIES \
        --board-size $BOARD_SIZE \
        --out $HYBRID_TRAJECTORY_FILE \
        --seed $SEED \
        --use-maskable  # Required for MaskablePPO models
else
    echo "=== Skipping Step 2: Trajectory collection ==="
fi

# Step 3: LPML annotation of hybrid trajectories
LPML_FILE="$LPML_DIR/hybrid_battleship.xml"
if [ "$SKIP_LPML" = false ]; then
    echo "=== Step 3: LPML annotation of hybrid model trajectories ==="
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable not set. Cannot perform LPML annotation."
        exit 1
    fi
    python lpml/async_annotate.py \
        --traj $HYBRID_TRAJECTORY_FILE \
        --out $LPML_FILE \
        --n_candidates 3 \
        --max_trajectories $MAX_LPML_TRAJECTORIES \
        --max_concurrent $MAX_CONCURRENT
else
    echo "=== Skipping Step 3: LPML annotation ==="
fi

# Step 4A: Train strategy-based student policy (LPML approach)
STRATEGY_MODEL="$DISTILL_DIR/hybrid_student_strategy.pth"
if [ "$SKIP_STRATEGY" = false ] && [ "$SKIP_LPML" = false ] && [ -f "$LPML_FILE" ]; then
    echo "=== Step 4A: Training strategy-based student policy from LPML ==="
    python examples/train_student_policy.py \
        --lpml $LPML_FILE \
        --model-type strategy \
        --out $STRATEGY_MODEL \
        --grid-size $BOARD_SIZE \
        --epochs $EPOCHS
else
    echo "=== Skipping Step 4A: Strategy-based distillation ==="
    if [ "$SKIP_LPML" = true ]; then
        echo "   (LPML annotation was skipped, so strategy distillation cannot proceed)"
    elif [ ! -f "$LPML_FILE" ]; then
        echo "   (LPML file not found, so strategy distillation cannot proceed)"
    fi
fi

# Step 4B: Train KL-based student policy from hybrid teacher
KL_MODEL="$DISTILL_DIR/hybrid_student_kl.pth"
if [ "$SKIP_KL" = false ] && [ -f "$HYBRID_MODEL" ]; then
    echo "=== Step 4B: Training KL-based student policy from hybrid teacher ==="
    python examples/train_student_policy.py \
        --teacher $HYBRID_MODEL \
        --model-type kl \
        --out $KL_MODEL \
        --grid-size $BOARD_SIZE \
        --epochs $EPOCHS
else
    echo "=== Skipping Step 4B: KL-based distillation ==="
    if [ ! -f "$HYBRID_MODEL" ]; then
        echo "   (Hybrid expert model not found, so KL distillation cannot proceed)"
    fi
fi

# Step 5: Evaluate all models
if [ "$SKIP_EVAL" = false ]; then
    echo "=== Step 5: Evaluating models ==="
    
    # Prepare evaluation output directory
    mkdir -p "$EVAL_DIR/results"
    RESULTS_FILE="$EVAL_DIR/results/$(date +%Y%m%d_%H%M%S)_hybrid_eval_results.txt"
    echo "Hybrid Model Evaluation Results (Board Size: $BOARD_SIZE, Episodes: $EVAL_EPISODES)" > $RESULTS_FILE
    echo "=======================================================" >> $RESULTS_FILE
    
    # Evaluate hybrid expert model
    echo "Evaluating hybrid expert model..." | tee -a $RESULTS_FILE
    python examples/evaluate_model.py \
        --agent $HYBRID_MODEL \
        --episodes $EVAL_EPISODES | tee -a $RESULTS_FILE
    echo "-------------------------------------------------------" >> $RESULTS_FILE
    
    # Evaluate strategy-based student model (LPML-based)
    if [ -f $STRATEGY_MODEL ]; then
        echo "Evaluating strategy-based student model..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent $STRATEGY_MODEL \
            --episodes $EVAL_EPISODES | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Evaluate KL-based student model
    if [ -f $KL_MODEL ]; then
        echo "Evaluating KL-based student model..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent $KL_MODEL \
            --episodes $EVAL_EPISODES | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Transfer task evaluation (larger grid)
    if [ $BOARD_SIZE -lt 10 ]; then
        LARGE_GRID_SIZE=$((BOARD_SIZE * 2))
        echo "Transfer task evaluation (larger grid: ${LARGE_GRID_SIZE}x${LARGE_GRID_SIZE})..." | tee -a $RESULTS_FILE
        
        # Evaluate hybrid expert on larger grid
        echo "Evaluating hybrid expert on larger grid..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent $HYBRID_MODEL \
            --variant large_grid \
            --episodes $((EVAL_EPISODES / 2)) | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
        
        # Evaluate strategy-based student on larger grid
        if [ -f $STRATEGY_MODEL ]; then
            echo "Evaluating strategy-based student on larger grid..." | tee -a $RESULTS_FILE
            python examples/evaluate_model.py \
                --agent $STRATEGY_MODEL \
                --variant large_grid \
                --episodes $((EVAL_EPISODES / 2)) | tee -a $RESULTS_FILE
            echo "-------------------------------------------------------" >> $RESULTS_FILE
        fi
        
        # Evaluate KL-based student on larger grid
        if [ -f $KL_MODEL ]; then
            echo "Evaluating KL-based student on larger grid..." | tee -a $RESULTS_FILE
            python examples/evaluate_model.py \
                --agent $KL_MODEL \
                --variant large_grid \
                --episodes $((EVAL_EPISODES / 2)) | tee -a $RESULTS_FILE
            echo "-------------------------------------------------------" >> $RESULTS_FILE
        fi
    fi
    
    # Run comparison for more detailed analysis if episodes > 0
    if [ "$EVAL_EPISODES" -gt 0 ]; then
        echo "Running detailed comparison of models..."
        python compare_strategies.py \
            --board-size $BOARD_SIZE \
            --episodes 10 \
            --hybrid-model $HYBRID_MODEL | tee -a $RESULTS_FILE
    fi
    
    echo "All evaluation results saved to $RESULTS_FILE"
else
    echo "=== Skipping Step 5: Evaluation ==="
fi

echo "=== Hybrid model experiment completed ==="
echo ""
echo "Models trained and evaluated:"
echo "  1. Hybrid Expert: $HYBRID_MODEL (heuristic_prob: $HEURISTIC_PROB)"
echo "  2. Strategy-based Student: $STRATEGY_MODEL (from LPML)"
echo "  3. KL-based Student: $KL_MODEL (from hybrid teacher)"
echo ""
echo "To run a custom comparison, use:"
echo "  python compare_strategies.py --hybrid-model $HYBRID_MODEL --board-size $BOARD_SIZE --episodes 20"
