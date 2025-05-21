#!/bin/bash
# Enhanced Battleship experiment with LPML, RAG, and hybrid approach
# This pipeline trains various models and compares them in a single flow

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
MAX_CONCURRENT=10  # Maximum number of concurrent API requests for LPML annotation
EPOCHS=5  # Training epochs for the student models
EVAL_EPISODES=50  # Number of episodes for evaluation
HEURISTIC_PROB=0.8  # Probability of using heuristic strategy in hybrid model
RAG_EPISODES=10  # Number of episodes for RAG vs Student comparison
OPENAI_MODEL="gpt-4o"  # Model to use for RAG and LPML annotation

# Parse command line arguments
SKIP_HEURISTIC=false
SKIP_HYBRID=false
SKIP_TRAJECTORIES=false
SKIP_LPML=false
SKIP_STUDENT=false
SKIP_RAG=false
SKIP_LPML_LLM=false   # Skip LPML vs. vanilla LLM comparison
SKIP_EVAL=false
FORCE_OPENAI=false    # Flag to force OpenAI-dependent steps even without API key

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --skip-heuristic)
        SKIP_HEURISTIC=true
        shift
        ;;
        --skip-hybrid)
        SKIP_HYBRID=true
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
        --skip-student)
        SKIP_STUDENT=true
        shift
        ;;
        --skip-rag)
        SKIP_RAG=true
        shift
        ;;
        --skip-lpml-llm)
        SKIP_LPML_LLM=true
        shift
        ;;
        --skip-eval)
        SKIP_EVAL=true
        shift
        ;;
        --force-openai)
        FORCE_OPENAI=true
        shift
        ;;
        --board-size)
        BOARD_SIZE="$2"
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
        --heuristic-prob)
        HEURISTIC_PROB="$2"
        shift
        shift
        ;;
        --rag-episodes)
        RAG_EPISODES="$2"
        shift
        shift
        ;;
        --openai-model)
        OPENAI_MODEL="$2"
        shift
        shift
        ;;
        --help)
        echo "Usage: ./run_experiment.sh [options]"
        echo "Options:"
        echo "  --skip-heuristic     Skip creating heuristic expert"
        echo "  --skip-hybrid        Skip creating hybrid expert"
        echo "  --skip-trajectories  Skip collecting trajectories"
        echo "  --skip-lpml          Skip LPML annotation"
        echo "  --skip-student       Skip student model training"
        echo "  --skip-rag           Skip RAG vs Student comparison"
        echo "  --skip-lpml-llm      Skip LPML vs vanilla LLM comparison"
        echo "  --skip-eval          Skip evaluation"
        echo "  --force-openai       Force LPML and RAG steps even if no API key is found"
        echo "  --board-size N       Set the board size (default: $BOARD_SIZE)"
        echo "  --trajectories N     Set number of trajectories to collect (default: $TRAJECTORIES)"
        echo "  --max-lpml N         Set max trajectories for LPML annotation (default: $MAX_LPML_TRAJECTORIES)"
        echo "  --eval-episodes N    Set number of episodes for evaluation (default: $EVAL_EPISODES)"
        echo "  --heuristic-prob N   Set probability of using heuristic in hybrid model (default: $HEURISTIC_PROB)"
        echo "  --rag-episodes N     Set number of episodes for RAG comparison (default: $RAG_EPISODES)"
        echo "  --openai-model NAME  Set OpenAI model to use for LPML and RAG (default: $OPENAI_MODEL)"
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

# Check for OpenAI API key and adjust steps accordingly
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set."
    echo "The LPML annotation, RAG comparison, and LPML vs. LLM steps require a valid API key."
    echo ""
    echo "To set the API key, run:"
    echo "  export OPENAI_API_KEY=your-api-key-here"
    echo ""
    
    if [ "$FORCE_OPENAI" = false ]; then
        echo "Automatically skipping OpenAI API-dependent steps."
        echo "Use --force-openai to attempt these steps anyway."
        SKIP_LPML=true
        SKIP_RAG=true
        SKIP_LPML_LLM=true
    else
        echo "Warning: Attempting to proceed with OpenAI API-dependent steps despite missing API key."
        echo "This will likely result in errors unless API key is configured elsewhere."
    fi
    echo ""
fi

# Step 1A: Create pure heuristic expert
HEURISTIC_MODEL="$EXPERT_DIR/heuristic_expert.zip"
if [ "$SKIP_HEURISTIC" = false ]; then
    echo "=== Step 1A: Creating pure heuristic expert ==="
    python expert/train_expert.py \
        --episodes 10 \
        --save-path $HEURISTIC_MODEL \
        --board-size $BOARD_SIZE \
        --seed $SEED
else
    echo "=== Skipping Step 1A: Heuristic expert creation ==="
fi

# Step 1B: Create hybrid expert (RL + heuristic)
HYBRID_MODEL="$EXPERT_DIR/hybrid_expert.zip"
if [ "$SKIP_HYBRID" = false ]; then
    echo "=== Step 1B: Creating hybrid expert (RL + heuristic) ==="
    echo "Board size: $BOARD_SIZE, Timesteps: $EXPERT_TIMESTEPS, Heuristic probability: $HEURISTIC_PROB"
    python expert/train_basic.py \
        --total-timesteps $EXPERT_TIMESTEPS \
        --save-path $HYBRID_MODEL \
        --board-size $BOARD_SIZE \
        --seed $SEED \
        --heuristic-prob $HEURISTIC_PROB
else
    echo "=== Skipping Step 1B: Hybrid expert creation ==="
fi

# Step 2A: Collect trajectories from heuristic expert
HEURISTIC_TRAJECTORY_FILE="$TRAJ_DIR/heuristic_trajectories.pkl"
if [ "$SKIP_TRAJECTORIES" = false ] && [ "$SKIP_HEURISTIC" = false ]; then
    echo "=== Step 2A: Collecting trajectories from heuristic expert ==="
    if [ ! -f "$HEURISTIC_MODEL" ]; then
        echo "Error: Heuristic expert model not found at $HEURISTIC_MODEL"
        echo "Run without --skip-heuristic first or ensure the model exists"
        exit 1
    fi
    
    python expert/collect_heuristic_trajectories.py \
        --model $HEURISTIC_MODEL \
        --episodes $TRAJECTORIES \
        --board-size $BOARD_SIZE \
        --out $HEURISTIC_TRAJECTORY_FILE \
        --seed $SEED || { echo "Error collecting heuristic trajectories"; exit 1; }
else
    echo "=== Skipping Step 2A: Heuristic trajectory collection ==="
fi

# Step 2B: Collect trajectories from hybrid expert
HYBRID_TRAJECTORY_FILE="$TRAJ_DIR/hybrid_trajectories.pkl"
if [ "$SKIP_TRAJECTORIES" = false ] && [ "$SKIP_HYBRID" = false ]; then
    echo "=== Step 2B: Collecting trajectories from hybrid expert ==="
    if [ ! -f "$HYBRID_MODEL" ]; then
        echo "Error: Hybrid expert model not found at $HYBRID_MODEL"
        echo "Run without --skip-hybrid first or ensure the model exists"
        exit 1
    fi
    
    python expert/collect_trajectories.py \
        --model $HYBRID_MODEL \
        --episodes $TRAJECTORIES \
        --board-size $BOARD_SIZE \
        --out $HYBRID_TRAJECTORY_FILE \
        --seed $SEED \
        --use-maskable  # Required for MaskablePPO models
else
    echo "=== Skipping Step 2B: Hybrid trajectory collection ==="
fi

# Function to verify API key validity
verify_api_key() {
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable not set."
        return 1
    fi
    
    # Attempt a minimal API call to verify key validity
    echo "Verifying OpenAI API key validity..."
    MODELS_RESPONSE=$(curl -s -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models)
    
    if [[ $MODELS_RESPONSE == *"error"* ]] && [[ $MODELS_RESPONSE == *"invalid_api_key"* ]]; then
        echo "Error: Invalid OpenAI API key. Please check your key and try again."
        return 1
    fi
    
    if [[ $MODELS_RESPONSE == *"error"* ]]; then
        echo "Warning: Unexpected error when verifying API key: $MODELS_RESPONSE"
        echo "Proceeding anyway, but this may cause issues..."
        return 0
    fi
    
    echo "API key verified successfully."
    return 0
}

# Step 3A: LPML annotation of heuristic expert trajectories
HEURISTIC_LPML_FILE="$LPML_DIR/heuristic_battleship.xml"
if [ "$SKIP_LPML" = false ] && [ -f "$HEURISTIC_TRAJECTORY_FILE" ]; then
    echo "=== Step 3A: LPML annotation of heuristic expert trajectories ==="
    
    # Verify API key before proceeding
    if ! verify_api_key; then
        if [ "$FORCE_OPENAI" = true ]; then
            echo "Warning: Attempting to proceed despite API key issues (--force-openai is set)."
        else
            echo "Skipping LPML annotation due to API key issues."
            echo "Use --force-openai to attempt anyway."
            SKIP_LPML=true
        fi
    else
        python lpml/async_annotate.py \
            --traj $HEURISTIC_TRAJECTORY_FILE \
            --out $HEURISTIC_LPML_FILE \
            --n_candidates 3 \
            --max_trajectories $MAX_LPML_TRAJECTORIES \
            --max_concurrent $MAX_CONCURRENT \
            --model $OPENAI_MODEL
    fi
else
    echo "=== Skipping Step 3A: Heuristic LPML annotation ==="
    if [ ! -f "$HEURISTIC_TRAJECTORY_FILE" ]; then
        echo "   (Heuristic trajectory file not found: $HEURISTIC_TRAJECTORY_FILE)"
    fi
fi

# Step 3B: LPML annotation of hybrid expert trajectories
HYBRID_LPML_FILE="$LPML_DIR/hybrid_battleship.xml"
if [ "$SKIP_LPML" = false ] && [ -f "$HYBRID_TRAJECTORY_FILE" ]; then
    echo "=== Step 3B: LPML annotation of hybrid expert trajectories ==="
    
    # No need to verify API key again if already done in step 3A
    if [ ! -f "$HEURISTIC_TRAJECTORY_FILE" ] && ! verify_api_key; then
        if [ "$FORCE_OPENAI" = true ]; then
            echo "Warning: Attempting to proceed despite API key issues (--force-openai is set)."
        else
            echo "Skipping LPML annotation due to API key issues."
            echo "Use --force-openai to attempt anyway."
            SKIP_LPML=true
        fi
    else
        python lpml/async_annotate.py \
            --traj $HYBRID_TRAJECTORY_FILE \
            --out $HYBRID_LPML_FILE \
            --n_candidates 3 \
            --max_trajectories $MAX_LPML_TRAJECTORIES \
            --max_concurrent $MAX_CONCURRENT \
            --model $OPENAI_MODEL
    fi
else
    echo "=== Skipping Step 3B: Hybrid LPML annotation ==="
    if [ ! -f "$HYBRID_TRAJECTORY_FILE" ]; then
        echo "   (Hybrid trajectory file not found: $HYBRID_TRAJECTORY_FILE)"
    fi
fi

# Step 4: Train student models from LPML annotations
if [ "$SKIP_STUDENT" = false ]; then
    # Step 4A: Train student from heuristic LPML
    HEURISTIC_STUDENT_MODEL="$DISTILL_DIR/heuristic_student.pth"
    if [ -f "$HEURISTIC_LPML_FILE" ]; then
        echo "=== Step 4A: Training student policy from heuristic LPML ==="
        python examples/train_student_policy.py \
            --lpml $HEURISTIC_LPML_FILE \
            --model-type strategy \
            --out $HEURISTIC_STUDENT_MODEL \
            --grid-size $BOARD_SIZE \
            --epochs $EPOCHS
    else
        echo "=== Skipping Step 4A: Heuristic student training ==="
        echo "   (LPML file not found: $HEURISTIC_LPML_FILE)"
    fi

    # Step 4B: Train student from hybrid LPML
    HYBRID_STUDENT_MODEL="$DISTILL_DIR/hybrid_student.pth"
    if [ -f "$HYBRID_LPML_FILE" ]; then
        echo "=== Step 4B: Training student policy from hybrid LPML ==="
        python examples/train_student_policy.py \
            --lpml $HYBRID_LPML_FILE \
            --model-type strategy \
            --out $HYBRID_STUDENT_MODEL \
            --grid-size $BOARD_SIZE \
            --epochs $EPOCHS
    else
        echo "=== Skipping Step 4B: Hybrid student training ==="
        echo "   (LPML file not found: $HYBRID_LPML_FILE)"
    fi
else
    echo "=== Skipping Step 4: Student model training ==="
fi

# Step 5: RAG vs Student comparison
if [ "$SKIP_RAG" = false ]; then
    echo "=== Step 5: Comparing RAG (with LPML access) vs Student (without LPML access) ==="
    
    # Verify API key before proceeding
    if ! verify_api_key; then
        if [ "$FORCE_OPENAI" = true ]; then
            echo "Warning: Attempting to proceed with RAG comparison despite API key issues (--force-openai is set)."
        else
            echo "Skipping RAG comparison due to API key issues."
            echo "Use --force-openai to attempt anyway."
            SKIP_RAG=true
        fi
    else
        # Choose the best available student model
        STUDENT_MODEL=""
        if [ -f "$HYBRID_STUDENT_MODEL" ]; then
            STUDENT_MODEL=$HYBRID_STUDENT_MODEL
            echo "Using hybrid student model for comparison"
        elif [ -f "$HEURISTIC_STUDENT_MODEL" ]; then
            STUDENT_MODEL=$HEURISTIC_STUDENT_MODEL
            echo "Using heuristic student model for comparison"
        elif [ -f "$DISTILL_DIR/student_strategy.pth" ]; then
            STUDENT_MODEL="$DISTILL_DIR/student_strategy.pth"
            echo "Using legacy student model for comparison"
        else
            echo "Error: No student model found for RAG comparison."
            echo "Make sure to run the student training step first or provide an existing model."
            SKIP_RAG=true
        fi
        
        if [ "$SKIP_RAG" = false ]; then
            python examples/rag_vs_student.py \
                --student-model $STUDENT_MODEL \
                --episodes $RAG_EPISODES \
                --board-size $BOARD_SIZE \
                --openai-model $OPENAI_MODEL
        fi
    fi
else
    echo "=== Skipping Step 5: RAG vs Student comparison ==="
fi

# Step 6: LPML vs Vanilla LLM comparison
if [ "$SKIP_LPML_LLM" = false ]; then
    echo "=== Step 6: Comparing LPML-guided LLM vs Vanilla LLM ==="
    
    # Verify API key before proceeding
    if ! verify_api_key; then
        if [ "$FORCE_OPENAI" = true ]; then
            echo "Warning: Attempting to proceed with LPML vs LLM comparison despite API key issues (--force-openai is set)."
        else
            echo "Skipping LPML vs LLM comparison due to API key issues."
            echo "Use --force-openai to attempt anyway."
            SKIP_LPML_LLM=true
        fi
    else
        # Select the best available LPML file
        LPML_FILE=""
        if [ -f "$HYBRID_LPML_FILE" ]; then
            LPML_FILE=$HYBRID_LPML_FILE
            echo "Using hybrid LPML file for comparison"
        elif [ -f "$HEURISTIC_LPML_FILE" ]; then
            LPML_FILE=$HEURISTIC_LPML_FILE
            echo "Using heuristic LPML file for comparison"
        elif [ -f "lpml/battleship.xml" ]; then
            LPML_FILE="lpml/battleship.xml"
            echo "Using legacy LPML file for comparison"
        else
            echo "Error: No LPML file found for LPML vs LLM comparison."
            echo "Make sure to run the LPML annotation step first."
            SKIP_LPML_LLM=true
        fi
        
        if [ "$SKIP_LPML_LLM" = false ]; then
            python examples/lpml_vs_llm.py \
                --lpml-file $LPML_FILE \
                --episodes $RAG_EPISODES \
                --board-size $BOARD_SIZE \
                --openai-model $OPENAI_MODEL
        fi
    fi
else
    echo "=== Skipping Step 6: LPML vs Vanilla LLM comparison ==="
fi

# Step 7: Evaluate models
if [ "$SKIP_EVAL" = false ]; then
    echo "=== Step 7: Evaluating all models ==="
    
    # Prepare evaluation output directory
    mkdir -p "$EVAL_DIR/results"
    RESULTS_FILE="$EVAL_DIR/results/$(date +%Y%m%d_%H%M%S)_full_evaluation.txt"
    echo "Full Model Evaluation Results (Board Size: $BOARD_SIZE, Episodes: $EVAL_EPISODES)" > $RESULTS_FILE
    echo "=======================================================" >> $RESULTS_FILE
    
    # Evaluate random baseline
    echo "Evaluating random baseline for comparison..." | tee -a $RESULTS_FILE
    python examples/evaluate_model.py \
        --random \
        --episodes $EVAL_EPISODES \
        --grid-size $BOARD_SIZE | tee -a $RESULTS_FILE
    echo "-------------------------------------------------------" >> $RESULTS_FILE
    
    # Evaluate heuristic expert
    if [ -f "$HEURISTIC_MODEL" ]; then
        echo "Evaluating heuristic expert..." | tee -a $RESULTS_FILE
        python evaluate_expert.py \
            --expert-path $HEURISTIC_MODEL \
            --episodes $EVAL_EPISODES \
            --board-size $BOARD_SIZE | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Evaluate hybrid expert
    if [ -f "$HYBRID_MODEL" ]; then
        echo "Evaluating hybrid expert..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent $HYBRID_MODEL \
            --episodes $EVAL_EPISODES \
            --grid-size $BOARD_SIZE | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Evaluate heuristic student model
    if [ -f "$HEURISTIC_STUDENT_MODEL" ]; then
        echo "Evaluating heuristic student model..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent $HEURISTIC_STUDENT_MODEL \
            --episodes $EVAL_EPISODES \
            --grid-size $BOARD_SIZE | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Evaluate hybrid student model
    if [ -f "$HYBRID_STUDENT_MODEL" ]; then
        echo "Evaluating hybrid student model..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent $HYBRID_STUDENT_MODEL \
            --episodes $EVAL_EPISODES \
            --grid-size $BOARD_SIZE | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Evaluate legacy student model if present
    if [ -f "$DISTILL_DIR/student_strategy.pth" ] && [ ! -f "$HEURISTIC_STUDENT_MODEL" ]; then
        echo "Evaluating legacy student model..." | tee -a $RESULTS_FILE
        python examples/evaluate_model.py \
            --agent "$DISTILL_DIR/student_strategy.pth" \
            --episodes $EVAL_EPISODES \
            --grid-size $BOARD_SIZE | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    # Run comprehensive comparison if we have both models
    if [ -f "$HEURISTIC_MODEL" ] && [ -f "$HYBRID_MODEL" ]; then
        echo "Running comprehensive model comparison..." | tee -a $RESULTS_FILE
        python compare_strategies.py \
            --board-size $BOARD_SIZE \
            --episodes $((EVAL_EPISODES / 5)) \
            --hybrid-model $HYBRID_MODEL \
            --rl-model $HYBRID_MODEL \
            --seed $SEED | tee -a $RESULTS_FILE
        echo "-------------------------------------------------------" >> $RESULTS_FILE
    fi
    
    echo "All evaluation results saved to $RESULTS_FILE"
else
    echo "=== Skipping Step 7: Evaluation ==="
fi

echo "=== Experiment completed ==="
echo ""
echo "Models trained and evaluated:"
if [ -f "$HEURISTIC_MODEL" ]; then
    echo "  - Heuristic Expert: $HEURISTIC_MODEL"
fi
if [ -f "$HYBRID_MODEL" ]; then
    echo "  - Hybrid Expert: $HYBRID_MODEL (heuristic_prob: $HEURISTIC_PROB)"
fi
if [ -f "$HEURISTIC_STUDENT_MODEL" ]; then
    echo "  - Heuristic Student: $HEURISTIC_STUDENT_MODEL (trained from LPML)"
fi
if [ -f "$HYBRID_STUDENT_MODEL" ]; then
    echo "  - Hybrid Student: $HYBRID_STUDENT_MODEL (trained from LPML)"
fi
if [ -f "$DISTILL_DIR/student_strategy.pth" ] && [ ! -f "$HEURISTIC_STUDENT_MODEL" ]; then
    echo "  - Legacy Student: $DISTILL_DIR/student_strategy.pth"
fi
echo ""

# Print information about OpenAI key if relevant features were skipped
if [ "$SKIP_LPML" = true ] || [ "$SKIP_RAG" = true ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Note: LPML annotation and/or RAG comparison were skipped due to missing OpenAI API key."
        echo "To use these features, set your API key with:"
        echo "  export OPENAI_API_KEY=your-key-here"
        echo ""
    fi
fi

echo "To run custom comparisons, use any of these commands:"
echo "  - python compare_heuristic_lpml.py --heuristic-expert $HEURISTIC_MODEL --lpml-agent $HEURISTIC_STUDENT_MODEL --board-size $BOARD_SIZE --episodes 20"
echo "  - python compare_strategies.py --hybrid-model $HYBRID_MODEL --board-size $BOARD_SIZE --episodes 20"
if [ -f "$HEURISTIC_STUDENT_MODEL" ] || [ -f "$HYBRID_STUDENT_MODEL" ]; then
    STUDENT=$([ -f "$HYBRID_STUDENT_MODEL" ] && echo "$HYBRID_STUDENT_MODEL" || echo "$HEURISTIC_STUDENT_MODEL")
    echo "  - python examples/rag_vs_student.py --student-model $STUDENT --episodes 10 --board-size $BOARD_SIZE"
fi
if [ -f "$HEURISTIC_LPML_FILE" ] || [ -f "$HYBRID_LPML_FILE" ]; then
    LPML=$([ -f "$HYBRID_LPML_FILE" ] && echo "$HYBRID_LPML_FILE" || echo "$HEURISTIC_LPML_FILE")
    echo "  - python examples/lpml_vs_llm.py --lpml-file $LPML --episodes 5 --board-size $BOARD_SIZE"
fi
