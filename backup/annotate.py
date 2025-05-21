#!/usr/bin/env python
"""
Annotate Battleship gameplay trajectories with LPML using GPT-4o.

This script processes trajectories from an expert agent and generates
LPML annotations using GPT-4o to extract strategic insights.
"""

import os
import argparse
import logging
from gym_battleship.lpml import LPMLAnnotator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Annotate Battleship trajectories with LPML")
    
    parser.add_argument(
        "--traj",
        type=str,
        required=True,
        help="Path to the trajectory pickle file"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save the LPML annotations XML file"
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=3,
        help="Number of annotation candidates to generate per trajectory"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=10,
        help="Maximum number of trajectories to annotate (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="OpenAI model to use (default: gpt-4o-mini-2024-07-18)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not provided, uses OPENAI_API_KEY environment variable)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Check for API key
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key must be provided either as command-line argument or as OPENAI_API_KEY environment variable")
            return 1
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
        # Create annotator
        logger.info(f"Creating LPMLAnnotator with model {args.model}")
        annotator = LPMLAnnotator(api_key=api_key, model=args.model)
        
        # Annotate trajectories
        logger.info(f"Annotating trajectories from {args.traj}")
        logger.info(f"Generating {args.n_candidates} candidates per trajectory")
        logger.info(f"Processing up to {args.max_trajectories} trajectories")
        
        # Process the trajectories
        output_path = annotator.annotate_from_file(
            trajectory_path=args.traj,
            output_path=args.out,
            n_candidates=args.n_candidates,
            max_trajectories=args.max_trajectories
        )
        
        logger.info(f"LPML annotations saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating LPML annotations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
