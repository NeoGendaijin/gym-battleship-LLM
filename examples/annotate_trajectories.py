#!/usr/bin/env python
"""
Example script for annotating Battleship trajectories with LPML using GPT-4o.

This script demonstrates how to use the LPML Annotator to generate
strategic annotations for Battleship gameplay trajectories.
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
        help="Path to the pickle file containing trajectories"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        required=True,
        help="Path to save the LPML XML file"
    )
    parser.add_argument(
        "--n_candidates", 
        type=int, 
        default=1,
        help="Number of annotation candidates to generate per trajectory"
    )
    parser.add_argument(
        "--max_trajectories", 
        type=int, 
        default=None,
        help="Maximum number of trajectories to annotate (default: all)"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None,
        help="OpenAI API key (if not set as environment variable)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        help="OpenAI model to use for annotations"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check for API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key must be provided either as parameter or as OPENAI_API_KEY environment variable")
        return 1
    
    try:
        # Create LPML Annotator
        logger.info(f"Creating LPML Annotator with model: {args.model}")
        annotator = LPMLAnnotator(api_key=api_key, model=args.model)
        
        # Annotate trajectories
        logger.info(f"Annotating trajectories from: {args.traj}")
        logger.info(f"Number of candidates per trajectory: {args.n_candidates}")
        if args.max_trajectories:
            logger.info(f"Maximum trajectories to annotate: {args.max_trajectories}")
        
        output_path = annotator.annotate_from_file(
            trajectory_path=args.traj,
            output_path=args.out,
            n_candidates=args.n_candidates,
            max_trajectories=args.max_trajectories
        )
        
        logger.info(f"LPML annotations saved to: {output_path}")
        return 0
    
    except Exception as e:
        logger.error(f"Error annotating trajectories: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
