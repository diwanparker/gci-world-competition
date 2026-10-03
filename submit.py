"""Automated submission script to Omnicampus Autograder.

Uploads submission.csv to Assignment 1038, polls scoring output,
and displays current public leaderboard standing.
"""
import argparse
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/diwan/core/automation/omnicampus-portal-mcp")
from omnicampus_client import OmnicampusClient

CONFIG_PATH = "/home/diwan/core/automation/omnicampus-portal-mcp/config.json"
COURSE_ID = 170
DEFAULT_ASSIGNMENT_ID = 1038
DEFAULT_CSV_PATH = "./output/submission.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit predictions to Omnicampus")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to submission.csv")
    parser.add_argument("--assignment-id", type=int, default=DEFAULT_ASSIGNMENT_ID, help="Assignment ID (default: 1038)")
    return parser.parse_args()


def submit_and_poll(csv_path: str, assignment_id: int):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Submission file not found: {csv_path}")

    client = OmnicampusClient(config_path=CONFIG_PATH)

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    file_size_kb = len(csv_bytes) / 1024
    logger.info("Uploading %s (%.1f KB) to Assignment %d...", csv_path, file_size_kb, assignment_id)

    initial_sub = client.get_user_submission(assignment_id=assignment_id, course_id=COURSE_ID)
    prev_count = initial_sub.get("submit_count", 0)

    res = client.submit_assignment(assignment_id=assignment_id, code_or_bytes=csv_bytes, course_id=COURSE_ID)
    logger.info("Upload response: %s", res)

    logger.info("Waiting for autograder scoring...")
    for attempt in range(40):
        time.sleep(3)
        sub_info = client.get_user_submission(assignment_id=assignment_id, course_id=COURSE_ID)
        current_count = sub_info.get("submit_count", 0)
        status = sub_info.get("status")

        if current_count != prev_count and status in ("success", "failed"):
            score = sub_info.get("score")
            output = sub_info.get("output", "")
            logger.info("Evaluation finished with status: %s | Score: %s", status, score)
            print("-" * 50)
            print(output.strip())
            print("-" * 50)
            break
    else:
        logger.warning("Scoring polling timed out. Please check portal manually.")

    # Leaderboard display
    lb_response = client.client.get(f"/v3/courses/{COURSE_ID}/assignments/{assignment_id}/leader-board")
    if lb_response.status_code == 200:
        leaderboard = lb_response.json()
        print("\n=== TOP 10 LEADERBOARD STANDINGS ===")
        for idx, entry in enumerate(leaderboard[:10]):
            handle = entry.get("user", {}).get("githubId", "Anonymous")
            score = entry.get("score", 0.0)
            marker = " <=== (YOU)" if handle == "diwanparker" else ""
            print(f"#{idx+1:02d} | Score: {score:.5f} | User: {handle}{marker}")

        for idx, entry in enumerate(leaderboard):
            if entry.get("user", {}).get("githubId") == "diwanparker":
                print(f"\nYour Standing: Rank #{idx+1} of {len(leaderboard)} (Score: {entry.get('score'):.5f})")
                break


if __name__ == "__main__":
    args = parse_args()
    submit_and_poll(args.csv, args.assignment_id)
