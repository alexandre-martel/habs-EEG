"""Regenerate results/leaderboard.csv and results/figures/ from results/runs/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.eval.report import write_report

if __name__ == "__main__":
    board = write_report()
    if len(board):
        cols = [
            "model",
            "scenario",
            "task",
            "top1",
            "top1_lo",
            "top1_hi",
            "chance_top1",
        ]
        print(board[cols].round(3).to_string(index=False))
