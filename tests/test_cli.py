"""
Tests for the CLI entry point (chess_robotic.py).

Tests argument parsing and command dispatch without actually running the game.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

from chess_robotic import main


class TestCLIHelp:
    def test_no_command_returns_zero(self):
        with patch("sys.argv", ["chess-robotic"]):
            result = main()
            assert result == 0

    def test_help_flag(self):
        with patch("sys.argv", ["chess-robotic", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0


class TestCLIParsing:
    def test_play_parses(self):
        """Verify play subcommand parses without error."""
        import argparse
        from chess_robotic import main as _main

        # Just test that the parser works (don't actually run the game)
        with patch("sys.argv", ["chess-robotic", "play", "--both-human", "--max-moves", "2"]):
            with patch("chess_robotic.cmd_play") as mock_play:
                mock_play.return_value = 0
                result = _main()
                assert result == 0
                mock_play.assert_called_once()

    def test_eval_parses(self):
        with patch("sys.argv", ["chess-robotic", "eval", "--episodes", "10"]):
            with patch("chess_robotic.cmd_eval") as mock_eval:
                mock_eval.return_value = 0
                result = main()
                assert result == 0
                mock_eval.assert_called_once()

    def test_train_parses(self):
        with patch("sys.argv", ["chess-robotic", "train", "--steps", "1000"]):
            with patch("chess_robotic.cmd_train") as mock_train:
                mock_train.return_value = 0
                result = main()
                assert result == 0
                mock_train.assert_called_once()

    def test_calibrate_parses(self):
        with patch("sys.argv", ["chess-robotic", "calibrate", "--synthetic"]):
            with patch("chess_robotic.cmd_calibrate") as mock_cal:
                mock_cal.return_value = 0
                result = main()
                assert result == 0
                mock_cal.assert_called_once()
