"""ViZDoom BASIC random-action baseline.

Run:
    python random_baseline.py --episodes 50
    python random_baseline.py --episodes 5 --visible
    python random_baseline.py --config ../ViZDoom/scenarios/basic.cfg

Requires: pip install vizdoom
No ANN or SNN is used. This script measures the environment and records a baseline.
"""

import argparse
import csv
import random
from pathlib import Path

import vizdoom as vzd


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate random actions on ViZDoom BASIC")
    p.add_argument("--config", type=Path, default=None,
                   help="Optional custom path to basic.cfg (otherwise use installed ViZDoom scenarios)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--frame-skip", type=int, default=4,
                   help="Game tics per chosen action (keep identical for future evaluations)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visible", action="store_true", help="Show the game window")
    p.add_argument("--output", type=Path, default=Path("results/random_baseline.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1 or args.frame_skip < 1:
        raise ValueError("--episodes and --frame-skip must be positive")
    rng = random.Random(args.seed)

    config = args.config if args.config is not None else Path(vzd.scenarios_path) / "basic.cfg"
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Missing scenario config: {config}")

    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_window_visible(args.visible)
    game.set_seed(args.seed)

    # Match the configured button order, rather than assuming a fixed order.
    buttons = list(game.get_available_buttons())
    print("Configured buttons:", [b.name for b in buttons])
    expected = {vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK}
    if set(buttons) != expected:
        raise ValueError("This baseline expects the BASIC scenario buttons: "
                         "MOVE_LEFT, MOVE_RIGHT, ATTACK")

    # A discrete action space: the same choices can later be used by DQN/SNN.
    choices = {
        "noop": set(),
        "left": {vzd.Button.MOVE_LEFT},
        "right": {vzd.Button.MOVE_RIGHT},
        "shoot": {vzd.Button.ATTACK},
        "left_shoot": {vzd.Button.MOVE_LEFT, vzd.Button.ATTACK},
        "right_shoot": {vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK},
    }
    actions = [(label, [int(b in active) for b in buttons])
               for label, active in choices.items()]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        game.init()
        for ep in range(1, args.episodes + 1):
            game.new_episode()
            decisions = 0
            while not game.is_episode_finished():
                label, action = rng.choice(actions)
                game.make_action(action, args.frame_skip)
                decisions += 1

            # The BASIC scenario ends on monster kill or timeout; classify
            # terminal cause explicitly rather than thresholding reward.
            timed_out = bool(game.is_episode_timeout_reached())
            kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            record = {
                "episode": ep,
                "seed": args.seed,
                "frame_skip": args.frame_skip,
                "total_reward": round(float(game.get_total_reward()), 3),
                "episode_tics": int(game.get_episode_time()),
                "action_decisions": decisions,
                "timed_out": int(timed_out),
                "kills": kills,
                "success": int(kills > 0),
            }
            records.append(record)
            print(f"Episode {ep:03d}: reward={record['total_reward']:7.2f} "
                  f"kills={kills} timeout={timed_out} tics={record['episode_tics']}")
    finally:
        game.close()

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    mean_reward = sum(x["total_reward"] for x in records) / len(records)
    success_rate = 100.0 * sum(x["success"] for x in records) / len(records)
    print(f"\nMean reward: {mean_reward:.2f}")
    print(f"Kill success rate: {success_rate:.1f}%")
    print(f"CSV saved: {args.output.resolve()}")
    print("Note: verify KILLCOUNT reflects your installed BASIC WAD on a test kill.")


if __name__ == "__main__":
    main()
