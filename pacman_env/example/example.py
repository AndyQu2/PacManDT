"""Play Pac-Man in a window (human rendering mode).

Controls
--------
Arrow keys / WASD : move
R                 : restart the episode
Esc               : quit

Run ``python example.py --help`` for the available options.
"""

import argparse

import gymnasium as gym

import pacman_env
from pacman_env.game.action import Action

KEY_TO_ACTION = {
    "K_LEFT": Action.LEFT,
    "K_a": Action.LEFT,
    "K_RIGHT": Action.RIGHT,
    "K_d": Action.RIGHT,
    "K_UP": Action.UP,
    "K_w": Action.UP,
    "K_DOWN": Action.DOWN,
    "K_s": Action.DOWN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Pac-Man with a keyboard.")
    parser.add_argument("--layout", default="classic",
                        help="layout name: classic, google, medium, medium_small, small")
    parser.add_argument("--lives", type=int, default=3, help="number of lives")
    parser.add_argument("--max-steps", type=int, default=10000, help="episode step limit")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--state", action="store_true",
                        help="also draw the observation matrix next to the maze")
    return parser.parse_args()


def keyboard_action(pg) -> Action:
    keys = pg.key.get_pressed()
    for key_name, action in KEY_TO_ACTION.items():
        if keys[getattr(pg, key_name)]:
            return action
    return Action.NOOP


def wait_for_restart(pg, env) -> str:
    """Show the end-of-episode screen until the player restarts or quits."""
    clock = pg.time.Clock()
    while True:
        env.render()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit"
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return "quit"
                if event.key in (pg.K_r, pg.K_RETURN):
                    return "restart"
        clock.tick(env.render_fps)


def main() -> None:
    args = parse_args()

    env = gym.make(
        pacman_env.ENV_ID,
        layout=args.layout,
        render_mode="human",
        player_lives=args.lives,
        max_steps=args.max_steps,
        draw_state=args.state,
        render_fps=args.fps,
    )
    env = env.unwrapped

    import pygame as pg

    _, info = env.reset(seed=args.seed)
    env.render()  # create the window before reading keyboard events
    clock = pg.time.Clock()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_r:
                    _, info = env.reset(seed=args.seed)

        _, reward, terminated, truncated, info = env.step(keyboard_action(pg))
        env.render()
        clock.tick(args.fps)

        if terminated or truncated:
            if info["win"]:
                outcome = "level cleared"
            elif terminated:
                outcome = "game over"
            else:
                outcome = "time limit reached"
            print(f"{outcome}: score={info['score']} pellets left={info['pellets_remaining']} "
                  f"(R/Enter to play again, Esc to quit)")
            if wait_for_restart(pg, env) == "restart":
                _, info = env.reset(seed=args.seed)
            else:
                running = False

    env.close()


if __name__ == "__main__":
    main()
