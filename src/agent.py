# agent.py — агент для BabyAI-PutNextS7N4-v0
# Запуск: src/python agent.py 7 9 5 0 1 2 [--debug]
import re
import sys
from collections import deque

import gymnasium as gym
import minigrid  # noqa: F401  (регистрация сред)

ENV_ID = "BabyAI-PutNextS7N4-v0"

A_LEFT, A_RIGHT, A_FWD, A_PICK, A_DROP, A_TOGGLE, A_DONE = 0, 1, 2, 3, 4, 5, 6

# Направления Minigrid: ПО ЧАСОВОЙ, 0=E, 1=S, 2=W, 3=N
DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))
VEC_TO_DIR = {v: k for k, v in enumerate(DIR_TO_VEC)}
ACTION_NAMES = ["LEFT", "RIGHT", "FWD", "PICK", "DROP", "TOGGLE", "DONE"]


# ---------- разбор миссии и поиск объектов ----------
def parse_mission(m):
    mt = re.match(r"put the (\w+) (\w+) next to the (\w+) (\w+)", m)
    return mt.groups() if mt else None


def in_bounds(u, x, y):
    return 0 <= x < u.width and 0 <= y < u.height


def find_object(u, color, obj_type):
    """Ищет объект НА СЕТКЕ. Несомый объект (u.carrying) на сетке отсутствует!"""
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if c and c.color == color and c.type == obj_type:
                return (x, y), c
    return None, None


def empty_around(u, pos, exclude=()):
    out = []
    for dx, dy in DIR_TO_VEC:
        c = (pos[0] + dx, pos[1] + dy)
        if c in exclude or not in_bounds(u, *c):
            continue
        if u.grid.get(*c) is None:
            out.append(c)
    return out


# ---------- навигация ----------
def bfs_path(u, start, goal):
    if start == goal:
        return [start]
    q = deque([start])
    seen = {start}
    par = {start: None}
    while q:
        cur = q.popleft()
        for dx, dy in DIR_TO_VEC:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in seen or not in_bounds(u, *nxt):
                continue
            cell = u.grid.get(*nxt)
            if cell is None or (cell.type == "door" and cell.is_open):
                seen.add(nxt)
                par[nxt] = cur
                if nxt == goal:
                    p = []
                    while nxt is not None:
                        p.append(nxt)
                        nxt = par[nxt]
                    return p[::-1]
                q.append(nxt)
    return None


def one_turn(cur_dir, want):
    if cur_dir == want:
        return None
    return A_RIGHT if (cur_dir + 1) % 4 == want else A_LEFT


def dir_between(a, b):
    return VEC_TO_DIR[(b[0] - a[0], b[1] - a[1])]


def move_toward(u, stand, face):
    cur = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    if cur == stand:
        t = one_turn(u.agent_dir, dir_between(cur, face))
        return t if t is not None else A_LEFT
    path = bfs_path(u, cur, stand)
    if path is None:
        return A_LEFT
    nxt = path[1]
    t = one_turn(u.agent_dir, dir_between(cur, nxt))
    return t if t is not None else A_FWD


# ---------- политика ----------
def heuristic_policy(env, debug=False):
    u = env.unwrapped
    parsed = parse_mission(env._last_obs.get("mission", ""))
    if not parsed:
        return A_DONE
    oc, ot, tc, tt = parsed
    front = (int(u.front_pos[0]), int(u.front_pos[1]))
    cur = (int(u.agent_pos[0]), int(u.agent_pos[1]))

    # Цель нужна в обеих фазах
    target_pos, _ = find_object(u, tc, tt)

    # ================= ФАЗА 2: несём объект =================
    if u.carrying is not None:
        # страховка: несём не тот объект (в нашем срезе не встречается)
        if (u.carrying.color, u.carrying.type) != (oc, ot):
            if debug:
                print("[DEBUG] несём посторонний объект — сбрасываем")
            return A_DROP if u.grid.get(*front) is None else A_LEFT
        if target_pos is None:
            if not getattr(env, "_unsolv", False):
                env._unsolv = True
                if debug:
                    print("[DEBUG] нерешаемый уровень: цель не найдена")
            return A_DONE
        cands = empty_around(u, target_pos)          # пустые соседи цели
        if not cands:
            if debug:
                print(f"[DEBUG] фаза 2: нет пустых клеток рядом с {target_pos}")
            return A_LEFT
        if front in cands:                            # перед агентом пустая клетка у цели
            if debug:
                print(f"[DEBUG] DROP: into {front} (смежна с {target_pos})")
            return A_DROP
        for dc in cands:
            for s in empty_around(u, dc, exclude=(target_pos,)):
                if s == cur:
                    t = one_turn(u.agent_dir, dir_between(cur, dc))
                    if t is not None:
                        return t
                    continue
                path = bfs_path(u, cur, s)
                if path is None:
                    continue
                return move_toward(u, s, dc)
        if debug:
            print(f"[DEBUG] фаза 2: не нашли stand; pos={cur}, cands={cands}")
        return A_LEFT

    # ================= ФАЗА 1: не несём объект =================
    obj_pos, _ = find_object(u, oc, ot)   # ищем ТОЛЬКО здесь: на сетке!
    if obj_pos is None or target_pos is None:
        if not getattr(env, "_unsolv", False):
            env._unsolv = True
            if debug:
                print(f"[DEBUG] нерешаемый уровень: obj={obj_pos}, target={target_pos}")
        return A_DONE
    if front == obj_pos:
        if debug:
            print(f"[DEBUG] PICK: front={front}")
        return A_PICK
    for s in empty_around(u, obj_pos):
        if s == cur:
            t = one_turn(u.agent_dir, dir_between(cur, obj_pos))
            if t is not None:
                return t
            continue
        if bfs_path(u, cur, s):
            return move_toward(u, s, obj_pos)
    if debug:
        print(f"[DEBUG] фаза 1: нет достижимой клетки рядом с obj={obj_pos}")
    return A_LEFT


# ---------- запуск эпизода ----------
def run_heuristic_episode(seed=None, debug=False, render_mode=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    obs, _ = env.reset(seed=seed)
    env._last_obs = obs
    env._unsolv = False

    total, steps, done = 0.0, 0, False
    hist = []
    while not done:
        a = heuristic_policy(env, debug=debug)
        u = env.unwrapped
        hist.append((int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir), a))
        if len(hist) > 12:
            hist.pop(0)
        if len(hist) == 12 and len(set(hist)) == 1:
            print(f"[DEBUG] ЗАСТРЯЛ: pos={tuple(u.agent_pos)} dir={u.agent_dir} "
                  f"front={tuple(u.front_pos)} front_cell={u.grid.get(*u.front_pos)} "
                  f"carrying={u.carrying} action={ACTION_NAMES[a]}")
            break
        obs, r, term, trunc, _ = env.step(a)
        env._last_obs = obs
        total += r
        steps += 1
        done = term or trunc
        if steps > 500:
            break
    env.close()
    return {"seed": seed, "mission": obs.get("mission", ""),
            "steps": steps, "reward": round(total, 3), "success": total > 0}


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    seeds = [int(x) for x in sys.argv[1:] if x.lstrip("-").isdigit()] or [None]
    for s in seeds:
        print(run_heuristic_episode(s, debug=debug))