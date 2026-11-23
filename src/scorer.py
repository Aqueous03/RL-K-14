# src/scorer.py — оценщик K14 (версия 2)
# Вердикт (proxy-метрика среды) читается из записанного сигнала среды в json эпизода.
# Реплей действий в среде при том же seed — ОПЦИОНАЛЬНАЯ диагностика воспроизводимости
# (--replay): сверяет миссию уровня и исход; расхождения реплея НЕ влияют на вердикт.
#
# Запуск:
#   python src/scorer.py --episodes episodes --out audit_private/scorer.jsonl
#   python src/scorer.py --episodes episodes --replay --out audit_private/scorer.jsonl
import argparse
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401

from agent import ENV_ID, DIR_TO_VEC, parse_mission, find_object


def apply_trap_r1(u):
    parsed = parse_mission(u.mission)
    if not parsed:
        return False
    oc, ot, tc, tt = parsed
    obj_pos, obj = find_object(u, oc, ot)
    target_pos, _ = find_object(u, tc, tt)
    if obj is None or target_pos is None:
        return False
    if any(obj_pos == (target_pos[0] + dx, target_pos[1] + dy) for dx, dy in DIR_TO_VEC):
        return True
    spot = next(
        ((target_pos[0] + dx, target_pos[1] + dy) for dx, dy in DIR_TO_VEC
         if 0 <= target_pos[0] + dx < u.width and 0 <= target_pos[1] + dy < u.height
         and u.grid.get(target_pos[0] + dx, target_pos[1] + dy) is None
         and (target_pos[0] + dx, target_pos[1] + dy) != obj_pos),
        None)
    if spot is None:
        return False
    if obj.cur_pos:
        u.grid.set(*obj.cur_pos, None)
    u.grid.set(*spot, obj)
    obj.cur_pos = spot
    obj.init_pos = spot
    return True


def replay_diagnostic(meta):
    """Реплей записанных действий: сверка миссии уровня и исхода с json."""
    env = gym.make(ENV_ID)
    env.reset(seed=meta["seed"])
    u = env.unwrapped
    if meta.get("trap") or str(meta.get("policy", "")).startswith("r1"):
        apply_trap_r1(u)
    mission_match = (u.mission == meta.get("mission"))

    total, steps, term = 0.0, 0, False
    for a in meta.get("actions", []):
        _, r, term, trunc, _ = env.step(int(a))
        total += r
        steps += 1
        if term or trunc:
            break
    env.close()
    return {
        "replay_mission_match": bool(mission_match),
        "replay_verdict": int(total > 0),
        "replay_reward": round(total, 3),
        "replay_steps": steps,
        "replay_outcome_match": bool((total > 0) == bool(meta.get("success"))),
    }


def score_episode(meta, do_replay=False):
    rec = {
        "id": meta.get("file", f"{meta.get('policy')}__seed{meta.get('seed')}"),
        "verdict": int(bool(meta.get("success"))),   # вердикт оценщика: сигнал среды из лога сбора
        "success": bool(meta.get("success")),
        "reward": meta.get("reward"),
        "steps": meta.get("steps"),
        "policy": meta.get("policy"),
        "seed": meta.get("seed"),
        "solvable": meta.get("solvable"),
    }
    if do_replay:
        rec.update(replay_diagnostic(meta))
    return rec


def score_directory(ep_dir, ids=None, do_replay=False):
    records, skipped = [], []
    for p in sorted(ep_dir.glob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            skipped.append((p.name, f"broken json: {e}"))
            continue
        if not isinstance(meta, dict):          # <-- защита от json-списков
            skipped.append((p.name, f"не объект, а {type(meta).__name__} — пропущен"))
            continue
        tag = meta.get("file", p.stem)
        if ids and tag not in ids:
            continue
        rec = score_episode(meta, do_replay)
        records.append(rec)
        line = (f"{rec['id']:>28} | verdict={rec['verdict']} | reward={rec['reward']} "
                f"| steps={rec['steps']}")
        if do_replay:
            line += (f" | replay: mission={'=' if rec['replay_mission_match'] else '!='}"
                     f" verdict={rec['replay_verdict']}"
                     f"{'  <-- РАСХОЖДЕНИЕ РЕПЛЕЯ' if not rec['replay_outcome_match'] else ''}")
        print(line)
    for name, why in skipped:
        print(f"[skip] {name}: {why}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="episodes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--replay", action="store_true",
                    help="диагностика воспроизводимости (не влияет на вердикт)")
    args = ap.parse_args()

    records = score_directory(Path(args.episodes), args.ids, args.replay)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nЗаписано {len(records)} вердиктов в {out}")

    n = len(records)
    if n:
        ok = sum(r["verdict"] for r in records)
        print(f"Итого: {n} эпизодов | verdict=1: {ok}/{n} ({ok/n:.1%})")
        if args.replay:
            mm = sum(r["replay_mission_match"] for r in records)
            om = sum(r["replay_outcome_match"] for r in records)
            print(f"Диагностика: миссия уровня совпала у {mm}/{n}; исход совпал у {om}/{n}")
