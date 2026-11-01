"""Episode plans (jsonl) for LIBERO-Goal. All 10 tasks share one scene and one init distribution,
so any init state can be paired with any instruction; the instruction is the only task cue."""
import json, os, sys

TASKS = [  # libero_goal order (benchmark.get_task(k).language)
    "open the middle drawer of the cabinet", "put the bowl on the stove",
    "put the wine bottle on top of the cabinet", "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet", "push the plate to the front of the stove",
    "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate",
    "put the wine bottle on the rack"]

PARA = [
    ["pull open the cabinet's middle drawer", "open the drawer in the middle of the cabinet", "slide the middle cabinet drawer open"],
    ["place the bowl onto the stove", "move the black bowl to the stove", "set the bowl on top of the stove"],
    ["place the wine bottle on the cabinet top", "move the bottle of wine onto the top of the cabinet", "set the wine bottle on the cabinet"],
    ["open the upper drawer and place the bowl in it", "pull out the top drawer, then put the bowl in the drawer", "put the bowl into the top drawer after opening it"],
    ["place the bowl on the cabinet top", "move the black bowl onto the top of the cabinet", "set the bowl on the cabinet"],
    ["slide the plate in front of the stove", "move the plate to the front of the stove", "push the plate toward the front of the stove"],
    ["place the cream cheese into the bowl", "move the cream cheese box into the black bowl", "drop the cream cheese in the bowl"],
    ["switch on the stove", "turn the stove on", "activate the stove burner"],
    ["place the bowl onto the plate", "move the black bowl to the plate", "set the bowl on top of the plate"],
    ["place the wine bottle in the rack", "move the bottle of wine onto the wine rack", "set the wine bottle on the rack"]]

UNDER = ["put the bowl somewhere", "pick up the bowl and place it", "put the wine bottle somewhere",
         "open a drawer", "use the stove", "tidy up the table", "", "do the task"]

SWITCH_PAIRS = [(1, 8), (4, 8), (2, 9), (0, 3), (7, 1), (6, 8)]


def ep(i, task, init, instr, cond, **kw):
    return dict(id=i, task=task, init=init, instruction=instr, cond=cond, **kw)


def build(name):
    out = []
    if name == "pilot":            # 20 eps, ids 0-19
        out = [ep(k * 2 + r, k, r, TASKS[k], "c0") for k in range(10) for r in range(2)]
    elif name == "c0":             # 500 eps, ids 1000-1499 (inits 0..49)
        out = [ep(1000 + k * 50 + r, k, r, TASKS[k], "c0") for k in range(10) for r in range(50)]
    elif name == "under_pilot":    # 8 prompts x 10 eps, ids 2000+
        i = 2000
        for u, s in enumerate(UNDER):
            for r in range(10):
                out.append(ep(i, r, 40 + u, s, f"u{u}")); i += 1
    elif name == "para":           # 10 tasks x 3 paraphrases x 10 inits, ids 3000+
        i = 3000
        for k in range(10):
            for p in range(3):
                for r in range(10):
                    out.append(ep(i, k, 10 + r, PARA[k][p], f"p{p}")); i += 1
    elif name == "switch":         # pairs x 2 dirs x 4 switch times x 4 inits, ids 4000+
        i = 4000
        for a, b in SWITCH_PAIRS:
            for src, dst in ((a, b), (b, a)):
                for kt in (15, 30, 50, 80):
                    for r in range(4):
                        out.append(ep(i, src, 20 + r, TASKS[src], f"sw{kt}", instruction2=TASKS[dst],
                                      switch_t=kt, task2=dst)); i += 1
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    plan = build(name)
    extra = sys.argv[2:]  # optional: under prompts chosen after the pilot, e.g. "under 0,1,2,6"
    os.makedirs("plans", exist_ok=True)
    with open(f"plans/{name}.jsonl", "w") as f:
        for p in plan:
            f.write(json.dumps(p) + "\n")
    print(name, len(plan))
