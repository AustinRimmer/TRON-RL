# mcts_agent.py
# Monte Carlo Tree Search for the provided Tron environment.
# Assumes: env.step(our_action, opp_action) -> (obs, (r_our, r_opp), terminated, truncated, info)
# Controls the "blue" agent (the first action parameter in env.step).

import math
import random
import copy
from collections import defaultdict, deque

# ---------- Lightweight forward-simulation state ----------

class SimTronState:
    """
    Minimal, fast, deterministic simulator that mirrors Environment.Tron.step()
    for planning. We read values once from the live env and then simulate here.
    """
    __slots__ = (
        "size", "step_reward", "trail_length",
        "agent_pos", "target_pos", "agent_trail", "target_trail"
    )

    # Directions must match Environment._action_to_direction
    DIRS = {
        0: (1, 0),    # right
        1: (0, 1),    # up
        2: (-1, 0),   # left
        3: (0, -1),   # down
    }

    def __init__(self, size, step_reward, trail_length,
                 agent_pos, target_pos, agent_trail, target_trail):
        self.size = size
        self.step_reward = step_reward
        self.trail_length = trail_length
        self.agent_pos = tuple(int(x) for x in agent_pos)
        self.target_pos = tuple(int(x) for x in target_pos)
        # store as tuples for O(1) membership and deterministic behavior
        self.agent_trail = deque(agent_trail, maxlen=trail_length)
        self.target_trail = deque(target_trail, maxlen=trail_length)

    @classmethod
    def from_env(cls, env):
        # Pull internal state directly (fast).
        size = int(env.size)
        step_reward = float(env.step_reward)
        # trail_length is set in reset()
        trail_length = int(getattr(env, "trail_length", 1000))
        agent_pos = tuple(env._agent_location.tolist())
        target_pos = tuple(env._target_location.tolist())
        agent_trail = list(env.agent_trail)
        target_trail = list(env.target_trail)
        return cls(size, step_reward, trail_length, agent_pos, target_pos, agent_trail, target_trail)

    def clone(self):
        return SimTronState(
            self.size, self.step_reward, self.trail_length,
            self.agent_pos, self.target_pos,
            list(self.agent_trail), list(self.target_trail)
        )

    def _clip(self, x, lo, hi):
        return max(lo, min(hi, x))

    def step(self, our_action, opp_action):
        """
        Returns: next_state, (r_our, r_opp), terminated
        This mirrors Environment.step() reward/termination logic.
        """
        # NOTE: env.step() terminates if positions already equal BEFORE moving.
        # We do the same.
        terminated = (self.agent_pos == self.target_pos)
        if terminated:
            # outcome is ambiguous; treat as neutral step
            return self.clone(), (self.step_reward, self.step_reward), True

        ax, ay = self.agent_pos
        tx, ty = self.target_pos
        dx1, dy1 = self.DIRS[our_action]
        dx2, dy2 = self.DIRS[opp_action]

        # Proposed new positions with boundary clipping
        new_agent_pos = (self._clip(ax + dx1, 0, self.size - 1),
                         self._clip(ay + dy1, 0, self.size - 1))
        new_target_pos = (self._clip(tx + dx2, 0, self.size - 1),
                          self._clip(ty + dy2, 0, self.size - 1))

        # Collision check (uses new positions vs existing trails, like the env)
        loss = False
        if (new_agent_pos in self.target_trail or
                new_target_pos in self.agent_trail or
                new_agent_pos in self.agent_trail or
                new_target_pos in self.target_trail):
            loss = True
            terminated = True

        # Prepare next state
        ns = self.clone()

        r_our = ns.step_reward
        r_opp = ns.step_reward

        if not terminated:
            # Commit movement
            ns.agent_pos = new_agent_pos
            ns.target_pos = new_target_pos

            ns.agent_trail.append(ns.agent_pos)
            ns.target_trail.append(ns.target_pos)
            # deque with maxlen handles trimming

        # If terminated after movement/collision, assign terminal rewards
        if terminated:
            blue_hit = (new_agent_pos in self.target_trail) or (new_agent_pos in self.agent_trail)
            red_hit  = (new_target_pos in self.agent_trail) or (new_target_pos in self.target_trail)

            if blue_hit and red_hit:
                r_our = r_opp = 0.0
            elif blue_hit:
                r_our, r_opp = -20.0, 20.0
            elif red_hit:
                r_our, r_opp = 20.0, -20.0
            else:
                r_our = r_opp = ns.step_reward

        return ns, (r_our, r_opp), terminated


# ----------------------- MCTS Core (single-agent branching) --------------------

class MCTSNode:
    __slots__ = ("state", "parent", "action_from_parent", "children",
                 "N", "W", "untried_actions")

    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent  # our action that led here
        self.children = {}       # action -> node
        self.N = 0               # visit count
        self.W = 0.0             # total value (sum of returns for our agent)
        self.untried_actions = {0, 1, 2, 3}  # our actions

    def is_expanded(self):
        return len(self.untried_actions) == 0

    def best_child_ucb(self, c=1.4):
        """UCB1 over our actions (children)."""
        best, best_score = None, -1e18
        for a, child in self.children.items():
            if child.N == 0:
                score = float('inf')  # ensure each child is tried once
            else:
                exploit = child.W / child.N
                explore = math.sqrt(math.log(self.N + 1) / child.N)
                score = exploit + c * explore
            if score > best_score:
                best, best_score = child, score
        return best


class MCTSAgent:
    def __init__(self,
                 simulations_per_move=400,
                 c_ucb=1.4,
                 rollout_depth=50,
                 opponent_policy="random",
                 gamma=1.0):
        self.simulations_per_move = simulations_per_move
        self.c_ucb = c_ucb
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.opponent_policy = opponent_policy

        self.root = None  # MCTSNode

    # ---------- opponent & rollout policies ----------

    def _opp_action(self, state: SimTronState):
        # simple random policy, replace with heuristics if desired
        safe_action = self._safe_opp_actions_from_state(state)
        return random.choice(safe_action)

    def _rollout_policy(self, state: SimTronState):
        # our rollout aciton: only picks a safe aciton
        our_safe = self._safe_our_actions_from_state(state)
        a_our = random.choice(our_safe)

        # opponent rollout action: also only picks safe action
        opp_safe = self._safe_opp_actions_from_state(state)
        a_opp = random.choice(opp_safe)

        return a_our, a_opp

    # ---------- main public API ----------

    def reset_tree(self):
        self.root = None

    def set_root_from_env(self, env):
        s = SimTronState.from_env(env)
        self.root = MCTSNode(s)

    def act(self, env):
        """
        Choose an action for the current env state. 
        Builds/updates a search tree rooted at the current state
        """
        if self.root is None:
            self.set_root_from_env(env)
        else:
            #try to reuse subtree if possible (if previous chosen action child exists and matches env)
            #Since opponent actions are sampled, we can't perfectly match so safest is to rebuild
            self.set_root_from_env(env)

        for _ in range(self.simulations_per_move):
            self._simulate_once()

        # pick the most visited child (robust)
        if not self.root.children:
            return random.randint(0, 3)

        # for a, child in self.root.children.items():
        #     print(f"Action {a}: N={child.N}, W={child.W:.2f}, Avg={child.W / child.N:.2f}")

        best_a = max(self.root.children.items(), key=lambda kv: (kv[1].W / kv[1].N) if kv[1].N > 0 else float('-inf'))[0]
        
        # Check if best_a would immediately kill us
        state = SimTronState.from_env(env)
        dx, dy = SimTronState.DIRS[best_a]
        ax, ay = state.agent_pos
        nx = max(0, min(state.size - 1, ax + dx))
        ny = max(0, min(state.size - 1, ay + dy))
        new_pos = (nx, ny)

        if (new_pos == (ax, ay)) or (new_pos in state.agent_trail) or (new_pos in state.target_trail):
            # Suicide detected - fallback to safe random
            safe_actions = self._safe_our_actions_from_state(state)
            fallback = random.choice(safe_actions)
            #print(f"[MCTS] Avoided suicidal action {best_a}, picked fallback {fallback}")
            return fallback
        return best_a

    # ---------- one simulation ----------

    def _simulate_once(self):
        node = self.root

        # 1) Selection
        path = [node]
        while node.is_expanded() and node.children:
            node = node.best_child_ucb(self.c_ucb)
            path.append(node)

        # 2) Expansion (if non-terminal and we still have actions)
        terminal = False
        if node.untried_actions:
            a = node.untried_actions.pop()  # try a new action
            # sample opponent action
            opp_a = self._opp_action(node.state)
            next_state, (r_our, _), term = node.state.step(a, opp_a)
            child = MCTSNode(next_state, parent=node, action_from_parent=a)
            node.children[a] = child
            path.append(child)
            node = child
            terminal = term
            immediate_reward = r_our
        else:
            #theres noo untried actions since we're at a leaf that's fully expanded.
            # we'll start rollout from here.
            pass

        # 3) Simulation (rollout) from 'node'
        G = immediate_reward if 'immediate_reward' in locals() else 0.0
        discount = self.gamma
        steps = 0

        cur_state = node.state
        if terminal:
            G = 0.0  # the step reward was already applied in transition to node, we back up from parent using W updates below
        else:
            while steps < self.rollout_depth:
                a_our, a_opp = self._rollout_policy(cur_state)
                cur_state, (r_our, _), term = cur_state.step(a_our, a_opp)
                G += discount * r_our
                discount *= self.gamma
                steps += 1
                if term:
                    break

        # 4) Backup: along the path, propagate returns.
        # We also add the immediate reward (if any) from the expansion step for the child we created.
        # To keep it simple (and comparable across sims), we only back up the rollout return G here.
        for nd in path:
            nd.N += 1
            nd.W += G

    def _safe_our_actions_from_state(self, state: SimTronState):
        """
        Our agent (blue): picks actions that do not lose the game
        """
        safe = []
        ax, ay = state.agent_pos
        for a in (0, 1, 2, 3):
            dx, dy = SimTronState.DIRS[a]
            nx = max(0, min(state.size - 1, ax + dx))
            ny = max(0, min(state.size - 1, ay + dy))
            new_pos = (nx, ny)

            # 'wall' avoidance: if clipping keeps us in place, treat as unsafe
            if new_pos == (ax, ay):
                continue

            # avoid stepping onto any existing trail immediately
            if (new_pos in state.agent_trail) or (new_pos in state.target_trail):
                continue

            safe.append(a)

        # if boxed-in, we must return something to keep the sim moving.
        return safe if safe else [0, 1, 2, 3]


    def _safe_opp_actions_from_state(self, state: SimTronState):
        """
        Opponent (red): will pick actions that do not lose the game
        """
        safe = []
        tx, ty = state.target_pos
        for a in (0, 1, 2, 3):
            dx, dy = SimTronState.DIRS[a]
            nx = max(0, min(state.size - 1, tx + dx))
            ny = max(0, min(state.size - 1, ty + dy))
            new_pos = (nx, ny)

            if new_pos == (tx, ty):
                continue
            if (new_pos in state.agent_trail) or (new_pos in state.target_trail):
                continue

            safe.append(a)
        return safe if safe else [0, 1, 2, 3]

#just testing
def run_mcts_vs_random(env, agent: MCTSAgent, render=True, reset_on_terminal=True, max_steps=10_000):
    """
    Example loop: our MCTS agent vs random opponent.
    Returns total episodic reward for our agent.
    """
    obs, info = env.reset()
    agent.reset_tree()
    total = 0.0
    steps = 0
    while steps < max_steps:
        if render and getattr(env, "render_mode", None) == "human":
            env.render()

        # Our move via MCTS
        our_action = agent.act(env)
        # Opponent random
        opp_action = random.randint(0, 3)

        # NOTE: Environment expects (agent_action, target_action)
        obs, (r_our, r_opp), terminated, truncated, info = env.step(our_action, opp_action)
        total += r_our
        steps += 1

        if terminated or truncated:
            if reset_on_terminal:
                obs, info = env.reset()
                agent.reset_tree()
            else:
                break
    return total
