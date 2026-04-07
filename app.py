import gradio as gr
import numpy as np
from env import MazeEnv
from q_agent import QLearningAgent

# Initialize
env = MazeEnv()
agent = QLearningAgent()

GRID_SIZE = 5
CELL_SIZE = 60

# 🎨 Render grid (safe)
def render_grid(state):
    grid = np.zeros((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, 3), dtype=np.uint8)

    try:
        x, y = state
    except:
        x, y = 0, 0  # fallback safety

    gx, gy = env.goal

    # Agent → Green
    grid[x*CELL_SIZE:(x+1)*CELL_SIZE, y*CELL_SIZE:(y+1)*CELL_SIZE] = [0, 255, 0]

    # Goal → Red
    grid[gx*CELL_SIZE:(gx+1)*CELL_SIZE, gy*CELL_SIZE:(gy+1)*CELL_SIZE] = [255, 0, 0]

    # Grid lines
    for i in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
        grid[i:i+2, :] = [255, 255, 255]
        grid[:, i:i+2] = [255, 255, 255]

    return grid

# ▶ Random agent
def run_episode():
    state = tuple(env.reset())
    total_reward = 0
    steps = 0

    for _ in range(25):  # reduced for stability
        action = np.random.choice([0, 1, 2, 3])
        next_state, reward, done = env.step(action)

        state = tuple(next_state)
        total_reward += reward
        steps += 1

        if done:
            break

    return render_grid(state), f"Random → Steps: {steps} | Reward: {total_reward}"

# ➡ Step once
def step_once():
    state = tuple(env.agent_pos)
    next_state, reward, done = env.step(np.random.choice([0,1,2,3]))

    return render_grid(tuple(next_state)), f"Reward: {reward} | Done: {done}"

# 🔄 Reset
def reset_env():
    state = tuple(env.reset())
    return render_grid(state), "Environment Reset"

# 🧠 Train agent (safe)
def train_agent():
    for _ in range(100):  # reduced from 200 → HF safe
        state = tuple(env.reset())

        for _ in range(30):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            next_state = tuple(next_state)
            agent.update(state, action, reward, next_state)

            state = next_state

            if done:
                break

    return render_grid(tuple(env.agent_pos)), "Training Completed ✅"

# 🚀 Run trained agent
def run_trained_agent():
    state = tuple(env.reset())
    total_reward = 0
    steps = 0

    for _ in range(25):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        state = tuple(next_state)
        total_reward += reward
        steps += 1

        if done:
            break

    return render_grid(state), f"Trained → Steps: {steps} | Reward: {total_reward}"

# 📊 Evaluate
def evaluate():
    success = 0

    for _ in range(30):  # reduced for speed
        state = tuple(env.reset())

        for _ in range(25):
            action = agent.choose_action(state)
            next_state, _, done = env.step(action)

            state = tuple(next_state)

            if done and state == tuple(env.goal):
                success += 1
                break

    return render_grid(tuple(env.agent_pos)), f"Success Rate: {success}/30"

# 🎯 UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 RL Maze (Q-Learning)")

    grid_output = gr.Image(label="Maze Grid")
    stats = gr.Textbox(label="Stats")

    with gr.Row():
        run_btn = gr.Button("▶ Random Run")
        step_btn = gr.Button("➡ Step Once")
        reset_btn = gr.Button("🔄 Reset")

    with gr.Row():
        train_btn = gr.Button("🧠 Train Agent")
        trained_btn = gr.Button("🚀 Run Trained Agent")
        eval_btn = gr.Button("📊 Evaluate")

    # Basic
    run_btn.click(run_episode, outputs=[grid_output, stats])
    step_btn.click(step_once, outputs=[grid_output, stats])
    reset_btn.click(reset_env, outputs=[grid_output, stats])

    # Advanced
    train_btn.click(train_agent, outputs=[grid_output, stats])
    trained_btn.click(run_trained_agent, outputs=[grid_output, stats])
    eval_btn.click(evaluate, outputs=[grid_output, stats])

# 🚀 Launch (HF safe)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)