---
title: RL Maze Visualization
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🧠 RL Maze Environment

# 🧠 RL Maze Environment (OpenEnv + Q-Learning + Visualization)

## 🔗 Live Demo

👉 *(Add your Hugging Face link here)*

---

## 📌 Overview

This project implements a **Reinforcement Learning (RL) environment** where an agent learns to navigate a grid-based maze to reach a goal.

It combines:

* ✅ **OpenEnv-compatible APIs** (`/reset`, `/step`)
* ✅ **Interactive Gradio UI**
* ✅ **Q-learning agent for intelligent behavior**

---

## 🎯 Objective

To design a complete RL system demonstrating:

* State transitions
* Action space
* Reward-driven learning
* Agent training (Q-learning)
* Performance evaluation

---

## 🚀 Key Features

* 🧩 Grid-based RL environment
* 🌐 OpenEnv REST APIs
* 🖥️ Interactive visualization (Gradio UI)
* 🧠 Q-learning agent (learning from experience)
* 📊 Evaluation (success rate, reward tracking)
* 🎯 Train → Run → Evaluate workflow

---

## 🧩 Environment Design

### 🔹 Grid Details

* Grid Size: **5 × 5**
* Start Position: **(0, 0)**
* Goal Position: **(4, 4)**

---

### 🔹 State Space

The state represents the **agent’s position**:

```
(x, y)
```

Example:

```
[0, 0] → Start  
[4, 4] → Goal
```

---

### 🔹 Action Space

| Action | Description |
| ------ | ----------- |
| 0      | Move Up     |
| 1      | Move Down   |
| 2      | Move Left   |
| 3      | Move Right  |

---

### 🔹 Reward Function (Core Design)

| Condition          | Reward |
| ------------------ | ------ |
| Reaching Goal      | +10    |
| Each Step          | -1     |
| Max Steps Exceeded | -5     |

> This reward structure encourages **shortest-path learning** and discourages inefficient exploration.

---

### 🔹 Episode Termination

An episode ends when:

* The agent reaches the goal ✅
* Maximum steps are exceeded ❌

---

## 🧠 Q-Learning Agent (Key Highlight)

The agent uses **Q-learning** to learn optimal actions.

### ⚙️ Parameters

* Learning rate (α): 0.1
* Discount factor (γ): 0.9
* Exploration rate (ε): 0.2

### 💡 Working

* Maintains a **Q-table (state-action values)**
* Updates values using reward feedback
* Gradually learns the **optimal path to goal**

> This transforms the agent from random behavior → intelligent navigation.

---

## 🌐 API Endpoints (OpenEnv Compatible)

### 🔸 POST `/reset`

Initializes environment

```json
{
  "state": [0, 0]
}
```

---

### 🔸 POST `/step`

```json
{
  "action": 1
}
```

Response:

```json
{
  "state": [1, 0],
  "reward": -1,
  "done": false
}
```

---

## ⚙️ Execution Flow

1. `/reset` initializes environment
2. Agent selects action
3. `/step` updates state + reward
4. Loop continues until termination

---

## 🖥️ Interactive UI (Gradio)

The project includes a visual interface where users can:

* ▶ Run random agent
* 🧠 Train Q-learning agent
* 🚀 Run trained agent
* 📊 Evaluate performance

---

## 🧪 Example Interaction

```
Reset → [0,0]
Step → [0,1], reward = -1
...
Goal → [4,4], reward = +10
```

---

## 📊 Evaluation Metrics

* Total reward per episode
* Number of steps
* Success rate (goal reached)

---

## 🛠️ Tech Stack

* Python
* FastAPI
* Uvicorn
* NumPy
* Gradio

---

## ▶️ How to Run Locally

### Install dependencies

```
pip install -r requirements.txt
```

### Run API

```
uvicorn inference:app --host 0.0.0.0 --port 8000
```

### Run UI

```
python app.py
```

---

## 🐳 Docker Support

```
docker build -t rl-maze .
docker run -p 8000:8000 rl-maze
```

---

## 📸 Demo Screenshot

*(Add screenshot here for better impact)*

---

## 🚀 Future Improvements

* Dynamic obstacles
* Deep RL (DQN)
* Path optimization comparison
* Reward visualization graphs
* Multi-agent environments

---

## 🧠 Key Learning Outcome

This project demonstrates:

* RL environment design
* Reward engineering
* Agent learning via Q-learning
* API-based agent interaction
* Visualization of learning behavior

---

## 🏁 Conclusion

This project provides a **complete RL pipeline**:

> Environment → API → Learning Agent → Visualization → Evaluation

It is **OpenEnv-compatible**, interactive, and extensible — making it suitable for both experimentation and real-world RL system design.

---
