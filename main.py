import random
from collections import defaultdict

ACTIONS = ["keep", "switch"]  # [da ostane ista boja na semafor, da ja promeni bojata]

class Grid:
    def __init__(self, lights, roads):
        self.lights = lights
        self.roads = roads

        self.Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

        self.alpha = 0.1   # learning rate
        self.gamma = 0.9   # discount
        self.epsilon = 0.2 # exploration

    def get_state(self, light):
        data = self.lights[light]
        return (data["cars"], data["color"])

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return max(self.Q[state], key=self.Q[state].get)

    def update_Q(self, state, action, reward, next_state):
        best_next = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )

    def change_light(self, light, action):
        if action == "switch":
            current = self.lights[light]["color"]
            self.lights[light]["color"] = "green" if current == "red" else "red"

    def simulate_cars(self, light):
        self.lights[light]["cars"] += random.randint(0, 3)

        if self.lights[light]["color"] == "green":
            passed = random.randint(1, 4)
            self.lights[light]["cars"] = max(
                0, self.lights[light]["cars"] - passed
            )

        return -self.lights[light]["cars"]

    def step(self, light):
        state = self.get_state(light)
        action = self.choose_action(state)

        self.change_light(light, action)
        reward = self.simulate_cars(light)

        next_state = self.get_state(light)
        self.update_Q(state, action, reward, next_state)

    def train(self, episodes=1000):
        for _ in range(episodes):
            for light in self.lights:
                self.step(light)



if __name__ == "__main__":
    lights = {
        (1, 0): {"color": "red", "cars": 0},
        (2, 3): {"color": "green", "cars": 0},
        (3, 3): {"color": "red", "cars": 0},
    }

    roads = {}
    game = Grid(lights, roads)
    game.train(episodes=5000)

    print("Learned Q-table:")
    for state, actions in list(game.Q.items())[:10]:
        print(state, actions)
