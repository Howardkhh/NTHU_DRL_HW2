from collections import deque
import numpy as np
import torch
from PIL import Image

class Network(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.valueNet = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.advantageNet = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor):
        # assumes input is uint8
        x = x.to(torch.float32) / 255.
        conv_out = self.conv(x).view(x.size()[0], -1)
        value = self.valueNet(conv_out)
        advantage = self.advantageNet(conv_out)
        return value + advantage - advantage.mean()


class Agent():
    def __init__(self):
        self.behavior_network = Network((12, 84, 84), 12)
        self.behavior_network.load_state_dict(torch.load("109062102_hw2_data.py", map_location="cpu"))
        self.behavior_network.eval()
        self.counter = 0
        self.last_action = None
        self.frames = deque([], maxlen=4)
        self.start_frame = None

    def preprocess(self, state):
        processed_state = np.array(Image.fromarray(state.astype(np.uint8)).resize((84, 84), resample=Image.BILINEAR), dtype=np.uint8)
        while len(self.frames) < 4:
            self.frames.append(processed_state)
            self.start_frame = state.copy()
        self.frames.append(processed_state)
        processed_state = np.concatenate(list(self.frames), axis=-1)
        processed_state = torch.tensor(np.array(processed_state).transpose((2, 0, 1)), device="cpu").unsqueeze(0)
        return processed_state

    def act(self, observation):
        if np.all(observation == self.start_frame) and self.counter > 10:
            self.counter = 0
            self.frames.clear()
        
        if self.counter % 4 == 0:
            self.last_action = self.behavior_network(self.preprocess(observation)).max(1)[1].view(1, 1)
        self.counter += 1
        
        return self.last_action.item()