import torch

state = torch.load('backbone.pth', map_location='cpu')

print(type(state))
print('Number of keys:', len(state))
print('First 10 keys:', list(state.keys())[:10])
print('Last 10 keys:', list(state.keys())[-10:])

if isinstance(state, dict):

    print(list(state.keys())[:10])

