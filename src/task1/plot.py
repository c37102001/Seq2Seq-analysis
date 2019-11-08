import matplotlib.pyplot as plt
import json

with open('../../models/task2/2-2-1_b32e512h128_lr0.5p5/history.json', 'r') as f:
    history = json.loads(f.read())

train_loss = [l['loss'] for l in history['train']]
valid_loss = [l['loss'] for l in history['valid']]
train_acc = [l['ctrl_acc'] for l in history['train']]
valid_acc = [l['ctrl_acc'] for l in history['valid']]

plt.figure(figsize=(7, 5))
plt.title('Task 2-1-1 loss')
plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='valid')
plt.legend()

plt.figure(figsize=(7, 5))
plt.title('Task 2-1-1 accuracy')
plt.plot(train_acc, label='train')
plt.plot(valid_acc, label='valid')
plt.legend()

plt.xlabel('Epochs')
plt.ylabel('Control signal accuracy')

plt.savefig('../../result/task2-1.png')

print('Figure saved.')
