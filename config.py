import os

classes = ['cat', 'dog']
class_num = 2

batch_size = 32
image_H, image_W, image_C = 227, 227, 3
train_set_mean = [124.6629, 116.2139, 106.5368]
data_set_path = './DataSet'
train_list_name = 'train_list.txt'
val_list_name = 'val_list.txt'
test_list_name = 'test_list.txt'

reg_scale = 5e-4
keep_prob = 0.5
initial_weights_path = './Alexnet_weights/Alexnet.ckpt'
weights_path = './My_weights/Alexnet.ckpt'
config_dir = './My_weights'

initial_learning_rate = 1e-5
decay_steps = 187
decay_rate = 0.8
staircase = True
num_epoch = 10
summary_path = './Summary'
summary_steps = 10

