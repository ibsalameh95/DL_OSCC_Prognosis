import argparse
import matplotlib.pyplot as plt
import numpy as np
import math 

parser = argparse.ArgumentParser(description='Plot the loss vs iteration and accuracy vs iteration for given data file')
parser.add_argument('--data_file', default='Results/prognosis_model/metrics/epoch_loss_acc__2024_05_04__14_53_15.txt', help='Data file path', dest='data_file')
parser.add_argument('--step_size', default=1, type=int, help='Data file path', dest='step_size')
parser.add_argument('--filter_size', default=50, type=int, help='Data file path', dest='filter_size')
parser.add_argument('--num_epochs', default=500, type=int, help='Data file path', dest='num_epochs')

FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')


# steps = data_arr[:,0]
steps = np.arange(data_arr.shape[0])
train_acc = data_arr[:,2] * 100
train_loss = data_arr[:,1]
val_acc = data_arr[:,4] * 100
val_loss = data_arr[:,3]


def moving_avg_filter(data_arr, w):
	data_arr_cumsum = np.cumsum(data_arr)
	data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
	data_arr_filtered = data_arr_cumsum[w-1:]/w

	return data_arr_filtered

if w>1:
	steps = steps[w-1:]
	train_acc = moving_avg_filter(train_acc,w)
	train_loss = moving_avg_filter(train_loss,w)
	val_acc = moving_avg_filter(val_acc,w)
	val_loss = moving_avg_filter(val_loss,w)


ind_start = 0
ind_step = FLAGS.step_size
# ind_end = min(110,len(steps))
ind_end = len(steps)

# fig = plt.figure(1)
# plt.plot(steps[ind_start:ind_end:ind_step], train_loss[ind_start:ind_end:ind_step], 'r', label="train")
# plt.plot(steps[ind_start:ind_end:ind_step], val_loss[ind_start:ind_end:ind_step], 'b', label="val")
# plt.title('loss vs epoch')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid(linestyle='--')
# plt.legend()
# # fig.savefig('{}__loss.png'.format(FLAGS.data_file[:-4]), bbox_inches='tight')
# # plt.show()

# fig = plt.figure(2)
# plt.plot(steps[ind_start:ind_end:ind_step], train_acc[ind_start:ind_end:ind_step], 'r', label="train")
# plt.plot(steps[ind_start:ind_end:ind_step], val_acc[ind_start:ind_end:ind_step], 'b', label="val")
# plt.title('acc vs epoch')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.grid(linestyle='--')
# plt.legend()
# # fig.savefig('{}__acc.png'.format(FLAGS.data_file[:-4]), bbox_inches='tight')
# plt.show()

plt.figure('train', (12,6))
plt.subplot(1,2,1)
plt.title('Epoch vs Loss')


epoch = [i + 1 for i in range(FLAGS.num_epochs)]

new_list = list(range(0, FLAGS.num_epochs +1, math.ceil(FLAGS.num_epochs/5)))


plt.xlabel('epoch')
plt.xticks(new_list)

plt.plot(steps[ind_start:ind_end:ind_step], train_loss[ind_start:ind_end:ind_step], label = 'Train Loss')
plt.plot(steps[ind_start:ind_end:ind_step], val_loss[ind_start:ind_end:ind_step], color='r', label = 'Validation Loss')
plt.grid()
plt.legend(loc="best")

plt.subplot(1,2,2)
plt.title('Epoch vs Accuracy')

plt.xlabel('epoch')
plt.xticks(new_list)

plt.plot(steps[ind_start:ind_end:ind_step], train_acc[ind_start:ind_end:ind_step], label='Train Accuracy')
plt.plot(steps[ind_start:ind_end:ind_step], val_acc[ind_start:ind_end:ind_step], color='r', label='Validation Accuracy')
plt.grid()
plt.legend(loc="best")
#plt.show

plt.tight_layout()
plt.savefig('{}__acc_cutted.png'.format(FLAGS.data_file[:-4]), dpi=300)
# fig.savefig('{}__acc.png'.format(FLAGS.data_file[:-4]), bbox_inches='tight')
# fig, ax = plt.subplots(1,2,sharex=True)
# ax[0].plot(steps[ind_start:ind_end:ind_step], train_loss[ind_start:ind_end:ind_step], label="train")
# ax[0].plot(steps[ind_start:ind_end:ind_step], val_loss[ind_start:ind_end:ind_step], label="val")
# ax[0].set_title('Loss vs epoch')
# ax[0].set_xlabel('epoch')
# ax[0].set_ylabel('Loss')
# ax[0].grid(linestyle='--')
# ax[0].legend()

# ax[1].plot(steps[ind_start:ind_end:ind_step], train_acc[ind_start:ind_end:ind_step], label="train")
# ax[1].plot(steps[ind_start:ind_end:ind_step], val_acc[ind_start:ind_end:ind_step], label="val")
# ax[1].set_title('Accuracy vs epoch')
# ax[1].set_xlabel('epoch')
# ax[1].set_ylabel('acc')
# ax[1].grid(linestyle='--')
# ax[1].legend()


print('{}.png'.format(FLAGS.data_file[:-4]))
# fig.savefig('{}__acc.png'.format(FLAGS.data_file[:-4]), bbox_inches='tight')
# plt.show()
