import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, precision_recall_curve
import math
import os

# plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})
plt.rcParams.update({'font.size':10})

def score_fnc(data_arr1, data_arr2):
	auc = roc_auc_score(data_arr1, data_arr2)
	return auc

def BootStrap(data_arr1, data_arr2, n_bootstraps):

	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []
	# print(data_arr2)
	# print(data_arr2)

	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(data_arr2), len(data_arr2))

		if len(np.unique(data_arr1[indices])) < 2:
			# We need at least one sample from each class
			# otherwise reject the sample
			#print("We need at least one sample from each class")
			continue
		else:
			score = score_fnc(data_arr1[indices], data_arr2[indices])
			bootstrapped_scores.append(score)
			#print("score: %f" % score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	# Computing the lower and upper bound of the 95% confidence interval
	# You can change the bounds percentiles to 0.025 and 0.975 to get
	# a 95% confidence interval instead.

	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	# print(confidence_lower)
	# print(confidence_upper)
	# print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
	return sorted_scores, confidence_lower, confidence_upper


parser = argparse.ArgumentParser(description='')

parser.add_argument('--metrics_file', default='Results/prognosis_model/test_metrics/2024_08_19__17_31_46__369/test/slide_scores.txt', help='Text file to write metrics', dest='metrics_file')
parser.add_argument('--title', default='slide_scores', help='Text file to write metrics', dest='title')

FLAGS = parser.parse_args()

def plot_auc(metrics_file):
	data_arr = np.loadtxt(metrics_file, delimiter='\t',comments='#',dtype=str)
	label_arr = np.asarray(data_arr[:,1],dtype=int)
	positive_score_arr = np.asarray(data_arr[:,2],dtype=float)

	predicted_labels = [0 if x <= 0.5 else 1 for x in positive_score_arr]

	fpr, tpr, _ = roc_curve(label_arr, positive_score_arr, pos_label=1)
	roc_auc = auc(fpr, tpr)
	# print(roc_auc)

	report = classification_report(label_arr, predicted_labels, target_names=['Good Prognosis', 'Poor Prognosis'])

	out_file = metrics_file[:-4]

	with open(out_file + '_classification_report.txt', 'w') as f:
		f.write(report)


	sorted_scores, scores_lower, scores_upper = BootStrap(label_arr, positive_score_arr, n_bootstraps=2000)


	# title_text = 'AUROC = {:.3f} (CI: {:.3f} - {:.3f})'.format(roc_auc, scores_lower, scores_upper)
	title_text = 'AUROC = {:.3f} ({:.3f} - {:.3f})'.format(roc_auc, scores_lower, scores_upper)

	print(title_text)

	plt.figure(figsize=(8, 6))
	plt.plot(fpr, tpr, lw=2, label=title_text)
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xticks(np.arange(0,1.05,0.2))
	plt.yticks(np.arange(0,1.05,0.2))
	plt.xlabel('1 - Specificity')
	plt.ylabel('Sensitivity')
	# plt.title('ROC Curve')
	plt.grid()
	plt.legend(loc='lower right')
	plt.tight_layout()

	png_filename = '{}__roc.png'.format(out_file)
	png_filename = png_filename.replace('slide_scores_', FLAGS.title)
	pdf_filename = '{}__roc.pdf'.format(out_file)
	pdf_filename = pdf_filename.replace('slide_scores_', FLAGS.title)

	plt.savefig(png_filename, dpi= 300)
	plt.savefig(pdf_filename, dpi= 300)


plot_auc(FLAGS.metrics_file)

# for root, dirs, files in os.walk('Results/prognosis_model/test_metrics/'):
# 	for file in files:
# 		if file == 'slide_scores.txt':
# 			plot_auc(os.path.join(root, file))

def plot_auprc(metrics_file):
	
	data_arr = np.loadtxt(metrics_file, delimiter='\t',comments='#',dtype=str)
	label_arr = np.asarray(data_arr[:,1],dtype=int)
	positive_score_arr = np.asarray(data_arr[:,2],dtype=float)

	precision, recall, _ = precision_recall_curve(label_arr, positive_score_arr)

	# Add points at the beginning and end to ensure the curve spans the entire range
	# precision = np.concatenate(([precision[0]], precision, [0]))
	# recall = np.concatenate(([0], recall, [1]))

	# Insert 0 at the beginning of precision and 1 at the end recall

	precision = np.concatenate(([0], precision))
	recall = np.concatenate(([1], recall))

	auprc = auc(recall, precision)

	baseline = len(label_arr[label_arr==1]) / len(label_arr)

	sorted_scores, scores_lower, scores_upper = BootStrap(label_arr, positive_score_arr, n_bootstraps=2000)

	# title_text = 'AUROC = {:.3f} (CI: {:.3f} - {:.3f})'.format(roc_auc, scores_lower, scores_upper)

	title_text = 'AUPRC = {:.3f} ({:.3f} - {:.3f})'.format(auprc, scores_lower, scores_upper)

	print(title_text)

	plt.figure(figsize=(8, 6))
	plt.plot(recall, precision, lw=2, label=title_text)
	plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline', color='r')
	# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xticks(np.arange(0,1.05,0.2))
	plt.yticks(np.arange(0,1.05,0.2))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.grid()
	plt.legend(loc='lower right')
	plt.tight_layout()

	out_file = metrics_file[:-4]

	png_filename = '{}__auprc.png'.format(out_file)
	png_filename = png_filename.replace('slide_scores_', FLAGS.title)
	pdf_filename = '{}__auprc.pdf'.format(out_file)
	pdf_filename = pdf_filename.replace('slide_scores_', FLAGS.title)

	plt.savefig(png_filename, dpi= 300)
	plt.savefig(pdf_filename, dpi= 300)

	plt.show()

plot_auprc(FLAGS.metrics_file)	