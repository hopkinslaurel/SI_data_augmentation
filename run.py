# Training.py -- start here for Data Augmentation Approaches for Satellite Imagery
# Code based on https://github.com/ermongroup/tile2vec

import sys
import yaml
import argparse
import paths

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_file', default='config.yml')
args = parser.parse_args()

# read in config file
config = yaml.safe_load(open(args.config_file))
task = config['model']['task']
paths = paths.get_paths(task)

sys.path.append('../')
sys.path.append(paths['home_dir'])

import os
import glob
import pandas
import torch
from torch import optim
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from time import time
import random
import numpy as np
from tensorboardX import SummaryWriter
import pickle
import csv
import pandas as pd
from datetime import date
import shutil
import utils
from src.data_utils import *
from src.training import train_model, validate_model

# read in config file
config = yaml.safe_load(open(args.config_file))
training = config['training']
data = config['data']
eval = config['eval']
pre = config['preprocessing']

# environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cuda = torch.cuda.is_available()

# torch.manual_seed(50)

# create necessary directories if they do not already exist
if not os.path.exists(paths['log_dir']):
	os.makedirs(paths['log_dir'])
if not os.path.exists(paths['model_dir']):
	os.makedirs(paths['model_dir'])
save_dir = paths['model_dir'] + config['model']['name']
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# data parameters
img_type = data['img_type']
img_size = data['img_size']
bands = data['bands']
img_ext = data['img_ext']
augment = training['augment']
augment_type = training['augment_type']
augment_random = training['augment_random']
batch_size = training['batch_size']
shuffle = training['shuffle']
num_workers = training['num_workers']
labels_path = data['labels']
mixing_method = training['mixing_method']
cutmix_alpha = training['cutmix_alpha']
cutmix_num_pairs = training['cutmix_num_pairs']

if augment_random:
	from src.datasets_random_augment import satellite_dataloader
else:
	from src.datasets import satellite_dataloader

##### Preprocessing #####
# Convert tifs to npy - do this once before training (if needed)
if pre['tif2npy']:
	print('\nConverting tif to npy')
	tif2npy(paths['tif_dir'], paths['npy_dir'], img_ext, bands)

if pre['csv2npy']:  # Convert label csv to numpy
	print('\nConverting csv to npy')
	utils.csv_to_npy(labels_path, paths['home_dir'])

# Calculate dataset means/std devs for preprocessing - do this once before training
if pre['calc_channel_means'] or pre['calc_channel_means_stdDevs']:
	split = pre['split']  # which split to calculate means over

	if pre['calc_channel_means']:
		print(f'\n\nCalculating channel means for {split}')
	else:
		print(f'\n\nCalculating channel means & standard deviations for {split}')

	dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=split,
									  size=img_size, img_ext=img_ext, bands=bands,
									  augment=False, augment_type=[], augment_random=augment_random, batch_size=1,
									  shuffle=False, num_workers=num_workers, means=None)  # means MUST be None

	means, stds = utils.get_channel_mean_stdDev(dataloader, bands)

	print(f'Means: {means}')
	if pre['calc_channel_means_stdDevs']:
		print(f'Standard deviations: {stds}')

	# save means
	img_path = os.path.normpath(paths['img_dir']).split(os.path.sep)[-3]
	today = date.today()
	d = today.strftime('%b-%d-%Y')
	if pre['calc_channel_means']:
		np.savetxt('channel_means_' + img_path + '_' + d + '.txt', means)
	else:
		means_stds = torch.cat((means, stds))
		np.savetxt('channel_means_stds_' + img_path + '_' + d + '.txt', means_stds)

##### Define model #####
print(config['model']['arch'])
if config['model']['arch'] == 'resnet18':
	if task == 'eurosat':
		from src.resnet_eurosat import make_cnn
		CNN = make_cnn(num_classes=data['num_classes'], in_channels=bands)
	else:
		from src.resnet import make_cnn
		CNN = make_cnn(num_classes=data['num_classes'])  # assumed 3 input channels
else:
	raise Exception('Model not defined')

# define criterion
if config['model']['mode'] == 'regression':
	criterion = nn.MSELoss()
else:
	print(f"Num classes: {data['num_classes']}")
	if data['num_classes'] == 1:
		criterion = nn.BCEWithLogitsLoss(reduction='mean')
	else:
		criterion = nn.CrossEntropyLoss()

if cuda:
	CNN.cuda()
print('\nCuda available: ' + str(cuda))


#### Dataloaders #####
if training['train'] or eval['post_training']:
	train_split = 'train'
	test_split = 'test'
	val_split = 'val'

	# read in means
	train_means = np.loadtxt(paths['means'])
	train_means = tuple(train_means)

	train_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=train_split,
											size=img_size, img_ext=img_ext, bands=bands,
											augment=augment, augment_type=augment_type, augment_random=augment_random,
											batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
											means=train_means)
	print('\nTrain Dataloader set up complete.')
	print(len(train_dataloader))

	if training['val']:
		val_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=val_split,
											  size=img_size, img_ext=img_ext, bands=bands,
											  augment=augment, augment_type=augment_type, augment_random=augment_random,
											  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
											  means=train_means)
		print('Val Dataloader set up complete.')

	test_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=test_split,
										   size=img_size, img_ext=img_ext, bands=bands,
										   augment=augment, augment_type=augment_type, augment_random=augment_random,
										   batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
										   means=train_means)
	print('Test Dataloader set up complete.')

	if eval['post_training']:
		eval_split = test_split
		eval_means = None

		eval_dataloader = satellite_dataloader(img_type, paths['img_dir'], labels_path, split=eval_split,
											   size=img_size, img_ext=img_ext, bands=bands,
											   augment=False, augment_type=[], augment_random=augment_random,
											   batch_size=batch_size, shuffle=False, num_workers=1, means=eval_means)
		print(len(eval_dataloader))

if training['train']:
	# print summary
	print('\nModel summary:')
	print(CNN)
	summary(CNN, (bands, img_size, img_size))

	##### Training #####
	# load saved model
	if config['model']['model_fn']:
		CNN.load_state_dict(torch.load(config['model']['model_fn']))
		print('\nLoaded saved model')

	print(f"\nTraining model: {config['model']['arch']}")
	print(f"Name: {config['model']['name']}")

	if training['save_models']:
		print('\nSaving checkpoints')
	else:
		print('\nNot saving checkpoints')

	lr = float(training['lr'])
	weight_decay = training['weight_decay']
	beta1 = training['beta1']
	lr_gamma = training['lr_gamma']
	optimizer = optim.AdamW(CNN.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)

	if lr_gamma:
		print('Using learning rate scheduler')
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
	else:
		scheduler = None

	print_every = 2000

	# logging
	writer = SummaryWriter(paths['log_dir'] + config['model']['name'])

	# copy config file to model_dir
	shutil.copyfile(args.config_file, os.path.join(save_dir, os.path.basename(args.config_file)))

	train_loss = []
	test_loss = []
	val_loss = []

	##### Training #####
	t0 = time()
	for epoch in range(training['epochs_start'], training['epochs_end']):
		if training['train']:
			avg_loss_train = train_model(CNN, cuda, train_dataloader, optimizer,
										 epoch, criterion, data['num_classes'], config['model']['mode'],
										 print_every, mixing_method, cutmix_alpha, cutmix_num_pairs)
			train_loss.append(avg_loss_train)
			writer.add_scalar('loss/train', avg_loss_train, epoch)

		if training['test']:
			avg_loss_test = validate_model(CNN, cuda, val_dataloader, epoch,
										   criterion, data['num_classes'])
			test_loss.append(avg_loss_test)
			writer.add_scalar('loss/test', avg_loss_test, epoch)

		if training['val']:
			avg_loss_val = validate_model(CNN, cuda, val_dataloader, epoch,
										  criterion, data['num_classes'])
			val_loss.append(avg_loss_val)
			writer.add_scalar('loss/val', avg_loss_val, epoch)

		if epoch % 50 == 0 and scheduler:
			print("STEPPING")
			scheduler.step()

		if training['save_models'] & (epoch % 5 == 0):
			print('Saving')
			save_name = 'CNN' + str(epoch) + '.ckpt'
			model_path = os.path.join(save_dir, save_name)
			torch.save(CNN.state_dict(), model_path)
			if training['train']:
				with open(save_dir + '/train_loss.p', 'wb') as f:
					pickle.dump(train_loss, f)
			if training['test']:
				with open(save_dir + '/test_loss.p', 'wb') as f:
					pickle.dump(test_loss, f)

		##### Evaluation #####
		results = []

		# Calculate model R^2 (coefficient of determination)
		if eval['r2'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating R^2 for {split}')
			r2 = utils.calc_r2(CNN, test_dataloader, cuda)
			results.append(f'R2 - epoch{str(epoch)}: {str(r2)}')

		# Calculate mean squared error
		if eval['mse'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating MSE for {split}')
			mean_sq_loss = utils.calc_mse(CNN, test_dataloader, cuda)
			results.append(f'MSE - epoch{str(epoch)}: {str(mean_sq_loss)}')

		# Calculate mean absolute error
		if eval['mae'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating MAE for {split}')
			mean_abs_loss = utils.calc_mae(CNN, test_dataloader, cuda)
			results.append(f'MAE - epoch{str(epoch)}: {str(mean_abs_loss)}')

		# Calculate accuracy
		if eval['acc'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating accuracy for {split}')
			acc = utils.calc_acc(CNN, test_dataloader, cuda, data['num_classes'])
			results.append(f'Accuracy - epoch{str(epoch)}: {str(acc)}')

		# Calculate precision, recall & F1-score
		if eval['pr'] & (epoch % 5 == 0):
			split = 'test'
			print(f'Calculating precision & recall for {split}')

			macro, weighted = utils.calc_PR(CNN, test_dataloader, cuda, data['num_classes'])
			results.append(f'Macro - epoch{str(epoch)}: {str(macro)}')
			results.append(f'Weighted - epoch{str(epoch)}: {str(weighted)}')

		# print time
		if epoch % 5 == 0:
			print(f'Training time: {(time() - t0):.2f} sec')

		# save results to txt
		txt = os.path.join(save_dir, 'results_' + config['model']['name'] + '.txt')
		with open(txt, 'a') as f:
			for line in results:
				f.write(f'{line}\n')

	if eval['r2']:
		print(f'R2 = {str(r2)}')
	if eval['mse']:
		print(f'MSE = {str(mean_sq_loss)}')
	if eval['mae']:
		print(f'MAE = {str(mean_abs_loss)}')
	if eval['acc']:
		print(f'Accuracy = {str(acc)}')
	if eval['pr']:
		print(f'Macro = {str(macro)}')
		print(f'Weighted = {str(weighted)}')
	print('Done training!')

# Calculate model predictions (to later be used for evaluation metrics, etc.)
if eval['post_training']:
	# print summary
	print('\nModel summary:')
	print(CNN)
	summary(CNN, (bands, img_size, img_size))


	CNN.load_state_dict(torch.load(eval['model_weights']))
	model_name_lst = eval['model_weights'].split('/')
	model_name = model_name_lst[-2]
	model_name += '_' + model_name_lst[-1].split('.')[-2]
	print(f'\nWeights loaded from {model_name}')

	print(f'\nEvaluating model on {eval_split} split\n')

	post_training_results = []

	eval_dataloader = test_dataloader

	if eval['r2']:
		print(f'\nCalculating R^2')
		r2 = utils.calc_r2(CNN, eval_dataloader, cuda)
		post_training_results.append(f'R2: {str(r2)}')
		print(f'R2 = {str(r2)}')
	if eval['mse']:
		print(f'\nCalculating MSE')
		mse, mse2 = utils.calc_mse(CNN, eval_dataloader, cuda, criterion)
		post_training_results.append(f'MSE: {str(mse)}')
		print(f'MSE = {str(mse)}')
		print(f'MSE2 = {str(mse2)}')
	if eval['mae']:
		print(f'\nCalculating MAE')
		mae = utils.calc_mae(CNN, eval_dataloader, cuda)
		post_training_results.append(f'MAE: {str(mae)}')
		print(f'MAE = {str(mae)}')
	if eval['acc']:
		print(f'\nCalculating accuracy')
		acc = utils.calc_acc(CNN, eval_dataloader, cuda)
		post_training_results.append(f'Accuracy: {str(acc)}')
		print(f'Accuracy = {str(acc)}')
	if eval['confusion_matrix']:
		print(f'\nCalculating confusion matrix')
		confusion = utils.calc_confusion_matrix(CNN, eval_dataloader, cuda)
		post_training_results.append(f'Confusion matrix: {str(confusion)}')
		print(f'Confusion matrix = \n{str(confusion)}')
	if eval['pr']:
		print(f'\nCalculating precision & recall')
		macro, weighted = utils.calc_PR(CNN, eval_dataloader, cuda)
		post_training_results.append(f'Macro: {str(macro)}')
		post_training_results.append(f'Weighted: {str(weighted)}')
		print(f'Macro = {str(macro)}')
		print(f'Weighted = {str(weighted)}')
	if eval['inference_time']:
		_, val_time = validate_model(CNN, cuda, eval_dataloader, 0, criterion, timeit=True)
		num_images = len(eval_dataloader) * batch_size
		print(f'Inference performed on {num_images} images in {val_time} seconds')

	# Get model predictions post-training
	if eval['predictions']:
		print(f'\nCalculating predictions')
		if config['model']['mode'] == 'classification':
			outputs, preds, labels, ids = utils.get_predictions(CNN, eval_dataloader, cuda, config['model']['mode'])

			to_df = np.concatenate((ids, labels, preds, outputs), axis=1)
			df_cols = ['ids', 'labels', 'predictions']
			output_cols = ['softmax_output_' + str(i) for i in range(data['num_classes'])]
			print(len(output_cols))
			df_cols += output_cols
			df_out = pd.DataFrame(to_df, columns=df_cols)
		else:
			preds, labels, ids = utils.get_predictions(CNN, eval_dataloader, cuda, config['model']['mode'])
			to_df = np.stack((ids, labels, preds), axis=1)
			df_cols = ['ids', 'labels', 'predictions']
			df_out = pd.DataFrame(to_df, columns=df_cols)

		fp = os.path.join(paths['home_dir'], 'predictions_' + model_name + '.csv')
		print(fp)
		df_out.to_csv(fp, sep=',')

	# save results to txt
	if len(post_training_results) > 0:
		eval_txt = os.path.join(paths['home_dir'], 'results_' + model_name + '.txt')
		with open(eval_txt, 'a') as f:
			for line in post_training_results:
				f.write(f'{line}\n')
		# use this code to save predictions
		print('Done!')

