import argparse
from datetime import datetime
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from model.additive_model import get_additive_mil_model
from dataset.dataset_train import Dataset, custom_collate_fn, worker_init_fn
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--model_dir', default='Results/prognosis_model/saved_models/', help='Directory to save models', dest='model_dir')
parser.add_argument('--slide_list_filename_train', default='prognosis_model/data/seg14/train.txt', help='slide list train', dest='slide_list_filename_train')
parser.add_argument('--slide_list_filename_valid', default='prognosis_model/data/seg14/valid.txt', help='slide list valid', dest='slide_list_filename_valid')
parser.add_argument('--slide_list_filename_test', default='prognosis_model/data/seg14/test.txt', help='slide list test', dest='slide_list_filename_test')
parser.add_argument('--patch_size', default='0', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='0', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_features', default='64', type=int, help='number of features', dest='num_features')
parser.add_argument('--pretrained', default=False, help='Pretrain model on ImageNet', dest='pretrained')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--learning_rate', default='1e-4', type=float, help='Learning rate', dest='learning_rate')
parser.add_argument('--weight_decay', default='1e-5', type=float, help='Weight decay', dest='weight_decay')
parser.add_argument('--num_epochs', default=500, type=int, help='Number of epochs', dest='num_epochs')
parser.add_argument('--save_interval', default=100, type=int, help='Model save interval (default: 1000)', dest='save_interval')
parser.add_argument('--metrics_dir', default='Results/prognosis_model/metrics/', help='Text file to write step, loss, accuracy metrics', dest='metrics_dir')
parser.add_argument('--validation_dir', default='Results/prognosis_model/validation/', help='Text file to write step, loss, accuracy metrics', dest='validation_dir')
parser.add_argument('--cz_imgs_list', default='prognosis_model/data/cz_img_list.txt', help='Image directory', dest='cz_imgs_list')
parser.add_argument('--he_imgs_list', default='prognosis_model/data/he_img_list.txt', help='Image directory', dest='he_imgs_list')
parser.add_argument('--imgs_type', default='CZ_HE', help='Image directory', dest='imgs_type')
parser.add_argument('--device_num', default='0', type=int, help='Batch size', dest='device_num')

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    
if not os.path.exists(FLAGS.metrics_dir):
    os.makedirs(FLAGS.metrics_dir)

if not os.path.exists(FLAGS.validation_dir):
    os.makedirs(FLAGS.validation_dir)

current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
metrics_file = FLAGS.metrics_dir + 'epoch_loss_acc' + current_time + '.txt'
validation_file = FLAGS.validation_dir + 'validation_results__' + current_time + '.txt'

print('current_time: {}'.format(current_time))
print('model_dir: {}'.format(FLAGS.model_dir))
print('init_model_file: {}'.format(FLAGS.init_model_file))
print('slide_list_filename_train: {}'.format(FLAGS.slide_list_filename_train))
print('slide_list_filename_valid: {}'.format(FLAGS.slide_list_filename_valid))
print('slide_list_filename_test: {}'.format(FLAGS.slide_list_filename_test))
print('patch_size: {}'.format(FLAGS.patch_size))
print('imgs_type: {}'.format(FLAGS.imgs_type))
print('cz_imgs_list: {}'.format(FLAGS.cz_imgs_list))
print('he_imgs_list: {}'.format(FLAGS.he_imgs_list))
print('num_instances: {}'.format(FLAGS.num_instances))
print('num_features: {}'.format(FLAGS.num_features))
print('num_classes: {}'.format(FLAGS.num_classes))
print('pretrained: {}'.format(FLAGS.pretrained))
print('batch_size: {}'.format(FLAGS.batch_size))
print('learning_rate: {}'.format(FLAGS.learning_rate))
print('weight_decay: {}'.format(FLAGS.weight_decay))
print('num_epochs: {}'.format(FLAGS.num_epochs))
print('save_interval: {}'.format(FLAGS.save_interval))
print('metrics_dir: {}'.format(FLAGS.metrics_dir))
print('metrics_file: {}'.format(metrics_file)) 


train_dataset = Dataset(slide_list_filename=FLAGS.slide_list_filename_train, cz_imgs_list= FLAGS.cz_imgs_list, he_imgs_list= FLAGS.he_imgs_list, dataset_type= 'train', imgs_type= FLAGS.imgs_type)
num_slides_train = train_dataset.num_slides
print("Training Data - num_slides: {}".format(train_dataset.num_slides))

valid_dataset = Dataset(slide_list_filename=FLAGS.slide_list_filename_valid, cz_imgs_list= FLAGS.cz_imgs_list, he_imgs_list= FLAGS.he_imgs_list, dataset_type= 'valid', imgs_type= FLAGS.imgs_type)
num_slides_valid = valid_dataset.num_slides
print("Validation Data - num_slides: {}".format(valid_dataset.num_slides))

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=16, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=5, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

device = torch.device('cuda:{}'.format(FLAGS.device_num)) if torch.cuda.is_available() else torch.device('cpu')

# get the model using helper function
model = get_additive_mil_model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, device_num= FLAGS.device_num, requires_grad= False)

#Model(num_classes=FLAGS.num_classes, features_size=1024, num_features=FLAGS.num_features, device_num= FLAGS.device_num, requires_grad= False)

# move model to the right device
model.to(device)

# define criterion
criterion = nn.CrossEntropyLoss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
# optimizer = torch.optim.SGD(params, lr=FLAGS.learning_rate, momentum=0.9, weight_decay=FLAGS.weight_decay)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, cooldown=50)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        state_dict = torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model_state_dict'])
        # optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        print("Model weights loaded successfully from file: ", FLAGS.init_model_file)


with open(metrics_file, 'w') as f:
    f.write('# current_time: {}\n'.format(current_time))
    f.write('# model_dir: {}\n'.format(FLAGS.model_dir))
    f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f.write('# slide_list_filename_train: {}\n'.format(FLAGS.slide_list_filename_train))
    f.write('# slide_list_filename_valid: {}\n'.format(FLAGS.slide_list_filename_valid))
    f.write('# slide_list_filename_test: {}\n'.format(FLAGS.slide_list_filename_test))
    f.write('# patch_size: {}\n'.format(FLAGS.patch_size))
    f.write('# cz_img_list: {}\n'.format(FLAGS.cz_imgs_list))
    f.write('# he_imgs_list: {}\n'.format(FLAGS.he_imgs_list))
    f.write('# imgs_type: {}\n'.format(FLAGS.imgs_type))
    f.write('# normalizer: \n')
    f.write('# num_instances: {}\n'.format(FLAGS.num_instances))
    f.write('# num_features: {}\n'.format(FLAGS.num_features))
    f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f.write('# pretrained: {}\n'.format(FLAGS.pretrained))
    f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f.write('# learning_rate: {}\n'.format(FLAGS.learning_rate))
    f.write('# weight_decay: {}\n'.format(FLAGS.weight_decay))
    f.write('# num_epochs: {}\n'.format(FLAGS.num_epochs))
    f.write('# save_interval: {}\n'.format(FLAGS.save_interval))
    f.write('# metrics_dir: {}\n'.format(FLAGS.metrics_dir))
    f.write('# metrics_file: {}\n'.format(metrics_file))
    f.write('# epoch\ttraining_loss\ttraining_acc\tvalidation_loss\tvalidation_acc\n')


with open(validation_file, 'w') as f:
    f.write('#epoch\tslide_id\tlabel\tpredicted\n')

total_steps = len(data_loader_train)
max_val_acc = 0.0
min_val_loss = 100.0

for epoch in range(FLAGS.num_epochs):
    
    ### training ###
    model.train()

    pbar = tqdm(total=len(data_loader_train))
        
    num_predictions = 0
    running_loss = 0.0
    running_correct = 0
    
    label_list = []
    predicted_list = []
    for i, (img, label, slide_id) in enumerate(data_loader_train):

        img, label = img.to(device), label.to(device)
        output = model(img)

        output = output['value']

        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        
        optimizer.step()
                                   
        _, predicted = torch.max(output, 1)
        
        correct = (predicted == label).sum().item()
        
        num_predictions += label.size(0)
        
        running_loss += loss.item() * label.size(0)
        
        running_correct += correct
            
        label_list += list(label.cpu().numpy())
        predicted_list += list(predicted.cpu().numpy())
            
        pbar.update(1)
        
    pbar.close()

    train_loss = running_loss / num_predictions
    train_acc = running_correct / num_predictions


    ### validation ###
    pbar2 = tqdm(total=len(data_loader_valid))

    num_predictions = 0
    running_loss = 0.0
    running_correct = 0
        
    label_list = []
    predicted_list = []
    slide_id_list = []

    model.eval()
    with torch.no_grad():
        for i, (img, label, slide_id) in enumerate(data_loader_valid):
                        
            img, label = img.to(device), label.to(device)
            output = model(img)

            output = output['value']
            
            loss = criterion(output, label)

            # print('loss_total: {}'.format(loss_total))
                        
            _, predicted = torch.max(output, 1)
        
            correct = (predicted == label).sum().item()

            num_predictions += label.size(0)
            
            running_loss += loss.item() * label.size(0)
        
            running_correct += correct
            
            label_list += list(label.cpu().numpy())
            predicted_list += list(predicted.cpu().numpy())
            slide_id_list += list(slide_id)
    
            with open(validation_file, 'a') as f:
                f.write('{}\t{}\t{}\t{}\n'.format(epoch + 1,slide_id[0], str(label[0].cpu().numpy()), str(predicted[0].cpu().numpy())))   

            # scheduler.step(running_loss)
            pbar2.update(1)

    pbar2.close()
                
    valid_loss = running_loss / num_predictions
    valid_acc = running_correct / num_predictions

    # print('Epoch : {:d}'.format(epoch + 1))
    print('Epoch - {}: train_loss={:.4f}, train_acc={:.4f}, valid_loss={:.4f}, valid_acc={:.4f}'.format(epoch+1, train_loss, train_acc, valid_loss, valid_acc))
    
    with open(metrics_file, 'a') as f:
        f.write('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))


    if ((valid_loss < min_val_loss) or (valid_acc > max_val_acc) or ((epoch + 1) % FLAGS.save_interval == 0)) and (epoch + 1)>10:
        model_weights_filename = FLAGS.model_dir + 'model_weights' + current_time + '__' + str(epoch + 1) + '.pth'
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state_dict, model_weights_filename)
        print('Model weights saved in file: {}'.format(model_weights_filename))
           
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss

        if valid_acc > max_val_acc:
            max_val_acc = valid_acc
        
