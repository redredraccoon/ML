import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import matplotlib.pyplot as plt

##REPRODUCIBILITY
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = './seg_train'
DATASET_ROOT_TEST = './seg_test'

def train():
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	#print(DATASET_ROOT)
	#train_load
	train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
	train_data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
	#test_load
	test_set = IMAGE_Dataset(Path(DATASET_ROOT_TEST), data_transform)
	test_data_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
	#print(train_set.num_classes)
	model = VGG16(num_classes=train_set.num_classes)
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 10
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)


	train_loss_list = [] 
	train_acc_list = [] 
	epoch_list = []
	
	test_loss_list = [] 
	test_acc_list = [] 
	test_epoch_list = []
	
	for epoch in range(num_epochs):
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0
		
		testing_loss = 0.0
		testing_corrects = 0

		for i, (inputs, labels) in enumerate(train_data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			# train step
			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)

		train_loss_list.append(training_loss)
		train_acc_list.append(training_acc)
		epoch_list.append(epoch+1)
		
		with torch.no_grad():	
                        for inputs, labels in test_data_loader:
                                inputs = Variable(inputs.cuda(CUDA_DEVICES))
                                labels = Variable(labels.cuda(CUDA_DEVICES))
                                outputs = model(inputs)
                                _, preds = torch.max(outputs.data, 1)
                                loss_test = criterion(outputs, labels)
                                # test
                                testing_loss += loss_test.item() * inputs.size(0)
                                testing_corrects += torch.sum(preds == labels.data)
                                
		testing_loss = testing_loss / len(test_set)
		testing_acc = testing_corrects.double() / len(test_set)
                

		test_loss_list.append(testing_loss)
		test_acc_list.append(testing_acc)
		test_epoch_list.append(epoch+1)

		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')


		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_model_params)
	torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')

	plt.plot(epoch_list, train_loss_list, label='train')
	plt.plot(test_epoch_list, test_loss_list, label='test')
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(['train','test'], loc='upper right')
	plt.savefig("loss.png",dpi=250,format="png")
	plt.clf()
	plt.cla()

	plt.plot(epoch_list, train_acc_list, label='train')
	plt.plot(test_epoch_list, test_acc_list, label='test')
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(['train','test'], loc='upper right')
	plt.savefig("accuracy.png",dpi=250,format="png")
	plt.clf()
	plt.cla()

if __name__ == '__main__':
	train()
