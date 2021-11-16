# importing modules
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torchsummary
import torch
import PIL
import sys
from time import time
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

# Loading dataset labels
BASE_PATH=r"C:\Users\abhin\Desktop\AI-Project"
train_dataset=pd.read_csv(os.path.join(BASE_PATH,'train_labels.csv'))
test_dataset=pd.read_csv(os.path.join(BASE_PATH,'test_labels.csv'))

# code to view some images of training dataset
# figure size
figure=plt.figure(figsize=(32, 32))
# rows and columns
columns = 3
rows = 5
#Total images = row* columns
for i in range(1,rows*columns+1):
    IMG_PATH=BASE_PATH+'\\train\\'
    img=Image.open(os.path.join(IMG_PATH,train_dataset.iloc[i][0]))
    figure.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

# loading images of training and testing dataset according to the labels mentioned
class Dataset(data.Dataset):
    def __init__(self,csv_path,images_path,transform=None):
        self.train_set=pd.read_csv(csv_path) #Read The CSV and create the dataframe
        self.train_path=images_path #Images Path
        self.transform=transform # Augmentation Transforms
    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self,idx):
        file_name=self.train_set.iloc[idx][0] 
        label=self.train_set.iloc[idx][1]
        img=Image.open(os.path.join(self.train_path,file_name)) #Loading Image
        if self.transform is not None:
            img=self.transform(img)
        return img,label

# learning rate
learning_rate=1e-4

# Now used Dataloader class from pytorch to load images and it returned a untransformed data 
training_set_untransformed=Dataset(os.path.join(BASE_PATH,'train_labels.csv'),os.path.join(BASE_PATH,'train/'))
print(len(training_set_untransformed))

# transforming data like rotation and translation
transform_train = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
		transforms.ToTensor()])

# creating new images only for those images which are labelled as covid positive
new_created_images=[]
for j in range (len(training_set_untransformed)):
    if training_set_untransformed[j][1]==1:
        for k in range(8):
            transformed_image = transform_train(training_set_untransformed[j][0])
            new_created_images.append((transformed_image,1))
    else:
        transformed_image = transform_train(training_set_untransformed[j][0])
        new_created_images.append((transformed_image,0))

print(len(new_created_images))

# splitting the newly created images into training and validation set in the ratio 80:20
train_size = int(0.8 * len(new_created_images))
validation_size = len(new_created_images) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(new_created_images, [train_size,validation_size])
print(len(train_dataset))

# using dataloader function of pytorch to roughly shufle the training images and creating a batch of 16 each
training_generator = data.DataLoader(train_dataset,shuffle=True,batch_size=16,pin_memory=True)

# enabling GPU if not GPU we can run it on CPU as well
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# Using efficientNet pretrained model
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)

# loading the model into device(GPU/CPU)
model.to(device)

# printing summary of the model
print(torchsummary.summary(model, input_size=(3, 224, 224)))

# saving the weights and epochs will be 2
criterion = nn.CrossEntropyLoss()
lr_decay=0.99
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eye = torch.eye(2).to(device)
classes=[0,1]
history_accuracy=[]
history_loss=[]
epochs = 2
# path to save
PATH_SAVE=r"C:\Users\abhin\Desktop\AI-Project\Weights"
# running epochs
for epoch in range(epochs):  
    running_loss = 0.0
    correct=0
    total=0
    class_correct = list(0. for _ in classes)
    class_total = list(0. for _ in classes)
    
    for i, data in enumerate(training_generator, 0):
        inputs, labels = data
        t0 = time()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = eye[labels]
        optimizer.zero_grad()
        #torch.cuda.empty_cache()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        c = (predicted == labels.data).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = float(correct) / float(total)
        
        history_accuracy.append(accuracy)
        history_loss.append(loss)
        
        loss.backward()
        optimizer.step()
        
        for j in range(labels.size(0)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        
        running_loss += loss.item()
        
        print( "Epoch : ",epoch+1," Batch : ", i+1," Loss :  ",running_loss/(i+1)," Accuracy : ",accuracy,"Time ",round(time()-t0, 2),"s" )
    for k in range(len(classes)):
        if(class_total[k]!=0):
            print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
        
    print('[%d epoch] Accuracy of the network on the Training images: %d %%' % (epoch+1, 100 * correct / total))
    
    if epoch%10==0 or epoch==0:
        torch.save(model.state_dict(), os.path.join(PATH_SAVE,str(epoch+1)+'_'+str(accuracy)+'.pth'))
# saving the model
torch.save(model.state_dict(), os.path.join(PATH_SAVE,'Last_epoch'+str(accuracy)+'.pth'))

# plotting history accuracy and history loss
plt.plot(history_accuracy)
plt.plot(torch.tensor(history_loss).detach().numpy())

# evaluating the model
model.eval()
# transformng the test images
test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
                                      transforms.ToTensor(),
                                     ])
# predicting that is the image covid + or - based on the model trained
def predict_image(image):
    image_tensor = test_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

# testing the accuracy for the validation test
correct_counter=0
for i in range(len(validation_dataset)):
    print(i)
    image_tensor = validation_dataset[i][0].unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    if index == validation_dataset[i][1]:
        correct_counter+=1
print("Accuracy=",correct_counter/len(validation_dataset))
# The accuracy is 97%
# creating a sample submission .csv files
submission=pd.read_csv(BASE_PATH+'\sample_submission.csv')
submission_csv=pd.DataFrame(columns=['File','Label'])
IMG_TEST_PATH=os.path.join(BASE_PATH,'test/')
for i in range(len(submission)):
    img=Image.open(IMG_TEST_PATH+submission.iloc[i][0])
    prediction=predict_image(img)
    submission_csv=submission_csv.append({'File': submission.iloc[i][0],'Label': prediction},ignore_index=True)
    if(i%10==0 or i==len(submission)-1):
        print('[',32*'=','>] ',round((i+1)*100/len(submission),2),' % Complete')
submission_csv.to_csv('submission_epoch2_2.csv',index=False)

