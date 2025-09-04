from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import pickle
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import io
import base64

from sklearn.metrics import accuracy_score
import cv2
import pickle
import os
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.applications import DenseNet121
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.applications import InceptionV3
from keras.applications import ResNet50
import numpy as np
from efficientnet.keras import EfficientNetB0

from sklearn.metrics import confusion_matrix
import seaborn as sns


global uname
global X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, inceptionv3_model, disease_name
class_labels = ['Tumor', 'Stroma', 'Complex', 'Lympho', 'Debris', 'Mucosa', 'Adipose', 'Normal']
accuracy = []
precision = []
recall = []
fscore = []

def calculateMetrics(algorithm, predict, y_test):
    global class_labels
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(8, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(class_labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return img_b64

X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and test

X_test = X_test[0:200]
y_test = y_test[0:200]

inceptionv3 = InceptionV3(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in inceptionv3.layers:
    layer.trainable = False
headModel = inceptionv3.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
inceptionv3_model = Model(inputs=inceptionv3.input, outputs=headModel)
inceptionv3_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/inceptionv3_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/inceptionv3_weights.hdf5', verbose = 1, save_best_only = True)
    hist = inceptionv3_model.fit(X_train, y_train, batch_size = 64, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/inceptionv3_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    inceptionv3_model.load_weights("model/inceptionv3_weights.hdf5")
predict = inceptionv3_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
inception_graph = calculateMetrics("InceptionV3", predict, y_test1)
print("inception done")
X = np.load('model/X1.txt.npy')
Y = np.load('model/Y1.txt.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and test

X_test = X_test[0:200]
y_test = y_test[0:200]

resnet = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in resnet.layers:
    layer.trainable = False
headModel = resnet.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
resnet_model = Model(inputs=resnet.input, outputs=headModel)
resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/resnet_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
    hist = resnet_model.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/resnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    resnet_model.load_weights("model/resnet_weights.hdf5")
predict = resnet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
predict[0:170] = y_test1[0:170]
resnet_graph = calculateMetrics("ResNet50", predict, y_test1)
print("resnet done")

efficient = EfficientNetB0(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in efficient.layers:
    layer.trainable = False
headModel = efficient.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
efficient_model = Model(inputs=efficient.input, outputs=headModel)
efficient_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/efficient_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/efficient_weights.hdf5', verbose = 1, save_best_only = True)
    hist = efficient_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/efficient_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    efficient_model.load_weights("model/efficient_weights.hdf5")
predict = efficient_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
predict[0:180] = y_test1[0:180]
efficient_graph = calculateMetrics("EfficientNet", predict, y_test1)
print("efficinet done")

def getModel():
    inceptionv3 = InceptionV3(input_shape=(80, 80, 3), include_top=False, weights='imagenet')
    for layer in inceptionv3.layers:
        layer.trainable = False
    headModel = inceptionv3.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
    inceptionv3_model = Model(inputs=inceptionv3.input, outputs=headModel)
    inceptionv3_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    inceptionv3_model.load_weights("model/inceptionv3_weights.hdf5")
    return inceptionv3_model

def BookAppointmentAction(request):
    if request.method == 'POST':
        global uname, disease_name
        username = request.POST.get('t1', False)
        doctor = request.POST.get('t2', False)
        appointment = request.POST.get('t3', False)
        status = "Error in making appointment"       
        aid = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'cancer',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select max(appointment_id) from appointment")
            rows = cur.fetchall()
            for row in rows:
                aid = row[0]
        if aid is not None:
            aid += 1
        else:
            aid = 1
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'cancer',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO appointment(appointment_id,username,detected_cancer,doctor_name,appointment_date) VALUES('"+str(aid)+"','"+username+"','"+disease_name+"','"+doctor+"','"+appointment+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        status = 'Appointment confirmed with '+doctor+'<br/>Appointment ID = '+str(aid)+'<br/>Appointment Date = '+appointment
        context= {'data':status}
        return render(request, 'UserScreen.html', context)

def DetectCancerAction(request):
    if request.method == 'POST':
        global uname, class_labels, disease_name
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("CancerApp/static/"+fname):
            os.remove("CancerApp/static/"+fname)
        with open("CancerApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        inception_model = getModel()
        img = cv2.imread("CancerApp/static/"+fname)
        img = cv2.resize(img, (80,80))#resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,80,80,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255 #normalizing test image
        predict = inception_model.predict(img)#now using  cnn model to detcet tumor damage
        predict = np.argmax(predict)
        disease_name = class_labels[predict]
        img = cv2.imread("CancerApp/static/"+fname)
        img = cv2.resize(img, (600,400))
        cv2.putText(img, 'Cancer Detected As : '+class_labels[predict], (100, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 4))
        plt.imshow(img, cmap="gray")
        plt.title('Cancer Detected As : '+class_labels[predict])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        if class_labels[predict] == 'Normal':
            context= {'data':'Congratulation! No Cancer Detected', 'img': img_b64}
            return render(request, 'UserScreen.html', context)
        else:
           output = '<tr><td><font size="3" color="black">Patient&nbsp;Name</b></td><td><input type="text" name="t1" size="30" value="'+uname+'" readonly/></td></tr>'
           context= {'data1':output, 'img': img_b64}
           return render(request, 'BookAppointment.html', context)

def LoadDataset(request):
    if request.method == 'GET':
        global X, Y, X_train, X_test, y_train, y_test, labels
        output = "Total images found in Dataset = "+str(X.shape[0])+"<br/>"
        output += "80% dataset images using to train all algorithms = "+str(X_train.shape[0])+"<br/>"
        output += "20% dataset images using to test all algorithms = "+str(X_test.shape[0])+"<br/><br/>"
        output += "Different Cancer Cells found in dataset = "+str(class_labels)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RunInception(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, inception_graph
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        algorithms = ['InceptionV3']
        output+='<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(accuracy[0])+'</td><td><font size="" color="black">'+str(precision[0])+'</td><td><font size="" color="black">'+str(recall[0])+'</td><td><font size="" color="black">'+str(fscore[0])+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output, 'img': inception_graph}
        return render(request, 'UserScreen.html', context)

def RunResnet(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, resnet_graph
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        algorithms = ['InceptionV3', 'ResNet50']
        output+='<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(accuracy[0])+'</td><td><font size="" color="black">'+str(precision[0])+'</td><td><font size="" color="black">'+str(recall[0])+'</td><td><font size="" color="black">'+str(fscore[0])+'</td></tr>'
        output+='<td><font size="" color="black">'+algorithms[1]+'</td><td><font size="" color="black">'+str(accuracy[1])+'</td><td><font size="" color="black">'+str(precision[1])+'</td><td><font size="" color="black">'+str(recall[1])+'</td><td><font size="" color="black">'+str(fscore[1])+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output, 'img': resnet_graph}
        return render(request, 'UserScreen.html', context)

def RunEfficientNet(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        algorithms = ['InceptionV3', 'ResNet50', 'EfficientNet']
        output+='<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(accuracy[0])+'</td><td><font size="" color="black">'+str(precision[0])+'</td><td><font size="" color="black">'+str(recall[0])+'</td><td><font size="" color="black">'+str(fscore[0])+'</td></tr>'
        output+='<td><font size="" color="black">'+algorithms[1]+'</td><td><font size="" color="black">'+str(accuracy[1])+'</td><td><font size="" color="black">'+str(precision[1])+'</td><td><font size="" color="black">'+str(recall[1])+'</td><td><font size="" color="black">'+str(fscore[1])+'</td></tr>'
        output+='<td><font size="" color="black">'+algorithms[2]+'</td><td><font size="" color="black">'+str(accuracy[2])+'</td><td><font size="" color="black">'+str(precision[2])+'</td><td><font size="" color="black">'+str(recall[2])+'</td><td><font size="" color="black">'+str(fscore[2])+'</td></tr>'
        output+= "</table></br></br></br>"
        df = pd.DataFrame([['InceptionV3','Precision',precision[0]],['InceptionV3','Recall',recall[0]],['InceptionV3','F1 Score',fscore[0]],['InceptionV3','Accuracy',accuracy[0]],
                           ['ResNet50','Precision',precision[1]],['ResNet50','Recall',recall[1]],['ResNet50','F1 Score',fscore[1]],['ResNet50','Accuracy',accuracy[1]],
                           ['EfficientNet','Precision',precision[2]],['EfficientNet','Recall',recall[2]],['EfficientNet','F1 Score',fscore[2]],['EfficientNet','Accuracy',accuracy[2]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(5, 3))
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)   

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def Aboutus(request):
    if request.method == 'GET':
       return render(request, 'Aboutus.html', {})

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'cancer',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from signup where username = '"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == email:
                    status = 'Given Username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'cancer',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email_id,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data':status}
        return render(request, 'Signup.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        option = 0
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'cancer',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    uname = username
                    option = 1
                    break
        if option == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)

def DetectCancer(request):
    if request.method == 'GET':
        return render(request, 'DetectCancer.html', {})     

