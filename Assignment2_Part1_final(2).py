import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

class Data():

    def __init__(self, relative_path="C:/Users/tobti/Desktop/Assignment 2 AI/keio2019aia-master/data/assignment2/"):
        
        self.glove = open(relative_path + "wv_50d.txt","r",encoding='utf8')
        self.binarytrain = open(relative_path + "senti_binary.train","r")
        self.binarytest = open(relative_path + "senti_binary.test","r")

        ##Preprocessing GloVe-Data##
        
        if self.glove.mode == "r":
            glove = self.glove.read()

        data_glove = glove.split()


        list_data_glove = []
        prearrayglove = []
        p =0

        for x in data_glove:
            if p == 0 or p%51 == 0:
                if prearrayglove != []:
                    list_data_glove += [(y, np.array(prearrayglove))]
                y = x
                prearrayglove = []
            else:
                prearrayglove.append(float(x))
            p +=1
            if p == 20400000:
                list_data_glove += [(y, np.array(prearrayglove))]

        self.glovedict = {}

        ##GloVe-Dictionary, word key, globle vector as corresponding value##
        
        for x,y in list_data_glove:
            self.glovedict[x] = y
        
    def data_train(self):
        
        ##Preprocessing Train-Dataset##
        
        list_data = []

        for line in self.binarytrain:
            line_split = line[:-2].split()
            list_data += [(np.array(line_split),line[-2])]

        list_data_final_train = []

        for x,y in list_data:
            wordemb = [self.glovedict.get(z) for z in x if self.glovedict.get(z) is not None] + [[0 for t in range(50)] for z in x if self.glovedict.get(z) is None]
            wordembfinal = np.array(wordemb)
            wordembfinal_mean = np.mean(wordembfinal, axis = 0)
            list_data_final_train += [(np.array(wordembfinal_mean),int(y))]
    
        return list_data_final_train    
            
    def data_test(self):
        
        ##Preprocessing Test-Dataset##
        
        list_data_test = []

        for linetest in self.binarytest:
            line_split_test = linetest[:-2].split()
            list_data_test += [(np.array(line_split_test),linetest[-2])]

        list_data_final_test = []

        for x,y in list_data_test:
            wordemb_test = [self.glovedict.get(z) for z in x if self.glovedict.get(z) is not None] + [[0 for t in range(50)] for z in x if self.glovedict.get(z) is None]
            wordembfinal_test = np.array(wordemb_test)
            wordembfinal_test_mean = np.mean(wordembfinal_test, axis = 0)
            list_data_final_test += [(np.array(wordembfinal_test_mean),int(y))]

        return list_data_final_test

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        
        super(Model, self).__init__()
        
        self.hl1 = nn.Linear(input_dim, hidden1_dim)
        self.hl1a = nn.ReLU()
        self.layer1 = [self.hl1, self.hl1a]
        
        self.hl2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hl2a = nn.ReLU()
        self.layer2 = [self.hl2, self.hl2a]
        
        self.ol = nn.Linear(hidden2_dim, output_dim)
        self.ola = (lambda x: x)
        self.layer3 = [self.ol, self.ola]
        
        self.layers = [self.layer1, self.layer2, self.layer3]
        
    def forward(self, x):
        
        out = x
        
        for pa, a in self.layers:
            
            out = a(pa(out))
        
        return out

class Trainer():
    
    def __init__(self, model, data):
        
        self.model = model
        self.data = data
        self.data_train = self.data.data_train()
        self.data_test = self.data.data_test()
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.data_train, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.data_test, batch_size=64, shuffle=True)
    
    def train(self, lr, ne):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)

        self.model.train()
        
        self.costs_train = []
        self.costs_test = []
        self.accuracy_train = []
        self.accuracy_test = []

        ##calculating initial accuracy of train and test set##
        correct_initial = 0
        train_cost_initial = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            train_cost_initial += loss
            pred = torch.max(outputs,1)[1]
            correct_initial += torch.eq(pred,targets).sum()
        
        initialcost_train = train_cost_initial/len(self.data_train)
        initialcost_test = self.test()[1]
        initialaccuracy_train = 100 * correct_initial.item() / len(self.data_train)
        initialaccuracy_test = self.test()[0]
        print("initial cost train set: %f, initial accuracy train set: %.3f" % (initialcost_train, initialaccuracy_train))     
        print("initial cost test set: %f, initial accuracy test set: %.3f" % (initialcost_test, initialaccuracy_test))
        
        
        ##training##
        for e in range(ne):
            
            print('training epoch %d / %d ...' %(e+1, ne))
            
            train_cost = 0
            correct = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):

                inputs = Variable(inputs)
                targets = Variable(targets)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                train_cost += loss
                pred = torch.max(outputs,1)[1]
                
                correct += torch.eq(pred,targets).sum()
                loss.backward()
                optimizer.step()
            
            self.costs_train.append(train_cost/len(self.data_train))
            self.costs_test.append(self.test()[1])
            self.accuracy_train.append(100 * correct.item() / len(self.data_train))
            self.accuracy_test.append(self.test()[0])
            print("cost train set: %f, cost test set: %f, accuracy train set=%.3f, accuracy test set=%.3f" %(self.costs_train[-1], self.costs_test[-1], self.accuracy_train[-1], self.accuracy_test[-1]))

        print("training complete")
        print("final accuracy train set: %.3f, final accuracy test set: %.3f " % (self.accuracy_train[-1], self.accuracy_test[-1]))
        self.costs_train.insert(0, initialcost_train)
        self.costs_train.insert(0, initialcost_test)
        self.accuracy_train.insert(0,initialaccuracy_train)
        self.accuracy_test.insert(0,initialaccuracy_test)
        self.output()

    def test(self):
        
        criterion = nn.CrossEntropyLoss()
        correct_test = 0
        test_cost = 0
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                
            inputs_test = Variable(inputs)
            targets_test = Variable(targets)
            outputs_test = self.model(inputs_test)
            loss_test = criterion(outputs_test, targets_test)
            test_cost += loss_test
            pred_test = torch.max(outputs_test,1)[1]
            correct_test += torch.eq(pred_test,targets_test).sum()
        
        testaccuracy = 100 * correct_test.item() / len(self.data_test)
        testcost = test_cost/len(self.data_test)
        
        return [testaccuracy, testcost]
    
    def output(self):
        
        txt_file = open("assignment2_part1_results", "w")
        txt_file.write(" %f \n %f " % (self.accuracy_train[-1], self.accuracy_test[-1]))
        txt_file.close()
        print("results text file successfully created!")
    
    def plots(self):
        
        ##plot accuracy##    
        plt.plot(range(len(self.accuracy_train)), self.accuracy_train, c="r", label="accuracy train set")
        plt.plot(range(len(self.accuracy_test)), self.accuracy_test, c="b", label="accuracy test set")
        plt.axis([0,25,0,100])
        plt.xlabel("epoch")
        plt.ylabel("accuracy in %")
        plt.legend(loc=4)
        plt.savefig("assignment2_part1_plots_accuracy")
        plt.show()  
    
        ##plot loss##
        plt.plot(range(len(self.costs_train)), self.costs_train, c="r", label="loss train set")
        plt.plot(range(len(self.costs_test)), self.costs_test, c="b", label="loss test set")
        plt.axis([0,25,0,0.015])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc=4)
        plt.savefig("assignment2_part1_plots_loss")
        plt.show()
    
    def predict(self, stringinput):
        
        textpre = np.array(stringinput.split())
        wordembtext = [self.data.glovedict.get(z) for z in textpre if self.data.glovedict.get(z) is not None] + [[0 for t in range(50)] for z in textpre if self.data.glovedict.get(z) is None]
        wordembtextfinal = np.array(wordembtext)
        wordembtextfinal_mean = np.mean(wordembtextfinal, axis = 0)
            
        texttensor = torch.tensor(wordembtextfinal_mean)
        inputtext = Variable(texttensor)
    
        yhatpredict = self.model(inputtext) 
        d = torch.max(yhatpredict,0)[1]
        
        return d.item()
    
    def save_checkpoint(self):
        
        checkpoint = {'state_dict': self.model.state_dict()}
        torch.save(checkpoint, 'part1_state.chkpt')
        print("model state successfully saved!")
        
def main():

    print("Data Preprocessing onging for approx. 3 Minutes")
    data = Data("C:/Users/tobti/Desktop/Assignment 2 AI/keio2019aia-master/data/assignment2/")   #specifiy relative path of input files
    model = Model(50, 100, 100, 2)
    model.double()
    trainer = Trainer(model, data)
    trainer.train(0.1, 25)
    trainer.plots()
    trainer.save_checkpoint()
    
    
if __name__ == '__main__':

    main()

