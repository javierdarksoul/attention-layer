import torch
import json

def train(model, train_loader , valid_loader,epochs, lr, usecuda,path,metricsname):
    criterion = torch.nn.CrossEntropyLoss()
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    den_train = train_loader.__len__()*train_loader.batch_size
    den_valid = valid_loader.__len__()*valid_loader.batch_size
    metrics_dict={}
    if usecuda:
        model = model.cuda()
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        for image,label in train_loader:
            optimizer.zero_grad()
            if usecuda:
                image = image.cuda()
                label = label.cuda()
            
            output=model(image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            train_loss+= loss.item()
            
        for image,label in valid_loader:
            if usecuda:
                image = image.cuda()
                label = label.cuda()
            output=model(image)
            loss = criterion(output,label)
            valid_loss+= loss.item()
        metrics_dict['Epoch'] = epoch
        metrics_dict['Train_Loss'] = train_loss
        metrics_dict['Valid_Loss'] = valid_loss
        with open(metricsname, "w") as f:
            f.write(json.dumps(metrics_dict))
        if valid_loss<global_loss:
            print("Se encontró un mejor modelo en la epoca: %i\nLoss Actual: =%f" % (epoch, valid_loss/den_valid))
            global_loss = valid_loss
            torch.save(model.state_dict(), path+"/best.pt")
            print("Guardando modelo en la siguiente ruta: %s" % (path))
            
