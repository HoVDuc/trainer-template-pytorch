import os
import torch
from tqdm import tqdm
from datetime import datetime

class Trainer:

    def __init__(self) -> None:
        self.history = {
            'loss': [],
            'val_loss': [],
        }

    def batch_device(self, batch):
        for item in batch:
            yield item.to(self.device)
            
    def fit(self, model, train_loader, valid_loader, critation, optimizer, lr_scheduler, epochs, device, save_dir):
        self.model = model
        self.critation = critation
        self.optimizer = optimizer
        self.device = device
        best_val_loss = 999
        
        # Create folder checkpoints:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        now_path = os.path.join(save_dir, now)
        if not os.path.isdir(now_path):
            os.mkdir(now_path)
        # Training process
        for epoch in range(1, epochs+1):
            print("Epoch: %3s/%3s" % (epoch, epochs))
            # Train phase
            total_loss = 0
            num_train_loader = len(train_loader)
            pbar = tqdm(train_loader, ncols=100)
            for i, batch in enumerate(pbar):
                total_loss += self.train_step(batch)
                pbar.set_description('train_loss: %.5f' % (total_loss / (i+1)))
            train_loss = total_loss / num_train_loader

            # Validation phase
            total_val_loss = 0
            num_val_loader = len(valid_loader)
            for batch in valid_loader:
                total_val_loss += self.valid_step(batch)
            val_loss = total_val_loss / num_val_loader
            
            #Save best val_loss each epoch 
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                self.save(os.path.join(now_path, 'best-checkpoint.pth'))
                print('Model saved at epoch {} with val_loss is {}'.format(epoch, val_loss))
            self.save(os.path.join(now_path, 'epoch-{}.pth'.format(epoch)))

            # Log loss and valid loss into history
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            lr_scheduler.step(train_loss)

        print('train_loss:', train_loss)
        print('valid_loss:', val_loss)

    def train_step(self, batch):
        self.model.train()
        input_1, input_2, targets = self.batch_device(batch)
        outputs = self.model(input_1, input_2)
        self.optimizer.zero_grad()
        loss = self.critation(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
            
    def valid_step(self, batch):
        with torch.no_grad():
            self.model.eval()
            input_1, input_2, targets = self.batch_device(batch)
            outputs = self.model(input_1, input_2)
            loss = self.critation(outputs, targets)
        return loss.item()

    def visualize(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history['loss'], label='loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
