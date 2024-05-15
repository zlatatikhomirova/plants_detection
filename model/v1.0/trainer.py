import torch
import matplotlib.pyplot as plt
import copy
import datetime as dt
import copy

class Trainer():
  """
  Parameters:
      loss_f: функция потерь
      learning_rate: величина градиентного шага
      epoch_amount: общее количество эпох
      batch_size: размер одного бача
      max_batches_per_epoch: максимальное количество бачей, 
                             подаваемых в модель в одну эпоху
      device: устройство для вычислений
      early_stopping: количество эпох без улучшений до остановки обучения
      optim: оптимизатор
      scheduler: регулятор градиентного шага
      permutate: перемешивание тренировочной выборки перед обучением

  Attributes:
      start_model: необученная модель
      best_model: модель, после обучения
      train_loss: средние значения функции потерь на тренировочных 
                  данных в каждой эпохе
      val_loss: средние значения функции потерь на валидационных 
                данных в каждой эпохе

  Methods:
      fit: обучение модели
      test: валидация модели

  """
  def __init__(self,  net, loss_f, metric, 
               optim, 
              learning_rate=1e-3, 
              epoch_amount=10, batch_size=5, 
              max_batches_per_epoch=None,
              device='cpu', early_stopping=10, 
              scheduler=None, permutate=True):

      self.loss_f = loss_f
      self.learning_rate = learning_rate
      self.epoch_amount = epoch_amount
      self.batch_size = batch_size
      self.max_batches_per_epoch = max_batches_per_epoch
      self.device = device
      self.early_stopping = early_stopping
      self.optim = optim
      self.scheduler = scheduler
      self.permutate = permutate
      self.start_model = net
      self.best_model = net
      self.metric = metric

      self.best_metric = 0
      # Лучшее значение функции потерь на валидационной выборке
      self.best_val_loss = float('inf')
      # Эпоха, на которой достигалось лучшее 
      # значение функции потерь на валидационной выборке
      self.best_epoch = 0
    
      self.train_loss = []
      self.train_time = []
      self.train_metric = []
      self.val_loss = []
      self.val_time = []
      self.val_metric = []

  def test(self, model, test):
    model.eval()
    
    mean_loss = 0
    mean_metric = 0
    batch_n = 0
    
    with torch.no_grad():
        for batch_X, target in test:
          if self.max_batches_per_epoch is not None:
              if batch_n >= self.max_batches_per_epoch:
                  break
          batch_X = batch_X.to(self.device)
          target = target.to(self.device)
          
          Y_pred = model(batch_X)
          
          pred = torch.clamp(Y_pred, min=0, max=1)
          loss = self.loss_f(pred, target)
          mean_loss += float(loss)
          
          Y_pred = (Y_pred > 0).type(torch.uint8)
          mean_metric += self.metric(Y_pred, target).mean().item()
          
          batch_n += 1

        mean_loss /= batch_n
        mean_metric /= batch_n

        return mean_loss, mean_metric
    
  def fit(self, model, train, val):
      optimizer = self.optim   
      if self.scheduler:
          scheduler = self.scheduler

      for epoch in range(self.epoch_amount): 
          start = dt.datetime.now()
          print('* Epoch %d/%d' % (epoch+1, self.epoch_amount))
          model.train()
          mean_loss = 0
          mean_metric = 0
          batch_n = 0

          for batch_X, target in train:
              if self.max_batches_per_epoch is not None:
                  if batch_n >= self.max_batches_per_epoch:
                      break
              optimizer.zero_grad()
              
              batch_X = batch_X.to(self.device)
              target = target.to(self.device)

              Y_pred = model(batch_X)
  
              pred = torch.clamp(Y_pred, min=0, max=1)
              loss = self.loss_f(pred, target)
              mean_loss += float(loss)

              Y_pred = (Y_pred > 0).type(torch.uint8)
              mean_metric += self.metric(Y_pred, 
                                         target).mean().item()
            
              loss.backward()
              optimizer.step()

              batch_n += 1

          mean_loss /= batch_n
          mean_metric /= batch_n
          time = dt.datetime.now() - start
          self.train_time.append(time)  
          self.train_loss.append(mean_loss)
          self.train_metric.append(mean_metric)
          print(f'Loss_train: {mean_loss}, {time} сек')
      
          mean_loss_val, mean_metric_val = self.test(model, val)
          time = dt.datetime.now() - start
          self.val_time.append(time)
          self.val_loss.append(mean_loss_val)
          self.val_metric.append(mean_metric_val)
          print(f'Loss_val: {mean_loss_val}, {time} сек')

          if mean_metric < self.best_metric:
              self.best_metric = mean_metric
              
          if mean_loss < self.best_val_loss:
              self.best_model = copy.deepcopy(model)
              self.best_val_loss = mean_loss
              self.best_epoch = epoch
          elif epoch - self.best_epoch > self.early_stopping:
              print(f'{self.early_stopping} без улучшений. Прекращаем обучение...')
              break
          if self.scheduler is not None:
              scheduler.step()
          print()
      torch.cuda.empty_cache()
      
  def plot_loss(self):
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(self.train_loss)),
             self.train_loss, color='orange',
             label='train', linestyle='--')
    plt.plot(range(len(self.val_loss)),
             self.val_loss, color='blue',
             marker='o', label='val')
    plt.legend()
    plt.show()

  def plot_metric(self):
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(self.train_metric)),
             self.train_metric, color='orange',
             label='train', linestyle='--')
    plt.plot(range(len(self.val_metric)),
             self.val_metric, color='blue',
             marker='o', label='val')
    plt.legend()
    plt.show()

  def make_report(self, path):
    with open(path, 'w') as f:
      for i in range(len(self.train_time)):
        line  = "Epoch: " + str(i+1)
        line += "\nTime: " + str(self.train_time[i])
        line += "\nTrain loss: " + str(self.train_loss[i])
        line += "\nTrain metric: " + str(self.train_metric[i])
        line += "\nVal loss: " + str(self.val_loss[i])
        line += "\nVal metric: " + str(self.val_metric[i])
        line += "\n\n"
        f.write(line)
  
  
