import torch
import numpy as np
import matplotlib.pyplot as plt
import params
import logging
import model,model1,model2,model3,model4,model5,model6
from load_data import data_loader

logger = logging.getLogger(__name__)

def train_model(start_epochs, n_epochs, valid_loss_min_input, training_loader, validation_loader, model,best_model_path):

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info(
        'batch_size: {}  lr: {} weight_decay: {}'.format(params.TRAIN_BATCH_SIZE, params.learning_rate,params.weight_decay))

    Train_Loss_list = []
    Valid_Loss_list = []
    print_loss_total = 0
    epoch_loss_total = 0
    loss2_total=0

    steps_per_epoch = len(training_loader)

    print_every = len(training_loader) // params.printEV + 1


    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

    valid_loss_min = valid_loss_min_input
    count = 0
    countless=0
    for epoch in range(start_epochs-1, n_epochs):
        model.train()
        logger.info('Epoch {}: Training Start'.format(epoch))
        for batch_idx, data in enumerate(training_loader):

            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            optimizer.zero_grad()
            clspre,outputs = model(ids, mask, token_type_ids)
            loss1 = loss_fn(outputs, targets)
            loss2= loss_fn(clspre, targets)
            loss=0.8*loss1+0.2*loss2

            loss.backward()
            optimizer.step()

            print_loss_total += loss1.item()
            epoch_loss_total += loss1.item()
            loss2_total+=loss2.item()

            if batch_idx % print_every == 0:

                print_loss_avg = print_loss_total / print_every
                l2=loss2_total/print_every
                print_loss_total = 0
                loss2_total=0

                log_msg = 'Epoch: %d, Iteration: %.2f%%, Loss: %.6f, Loss2: %.6f,BATCH:%d' % (
                    epoch, batch_idx / len(training_loader), print_loss_avg
                    ,l2
                    ,batch_idx
                )
                logger.info(log_msg)

            del data,batch_idx
        epoch_loss_avg = epoch_loss_total / len(training_loader)
        Train_Loss_list.append(epoch_loss_avg)
        epoch_loss_total = 0
        log_msg = 'Finished epoch %d: Train loss: %.6f' % (epoch, epoch_loss_avg)

        logger.info(log_msg)
        logging.info('Epoch {}: Validation Start'.format(epoch))

        if params.do_eval:
            model.eval()
            eval_loss_total = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(validation_loader):
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.float)

                    clspre,outputs = model(ids, mask, token_type_ids)

                    loss = loss_fn(outputs, targets)
                    eval_loss_total += loss.item()

            eval_loss_avg = eval_loss_total / max(1, len(validation_loader))
            Valid_Loss_list.append(eval_loss_total)
            log_msg += ', Dev loss: %.6f' % (eval_loss_avg)
            logger.info(log_msg)
            model.train()
            # save the model

            if params.do_save:
                if eval_loss_avg <= valid_loss_min:
                    logger.info('Validation loss decreased from {:.6f} to {:.6f}). Saving model'.format(valid_loss_min,eval_loss_avg))
                    valid_loss_min = eval_loss_avg
                    # count = 0
                    count = count + 1
                    torch.save(model.state_dict(), best_model_path + str(count))
                else:
                    count = count + 1
                    countless=countless+1
                    torch.save(model.state_dict(), best_model_path + str(count))
                    if countless > 3:
                        print("三次验证集损失不下降，停止训练")
                        print("min_loss:",valid_loss_min)
                        return
    plt.grid(True)
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.plot(Train_Loss_list, 'o-', label="TrainLoss")
    plt.legend()
    plt.savefig('img/trainLoss'+params.modelName_train+'.jpg')

    plt.plot(Valid_Loss_list, 'o-', label="ValidLoss")
    plt.legend()
    plt.savefig('img/validLoss'+params.modelName_train+'.jpg')

if __name__ == '__main__':

    print('model_train' + params.trainDataPath)
    print('model_train' + params.valDataPath)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available, '
              f'We will use the GPU: {torch.cuda.get_device_name(0)}.')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    # 创建模型
    allTag = list()
    with open(params.tagListPath, encoding='utf-8', mode='r') as tagData:
        for line in tagData:
            allTag.append(line.strip())
    meshNum=len(allTag)
    
    model = model6.BERTClass(meshNum,allTag)

    model.to(device)

    best_model = '../save_model/' + params.modelName_train + '.pt'
    logger.critical('EPOCH' + str(params.num_epochs))
    logger.critical('model_train标签数据路径' + params.tagListPath)
    logger.critical('model_train标签数量' + str(meshNum))
    logger.critical('model_train训练数据路径' + params.trainDataPath)
    logger.critical('model_train验证数据路径' + params.valDataPath)
    logger.critical('model_train模型保存路径' + best_model)
    
    training_loader, validation_loader = data_loader(params.trainDataPath, params.valDataPath, meshNum)


    train_model(1, params.num_epochs, np.Inf, training_loader, validation_loader, model, best_model)
                                
    print('model_train标签数据路径' + params.tagListPath)
    print('model_train标签数量'+str(meshNum))
    print('model_train训练数据路径' + params.trainDataPath)
    print('model_train验证数据路径' + params.valDataPath)
    print('model_train模型保存路径' +best_model)
