import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_acc(model, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_acc_5 = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()
        test_acc_5 += prec5.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Prec1: {}/{} ({:.2f}%), Prec5: ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader), test_acc_5 / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2), np.round(test_acc_5 / len(test_loader), 2)