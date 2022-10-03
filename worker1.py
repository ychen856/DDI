
import torch
import sys
import socket
import pickle
import os
from network import MobileNetV2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
if __name__ == "__main__":
    imageAmount = int(sys.argv[1])

    SocketToMaster = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("wait con")
    SocketToMaster.connect(('10.7.48.14', 5038))  # connect to worker 1
    #SocketToMaster.connect(('10.7.48.142', 5038))  # connect to worker 1
    print("conn")

    model = MobileNetV2(1000)
    model.load_state_dict(torch.load("imagenetmobilev2.pt"))                  #load model
    model.cuda()
    model.eval()
    startindex = 1                                         # initial start layer and end layer index
    endindex = 20

    #set layer assign update itreation
    while (1):
        newxlen = SocketToMaster.recv(4)
        size = int.from_bytes(newxlen, byteorder='big')
        totalSize = size

        newx = b''

        while size > 0:  # receive activation vector
            if size > 1024:

                data = SocketToMaster.recv(1024)
                size = size - 1024
                newx += data
            else:
                data = SocketToMaster.recv(size)
                newx += data
                break

        inputs = pickle.loads(newx)



        if inputs == 'end':
            break;

        inputs.cuda()
        outputs = getattr(model, 'forward' + '1')(inputs)
        for i in range(2, endindex + 1):  # process assign layer
            index = i
            index = str(index)
            outputs = getattr(model, 'forward' + index)(outputs)

        _, predicted = torch.max(outputs.data, 1)

        predicted = predicted.detach().cpu().numpy()[0]
        #print('w1 out: ', predicted)

        newx = pickle.dumps(predicted)
        SocketToMaster.send(newx)

    SocketToMaster.close()

