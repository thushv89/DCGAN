__author__ = 'Thushan Ganegedara'
import numpy as np
import matplotlib.pyplot as plt

linecount = 5
# didnt use all the weight indices (100) because some only have 64
indices_to_plot = np.random.permutation(np.arange(50))[:linecount]

reading_gen = False
reading_disc = False

gen_data,disc_data = {},{}

with open('grad_logger_test','r') as f:
    for line in f.readlines():
        if line.startswith('#Gen'):
            reading_gen,reading_disc = True,False
        elif line.startswith('#Disc'):
            reading_gen,reading_disc = False,True

        if reading_gen and not reading_disc and not line.startswith('#'):
            line_tokens = line.rstrip().split(',')
            arr = np.array([float(val) for val in line_tokens[1:] if len(val)>0]).reshape(1,-1)
            if line_tokens[0] not in gen_data:
                if arr.size>linecount:
                    gen_data[line_tokens[0]] = arr[0,indices_to_plot].reshape(1,-1)
                else:
                    gen_data[line_tokens[0]] = arr.reshape(1,-1)
            else:
                if arr.size>linecount:
                    gen_data[line_tokens[0]] = np.append(gen_data[line_tokens[0]],arr[0,indices_to_plot].reshape(1,-1),axis=0)
                else:
                    gen_data[line_tokens[0]] = np.append(gen_data[line_tokens[0]],arr,axis=0)

        if not reading_gen and reading_disc and not line.startswith('#'):
            line_tokens = line.rstrip().split(',')
            arr = np.array([float(val) for val in line_tokens[1:] if len(val)>0]).reshape(1,-1)
            if line_tokens[0] not in disc_data:
                if arr.size>linecount:
                    disc_data[line_tokens[0]] = arr[0,indices_to_plot].reshape(1,-1)
                else:
                    disc_data[line_tokens[0]] = arr.reshape(1,-1)
            else:
                if arr.size>linecount:
                    disc_data[line_tokens[0]] = np.append(disc_data[line_tokens[0]],arr[0,indices_to_plot].reshape(1,-1),axis=0)
                else:
                    disc_data[line_tokens[0]] = np.append(disc_data[line_tokens[0]],arr,axis=0)

print('Reading the log file completed')

genGradStat,discGradStat = {},{}
with open('gradStat_logger_test','r') as f:
    for line in f.readlines():
        if line.startswith('Gen'):
            line_tokens = line.rstrip().split(',')
            line_tokens = line_tokens[1:]
            arr = np.array([float(val) for val in line_tokens[1:] if len(val)>0]).reshape(1,-1)

            if line_tokens[0] not in genGradStat:
                genGradStat[line_tokens[0]] = arr.reshape(1,-1)
            else:
                genGradStat[line_tokens[0]] = np.append(genGradStat[line_tokens[0]],arr,axis=0)

        elif line.startswith('Disc'):
            line_tokens = line.rstrip().split(',')
            line_tokens = line_tokens[1:]
            arr = np.array([float(val) for val in line_tokens[1:] if len(val)>0]).reshape(1,-1)
            if line_tokens[0] not in discGradStat:
                discGradStat[line_tokens[0]] = arr.reshape(1,-1)
            else:
                discGradStat[line_tokens[0]] = np.append(discGradStat[line_tokens[0]],arr,axis=0)

plt.figure(1)

index = 1
for k,v in gen_data.items():
    x_axis = np.arange(v.shape[0])
    #str_subplot = (4,4,index)
    plt.subplot(4,4,index)
    print(v.shape)
    for i in range(v.shape[1]):

        if v.shape[1]==linecount:
            plt.plot(x_axis,v[:,i],label='index_'+str(indices_to_plot[i]))
        else:
            plt.plot(x_axis,v[:,i],label='index_'+str(i))
    #plt.xlabel('Position in the Dataset')
    plt.title('Generator ('+k+')')

    legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
    index += 1

plt.figure(2)

index = 1
for k,v in disc_data.items():
    x_axis = np.arange(v.shape[0])
    #str_subplot = (4,4,index)
    plt.subplot(4,4,index)
    print(v.shape)
    for i in range(v.shape[1]):

        if v.shape[1]==linecount:
            plt.plot(x_axis,v[:,i],label='index_'+str(indices_to_plot[i]))
        else:
            plt.plot(x_axis,v[:,i],label='index_'+str(i))
    #plt.xlabel('Position in the Dataset')
    plt.title('Discriminator ('+k+')')

    legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
    index += 1

plt.figure(3)

index = 1

for k,v in genGradStat.items():
    x_axis = np.arange(v.shape[0])
    #str_subplot = (4,4,index)
    plt.subplot(4,4,index)
    plt.plot(x_axis,v[:,4],label=k)
    #plt.xlabel('Position in the Dataset')
    plt.title('Generator ('+k+')')
    legend = plt.legend(loc='lower left', shadow=False, fontsize='small')
    index += 1

plt.show()
