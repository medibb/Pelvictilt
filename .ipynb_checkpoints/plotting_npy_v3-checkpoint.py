import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

################################### 사용법 #######################################
# 본 파일을 실행 시킨 다음 data file 번호와 해당 데이터 파일 내 실험 번호를 순서대로 입력
# 소스코드 수정할 필요 없음
#################################################################################

axis_list = ['axis-x', 'axis-y', 'axis-z']
dataset = ['BYK1_invest1', 'BYK2_invest2',
           'CHE1_invest2', 'CHE2_invest1',
          'KJH1_invest3', 'KJH2_invest1',
          'PJY1_invest2', 'PJY2_invest2']

font = {'family': 'Times New Roman',
      'color':  'black',
      'weight': 'bold',
      'size': 12,
      'alpha': 0.7}

def inflection_detection(data):
    dd = gaussian_filter1d(data, 1)
    ext_dd = np.gradient(dd)
    infls = np.where(np.diff(np.sign(ext_dd)))[0]
    return infls

while 1:
    print('=== Data File List ===')
    print(' No | Data file name')
    for i, data in enumerate(dataset):
        print('%3d | %s' % (i, data))
    data_idx = int(input('Choose a data file (0-7): '))
    data   = np.load('./NPY_data/' + dataset[data_idx] + '_data.npy')
    label  = np.load('./NPY_data/' + dataset[data_idx] + '_label.npy', allow_pickle=True)    
    depth  = np.load('./NPY_data/' + dataset[data_idx] + '_depth.npy')        
    exp_No = int(input('Choose an experiment number (0-%d): ' % (len(label)-1)))
    
    for j in range(3):
        plt.figure()
        plt.plot(data[exp_No, 0, j], 'r', label='sensor1', )
        plt.plot(data[exp_No, 1, j], 'g', label='sensor2', )
        plt.plot(data[exp_No, 2, j], 'b', label='sensor3', )
        plt.plot(data[exp_No, 3, j], 'y', label='sensor4', )
        plt.axvline(x=label[exp_No], color='r', linestyle='--', linewidth=2)

        if j == 2:
            inf = inflection_detection(data[exp_No, 1, j])

            start = np.where(inf < label[exp_No])[0][-1]
            finish = np.where(inf >= label[exp_No])[0][0]

            PPA = data[exp_No, 1, j, inf[finish]] - data[exp_No, 1, j, inf[start]]

            plt.text(inf[start], data[exp_No, 1, j, inf[start]], 'Start : ' + np.str(data[exp_No, 1, j, inf[start]]), fontdict=font)
            plt.text(inf[finish], data[exp_No, 1, j, inf[finish]], 'Finish : ' + np.str(data[exp_No, 1, j, inf[finish]]), fontdict=font)
            plt.text(label[exp_No]+1, 80, 'Posterior Pelvic Angle : ' + str(PPA))
            
            #start 지점에서 지면방향으로 내려간 거리
            arrowprops= dict(color='green', arrowstyle="<|-|>", alpha=0.5)
            min_idx1 = np.argmin(data[exp_No, 2, j])
            plt.hlines(y=data[exp_No, 2, j, min_idx1], xmin=0, xmax=min_idx1, linestyle='--', color='green')
            plt.annotate("",  xy=(inf[start], data[exp_No, 2, j, min_idx1]), xytext=(inf[start], data[exp_No, 1, j, inf[start]]), arrowprops = arrowprops)
            plt.text(inf[start]+2, (data[exp_No, 2, j, min_idx1]+data[exp_No, 1, j, inf[start]])/2, 'Squat Depth\n%f' % (depth[exp_No, 0, 0]-depth[exp_No, 0, min_idx1]))
            for k in inf:
                plt.scatter(k, data[exp_No, 1, j, k],  c='g', s=50)

        title = dataset[data_idx] + ', Experiment No.' + str(exp_No) +', ' + axis_list[j]
        plt.title(title)
        plt.legend(loc='upper right')

        if j == 2:
            plt.figure()
            
            KA = data[exp_No, 2, j] + data[exp_No, 3, j]
            LL = 180 - data[exp_No, 0, j] + data[exp_No, 1, j]            
            LL2 = gaussian_filter1d(LL, 1)
            plt.plot(KA, label='KA')            
            plt.plot(LL, label='LL')
            plt.plot(LL2, label='Smoothed LL')
            
            #스쿼트 깊이 - 시작지점에서 풀스쿼트 지점
            arrowprops= dict(color='blue', arrowstyle="<|-|>", alpha=0.5)
            min_idx2 = np.argmin(KA)            
            plt.hlines(y=KA[min_idx2], xmin=0, xmax=min_idx, linestyle='--')
            plt.annotate("",  xy=(0, KA[0]), xytext=(0, KA[min_idx2]), arrowprops = arrowprops)
            plt.text(2, (KA[0]+KA[min_idx2])/2, 'Squat Depth\n%f' % (depth[exp_No, 0, 0]-depth[exp_No, 0, min_idx2]))
            
            #LL time for180 to 180 
            ll_idx = np.where(LL2>180)[0]                        
            arrowprops= dict(color='red', arrowstyle="<|-|>", alpha=0.5)
            plt.annotate("",  ha='center', va='bottom', xy=(ll_idx[-1], LL2[ll_idx[-1]]), xytext=(ll_idx[0], LL2[ll_idx[0]]), arrowprops = arrowprops)
            plt.text((ll_idx[-1]+ll_idx[0])/2-15, 175, 'time for 180 to 180\n%.3f sec' % ((ll_idx[-1]-ll_idx[0])*0.05))
            
            title = dataset[data_idx] + ', Experiment No.' + np.str(exp_No) + ', ' + axis_list[j] + ', KA & LL'
            plt.title(title)
            plt.legend(loc='upper right')

    plt.show()