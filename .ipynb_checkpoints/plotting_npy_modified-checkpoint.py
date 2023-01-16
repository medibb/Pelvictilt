import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

################################### 사용법 #######################################
# 본 파일을 실행 시킨 다음 data file 번호와 해당 데이터 파일 내 실험 번호를 순서대로 입력
#################################################################################

axis_list = ['axis-x', 'axis-y', 'axis-z']
dataset = ['Total_Squat',
           'JHKim1_invest3', 'JHKim2_invest1',
           'JYPark1_invest2', 'JYPark2_invest2',
           'YKBae1_invest1', 'YKBae2_invest2',
           'HEChoi1_invest2','byk220306_a', 'byk220306_b']

font = {'family': 'Times New Roman',
    'color': 'black',
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
    data = np.load('./PlotData_parameters/data/220306/' + dataset[data_idx] + '_data.npy')
    label = np.load('./PlotData_parameters/data/220306/' + dataset[data_idx] + '_label.npy')
    exp_No = int(input('Choose an experiment number (0-%d): ' % (len(label))))

    for j in range(3):
        plt.figure()
        plt.plot(data[exp_No, 0, j], 'r', label='sensor1', )
        plt.plot(data[exp_No, 1, j], 'g', label='sensor2', )
        plt.plot(data[exp_No, 2, j], 'b', label='sensor3', )
        plt.plot(data[exp_No, 3, j], 'y', label='sensor4', )
        plt.axvline(x=label[exp_No], color='r', linestyle='--', linewidth=2)

        if j == 2:  # z축
            inf = inflection_detection(data[exp_No, 1, j])  #sacrum sensor
            
            start = np.where(inf < label[exp_No])[0][-1]
            finish = np.where(inf >= label[exp_No])[0][-0]

            PPA = data[exp_No, 1, j, inf[finish]] - data[exp_No, 1, j, inf[start]]

            plt.text(inf[start], data[exp_No, 1, j, inf[start]], 'Start : ' + np.str(data[exp_No, 1, j, inf[start]]), fontdict=font)
            plt.text(inf[finish], data[exp_No, 1, j, inf[finish]], 'Finish : ' + np.str(data[exp_No, 1, j, inf[finish]]), fontdict=font)
            plt.text(label[exp_No]+1, 80, 'Posterior Pelvic Angle : ' + str(PPA))
            for k in inf:
                plt.scatter(k, data[exp_No, 1, j, k],  c='g', s=50)


        title = dataset[data_idx] + ', Experiment No.' + str(exp_No) + ', ' + axis_list[j]
        plt.title(title)
        plt.legend(loc='upper right')

        if j == 2:
            plt.figure()
            KA = data[exp_No, 2, j] + data[exp_No, 3, j]
            LL = 180 - data[exp_No, 0, j] + data[exp_No, 1, j]
            over180 = np.where((LL - 180) > 0)[0]
            under180 = np.where((LL - 180) < 0)[0]
            for i, idx in enumerate(over180):
                if idx + 10 == over180[i + 10]:
                    s_180 = over180[i]
                    e_180 = under180[np.where(over180[i] < under180)][0]
                    break
            plt.text(s_180, LL[s_180], str(s_180), fontdict=font)
            plt.scatter(s_180, LL[s_180], c='g', s=50)
            plt.text(e_180, LL[e_180], str(e_180), fontdict=font)
            plt.scatter(e_180, LL[e_180], c='g', s=50)
            plt.plot(KA, label='KA')
            plt.plot(LL, label='LL')
            title = dataset[data_idx] + ', Experiment No.' + str(exp_No) + ', ' + axis_list[j] + ', KA & LL'
            plt.title(title)
            plt.legend(loc='upper right')

    plt.show()
    PPA = "%0.2f" % PPA
    print('inflection points of pelvis:', inf)
    #print('Over 180 degrees in LL:', over180)

    print(label[exp_No], inf[start], inf[finish], over180[0], over180[-1], PPA)


