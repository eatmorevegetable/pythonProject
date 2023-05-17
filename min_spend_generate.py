import operator
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib


def mincomptime_generate(num_accessed_plane, resource_conf):
    NP = 50  # 种群数量
    Pc = 0.8  # 交叉率
    Plm = 0.1
    Pum = 0.3  # 变异率
    Gmax = 100  # 最大遗传代数
    dr = 2
    I = num_accessed_plane
    J = 6  # 工序数量

    pop_encode_conf = {'NP': NP,'I': I, 'J': J, 'j_inertial_navigation': 6, 'j_transport': 3, 'num_land_station': num_accessed_plane,
                       'num_takeoff_station': num_accessed_plane}
    # 编码参数：飞机和工序数量，惯导、调运工序号和降落起飞站位数量

    duration_operation = resource_conf['duration_operation']
    # 保障工序时间参数表3*6
    #        挂弹1,挂弹2,调运,充氧,加油,惯导
    # 主体时长
    # 准备时间
    # 收尾时间

    num_people_each_type = resource_conf['num_people_each_type']
    # 人员数量[挂弹1、挂弹2、调运、充氧、加油]
    num_equipment_each_type = resource_conf['num_equipment_each_type']
    # 设备数量[调运、充氧、加油]
    pop_decode_conf = {'NP': NP, 'I': I, 'J': J,
                       'num_equipment_types': len(num_equipment_each_type), 'num_people_types': len(num_people_each_type),
                       'max_devices_same_type': max(num_equipment_each_type), 'max_people_same_type': max(num_people_each_type),
                       'duration_operation': duration_operation,
                       'num_people_each_type': num_people_each_type,'num_equipment_each_type':num_equipment_each_type}
    #设备种类数、人员种类数、同种最大设备数、同种最大人员数、保障时间参数表、每个工序的人员数量

    SpaCon = [[],
              [],
              [],
              [], [], [], [], [], [],  # 9
              [], [], []]  # 第0维index为i，spacon[i]表示对i站位有空间约束的最小站位集合的集合
    load_time = [0, 0, 0, 0, 0, 0]  # 飞机的降落时间
    parallel_or_not = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1]
    ])  # 工序之间的是否并行关系
    min_comp_time_conf = {'NP': NP, 'I': I, 'J': J,
                          'mini_station_set_with_spatial_constraint':SpaCon, 'load_time':load_time,
                          'duration_operation': duration_operation, 'parallel_or_not': parallel_or_not}
    calculate_startandend_time_conf = {'I': I, 'J': J,
                                       'num_equipment_types': len(num_equipment_each_type),
                                       'num_people_types': len(num_people_each_type),
                                       'max_devices_same_type': max(num_equipment_each_type),
                                       'max_people_same_type': max(num_people_each_type),
                                       'load_time': load_time, 'duration_operation': duration_operation,
                                       'parallel_or_not': parallel_or_not}





#编码函数，生成种群包含50个个体，每个个体包含工序排列信息,前i*j个三元组为(i,j,0)后i个三元组为(i，h1,h2)
# i：表示舰载机号；  j：表示工序号；  h1降落站位，h2起飞站位；
#约束条件：1.工序5 GD是每个舰载机的最后一道工序
#        2.对于同一JZJ 5GD、2调运、3充氧、4加油互相不能并行，加油和惯导可以并行
#pop_encode_conf = {'I':,'J':,j_inertial_navigation': , 'j_transport': , 'num_land_station': ,'num_takeoff_station': }
def encoder(pop_encode_conf):
    dna = []
    sr = []
    temp = []
    a = []  #dna的中转
    NP = pop_encode_conf['NP']
    I = pop_encode_conf['I']
    J = pop_encode_conf['J']
    jl = pop_encode_conf['j_inertial_navigation']
    jt = pop_encode_conf['j_transport']
    h1 = pop_encode_conf['num_land_station']
    h2 = pop_encode_conf['num_takeoff_station']
    for n in range(NP):  #生成50个种群
        for i in range(1, I + 1):
            for j in range(1, J):
                temp.append([i, j, 0])
            random.shuffle(temp)
            temp.append([i, jl, 0])
            sr.append(temp)        #每个舰载机的打乱的工序排列
            temp = []
        I_gat = [i for i in range(6)]
        while I_gat != []:
            cho = np.random.choice(I_gat)
            dna.append(sr[cho][0])
            del sr[cho][0]
            if sr[cho] == []:I_gat.remove(cho)
        sr = []
        l = np.random.choice([x for x in range(1, h1 + 1)], I, replace=False)  # 1~6中取6个
        f = np.random.choice([x for x in range(h1 + 1, h2 + h1 + 1)], I, replace=False)  # 7~12中取6个
        jtindex = []
        for m in range(I*J):
            i = dna[m][0] - 1
            #j = dna[m][1] - 1
            if m > dna.index([i + 1, jt, 0]):
                dna[m][2] = f[i]
            elif m < dna.index([i + 1, jt, 0]):
                dna[m][2] = l[i]
            elif m == dna.index([i + 1, jt, 0]):
                jtindex.append(m)
        for x in jtindex:
            dna[x][2] = l[dna[x][0]-1]
        a.append(dna)            #加入种群
        dna = []
    pop = np.array(a)
    return pop

#解码思路：先比较工序(i,j)与(i,jt)的前后关系，在jt前则在降落站位，在jt后则在起飞站位
#有四类保障人员KP1,2,3,4每类保障人员定义足够多的组数：6
#pop_decode_conf = {'NP': NP, 'I': I, 'J': J, 'num_equipment_types': 3, 'num_people_types': 5,
#                       'devices_same_type': 6, 'people_same_type': 6,'duration_operation': ,
#                        'num_people_each_type':,'num_equipment_each_type': }
def translate_dna(pop,pop_decode_conf):
    M = 250
    K = np.array([[1, 0], [2, 0], [3, 1], [4, 2], [5, 3], [0, 0]])
    # 5步工序所需要的人员和设备保障种类，每组第一个是KP人员，第二个是KR设备，0代表没有使用
    NP = pop_decode_conf['NP']
    I = pop_decode_conf['I']
    J = pop_decode_conf['J']
    krn = pop_decode_conf['num_equipment_types']
    kpn = pop_decode_conf['num_people_types']
    mkr = pop_decode_conf['max_devices_same_type']
    mkp = pop_decode_conf['max_people_same_type']
    duration_operation = pop_decode_conf['duration_operation']
    num_people = pop_decode_conf['num_people_each_type']
    num_equipment = pop_decode_conf['num_equipment_each_type']

    #创建各工序的人员分布表
    distribution_operation = np.zeros((kpn, mkp), dtype=int)
    for per_type_people in range(kpn):
        for i in range(num_people[per_type_people],mkp):
            distribution_operation[per_type_people][i] = M

    #创建各个飞机每个工序的总消耗时长(主时长+准备和收尾时长)
    #j\i 飞机1 飞机2 ... 飞机I
    #工序1
    #工序2
    #...
    #工序J
    Spt = np.zeros((J, I), dtype=int)
    for j in range(J):
        for i in range(I):
            Spt[j][i] = duration_operation['main_time'][j] + \
                        duration_operation['preparation_time'][j] + duration_operation['wend_up_time'][j]

    # 创建一个五维数组来存放50个种群的Xpijkl 50*6*6*5*6 5是k类保障人员，6是每类人数
    xp = np.zeros((NP, I, J, kpn, mkp), dtype=int)
    # 生成决策变量Xpijkl
    for n in range(NP):
        # kp_cum表示第k类第l个保障人员累计作业时间 挂弹1 挂弹2 调运 充氧 加油
        kp_cum = distribution_operation
        for m in range(I*J):
            i = pop[n][m][0]-1
            j = pop[n][m][1]-1
            k = K[j][0]   #通过工序号查找需要的人员种类1-5
            if k != 0:
                l = np.argmin(kp_cum[k - 1])  # 累计作业时间最少优先
                xp[n][i][j][k - 1][l] = 1
                kp_cum[k - 1][l] += Spt[j][i]


    xr = np.zeros((NP, I, J, krn, mkr), dtype=int)  #生成一个五维数组来存放Xrijkl 50*6*6*3*6
    #有3类保障设备，每类保障设备分布为4,6,6(调运,充氧,加油)
    #先算出每道工序的站位号根据站位号分配设备号

    #创建站位的设备覆盖表,下面的加油站覆盖可以再去耦合改进一下
    coverage_fuel = [[1, 2], [1, 2], [1, 2], [1, 2], [3], [3], [4], [4], [4], [5, 6], [5, 6], [5, 6]]
    Cbea = []
    for i in range(2*I):
        Cbea.append([])
        Cbea[i].append([n for n in range(1,num_equipment[0]+1)])
        Cbea[i].append([n for n in range(1,num_equipment[1]+1)])
        Cbea[i].append(coverage_fuel[i])

    for n in range(NP):
        kr_cum = np.zeros((krn, mkr), dtype=int)  # 定义一个3*6的矩阵来表示第k类第l个保障设备累计作业时间
        for m in range(I*J):
            i = pop[n][m][0]-1 #0~5
            j = pop[n][m][1]-1 #0~5
            h = pop[n][m][2]-1   #0~11
            k = K[j][1]   #通过工序号查找需要的设备种类k:1~3
            if k > 0 :
                cum_quetime = M
                if len(Cbea[h][k - 1]) == 1:  # 如果k类设备覆盖舰载机i的只有一台设备，则选择该设备号
                    l = Cbea[h][k - 1][0] - 1
                else:
                    for o in range(len(Cbea[h][k - 1])):  # 否则计算可用设备排队时间最短的设备
                        eq_num = Cbea[h][k - 1][o]  # 记录当下可用设备号
                        if kr_cum[k-1][eq_num-1] < cum_quetime:
                            cum_quetime = kr_cum[k - 1][eq_num - 1]
                            l = eq_num - 1
                xr[n][i][j][k - 1][l] = 1
                kr_cum[k - 1][l] += Spt[j][i]

    yp = np.zeros((NP, I, J, I, J), dtype=int)  # 生成一个五维数组来存放Ypijeg 50*6*6*6*6
    for n in range(NP):
        for m1 in range(I*J):
            i = pop[n][m1][0] - 1
            j = pop[n][m1][1] - 1
            for k in range(kpn):
                for l in range(mkp):
                   if xp[n][i][j][k][l] == 1:
                       for m2 in range(m1 + 1,I*J):
                           e = pop[n][m2][0] - 1
                           g = pop[n][m2][1] - 1
                           if xp[n][e][g][k][l] == 1:
                               yp[n][i][j][e][g] = 1

    yr = np.zeros((NP, I, J, I, J), dtype=int)  # 生成一个五维数组来存放Yrijeg 50*6*6*6*6
    for n in range(NP):
        for m1 in range(I * J):
            i = pop[n][m1][0] - 1
            j = pop[n][m1][1] - 1
            for k in range(krn):
                for l in range(mkr):
                    if xr[n][i][j][k][l] == 1:
                        for m2 in range(m1 + 1, I * J):
                            e = pop[n][m2][0] - 1
                            g = pop[n][m2][1] - 1
                            if xr[n][e][g][k][l] == 1:
                                yr[n][i][j][e][g] = 1
    xy = [xp,xr,yp,yr]
    return xy


def earlist_complet_time(pos, popn, ypn, yrn, end_timelist, load_time, duration_operation, parallel_or_not):
    # 计算工序e，g的最短完成时间
    e = popn[pos][0] -1
    g = popn[pos][1] -1

    lastpos = []
    # 求出同一飞机的紧前工序位置
    if pos != 0:
        for m0 in range(pos):
            # 每循环一次在pos为起点的基础上取往前一步的工序的JZJ号
            e1 = popn[pos - m0 - 1][0] - 1
            g1 = popn[pos - m0 - 1][1] - 1
            h = popn[pos][2] - 1
            h1 = popn[pos - m0 - 1][2] - 1
            # 同一飞机但不可并行或者可并行但在不同站位:把它们的基因位都加入到lastpos中
            if e1 == e and (parallel_or_not[g][g1] == 0 or h != h1):
                lastpos.append(pos - m0 - 1)

    # 计算工序（e，g）的紧前工序，所有紧前工序完成的时间加上工序保障时间为该工序的最早完成时间
    # 紧前工序的完成时间
    prior_time = 0
    for m1 in range(pos):
        i = popn[pos - m1 - 1][0] - 1
        j = popn[pos - m1 - 1][1] - 1
        # 如果工序(i,j)和(e,g)分配在同一机组且先于(e,g)保障，
        # 并且(i,j)的最早完成时间大于紧前工序完成时间，将更新紧前工序完成时间为(i,j)的完成时间
        if ypn[i][j][e][g] == 1 :
            prior_time = max(end_timelist[pos - m1 - 1], prior_time)
        # elif ypn[i][j][e][g] == 1 and end_timelist[pos - m1 - 1] == 0 and \
        #     earlist_complet_time(pos - m1 - 1, popn, ypn, yrn,end_timelist, load_time, duration_operation, parallel_or_not) > prior_time:
        #     prior_time = earlist_complet_time(pos - m1 - 1, popn, ypn, yrn,end_timelist, load_time, duration_operation, parallel_or_not)
        # 如果工序(i,j)和(e,g)分配在同一设备且先于(e,g)保障，
        # 并且(i,j)的最早完成时间大于紧前工序完成时间，将更新紧前工序完成时间为(i,j)的完成时间
        if yrn[i][j][e][g] == 1 :
            prior_time = max(end_timelist[pos - m1 - 1], prior_time)
        # elif yrn[i][j][e][g] == 1 and end_timelist[pos - m1 - 1] == 0 and \
        #     earlist_complet_time(pos - m1 - 1, popn, ypn, yrn,end_timelist, load_time, duration_operation, parallel_or_not) > prior_time:
        #     prior_time = earlist_complet_time(pos - m1 - 1, popn, ypn, yrn,end_timelist, load_time, duration_operation, parallel_or_not)

    # 若存在同一飞机不可并行且排在(e,g)前的工序
    if lastpos != []:
        for x in lastpos:
            prior_time = max(end_timelist[x] - duration_operation['wend_up_time'][popn[x][1] - 1] - duration_operation['preparation_time'][popn[pos][1] - 1],prior_time)
            # elif end_timelist[x] == 0 and earlist_complet_time(x, popn, ypn, yrn,end_timelist, load_time, duration_operation, parallel_or_not) \
            #         - duration_operation['wend_up_time'][popn[x][1] - 1] - duration_operation['preparation_time'][popn[pos][1] - 1] > prior_time:
            #     prior_time = earlist_complet_time(x, popn, ypn, yrn,end_timelist, load_time, duration_operation, parallel_or_not) - \
            #                  duration_operation['wend_up_time'][popn[x][1] - 1] - duration_operation['preparation_time'][popn[pos][1] - 1]
    if prior_time == 0:prior_time = load_time[e]

    # print(1)
    return (prior_time + duration_operation['main_time'][g]+duration_operation['preparation_time'][g]+duration_operation['wend_up_time'][g])


#min_comp_time_conf = {'NP': 50, 'I': 6, 'J': 6,
#                          'mini_station_set_with_spatial_constraint': SpaCon, 'load_time': load_time,
#                          'duration_operation': duration_operation, 'parallel_or_not': parallel_or_not}
def Min_comp_time(pop,xy,min_comp_time_conf):
    # 通过决策变量计算最后一道工序的最短完成时间
    M = 250
    NP = min_comp_time_conf['NP']
    I = min_comp_time_conf['I']
    J = min_comp_time_conf['J']
    SpaCon = min_comp_time_conf['mini_station_set_with_spatial_constraint']
    duration_operation = min_comp_time_conf['duration_operation']
    f = [0 for i in range(NP)]
    for n in range(NP):
        # 当ifspacon为0时表示违反了空间约束
        ifspacon = 1
        # 顺序遍历工序，对于一个(i,j,h)
        ifocc = np.zeros(2*I,dtype=int)
        for m in range(I*J):
            h = pop[n][m][2] - 1
            if SpaCon[h] != []:
                for x in SpaCon[h]:
                    jud = 1
                    for hx in x:
                        if ifocc[hx] == 0: jud = 0
                    if jud == 1:
                        ifspacon = 0
                        break
            if ifspacon == 0: break
            ifocc[h] = 1
        if ifspacon == 0: f[n] = M
        if ifspacon == 1:
            min_compl_t = 0
            # 用来缓存每个工序的结束时间
            endtlist = [0 for i in range(I * J)]
            for m in range(I*J):
                egt = earlist_complet_time(m, pop[n], xy[2][n], xy[3][n], endtlist,
                                           min_comp_time_conf['load_time'], duration_operation,
                                           min_comp_time_conf['parallel_or_not'])
                endtlist[m] = egt
                min_compl_t = max(egt,min_compl_t)
            f[n] = min_compl_t

    return f

def get_fitness(np_f,NP):
    # 计算适应度函数
    p_order = []
    # 选择压力
    secret_p = 1.5
    torch_f = torch.tensor(np_f)
    sorted, indices = torch.sort(torch_f, dim=0, descending=True)
    sorted = sorted.numpy()
    for i in range(len(np_f)):
        for j in range(len(sorted)):
            if np_f[i] == sorted[j]:
                p_order.append(j+1)
                break

    fitness = [2 - secret_p + (2 * (p - 1) * (secret_p - 1)) / (NP - 1) for p in p_order]
    return fitness

def choose(pop, fitness, NP):
    # 计算每个个体被选中的概率
    s = sum(fitness)
    p = [fitness[i]/s for i in range(len(fitness))]
    chosen = []
    # 通过赌盘法选择NP个染色体
    for i in range(NP):
        cum = 0
        m = random.random()
        for j in range(NP):
            cum += p[j]
            if cum >= m:
                chosen.append(pop[j])
                break
    chosen = np.array(chosen)
    return chosen

def Crossover(pop, I, J, Pc, jt, jl):
    crossed = np.array([])
    for i in range(0,len(pop),2):
        a = pop[i]
        b = pop[i+1]
        if random.random() < Pc:
            pos = random.randint(1,len(pop[i]) - 1)
            # print("交叉点是"+str(pos))
            temp1 = b[:pos]
            temp2 = a[:pos]
            temp = b[pos:]
            for j in range(len(a)):
                for k in range(len(temp)):
                    if a[j][0] == temp[k][0] and a[j][1] == temp[k][1]:
                        temp1 = np.concatenate((temp1, [temp[k]]))
                        break
            temp = a[pos:]
            for j in range(len(b)):
                for k in range(len(temp)):
                    if b[j][0] == temp[k][0] and b[j][1] == temp[k][1]:
                        temp2 = np.concatenate((temp2, [temp[k]]))
                        break
            a = temp1
            b = temp2
            # 将交叉后的站位按照调运前是降落站位，调运后是起飞站位的原则重新安排
            l = [0 for x in range(I)]
            f = [0 for x in range(I)]
            jtpos = [0 for x in range(I)]
            for m in range(I * J):
                i1 = a[m][0] - 1
                j1 = a[m][1] - 1
                if j1 == jt - 1:
                    l[i1] = a[m][2]
                    jtpos[i1] = m
                if j1 == jl - 1: f[i1] = a[m][2]
            for m in range(I * J):
                i2 = a[m][0] - 1
                if m > jtpos[i2]:
                    a[m][2] = f[i2]
                elif m < jtpos[i2]:
                    a[m][2] = l[i2]
            l = [0 for x in range(I)]
            f = [0 for x in range(I)]
            jtpos = [0 for x in range(I)]
            for m in range(I * J):
                i1 = b[m][0] - 1
                j1 = b[m][1] - 1
                if j1 == jt - 1:
                    l[i1] = b[m][2]
                    jtpos[i1] = m
                if j1 == jl - 1: f[i1] = b[m][2]
            for m in range(I * J):
                i2 = b[m][0] - 1
                if m > jtpos[i2]:
                    b[m][2] = f[i2]
                elif m < jtpos[i2]:
                    b[m][2] = l[i2]

        if i == 0: crossed = np.concatenate(([a], [b]))
        else: crossed = np.concatenate((crossed, [a], [b]))
    return crossed

def Variation(pop, min_time, D_max, Pum, Plm, NP, I, J):
    D_G = np.var(min_time)
    pm = Pum - (Pum - Plm) * D_G / D_max
    for n in range(NP):
        a = pop[n]
        if random.random() < pm:
            i = np.random.choice([x for x in range(1, I+1)], 1, replace=False)
            # 记录飞机i各工序在a中的位置
            poslist = []
            for m in range(I*J):
                if a[m][0] == i:poslist.append(m)
            chapos = np.random.choice([x for x in range(J-1)], 2, replace=False)
            temp = np.array(a[poslist[chapos[0]]])
            a[poslist[chapos[0]]] = a[poslist[chapos[1]]]
            a[poslist[chapos[1]]] = temp
        if n == 0 :variated = np.array([a])
        else:variated = np.concatenate((variated, [a]))
    return variated, D_G

#best_xy= [xp,xr,yp,yr]
# calculate_startandend_time_conf = {'I': 6, 'J': 6,
#                                        'num_equipment_types': krn,
#                                        'num_people_types': kpn,
#                                        'max_devices_same_type': mkr,
#                                        'max_people_same_type': mkp,
#                                        'load_time': load_time, 'duration_operation': duration_operation,
#                                        'parallel_or_not': parallel_or_not}
def calculate_startandend_time(best_ord,best_xy,calculate_startandend_time_conf):
    # best_ord指最佳的工序排列, best_xp: 6*5*4*6 ; best_xr: 6*5*3*6
    # 数列的每个元素是一个七元组：i,j,start-time,end-time,h,pk-l,rk-l
    I = calculate_startandend_time_conf['I']
    J = calculate_startandend_time_conf['J']
    start_and_end_time_list = []
    end_time_list = [0 for i in range(I * J)]
    for m in range(I*J):
        i = best_ord[m][0] - 1
        j = best_ord[m][1] - 1
        end_time = earlist_complet_time(m, best_ord, best_xy[2], best_xy[3],
                                        end_time_list,calculate_startandend_time_conf['load_time'],
                                        calculate_startandend_time_conf['duration_operation'],
                                        calculate_startandend_time_conf['parallel_or_not'])
        end_time_list[m] = end_time
        start_time = end_time - calculate_startandend_time_conf['duration_operation']['main_time'][j] - \
                     calculate_startandend_time_conf['duration_operation']['preparation_time'][j] - \
                     calculate_startandend_time_conf['duration_operation']['wend_up_time'][j]

        h = best_ord[m][2] - 1
        for k in range(calculate_startandend_time_conf['num_people_types']):
            for l in range(calculate_startandend_time_conf['max_people_same_type']):
                if best_xy[0][i][j][k][l] == 1:
                    people_illustrate = '-p' + str(l+1)
                    break
        for k in range(calculate_startandend_time_conf['num_equipment_types']):
            for l in range(calculate_startandend_time_conf['max_devices_same_type']):
                if best_xy[1][i][j][k][l] == 1:
                    equipment_illustrate = '-r' + str(l+1)
                    break
        start_and_end_time_list.append([i, j, start_time, end_time, h, people_illustrate, equipment_illustrate])
    return start_and_end_time_list

def Draw_gantt(best_setime, load_time, I, J, duration_operation):
    main_time = duration_operation['main_time']
    preparation_time = duration_operation['preparation_time']
    wend_up_time = duration_operation['wend_up_time']
    #best_setime为四维向量组(i,j,starttinme,endtime)i:0~5;j:0~4  1111
    color = ['red','brown','dodgerblue','mediumseagreen','blueviolet','orange', #主体颜色
             'darkred','darkred','blue','green','indigo','orangered',  #边框颜色
             'salmon','rosybrown','lightskyblue','aquamarine','plum','moccasin']  #准备和收尾阶段颜色
    name= ['挂弹1','挂弹2','调运','充氧','加油','惯导']
    #每个飞机有两个时间轴，将有时间重合的工序分开绘制，
    # 设置一个变量dura：I*4来更新每绘制一个工序每个飞机1,2,3,4四条时间轴的持续时间
    dura = [[i,i,i] for i in load_time]
    #绘制甘特图的横纵坐标和标题
    plt.xticks(np.arange(0,9,step = 1.5),[x for x in range(0,60,10)])
    plt.yticks(np.arange(0,9,step = 1.5),[x for x in range(1,I+1)])
    plt.title('甘特图',fontsize=15)
    plt.xlabel("时间/min", fontsize=15)
    plt.ylabel("JZJ号", fontsize=15)
    #画标签
    for j in range(J):
        plt.barh(I-j/2+2,width = 1,height = 0.3,left = 9,color = color[j])
        plt.text(x = 10,y= I-j/2-0.1+2,s = name[j],fontsize = 12)
    #画时间条
    for x in best_setime: #x[0]:i 0~5  x[1]:j 0~5 x[2]:starttime x[3]:endtime
        if x[2] < dura[x[0]][0] and x[2] < dura[x[0]][1] and x[2] < dura[x[0]][2]:
            plt.barh(x[0] * 1.5 + 0.45, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1]],
                     edgecolor=color[x[1] + J])  #总时间
            plt.barh(x[0] * 1.5 + 0.45, preparation_time[x[1]]*1.5 / 10, height=0.3, left=x[2] *1.5/ 10, color=color[x[1] + 2*J],
                     edgecolor=color[x[1] + J])   #准备时间
            plt.barh(x[0] * 1.5 + 0.45, wend_up_time[x[1]]*1.5 / 10, height=0.3, left=(x[3] - wend_up_time[x[1]])*1.5 / 10,
                     color=color[x[1] + 2*J], edgecolor=color[x[1] + J])   #收尾时间
            infmt = 'h:'+ str(x[4]+1) + x[6] + '-ct:' + str(x[2]) + '~'+ str(x[3])
            plt.text(x = (x[2] / 10 + (x[3] - x[2]) / 20 - 0.6)*1.5, y = x[0] * 1.5 + 0.35, s = infmt, fontsize=12)
        else:
            if x[2] >= dura[x[0]][0]:
                plt.barh(x[0] * 1.5 +0.15, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2] *1.5/ 10, color=color[x[1]],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 +0.15, preparation_time[x[1]] *1.5/ 10, height=0.3, left=x[2] *1.5/ 10, color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 +0.15, wend_up_time[x[1]] *1.5/ 10, height=0.3, left=(x[3] - Spt[x[1]][-1]) *1.5/ 10,
                         color=color[x[1] + 2*J], edgecolor=color[x[1] + J])
                infmt = 'h:' + str(x[4]+1) + x[5] + x[6] + '-ct:' + str(x[2])+'~' + str(x[3])
                plt.text(x=(x[2] / 10 + (x[3] - x[2]) / 20 - 0.6)*1.5, y=x[0] * 1.5 +0.05, s=infmt, fontsize=12)
                dura[x[0]][0] = x[3]
            elif x[2] < dura[x[0]][0] and x[2] >= dura[x[0]][1]:
                plt.barh(x[0] * 1.5 - 0.15, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1]],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.15, preparation_time[x[1]] *1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.15, wend_up_time[x[1]] *1.5 / 10, height=0.3, left=(x[3] - Spt[x[1]][-1])*1.5 / 10,
                         color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                infmt =  'h:' + str(x[4]+1) + x[5] + x[6] + '-ct:' + str(x[2])+'~' + str(x[3])
                plt.text(x=(x[2] / 10 + (x[3] - x[2]) / 20 - 0.6)*1.5, y=x[0] * 1.5 - 0.25, s=infmt, fontsize=12)
                dura[x[0]][1] = x[3]
            elif x[2] < dura[x[0]][0] and x[2] < dura[x[0]][1] and x[2] >= dura[x[0]][2]:
                plt.barh(x[0] * 1.5 - 0.45, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1]],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.45, preparation_time[x[1]] *1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.45, wend_up_time[x[1]] *1.5 / 10, height=0.3, left=(x[3] - Spt[x[1]][-1])*1.5 / 10,
                         color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                infmt =  'h:' + str(x[4]+1) + x[5] + x[6] + '-ct:' + str(x[2])+'~'+ str(x[3])
                plt.text(x=(x[2] / 10 + (x[3] - x[2]) / 20 - 0.6)*1.5, y=x[0] * 1.5 - 0.55, s=infmt, fontsize=12)
                dura[x[0]][2] = x[3]
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    # 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
    matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    NP = 50
    pop_encode_conf = {'NP': 50,'I': 6,'J': 6, 'j_inertial_navigation': 6, 'j_transport': 3, 'num_land_station': 6,
                       'num_takeoff_station': 6}
    duration_operation = {'main_time': [6, 2, 6, 3, 12, 12],'preparation_time': [6, 4, 2, 1, 2, 1],'wend_up_time': [2, 0, 2, 1, 2, 1]}
    num_people_each_type = [6, 6, 6, 6, 6]
    num_equipment_each_type = [4, 6, 6]
    pop_decode_conf = {'NP': 50, 'I': 6, 'J': 6, 'num_equipment_types': len(num_equipment_each_type), 'num_people_types': len(num_people_each_type),
                       'max_devices_same_type': max(num_equipment_each_type), 'max_people_same_type': max(num_people_each_type),'duration_operation': duration_operation,
                       'num_people_each_type': num_people_each_type,'num_equipment_each_type':num_equipment_each_type}
    SpaCon = [[],
              [],
              [],
              [], [], [], [], [], [],  # 9
              [], [], []]  # 第0维index为i，spacon[i]表示对i站位有空间约束的最小站位集合的集合
    load_time = [0, 0, 0, 0, 0, 0]  # 飞机的降落时间
    parallel_or_not = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1]
    ])  # 工序之间的是否并行关系
    min_comp_time_conf = {'NP': 50, 'I': 6, 'J': 6,
                          'mini_station_set_with_spatial_constraint': SpaCon, 'load_time': load_time,
                          'duration_operation': duration_operation, 'parallel_or_not': parallel_or_not}
    calculate_startandend_time_conf = {'I': 6, 'J': 6,
                                       'num_equipment_types': len(num_equipment_each_type),
                                       'num_people_types': len(num_people_each_type),
                                       'max_devices_same_type': max(num_equipment_each_type),
                                       'max_people_same_type': max(num_people_each_type),
                                       'load_time': load_time, 'duration_operation': duration_operation,
                                       'parallel_or_not': parallel_or_not}


    history = {
        'T': [],
        'y': []
    }
    best_popn = np.array([])
    pop = encoder(pop_encode_conf)
    time = 0
    for iter in range(100):
        xy = translate_dna(pop, pop_decode_conf)
        min_time_set_for_pop = Min_comp_time(pop, xy, min_comp_time_conf)
        if iter == 0: dmax = np.var(min_time_set_for_pop)
        min_number = min(min_time_set_for_pop)
        min_index = np.argmin(min_time_set_for_pop)
        print(min_time_set_for_pop[min_index])
        if (history['T'] != [] and min_time_set_for_pop[min_index] < history['T'][-1]) or history['T'] == []:
            history['T'].append(min_time_set_for_pop[min_index])
            best_popn = pop[min_index]
            best_xy = [xy[i][min_index] for i in range(4)]
        elif history['T'] != [] and min_time_set_for_pop[min_index] >= history['T'][-1]:
            history['T'].append(history['T'][-1])
        history['y'].append(pop[min_index])
        fitness = get_fitness(min_time_set_for_pop, NP)
        chosen = choose(pop, fitness, NP)
        crossed = Crossover(chosen, 6, 6, Pc=0.8, jt=3, jl=6)
        variated, d_g = Variation(crossed, min_time_set_for_pop, dmax, Pum=0.3, Plm=0.1, NP=NP, I=6, J=6)
        pop = variated
        time += 1
        dmax = max(dmax,d_g)
        print("迭代" + str(time) + "目前最优解" + str(history['T'][-1]))
    best_setime = calculate_startandend_time(best_popn, best_xy, calculate_startandend_time_conf)
    # history['T'].remove([M])
    # print(best_x)
    # print(best_setime)
    print("最短时长为" + str(history['T'][-1]))
    print("holle")
    Draw_gantt(best_setime, load_time, I=6, J=6, duration_operation=duration_operation)
    # plt.figure(1)
    # Draw_gantt(best_setime)
    # plt.figure(2)
    # plt.plot(history['T'])
    # plt.title('Minimum-completion-time value')
    # plt.xlabel('Iter')
    # plt.show()
    #return history['T'][-1], best_setime, history['T']


