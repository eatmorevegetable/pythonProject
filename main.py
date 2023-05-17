import operator
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib



NP = 50                  #种群数量
Pc = 0.8                 #交叉率
Plm = 0.1
Pum = 0.3                #变异率
Gmax = 100               #最大遗传代数
I = 6                    #舰载机数量
J = 6                    #工序数量
jl = 6                   # GD工序的工序号
jt = 3                   #调运工序的工序号
krn = 3                  #设备种类数
mkr = 6                  #同种最大设备数
kpn = 5                  #人员种类数
mkp = 6                  #同种最大人员数
h1 = 6                   #降落站位个数 降落站位序号1~6
h2 = 6                   #起飞站位个数  起飞站位序号7~12
dr = 2
M = 250                #表示一个足够大的实数
K = np.array([[1,0],[2,0],[3,1],[4,2],[5,3],[0,0]])
#5步工序所需要的人员和设备保障种类，每组第一个是KP人员，第二个是KR设备，0代表没有使用

Spt = np.array([[14,14,14,14,14,14,6,2],
                [6,6,6,6,6,6,4,0],
                [10,10,8,10,10,10,2,2],
                [5,5,5,5,5,5,1,1],
                [16,16,16,16,16,16,2,2],
                [14,14,14,14,14,14,1,1]])
#保障工序时间参数表6*6,每行的最后两位各表示该工序的准备、收尾时间
Cbea = np.array([
    [[1,2,3,4],[1,2,3,4,5,6],[1,2]],
    [[1,2,3,4],[1,2,3,4,5,6],[1,2]],
    [[1,2,3,4],[1,2,3,4,5,6],[1,2]],
    [[1,2,3,4],[1,2,3,4,5,6],[1,2]],
    [[1,2,3,4],[1,2,3,4,5,6],[3]],
    [[1,2,3,4],[1,2,3,4,5,6],[3]],
    [[1,2,3,4],[1,2,3,4,5,6],[4]],
    [[1,2,3,4],[1,2,3,4,5,6],[4]],
    [[1,2,3,4],[1,2,3,4,5,6],[4]],
    [[1,2,3,4],[1,2,3,4,5,6],[5,6]],
    [[1,2,3,4],[1,2,3,4,5,6],[5,6]],
    [[1,2,3,4],[1,2,3,4,5,6],[5,6]]
], dtype=object) #设备对站位保障覆盖关系
ifpara = np.array([
    [1,1,0,0,0,0],
    [1,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,1,1],
    [0,0,0,0,1,1]
])#工序之间的是否并行关系

SpaCon = [[],
          [],
          [],
          [],[],[],[],[],[], #9
          [],[],[]]   #第0维index为i，spacon[i]表示对i站位有空间约束的最小站位集合的集合

loadtime = [0,2,4,6,8,10] #飞机的降落时间

#编码函数，生成种群包含50个个体，每个个体包含工序排列信息,前i*j个三元组为(i,j,0)后i个三元组为(i，h1,h2)
# i：表示舰载机号；  j：表示工序号；  h1降落站位，h2起飞站位；
#约束条件：1.工序5 GD是每个舰载机的最后一道工序
#        2.对于同一JZJ 5GD、2调运、3充氧、4加油互相不能并行，加油和惯导可以并行
def encoder():
    dna = []
    sr =[]
    temp = []
    a = []  #dna的中转
    pop = []
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
            cho =  np.random.choice(I_gat)
            dna.append(sr[cho][0])
            del sr[cho][0]
            if sr[cho] == []:I_gat.remove(cho)
        sr = []
        l = np.random.choice([x for x in range(1,h1+1)],I,replace=False) # 1~6中取6个
        f = np.random.choice([x for x in range(h1+1,h2+h1+1)], I, replace=False) #7~12中取6个
        jtindex = []
        for m in range(I*J):
            i = dna[m][0] - 1
            j = dna[m][1] - 1
            if m > dna.index([i +1, jt, 0]):
                dna[m][2] = f[i]
            elif m < dna.index([i + 1, jt, 0]):
                dna[m][2] = l[i]
            elif m == dna.index([i + 1, jt, 0]):
                jtindex.append(m)
        for x in jtindex:
            dna[x][2] = l[dna[x][0]-1]
        jtindex = []
        a.append(dna)        #加入种群
        dna = []
    pop = np.array(a)
    #print(pop)
    return pop

#解码思路：先比较工序(i,j)与(i,jt)的前后关系，在jt前则在降落站位，在jt后则在起飞站位
#有四类保障人员KP1,2,3,4每类保障人员定义足够多的组数：6
def translate_dna(pop):
    # 创建一个五维数组来存放50个种群的Xpijkl 50*6*6*5*6 4是k类保障人员，6是每类人数
    xp = np.zeros((50, I, J, kpn, mkp), dtype=int)
    for n in range(NP):
        kp_cum = np.zeros((kpn, mkp), dtype=int)  # 定义一个4*6的矩阵来表示第k类第l个保障人员累计作业时间
        for m in range(I*J):
            i = pop[n][m][0]-1
            j = pop[n][m][1]-1
            k = K[j][0]   #通过工序号查找需要的人员种类1-4
            if k != 0:
                l = np.argmin(kp_cum[k - 1])  # 累计作业时间最少优先
                xp[n][i][j][k - 1][l] = 1
                kp_cum[k - 1][l] += Spt[j][i]
    #生成决策变量Xpijkl

    xr = np.zeros((50, I, J, krn, mkr), dtype=int)  #生成一个五维数组来存放Xrijkl 50*6*6*3*6
    #有3类保障设备，每类保障设备分布为4,6,6(调运,充氧,加油)
    #先算出每道工序的站位号根据站位号分配设备号
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

    yp = np.zeros((50, I, J, I, J), dtype=int)  # 生成一个五维数组来存放Ypijeg 50*6*6*6*6
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

    yr = np.zeros((50, I, J, I, J), dtype=int)  # 生成一个五维数组来存放Yrijeg 50*6*6*6*6
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

def earlist_complet_time(pos,popn,ypn,yrn):
    #计算工序e，g的最短完成时间
    e = popn[pos][0] -1
    g = popn[pos][1] -1

    lastpos = []
    #求出同一飞机的紧前工序位置
    if pos != 0:
        for m0 in range(pos):
            e1 = popn[pos - m0 - 1][0] - 1 #每循环一次在pos为起点的基础上取往前一步的工序的JZJ号
            g1 = popn[pos - m0 - 1][1] - 1
            h = popn[pos][2] - 1
            h1 = popn[pos - m0 - 1][2] - 1
            # 同一飞机但不可并行或者可并行但在不同站位
            if e1 == e and (ifpara[g][g1] == 0 or h != h1):
                lastpos.append(pos - m0 - 1)

    #计算工序（e，g）的紧前工序，所有紧前工序完成的时间加上工序保障时间为该工序的最早完成时间
    prior_time = 0 #紧前工序的完成时间
    for m1 in range (pos):
        i = popn[pos - m1 - 1][0] - 1
        j = popn[pos - m1 - 1][1] - 1
        #如果工序(i,j)和(e,g)分配在同一机组且先于(e,g)保障，
        #并且(i,j)的最早完成时间大于紧前工序完成时间，将更新紧前工序完成时间为(i,j)的完成时间
        if ypn[i][j][e][g] == 1 and earlist_complet_time(pos - m1 - 1, popn, ypn, yrn) > prior_time:
            prior_time = earlist_complet_time(pos - m1 - 1, popn, ypn, yrn)
        # 如果工序(i,j)和(e,g)分配在同一设备且先于(e,g)保障，
        # 并且(i,j)的最早完成时间大于紧前工序完成时间，将更新紧前工序完成时间为(i,j)的完成时间
        if yrn[i][j][e][g] == 1 and earlist_complet_time(pos - m1 - 1, popn, ypn, yrn) > prior_time:
            prior_time = earlist_complet_time(pos - m1 - 1, popn, ypn, yrn)
    if lastpos != []:#若存在同一飞机不可并行且排在(e,g)前的工序
        for x in lastpos:
            if earlist_complet_time(x, popn, ypn, yrn) \
                    - Spt[popn[x][1] - 1][-1] - Spt[popn[pos][1] - 1][-2] > prior_time:
                prior_time = earlist_complet_time(x, popn, ypn, yrn) - \
                             Spt[popn[x][1] - 1][-1] - Spt[popn[pos][1] - 1][-2]
    if prior_time == 0:prior_time = loadtime[e]

    #print(1)
    return (prior_time + Spt[g][e])

def Cal_SE_T(best_ord,best_xp,best_xr,best_yp,best_yr):
    #best_ord指最佳的工序排列, best_xp: 6*5*4*6 ; best_xr: 6*5*3*6
    se_list =[] #数列的每个元素是一个七元组：i,j,start-time,end-time,h,pk-l,rk-l
    for m in range(I*J):
        i = best_ord[m][0] - 1
        j = best_ord[m][1] - 1
        et = earlist_complet_time(m,best_ord,best_yp,best_yr)
        st = et - Spt[j][i]
        h = best_ord[m][2] - 1
        rdstr = ''
        for k in range(kpn):
            for l in range(mkp):
                if best_xp[i][j][k][l] == 1:
                    pdstr = 'p' + str(k+1) + '-' + str(l+1)
                    break
        for k in range(krn):
            for l in range(mkr):
                if best_xr[i][j][k][l] == 1:
                    rdstr = '-r' + str(l+1)
                    break
        se_list.append([i,j,st,et,h,pdstr,rdstr])
    return se_list

def Min_comp_time(pop,xy):  #pop是
    # 通过决策变量计算最后一道工序的最短完成时间
    f = np.zeros((50,1),dtype=int)
    for n in range(50):
        #当ifspacon为0时表示违反了空间约束
        ifspacon = 1
        #顺序遍历工序，对于一个(i,j,h),
        ifocc = np.zeros(18,dtype=int)
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
            if ifspacon ==0 :break
            ifocc[h] = 1
        if ifspacon == 0:f[n] = M
        if ifspacon == 1:
            min_compl_t = 0
            for m in range(I*J):
                egt = earlist_complet_time(m, pop[n], xy[2][n], xy[3][n])
                if egt > min_compl_t:
                    min_compl_t = egt
            f[n] = min_compl_t

    return f

def get_fitness(np_f):
    #计算适应度函数
    p = []
    secret_p = 1.5  #选择压力
    torch_f = torch.from_numpy(np_f)
    sorted, indices = torch.sort(torch_f, dim=0, descending=True)
    sorted = sorted.numpy()
    for i in range(len(np_f)):
        for j in range(len(sorted)):
            if np_f[i] == sorted[j]:
                p.append([j+1])
                break
    p = np.array(p)
    fitness = 2 - secret_p + (2 * (p - 1) * (secret_p - 1)) / (NP - 1)
    return  fitness

def choose(pop,fitness):
    #计算每个个体被选中的概率
    s = sum(fitness)
    p = [fitness[i]/s for i in range(len(fitness))]
    chosen = []
    #通过赌盘法选择NP个染色体
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

def Crossover(pop):
    crossed = np.array([])
    for i in range(0,len(pop),2):
        a = pop[i]
        b = pop[i+1]
        if random.random() < Pc:
            pos = random.randint(1,len(pop[i]) - 1)
            #print("交叉点是"+str(pos))
            temp1 = b[:pos]
            temp2 = a[:pos]
            temp = b[pos:]
            for j in range(len(a)):
                for k in range(len(temp)):
                    if a[j][0] == temp[k][0] and a[j][1] == temp[k][1]:
                        temp1 = np.concatenate((temp1,[temp[k]]))
                        break
            temp = a[pos:]
            for j in range(len(b)):
                for k in range(len(temp)):
                    if b[j][0] == temp[k][0] and b[j][1] == temp[k][1]:
                        temp2 = np.concatenate((temp2, [temp[k]]))
                        break
            a = temp1
            b = temp2
            #将交叉后的站位按照调运前是降落站位，调运后是起飞站位的原则重新安排
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

        if i == 0:crossed =  np.concatenate(([a],[b]))
        else:crossed = np.concatenate((crossed,[a],[b]))
    return crossed

def Variation(pop,min_time,D_max):
    D_G = np.var(min_time)
    pm = Pum - (Pum - Plm) * D_G / D_max
    # print(pm)
    for n in range(NP):
        a = pop[n]
        if random.random() < pm:
            # print("第"+str(n+1))
            i = np.random.choice([x for x in range(1,I+1)],1,replace = False)
            poslist = []  #记录飞机i各工序在a中的位置
            for m in range(I*J):
                if a[m][0] == i:poslist.append(m)
            chapos = np.random.choice([x for x in range(J-1)],2,replace = False)
            temp = np.array(a[poslist[chapos[0]]])
            # print("temp值"+str(temp))
            # print(a[poslist[chapos[0]]])
            # print(a[poslist[chapos[1]]])
            a[poslist[chapos[0]]] = a[poslist[chapos[1]]]
            a[poslist[chapos[1]]] = temp
            # print("temp值" + str(temp))
            # print(a[poslist[chapos[0]]])
            # print(a[poslist[chapos[1]]])
        if n == 0 :variated = np.array([a])
        else:variated = np.concatenate((variated,[a]))
    return variated,D_G

def Draw_gantt(best_setime):
    #best_setime为四维向量组(i,j,starttinme,endtime)i:0~5;j:0~4
    color = ['red','brown','dodgerblue','mediumseagreen','blueviolet','orange', #主体颜色
             'darkred','darkred','blue','green','indigo','orangered',  #边框颜色
             'salmon','rosybrown','lightskyblue','aquamarine','plum','moccasin']  #准备和收尾阶段颜色
    name= ['挂弹1','挂弹2','调运','充氧','加油','惯导']
    #每个飞机有两个时间轴，将有时间重合的工序分开绘制，
    # 设置一个变量dura：I*4来更新每绘制一个工序每个飞机1,2,3,4四条时间轴的持续时间
    dura = [[i,i,i] for i in loadtime]
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
            plt.barh(x[0] * 1.5 + 0.45, Spt[x[1]][-2]*1.5 / 10, height=0.3, left=x[2] *1.5/ 10, color=color[x[1] + 2*J],
                     edgecolor=color[x[1] + J])   #准备时间
            plt.barh(x[0] * 1.5 + 0.45, Spt[x[1]][-1]*1.5 / 10, height=0.3, left=(x[3] - Spt[x[1]][-1])*1.5 / 10,
                     color=color[x[1] + 2*J],
                     edgecolor=color[x[1] + J])   #收尾时间
            infmt = 'h:'+str(x[4]+1)+ x[6] + '-ct:' + str(x[3])
            plt.text(x = (x[2] / 10 + (x[3] - x[2]) / 15 - 0.5)*1.5, y = x[0] * 1.5 + 0.35, s = infmt, fontsize=12)
        else:
            if x[2] >= dura[x[0]][0]:
                plt.barh(x[0] * 1.5 +0.15, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2] *1.5/ 10, color=color[x[1]],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 +0.15, Spt[x[1]][-2] *1.5/ 10, height=0.3, left=x[2] *1.5/ 10, color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 +0.15, Spt[x[1]][-1] *1.5/ 10, height=0.3, left=(x[3] - Spt[x[1]][-1]) *1.5/ 10,
                         color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                infmt = 'h:' + str(x[4]+1) + x[6] + '-ct:' + str(x[3])
                plt.text(x=(x[2] / 10 + (x[3] - x[2]) / 15 - 0.5)*1.5, y=x[0] * 1.5 +0.05, s=infmt, fontsize=12)
                dura[x[0]][0] = x[3]
            elif x[2] < dura[x[0]][0] and x[2] >= dura[x[0]][1]:
                plt.barh(x[0] * 1.5 - 0.15, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1]],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.15, Spt[x[1]][-2]*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.15, Spt[x[1]][-1]*1.5 / 10, height=0.3, left=(x[3] - Spt[x[1]][-1])*1.5 / 10,
                         color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                infmt =  'h:' + str(x[4]+1) + x[6] + '-ct:' + str(x[3])
                plt.text(x=(x[2] / 10 + (x[3] - x[2]) / 15 - 0.5)*1.5, y=x[0] * 1.5 - 0.25, s=infmt, fontsize=12)
                dura[x[0]][1] = x[3]
            elif x[2] < dura[x[0]][0] and x[2] < dura[x[0]][1] and x[2] >= dura[x[0]][2]:
                plt.barh(x[0] * 1.5 - 0.45, (x[3] - x[2])*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1]],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.45, Spt[x[1]][-2]*1.5 / 10, height=0.3, left=x[2]*1.5 / 10, color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                plt.barh(x[0] * 1.5 - 0.45, Spt[x[1]][-1]*1.5 / 10, height=0.3, left=(x[3] - Spt[x[1]][-1])*1.5 / 10,
                         color=color[x[1] + 2*J],
                         edgecolor=color[x[1] + J])
                infmt =  'h:' + str(x[4]+1) + x[6] + '-ct:' + str(x[3])
                plt.text(x=(x[2] / 10 + (x[3] - x[2]) / 15 - 0.5)*1.5, y=x[0] * 1.5 - 0.55, s=infmt, fontsize=12)
                dura[x[0]][2] = x[3]
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    # 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
    matplotlib.rcParams['axes.unicode_minus'] = False

def run():
    history = {
        'T':[],
        'x':[]
    }
    best_x =np.array([])
    pop = encoder()
    time = 0
    for iter in range(Gmax):
        xy = translate_dna(pop)
        min_time = Min_comp_time(pop,xy)
        if iter == 0:dmax = np.var(min_time)
        min_index, min_number = min(enumerate(min_time), key=operator.itemgetter(0))
        print(min_time[min_index])
        if (history['T'] != [] and min_time[min_index][0] < history['T'][-1][0]) or history['T'] == []:
            history['T'].append(min_time[min_index])
            best_x = pop[min_index]
            best_xp = xy[0][min_index]
            best_xr = xy[1][min_index]
            best_yp = xy[2][min_index]
            best_yr = xy[3][min_index]
        elif history['T'] != [] and min_time[min_index][0] > history['T'][-1][0]:
            history['T'].append(history['T'][-1])
        history['x'].append(pop[min_index])
        fitness = get_fitness(min_time)
        chosen = choose(pop,fitness)
        crossed = Crossover(chosen)
        variated,d_g = Variation(crossed,min_time,dmax)
        pop = variated
        #pop = crossed
        time += 1
        if d_g > dmax: dmax = d_g
        print("迭代" + str(time))
    best_setime = Cal_SE_T(best_x,best_xp,best_xr,best_yp,best_yr)
    #history['T'].remove([M])
    print(best_x)
    print(best_setime)
    print("最短时长为" + str(history['T'][-1]))
    plt.figure(1)
    Draw_gantt(best_setime)
    plt.figure(2)
    plt.plot(history['T'])
    plt.title('Minimum-completion-time value')
    plt.xlabel('Iter')
    plt.show()

def test():
    # x = np.array([[1,2,3,4],[5,6,7,8],[9,1,2,3],[4,5,6,7]])
    # print(x.reshape(-1,1))
    a = torch.tensor([6,9,7])
    b = np.array([6,6,7])
    a.exp().max().values
    print(a)


test()











