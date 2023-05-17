import RF
import numpy as np
import torch

CIRCLE_TIME = 24
def takeoff(T,plan_plane_num,sortie,interval_time_takeoff):
    if T + plan_plane_num * interval_time_takeoff > CIRCLE_TIME:
        num_plane_takeoff = (CIRCLE_TIME-T)/interval_time_takeoff
        T = CIRCLE_TIME
    else:
        num_plane_takeoff = plan_plane_num
        T = T + plan_plane_num * interval_time_takeoff
    sortie += num_plane_takeoff
    t_takeoff = num_plane_takeoff * interval_time_takeoff
    return {'T': T, 'num_plane_takeoff': num_plane_takeoff, 'sortie': sortie, 't_takeoff': t_takeoff}

def maint(T, num_plane_land, p_fine, resource_conf, time_servable):
    row_actual = resource_conf.insert(0, time_servable)
    num_servable_plane = int(servable(row_actual))
    num_accessed_plane = np.sum(np.random.rand(num_plane_land) >= p_fine)
    if num_accessed_plane < num_servable_plane:
        min_comp_time = mincomptime_generate(num_accessed_plane, resource_conf)  # 这里有一个函数还没写！!!!
        if min_comp_time < time_servable:
            T += min_comp_time
            t_serving = min_comp_time
    else:
        T += time_servable
        t_serving = time_servable
    return {'T': T, 'num_plane_served': min(num_accessed_plane, num_servable_plane), 't_serving': t_serving}

def simulation():
    sum_plane = 14
    num_round_plane = 7 #一波次计划数
    period = 1
    round_times = 1
    T = 0
    success = 1
    num_plane_land = 0
    last_takeoff = 0
    num_plane_served = 0
    sortie = 0
    p_fine = 0.1
    p_ga = 0.1 # 复飞率
    interval_time_takeoff = 6/60
    interval_time_land = 6/60
    while(T < CIRCLE_TIME):
        # 起飞
        if success == 0:
            T_takeoff = T
            num_plane_takeoff = 0
            t_takeoff = 0
        else:
            T = max((round_times - 1) * period, T)
            T_takeoff = T
            by_plane = sum_plane - last_takeoff - num_plane_land
            plan_plane_num = min(num_plane_served + by_plane, num_round_plane)
            info_takeoff = takeoff(T, plan_plane_num, sortie, interval_time_takeoff)
            T = info_takeoff['T']
            sortie = info_takeoff['sortie']
            num_plane_takeoff = info_takeoff['num_plane_takeoff']
        # 降落
        if last_takeoff > 0:
            num_ga = np.sum(np.random.rand(last_takeoff) < p_ga)
            t_land = (num_ga + last_takeoff) * interval_time_land
            num_plane_land = last_takeoff
            T += t_land
        # 保障
        time_servable = period - info_takeoff['t_takeoff'] - t_land
        if time_servable <= 0:
            success = 0
            t_serving = 0
            num_plane_served = 0
        else:
            success = 1
            info_maint = maint(T, num_plane_land, p_fine, resource_conf, time_servable)
            T = info_maint['T']
            num_plane_served = info_maint['num_plane_served']

        last_takeoff = num_plane_takeoff
        round_times += 1














if __name__ == '__main__':
    # a = {'T': {'l':[77,99],'m':88}, 'num_plane_takeoff': 2, 'sortie': 3, 't_takeoff': 4}
    # print(min(a['T']['l']))
    a = [2,4,6,5,1]
    b = [1,1,1,1,1]
    g = np.var(b)
    p_order = []
    torch_f = torch.tensor(a)
    sorted, indices = torch.sort(torch_f, dim=0, descending=True)
    sorted = sorted.numpy()
    for i in range(len(a)):
        for j in range(len(sorted)):
            if a[i] == sorted[j]:
                p_order.append(j+1)
                break
    fitness = [2*p-1 for p in p_order]
    print(g)




    
