import randomforest as rf
import gatimegenerate as ga
import numpy as np
import torch


CIRCLE_TIME = 24
num_station = 14  # 保障站位数
num_repair_station = 7  # 带维修的起飞站位数
station_conf = {'num_station': num_station, 'num_repair_station': num_repair_station}
# 全为完好飞机的最短完成时间缓存表
mincomptime_cache = [-1 for i in range(num_station//2+1)]
# 详细(每行（列）下标为保障的故障飞机数，行下标为总保障飞机数)最短完成时间缓存表
detailed_mincomptime_cache = [[-1]*i for i in range(1, num_station//2+2)]


# 起飞模块输入：当前时刻、计划起飞飞机数、架次数、起飞间隔时间
#        输出：起飞后当前时刻、实际起飞飞机数、架次数、起飞时长
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


# 保障模块输入：当前时刻、降落飞机数、待保障飞机数、完好率、资源参数、可保障时长
#        输出：保障后当前时刻、保障的飞机数、保障时长、待保障飞机数
def maint(T, num_plane_land, plane_wait_for_service, plane_wait_for_repair, p_fine, resource_conf, time_servable):

    num_fine_plane = np.sum(np.random.rand(num_plane_land) <= p_fine)
    num_sick_plane = num_plane_land - num_fine_plane

    # 等待保障的正常飞机数量：本波次降落的完好飞机数 + 以前波次未保障的待保障正常飞机数
    sum_fine_plane = num_fine_plane + plane_wait_for_service
    # 等待维修保障的故障飞机数：本波次降落故障数 + 以前波次未维修保障的故障飞机数
    sum_sick_plane = num_sick_plane + plane_wait_for_repair

    # 返回已保障的飞机数，包含完成保障的飞机数和完成维修保障的飞机数
    num_plane_served = get_plane_numbydichoyomy(sum_fine_plane, sum_sick_plane, resource_conf, time_servable)

    plane_wait_for_service = sum_fine_plane - num_plane_served['num_fine_plane_served']
    plane_wait_for_repair = sum_sick_plane - num_plane_served['num_sick_plane_served']
    T += time_servable
    t_serving = time_servable
    return {'T': T,
            'num_plane_served': num_plane_served['num_plane_fine_before'] + num_plane_served['num_plane_sick_before'],
            't_serving': t_serving,
            'plane_wait_for_service': plane_wait_for_service,
            'plane_wait_for_repair': plane_wait_for_repair,
            'num_sick_plane': num_sick_plane}

def get_plane_numbydichoyomy(sum_fine_plane, sum_sick_plane, resource_conf, time_servable):
    upper_bound = min(num_station//2, sum_fine_plane+sum_sick_plane)
    lower_bound = 0
    plane_num = 0
    print(time_servable*60)
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound)//2
        print(mid)
        if mincomptime_cache[mid] != -1:
            mincomptime = mincomptime_cache[mid]
        else:
            mincomptime = ga.mincomptime_generate(mid, 0, station_conf, resource_conf)
            mincomptime_cache[mid] = mincomptime
            detailed_mincomptime_cache[mid][0] = mincomptime
        if mincomptime <= time_servable*60:
            plane_num = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    while plane_num >= 0:
        fine_plane_temp = min(sum_fine_plane, plane_num)
        sick_plane_temp = plane_num - fine_plane_temp
        if mincomptime_query(plane_num, sick_plane_temp, resource_conf) > time_servable*60:
            plane_num = plane_num - 1
            continue
        else:
            while 1:
                sick_plane_temp = sick_plane_temp + 1
                fine_plane_temp = fine_plane_temp - 1
                if mincomptime_query(plane_num, sick_plane_temp, resource_conf) > time_servable*60\
                        or sick_plane_temp > sum_sick_plane or fine_plane_temp < 0 \
                        or sick_plane_temp > station_conf['num_repair_station']:
                    sick_plane_served = sick_plane_temp - 1
                    fine_plane_served = fine_plane_temp + 1
                    break
            break
    return {'num_fine_plane_served': fine_plane_served, 'num_sick_plane_served': sick_plane_served}


def mincomptime_query(plane_num,sick_plane_num,resource_conf):
    if detailed_mincomptime_cache[plane_num][sick_plane_num] != -1:
        mincomptime = detailed_mincomptime_cache[plane_num][sick_plane_num]
    else:
        mincomptime = ga.mincomptime_generate(plane_num - sick_plane_num, sick_plane_num, station_conf, resource_conf)
        detailed_mincomptime_cache[plane_num][sick_plane_num] = mincomptime
    return mincomptime


def simulation():
    sum_plane = 14
    num_round_plane = 7 # 一波次计划数
    period = 1
    round_times = 1
    T = 0
    success = 1
    #num_plane_land = 0
    last_takeoff = 0
    last_served_plane = 0
    num_plane_served = 0
    by_plane = num_round_plane  # 保障就绪等待起飞飞机数
    plane_wait_for_service = sum_plane - by_plane  # 已降落等待保障正常(无故障)飞机数
    plane_wait_for_repair = 0  # 已降落等待维修保障故障飞机数
    sortie = 0
    p_fine = 0.9  # 完好率
    p_ga = 0.1  # 复飞率
    interval_time_takeoff = 60/3600
    interval_time_land = 90/3600
    resource_conf = {'duration_operation': {'main_time': [10, 6, 2, 6, 3, 12, 12],
                                            'preparation_time': [2, 6, 4, 2, 1, 2, 1],
                                             'wend_up_time': [2, 2, 0, 2, 1, 2, 1]},
                     'num_people_each_type': [7, 6, 6, 6, 6, 6],
                     'num_equipment_each_type': [7, 4, 6, 6]}
    t_land = 0

    while(T < CIRCLE_TIME):
        # 起飞
        if success == 0:
            T_takeoff = T
            num_plane_takeoff = 0
            t_takeoff = 0
        else:
            T = max((round_times - 1) * period, T)
            T_takeoff = T
            by_plane = by_plane + last_served_plane - last_takeoff
            plan_plane_num = min(num_plane_served + by_plane, num_round_plane)
            info_takeoff = takeoff(T, plan_plane_num, sortie, interval_time_takeoff)
            T = info_takeoff['T']
            sortie = info_takeoff['sortie']
            num_plane_takeoff = info_takeoff['num_plane_takeoff']
            t_takeoff = info_takeoff['t_takeoff']
        # 降落
        if last_takeoff > 0:
            num_ga = np.sum(np.random.rand(last_takeoff) < p_ga)
            t_land = (num_ga + last_takeoff) * interval_time_land
        else:
            t_land = 0
        num_plane_land = last_takeoff
        T += t_land
        # 保障
        last_served_plane = num_plane_served
        time_servable = period - t_takeoff - t_land
        if time_servable <= 0:
            success = 0
            t_serving = 0
            num_plane_served = 0
        else:
            success = 1
            info_maint = maint(T, num_plane_land , plane_wait_for_service, plane_wait_for_repair, p_fine, resource_conf, time_servable)
            T = info_maint['T']
            num_plane_served = info_maint['num_plane_served']
            t_serving = info_maint['t_serving']
            plane_wait_for_service = info_maint['plane_wait_for_service']

        last_takeoff = num_plane_takeoff
        print({'周期': round_times, '起飞时刻': T_takeoff, '架次数': sortie,
               '保障数': num_plane_served, 'by': by_plane, 'wait': plane_wait_for_service, '故障数':info_maint['sick_plane_num'],
               '起飞时间': t_takeoff, '降落时间': t_land, '保障时间': t_serving, '结束时间': T})
        round_times += 1


if __name__ == '__main__':
     simulation()





    
