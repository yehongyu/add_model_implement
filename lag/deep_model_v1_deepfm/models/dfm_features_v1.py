#!/usr/bin/env python
# -*- coding: utf-8 -*-


context = [
    #deprecated 7 f_chn_id 9 f_periodofday
    8,  #f_hourofday
    10,  #f_dayofweek
]


#####################
# author相关特征
author = [
    201,  #f_g_author
]

author_info = [
    217,  #f_g_author_gender
    # TODO: age, location etc
]

#author_stat = [
# TODO: 作者被推送量, impr like follower
#]
#####################


#####################
#  user相关特征
user = [
    591,  #f_user # TODO: combine user recall_strategy replace user
]

user_info = [
    4,  #f_os
    11,  #f_region
    15,  #f_lang
    16,  #f_tz_name
    18,  #f_did
    113,  #f_u_top_affinity
    500,  #f_u_push_os_version
    501,  #f_u_push_sender
    502,  #f_u_push_channel
]

user_stat = [
    # -- remove user show & click
    #503,  #f_u_push_send
    #504,  #f_u_push_show
    #506,  #f_u_push_click
    #507,  #f_u_push_send_to_now
    #508,  #f_u_push_show_to_now
    #510,  #f_u_push_click_to_now
    511,  #f_u_push_active
    512,  #f_u_push_active_to_now
    513,  #f_u_push_activate_to_now
    #518,  #f_u_push_src_send_ctr
    #519,  #f_u_push_src_show_ctr
    #521,  #f_u_push_top_src_click
    #522,  #f_u_push_recent_src_click
]



# match 类特征只作为bias, must name startswith('_')
_user_match = [
    # -- remove user show & click
    434,  #f_ua_affinity
    #530,  #f_ua_push_src_send_ctr
    #531,  #f_ua_push_src_show_ctr
]

#####################


#####################
# 以下是用gid标识的push组特征
push = [
    2,  #f_gid
]

push_stat = [
    # -- remove group counter
    # 541,  #f_g_push_send
    # 542,  #f_g_push_show
    # 544,  #f_g_push_click
    # 545,  #f_g_push_send_ctr
    # 546,  #f_g_push_show_ctr
]

push_info = [
    #548,  #f_g_push_type
    549,  #f_g_push_image
    551,  #f_g_push_title
    553,  #f_g_push_content
    554,  #f_g_push_emoji
    555,  #f_g_push_sound
    556,  #f_g_push_alert
    557,  #f_g_push_keep_top
    558,  #f_g_push_badge
    559,  #f_g_push_led
    560,  #f_g_push_vibrator
    561,  #f_g_push_recall_strategies
]

push_group_info = [
    202,  #f_g_loc
    203,  #f_g_ct
    204,  #f_g_rate
    205,  #f_g_msc
    206,  #f_g_ch
    207,  #f_g_lk
    208,  #f_g_sh
    209,  #f_g_cmmt
    210,  #f_g_ply
    213,  #f_g_dur
    218,  #f_g_lk_rate
    219,  #f_g_sh_rate
    220,  #f_g_cmmt_rate
    245,  #f_g_city
]


''' 
user_recent_gid = [
    524,  #f_u_push_click_2_gids
    525,  #f_u_push_click_4_gids
    526,  #f_u_push_click_8_gids
    527,  #f_u_push_click_16_gids
]

user_match = [
    583,  #f_ug_push_type_region_lang
]

device = [
    17,  #f_device
]

push_text = [
    562,  #f_g_origin_title_split_by_space
    563,  #f_g_origin_content_split_by_space
    564,  #f_g_origin_title_split_by_2char
    565,  #f_g_origin_content_split_by_2char
]
'''

#####################
#
# FIXME: 只能在这里定义数组变量
