#!/usr/bin/env python

FM_SLOTS = [
    (1, 2, 128),  # f_user, f_gid
    (1, 201, 128),  # f_user, f_g_author
    (1, 202, 64),  # f_user, f_g_loc
    (1, 800, 8),  # f_user, f_g_live_audience_count
    (1, 203, 4),  # f_user, f_g_ct
    (1, 801, 8),  # f_user, f_g_live_income
    (1, 7, 128),  # f_user, f_chn_id
    (1, 217, 16),  # f_user, f_g_author_gender
    (1, 806, 16),  # f_user, f_a_live_type
    (1, 805, 16),  # f_user, f_a_live_trade_union_id
    (1, 300, 16),  # f_user, f_a_fllwer_cnt
    (1, 802, 16),  # f_user, f_a_live_fans_club_count
    (1, 803, 16),  # f_user, f_a_live_income
    (1, 262, 8),  # f_user, f_g_author_is_followed
    (1, 263, 8),  # f_user, f_g_author_is_friend
    (1, 808, 8),  # f_user, f_g_author_is_fans_club
    (7, 2, 128),  # f_chn_id, f_gid
    (7, 201, 128),  # f_chn_id, f_g_author
    (1, 200, 32),  # f_user, f_g_tag_id
    (1, [301, 302, 303, 304, 305], 16),
    # f_user, [f_a_total_show,f_a_total_play,f_a_total_follow,f_a_total_comment,f_a_total_gift]
    (1, [306, 307, 308, 309], 16),
    # f_user, [f_a_total_play_rate,f_a_total_follow_rate,f_a_total_comment_rate,f_a_total_gift_rate]
    (1, [221, 222, 223, 224, 225], 16),
    # f_user, [f_g_total_show,f_g_total_play,f_g_total_follow,f_g_total_comment,f_g_total_gift]
    (1, [226, 227, 228, 229, 230], 16),
    # f_user, [f_g_merge_show,f_g_merge_play,f_g_merge_follow,f_g_merge_comment,f_g_merge_gift]
    (1, [231, 232, 233, 234, 235], 16),
    # f_user, [f_g_video_show,f_g_video_play,f_g_video_follow,f_g_video_comment,f_g_video_gift]
    (1, [241, 242, 243, 244], 16),
    # f_user, [f_g_total_play_rate,f_g_total_follow_rate,f_g_total_comment_rate,f_g_total_gift_rate]
    (1, [245, 246, 247, 248], 16),
    # f_user, [f_g_merge_play_rate,f_g_merge_follow_rate,f_g_merge_comment_rate,f_g_merge_gift_rate]
    (1, [249, 250, 251, 252], 16),
    # f_user, [f_g_video_play_rate,f_g_video_follow_rate,f_g_video_comment_rate,f_g_video_gift_rate]
    (1, 911, 32),  # f_user, f_g_live_content_classifier_record
    (1, 912, 32),  # f_user, f_g_pk_status
    (1, 917, 32),  # f_user, f_g_pk_rival_user
    (1, [913, 914, 915, 917], 16),  # f_user, [f_g_pk_match_type,f_g_pk_duration,f_g_pk_duration_rate,f_g_pk_rival_user]

    (2, 200, 32),  # f_gid, f_g_tag_id
    (2, [301, 302, 303, 304, 305], 8),
    # f_gid, [f_a_total_show,f_a_total_play,f_a_total_follow,f_a_total_comment,f_a_total_gift]
    (2, [306, 307, 308, 309], 8),
    # f_gid, [f_a_total_play_rate,f_a_total_follow_rate,f_a_total_comment_rate,f_a_total_gift_rate]
    (2, [221, 222, 223, 224, 225], 8),
    # f_gid, [f_g_total_show,f_g_total_play,f_g_total_follow,f_g_total_comment,f_g_total_gift]
    (2, [226, 227, 228, 229, 230], 8),
    # f_gid, [f_g_merge_show,f_g_merge_play,f_g_merge_follow,f_g_merge_comment,f_g_merge_gift]
    (2, [231, 232, 233, 234, 235], 8),
    # f_gid, [f_g_video_show,f_g_video_play,f_g_video_follow,f_g_video_comment,f_g_video_gift]
    (2, [241, 242, 243, 244], 8),
    # f_gid, [f_g_total_play_rate,f_g_total_follow_rate,f_g_total_comment_rate,f_g_total_gift_rate]
    (2, [245, 246, 247, 248], 8),
    # f_gid, [f_g_merge_play_rate,f_g_merge_follow_rate,f_g_merge_comment_rate,f_g_merge_gift_rate]
    (2, [249, 250, 251, 252], 8),
    # f_gid, [f_g_video_play_rate,f_g_video_follow_rate,f_g_video_comment_rate,f_g_video_gift_rate]
    (2, 911, 32),  # f_gid, f_g_live_content_classifier_record
    (2, 912, 32),  # f_gid, f_g_pk_status
    (2, 917, 32),  # f_gid, f_g_pk_rival_user
    (2, [913, 914, 915, 917], 16),  # f_gid, [f_g_pk_match_type,f_g_pk_duration,f_g_pk_duration_rate,f_g_pk_rival_user]

    (201, 200, 32),  # f_gid, f_g_tag_id
    (201, [301, 302, 303, 304, 305], 16),
    # f_user, [f_a_total_show,f_a_total_play,f_a_total_follow,f_a_total_comment,f_a_total_gift]
    (201, [306, 307, 308, 309], 16),
    # f_user, [f_a_total_play_rate,f_a_total_follow_rate,f_a_total_comment_rate,f_a_total_gift_rate]
    (201, [221, 222, 223, 224, 225], 16),
    # f_user, [f_g_total_show, f_g_total_play, f_g_total_follow, f_g_total_comment, f_g_total_gift]
    (201, [226, 227, 228, 229, 230], 16),
    # f_user, [f_g_merge_show, f_g_merge_play, f_g_merge_follow, f_g_merge_comment, f_g_merge_gift]
    (201, [231, 232, 233, 234, 235], 16),
    # f_user, [f_g_video_show, f_g_video_play, f_g_video_follow, f_g_video_comment, f_g_video_gift]
    (201, [241, 242, 243, 244], 16),
    # f_user, [f_g_total_play_rate, f_g_total_follow_rate, f_g_total_comment_rate, f_g_total_gift_rate]
    (201, [245, 246, 247, 248], 16),
    # f_user, [f_g_merge_play_rate, f_g_merge_follow_rate, f_g_merge_comment_rate, f_g_merge_gift_rate]
    (201, [249, 250, 251, 252], 16),
    # f_user, [f_g_video_play_rate, f_g_video_follow_rate, f_g_video_comment_rate, f_g_video_gift_rate]
    (201, 911, 32),  # f_author, f_g_live_content_classifier_record
    (201, 912, 32),  # f_author, f_g_pk_status
    (201, 917, 32),  # f_author, f_g_pk_rival_user
    (201, [913, 914, 915, 917], 16),
    # f_author, [f_g_pk_match_type,f_g_pk_duration,f_g_pk_duration_rate,f_g_pk_rival_user]

    (856, 201, 8),  # f_u_count_decay15_show, f_g_author
    (857, 201, 8),  # f_u_count_decay15_play, f_g_author
    (858, 201, 8),  # f_u_count_decay15_stay, f_g_author
    (859, 201, 8),  # f_u_count_decay15_comment, f_g_author
    (860, 201, 8),  # f_u_count_decay15_follow, f_g_author
    (861, 201, 8),  # f_u_count_decay15_gift, f_g_author
    (862, 201, 8),  # f_u_count_decay15_barrage, f_g_author
    (863, 201, 8),  # f_u_count_decay15_red_send, f_g_author
    (866, 201, 8),  # f_u_value_decay15_stay, f_g_author
    (867, 201, 8),  # f_u_value_decay15_gift, f_g_author
    (868, 201, 8),  # f_u_value_decay15_red_send, f_g_author
    (856, 2, 8),  # f_u_count_decay15_show, f_gid
    (857, 2, 8),  # f_u_count_decay15_play, f_gid
    (858, 2, 8),  # f_u_count_decay15_stay, f_gid
    (859, 2, 8),  # f_u_count_decay15_comment, f_gid
    (860, 2, 8),  # f_u_count_decay15_follow, f_gid
    (861, 2, 8),  # f_u_count_decay15_gift, f_gid
    (862, 2, 8),  # f_u_count_decay15_barrage, f_gid
    (863, 2, 8),  # f_u_count_decay15_red_send, f_gid
    (866, 2, 8),  # f_u_value_decay15_stay, f_gid
    (867, 2, 8),  # f_u_value_decay15_gift, f_gid
    (868, 2, 8),  # f_u_value_decay15_red_send, f_gid
    (856, 202, 8),  # f_u_count_decay15_show, f_g_loc
    (857, 202, 8),  # f_u_count_decay15_play, f_g_loc
    (858, 202, 8),  # f_u_count_decay15_stay, f_g_loc
    (859, 202, 8),  # f_u_count_decay15_comment, f_g_loc
    (860, 202, 8),  # f_u_count_decay15_follow, f_g_loc
    (861, 202, 8),  # f_u_count_decay15_gift, f_g_loc
    (862, 202, 8),  # f_u_count_decay15_barrage, f_g_loc
    (863, 202, 8),  # f_u_count_decay15_red_send, f_g_loc
    (866, 202, 8),  # f_u_value_decay15_stay, f_g_loc
    (867, 202, 8),  # f_u_value_decay15_gift, f_g_loc
    (868, 202, 8),  # f_u_value_decay15_red_send, f_g_loc

    (871, 201, 8),  # f_u_recent_play, f_g_author
    (872, 201, 8),  # f_u_recent_stay, f_g_author
    (873, 201, 8),  # f_u_recent_comment, f_g_author
    (874, 201, 8),  # f_u_recent_follow, f_g_author
    (875, 201, 8),  # f_u_recent_gift, f_g_author
    (876, 201, 8),  # f_u_recent_barrage, f_g_author
    (877, 201, 8),  # f_u_recent_red_send, f_g_author
    (878, 201, 8),  # f_u_recent_club_join, f_g_author
    (879, 201, 8),  # f_u_recent_deposit, f_g_author

    (871, 2, 8),  # f_u_recent_play, f_gid
    (872, 2, 8),  # f_u_recent_stay, f_gid
    (873, 2, 8),  # f_u_recent_comment, f_gid
    (874, 2, 8),  # f_u_recent_follow, f_gid
    (875, 2, 8),  # f_u_recent_gift, f_gid
    (876, 2, 8),  # f_u_recent_barrage, f_gid
    (877, 2, 8),  # f_u_recent_red_send, f_gid
    (878, 2, 8),  # f_u_recent_club_join, f_gid
    (879, 2, 8),  # f_u_recent_deposit, f_gid

    (871, 202, 8),  # f_u_recent_play, f_g_loc
    (872, 202, 8),  # f_u_recent_stay, f_g_loc
    (873, 202, 8),  # f_u_recent_comment, f_g_loc
    (874, 202, 8),  # f_u_recent_follow, f_g_loc
    (875, 202, 8),  # f_u_recent_gift, f_g_loc
    (876, 202, 8),  # f_u_recent_barrage, f_g_loc
    (877, 202, 8),  # f_u_recent_red_send, f_g_loc
    (878, 202, 8),  # f_u_recent_club_join, f_g_loc
    (879, 202, 8),  # f_u_recent_deposit, f_g_loc

    (885, 201, 8),  # f_u_recent_play, f_g_author
    (886, 201, 8),  # f_u_recent_stay, f_g_author
    (887, 201, 8),  # f_u_recent_comment, f_g_author
    (888, 201, 8),  # f_u_recent_follow, f_g_author
    (889, 201, 8),  # f_u_recent_gift, f_g_author
    (890, 201, 8),  # f_u_recent_barrage, f_g_author

    (885, 2, 8),  # f_u_recent_play, f_gid
    (886, 2, 8),  # f_u_recent_stay, f_gid
    (887, 2, 8),  # f_u_recent_comment, f_gid
    (888, 2, 8),  # f_u_recent_follow, f_gid
    (889, 2, 8),  # f_u_recent_gift, f_gid
    (890, 2, 8),  # f_u_recent_barrage, f_gid

    (885, 202, 8),  # f_u_recent_play, f_g_loc
    (886, 202, 8),  # f_u_recent_stay, f_g_loc
    (887, 202, 8),  # f_u_recent_comment, f_g_loc
    (888, 202, 8),  # f_u_recent_follow, f_g_loc
    (889, 202, 8),  # f_u_recent_gift, f_g_loc
    (890, 202, 8),  # f_u_recent_barrage, f_g_loc

    (7, [3, 4, 5, 17], 8),  # f_chn_id, [f_access_type,f_os,f_app_ver,f_device]
    (7, 102, 8),  # f_chn_id, f_u_loc

    (7, [8, 9, 10], 8),  # f_chn_id, [f_hourofday,f_periodofday,f_dayofweek]

    (7, [104, 105], 8),  # f_chn_id, [f_u_reg_time,f_u_gender]

    (2, [3, 4, 5, 17], 8),  # f_gid, [f_access_type,f_os,f_app_ver,f_device]
    (201, [3, 4, 5, 17], 8),  # f_g_author, [f_access_type,f_os,f_app_ver,f_device]

    ([3, 4, 5, 17], [300, 802, 803], 8),
    # [f_access_type,f_os,f_app_ver,f_device], [f_a_fllwer_cnt,f_a_live_fans_club_count,f_a_live_income]

    (102, 2, 8),  # f_u_loc, f_gid
    (102, 201, 8),  # f_u_loc, f_g_author
    (102, [300, 802, 803], 8),  # f_u_loc, [f_a_fllwer_cnt,f_a_live_fans_club_count,f_a_live_income]

    (2, [8, 9, 10], 8),  # f_gid, [f_hourofday,f_periodofday,f_dayofweek]
    (201, [8, 9, 10], 8),  # f_g_author, [f_hourofday,f_periodofday,f_dayofweek]
    ([300, 802, 803], [8, 9, 10], 8),
    # [f_a_fllwer_cnt,f_a_live_fans_club_count,f_a_live_income], [f_hourofday,f_periodofday,f_dayofweek]

    (2, [104, 105], 8),  # f_gid, [f_u_reg_time,f_u_gender]
    (201, [104, 105], 8),  # f_g_author, [f_u_reg_time,f_u_gender]
    ([300, 802, 803], [104, 105], 8),
    # [f_a_fllwer_cnt,f_a_live_fans_club_count,f_a_live_income], [f_u_reg_time,f_u_gender]

    (864, 201, 8),  # f_u_count_decay15_connection_stay, f_g_author
    (869, 201, 8),  # f_u_value_decay15_connection_stay, f_g_author
    (864, 2, 8),  # f_u_count_decay15_connection_stay, f_gid
    (869, 2, 8),  # f_u_value_decay15_connection_stay, f_gid
    (864, 202, 8),  # f_u_count_decay15_connection_stay, f_g_loc
    (869, 202, 8),  # f_u_value_decay15_connection_stay, f_g_loc
    (880, 201, 8),  # f_u_recent_connection_stay, f_g_author
    (880, 2, 8),  # f_u_recent_connection_stay, f_gid
    (880, 202, 8),  # f_u_recent_connection_stay, f_g_loc

    # 增加profile & pk_status
    (864, 912, 8),  # f_u_count_decay15_connection_stay, f_g_pk_status
    (869, 912, 8),  # f_u_value_decay15_connection_stay, f_g_pk_status
    (880, 912, 8),  # f_u_recent_connection_stay, f_g_pk_status

    # 增加profile & pk_rival_user
    (864, 917, 8),  # f_u_count_decay15_connection_stay, f_g_pk_rival_user
    (869, 917, 8),  # f_u_value_decay15_connection_stay, f_g_pk_rival_user
    (880, 917, 8),  # f_u_recent_connection_stay, f_g_pk_rival_user
]

UE_FM_SLOTS = [
    (2, 128),  # f_gid
    (201, 128),  # f_author
]

FM_SLOTS_ADD = [

]

ROOM_TOWER_SLOTS = [600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612]

ROOM_FM_SLOTS = [1, 7, 201, 290]

BIAS_NN_SLOTS = [
    1,
    2,
    3,
    4,
    5,
    7,
    17,
    100,
    102,
    104,
    105,
    200,
    201,
    202,
    203,
    209,
    210,
    211,
    214,
    215,
    217,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    262,
    263,
    290,
    291,
    292,
    293,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    408,
    600,
    601,
    602,
    603,
    604,
    605,
    606,
    607,
    608,
    609,
    610,
    611,
    612,
    800,
    801,
    802,
    803,
    804,
    805,
    806,
    807,
    808,
    809,
    810,
    856,
    857,
    858,
    859,
    860,
    861,
    862,
    863,
    864,
    866,
    867,
    868,
    869,
    871,
    872,
    873,
    874,
    875,
    876,
    877,
    878,
    879,
    880,
    885,
    886,
    887,
    888,
    889,
    890,
    911,
    912,
    913,
    914,
    915,
    917
]

BIAS_ONLY_SLOTS = [
    # Time features
    8,
    9,
    10,
]

VALID_SLOTS = BIAS_NN_SLOTS + BIAS_ONLY_SLOTS
VEC_SLOTS = set()
for x, y, _ in FM_SLOTS:
    if isinstance(x, list):
        for x_i in x:
            VEC_SLOTS.add(x_i)
    else:
        VEC_SLOTS.add(x)
    if isinstance(y, list):
        for y_i in y:
            VEC_SLOTS.add(y_i)
    else:
        VEC_SLOTS.add(y)
for slot in ROOM_TOWER_SLOTS:
    VEC_SLOTS.add(slot)
for slot in ROOM_FM_SLOTS:
    VEC_SLOTS.add(slot)

SELECT_CONCAT_SLOTS = [
    (1, 128), (2, 64), (201, 128), (7, 64),
    (200, 64),
    (202, 32), (917, 32),
    (203, 16), (801, 16), (805, 16), (802, 16), (803, 16), (800, 16), (806, 16), (911, 16),
    (217, 8)
]

if __name__ == '__main__':
    from collections import defaultdict

    slots_dim = defaultdict(int)
    for x, y, dim in FM_SLOTS:
        def get_slots(slots):
            if isinstance(slots, list):
                return slots
            else:
                return [slots]


        x_slots = get_slots(x)
        y_slots = get_slots(y)
        for slot in x_slots:
            slots_dim[slot] += dim

        for slot in y_slots:
            slots_dim[slot] += dim
    slots = slots_dim.keys()
    for slot in slots:
        print
        slot, slots_dim[slot]

