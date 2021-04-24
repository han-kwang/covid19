#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:37:44 2021

@author: @hk_nien
"""




import matplotlib.pyplot as plt
import tools
import nlcovidstats as nlcs
import os


#%%
if __name__ == '__main__':


    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    print('--Corrected data--')

    outpth = 'video_out-nogit'

    os.system(f'rm {outpth}/Rt_[0-9][0-9][0-9].png')

    pad_frames = 10

    for i, lastday in enumerate(range(-172, 0, 1)):
        fig = nlcs.plot_Rt(ndays=100, lastday=lastday, delay=nlcs.DELAY_INF2REP,
                           source='r7', correct_anomalies=True, mode='return_fig',
                           ylim=(0.65, 1.6))

        fname = f'{outpth}/Rt_{pad_frames+i:03d}.png'
        fig.savefig(fname)
        plt.close(fig)
        print(fname)

    n_last = i+pad_frames

    for i in range(pad_frames):
        os.system(f'ln -s Rt_{pad_frames:03d}.png {outpth}/Rt_{i:03d}.png')
        os.system(f'ln -s Rt_{n_last:03d}.png {outpth}/Rt_{n_last+i:03d}.png')
#%%
    os.system('rm video_out-nogit/Rt_video.mp4')

    # Convert to video:
    print(f'Converting to mp4')
    frame_rate = 5
    cmd = (
        f'ffmpeg -framerate {frame_rate}'
        f' -i {outpth}/Rt_%03d.png'
        f' -c:v libx264 -r {frame_rate} -pix_fmt yuv420p'
        f' {outpth}/Rt_video.mp4'
        )
    print(cmd)
    os.system(cmd)
