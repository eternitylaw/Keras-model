import time
import os
import sys

import dogtail.rawinput
from PIL import ImageGrab
from PIL import Image
import subprocess


def checkresolution(res):
    for characters in res:
        if characters not in '1234567890x':
            return False
    return True

def changeBG():
    bgpic = []
    path = '/usr/share/backgrounds/'

    #Read all available resolutions
    output = subprocess.run(['xrandr'], stdout = subprocess.PIPE).stdout.decode('utf-8')
    resolution = []
    for line in output.split('\n'):
        if 'x' in line:
            resolution.append(line.split()[0])

    #Read all background pics
    for i in os.listdir(path):
        if 'jpg' in i or 'png' in i:
            bgpic.append(i)

    #Create document to save original data pics
    for pics in bgpic:
        os.makedirs('/home/u/Pictures/' + pics)

    #Minimize all open windows
    subprocess.run(['wmctrl', '-k', 'on'])

    count1 = 0
    #Capture
    for res in resolution:
        if checkresolution(res) is False:
            continue
        subprocess.run(['xrandr', '--output', resolution[1], '--mode', res])
        time.sleep(10)
        for j in bgpic:
            os.system('/usr/bin/gsettings set org.gnome.desktop.background  picture-uri /usr/share/backgrounds/' + j)
            time.sleep(10)
            for screenshot_time in range(10):
                # im = pyautogui.screenshot()
                im = ImageGrab.grab()
                time.sleep(0.5)
                im.save('/home/u/Pictures/' + j + '/' + str(count1) + '_' + str(screenshot_time) + '.png')
                testshow = Image.open('/home/u/Pictures/' + j + '/' + str(count1) + '_' + str(screenshot_time) + '.png')
                testshow.show()
                time.sleep(0.6)
                dogtail.rawinput.pressKey('esc')
                testshow.close()
                time.sleep(0.5)
            time.sleep(1)
        count1+=1

# if __name__ == '__main__':
#     changeBG()
changeBG()