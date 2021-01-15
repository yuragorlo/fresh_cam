import mysql.connector
import cv2
import time
import logging
import sys
import os
import usb.core
import usb.util
import pyzbar.pyzbar as pyzbar
import numpy as np
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from collections import deque
from multiprocessing.pool import ThreadPool
from PIL import ImageFont, ImageDraw, Image
from pyzbar.pyzbar import ZBarSymbol


# CONFIGURE SYSTEM
# VIDEO
VIDEO_SOURCE_1 = '****'
VIDEO_SOURCE_2 = '****'
# HAND SCANNER
ID_VENDOR = '****'
ID_PRODUCT = '****'
# SOUND
DURATION = 1
FREQ = 740
# DATABASE
DB_LABEL_1 = 'up'
DB_LABEL_2 = 'right'
DEFAULT_FLAG = 0
DB_CONFIG = {
    'host': '****'
    'user': '****',
    'password': '****',
    'database': "qr"}
# DISPLAY
LIGHT_DELAY = 5 # second

# Logger functions
def get_file_handler():
    file_handler = TimedRotatingFileHandler('log.log', when='W0', interval=100, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [function %(funcName)s] [%(levelname)-5.5s] %(message)s"))
    return file_handler


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [function %(funcName)s] [%(levelname)-5.5s] %(message)s"))
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger


# MySQL functions
def create_table(config):
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS qr_table 
            (id INT NOT NULL AUTO_INCREMENT, 
            data VARCHAR(100) NOT NULL, 
            time TIMESTAMP NOT NULL, 
            route VARCHAR(10) NOT NULL, 
            check_flag INT NOT NULL, 
            PRIMARY KEY (id));''')
    conn.commit()
    cursor.close()
    conn.close()


def push_to_db(qr_dict, qr_logger):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    placeholders = ', '.join(['%s'] * len(qr_dict))
    columns = ', '.join(qr_dict.keys())
    sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % ('qr_table', columns, placeholders)
    cursor.execute(sql, list(qr_dict.values()))
    conn.commit()
    cursor.close()
    conn.close()
    qr_logger.info(f'New qr data insert in db. Data: {qr_dict}')


# Hand scanner functions
def hid2ascii(lst, qr_logger):
    conv_table = {30: ['1', '!'], 31: ['2', '@'], 32: ['3', '#'], 33: ['4', '$'], 34: ['5', '%'],
                  35: ['6', '^'], 36: ['7', '&'], 37: ['8', '*'], 38: ['9', '('], 39: ['0', ')']}
    if lst[0] == 2:
        shift = 1
    else:
        shift = 0
    ch = lst[2]
    if ch not in conv_table:
        qr_logger.info("Warning: data not in conversion table")
        return ''
    return conv_table[ch][shift]


def get_scanner():
    # Find our device using the VID (Vendor ID) and PID (Product ID)
    dev = usb.core.find(idVendor=ID_VENDOR, idProduct=ID_PRODUCT)
    if dev is None:
        raise ValueError('USB device not found')
    # Detach the kernel driver so we can use interface (one user per interface)
    for config in dev:
        for i in range(config.bNumInterfaces):
            if dev.is_kernel_driver_active(i):
                dev.detach_kernel_driver(i)
    # Set the active configuration. With no arguments, the first configuration will be the active one
    dev.set_configuration()
    # Get an endpoint instance
    cfg = dev.get_active_configuration()
    intf = cfg[(0, 0)]
    ep = usb.util.find_descriptor(intf, custom_match = lambda e: \
                usb.util.endpoint_direction(e.bEndpointAddress) == \
                usb.util.ENDPOINT_IN)
    return ep


def get_scanner_data(ep, qr_logger):
    try:
        # Wait up to 0.05 seconds for data.
        timeout = 50
        data = ep.read(1000, timeout)
        ch = hid2ascii(data, qr_logger)
        return ch
    except usb.core.USBError:
        # Timed out. End of the data stream. Print the scan line.
        return None


# Sound function
def play_sound(duration, freq):
    os.system(f'play -nq -t alsa synth {duration} sine {freq}')


# Async functions
def add_video_task(pending_task, pool, cap, fifo, DB_LABEL, qr_logger):
    frame_got, frame = cap.read()
    if frame_got:
        task = pool.apply_async(process_frame, (frame.copy(), fifo, DB_LABEL, qr_logger,))
        pending_task.append(task)


def add_scanner_task(pending_task, pool, ep, qr_logger):
    task = pool.apply_async(process_scanner, (ep, qr_logger, ))
    pending_task.append(task)


def add_sound_task(pending_task, pool):
    task = pool.apply_async(play_sound, (DURATION, FREQ, ))
    pending_task.append(task)


def get_frame(pending_task, last_time, FPS=True):
    res = pending_task.popleft().get()
    frame = res[0]
    fifo = res[1]
    decodedObjects = res[2]
    sound = res[3]
    # Display FPS:
    if FPS:
        cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame, fifo, decodedObjects, sound


def get_scanner_ch(pending_task):
    res = pending_task.popleft().get()
    return res


def get_sound(pending_task):
    pending_task.popleft().get()


# Process functions
def process_frame(frame, fifo, db_label, qr_logger):
    # read and transform frame to grey
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find qr code on frame
    decodedObjects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])
    if decodedObjects:
        # Update fifo and push to db if not in fifo
        fifo, sound = process_qrs(decodedObjects, fifo, db_label, qr_logger)
        # Add square around qr and text with data on frame
        add_qr_to_frame(frame, decodedObjects)
    else:
        sound = False
    return frame, fifo, decodedObjects, sound


def process_obj(data, fifo, camera_label, qr_logger):
    sound = False
    if len(data) >= 12:
        if data[:2] == '94' and data not in fifo:
            fifo.append(data)
            qr_logger.info( f'Add {data} to fifo. Current fifo: {fifo}')
            now = datetime.now()
            new_data = {'data': data,
                        'time': now.strftime("%Y/%m/%d %H:%M:%S"),
                        'route': camera_label,
                        'check_flag': DEFAULT_FLAG}
            # Insert new qr code with time registration to qr_table qr database
            push_to_db(new_data, qr_logger)
            sound = True
        data = ''
        # Delete part of fifo
        if len(fifo) > 7:
            fifo = fifo[-3:]
            qr_logger.info(f'Free some of fifo. Current fifo: {fifo}')
    return fifo, data, sound


def process_qrs(decodedObjects, fifo, camera_label, qr_logger):
    sound = False
    # Get list of current qr objects from video stream
    current_data_list = [x.data.decode("utf-8") for x in decodedObjects]
    for data in current_data_list:
        fifo, data, sound = process_obj(data, fifo, camera_label, qr_logger)
    return fifo, sound


def process_scanner(ep, qr_logger):
    return get_scanner_data(ep, qr_logger)


def get_last_or_empty(in_list, obj_process):
    if in_list:
        if obj_process:
            return in_list[-1].data.decode("utf-8")
        else:
            return in_list[-1]
    else:
        return ''


# Display functions
def add_qr_to_frame(frame, decodedObjects):
    for decodedObject in decodedObjects:
        # Draw square around qr
        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
        # Number of points in the convex hull
        n = len(hull)
        # Draw the convex hull
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (0, 0, 255), 3)


def create_bord(concat_video, show_data):
    fifo_1, fifo_2, fifo_3,\
    decodedObjects_1, decodedObjects_2, scanner_digits,\
    light_label_1, light_label_2, light_label_3 = show_data
    # Get data
    last_qr_1 = get_last_or_empty(fifo_1, False)
    last_qr_2 = get_last_or_empty(fifo_2, False)
    last_qr_scanner = get_last_or_empty(fifo_3, False)
    current_qr_1 = get_last_or_empty(decodedObjects_1, True)
    current_qr_2 = get_last_or_empty(decodedObjects_2, True)
    # Define colors
    colors = {'GREY': (51, 51, 51),
              'WHITE': (255, 255, 255),
              'WHITE_': (255, 255, 255, 0),
              'RED': (51, 51, 255, 0),
              'GREEN': (102, 153, 0)}
    # Change colors depend on lights
    if light_label_1:
        color_rec_1 = colors['GREEN']
        color_text_1 = colors['WHITE']
    else:
        color_rec_1 = colors['WHITE']
        color_text_1 = colors['GREY']
    if light_label_2:
        color_rec_2 = colors['GREEN']
        color_text_2 = colors['WHITE']
    else:
        color_rec_2 = colors['WHITE']
        color_text_2 = colors['GREY']
    if light_label_3:
        color_rec_3 = colors['GREEN']
        color_text_3 = colors['WHITE']
    else:
        color_rec_3 = colors['WHITE']
        color_text_3 = colors['GREY']
    # Make new result window
    img = np.ones((int(concat_video.shape[1] * 0.75), concat_video.shape[1], concat_video.shape[2]), np.uint8)
    # Fill background
    img[:,:] = colors['GREY']
    # Put video to result window
    img[:concat_video.shape[0], :concat_video.shape[1], :concat_video.shape[2]] = concat_video
    # Get shapes
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    # Draw rectangle on frame
    cv2.rectangle(img, (int(w * 0.01), int(h * 0.6)), (int(w * 0.32), int(h * 0.7)), colors['WHITE'], -1)
    cv2.rectangle(img, (int(w * 0.01), int(h * 0.8)), (int(w * 0.32), int(h * 0.9)), color_rec_1, -1)
    cv2.rectangle(img, (int(w * 0.34), int(h * 0.6)), (int(w * 0.65), int(h * 0.7)), colors['WHITE'], -1)
    cv2.rectangle(img, (int(w * 0.34), int(h * 0.8)), (int(w * 0.65), int(h * 0.9)), color_rec_2, -1)
    cv2.rectangle(img, (int(w * 0.67), int(h * 0.6)), (int(w * 0.99), int(h * 0.7)), colors['WHITE'], -1)
    cv2.rectangle(img, (int(w * 0.67), int(h * 0.8)), (int(w * 0.99), int(h * 0.9)), color_rec_3, -1)
    img_pil = Image.fromarray(img)
    # Write text on frame
    font = ImageFont.truetype("Gouranga-Pixel.ttf", 20)
    draw = ImageDraw.Draw(img_pil)
    draw.text((int(w * 0.01), int(h * 0.58)), "РАСПОЗНАНО С ВЕРХНЕЙ КАМЕРЫ", font=font, fill=(colors['WHITE_']))
    draw.text((int(w * 0.01), int(h * 0.78)), "ПРОВЕРЕНО С ВЕРХНЕЙ КАМЕРЫ", font=font, fill=(colors['WHITE_']))
    draw.text((int(w * 0.34), int(h * 0.58)), "РАСПОЗНАНО С БОКОВОЙ КАМЕРЫ", font=font, fill=(colors['WHITE_']))
    draw.text((int(w * 0.34), int(h * 0.78)), "ПРОВЕРЕНО С БОКОВОЙ КАМЕРЫ", font=font, fill=(colors['WHITE_']))
    draw.text((int(w * 0.67), int(h * 0.58)), "РАСПОЗНАНО РУЧНЫМ СКАНЕРОМ", font=font, fill=(colors['WHITE_']))
    draw.text((int(w * 0.67), int(h * 0.78)), "ПРОВЕРЕНО С РУЧНОГО СКАНЕРА", font=font, fill=(colors['WHITE_']))
    # Write qr values (digits) in rectangles
    font = ImageFont.truetype("Gouranga-Pixel.ttf", 40)
    draw.text((int(w * 0.07), int(h * 0.63)), current_qr_1, font=font, fill=(colors['RED']))
    draw.text((int(w * 0.39), int(h * 0.63)), current_qr_2, font=font, fill=(colors['RED']))
    draw.text((int(w * 0.73), int(h * 0.63)), scanner_digits, font=font, fill=(colors['RED']))
    draw.text((int(w * 0.07), int(h * 0.83)), last_qr_1, font=font, fill=color_text_1)
    draw.text((int(w * 0.39), int(h * 0.83)), last_qr_2, font=font, fill=color_text_2)
    draw.text((int(w * 0.73), int(h * 0.83)), last_qr_scanner, font=font, fill=color_text_3)
    return np.array(img_pil)


def main():
    # Init logger
    qr_logger = get_logger('qr_detector')
    qr_logger.debug('Start program')
    thread_num = cv2.getNumberOfCPUs()
    try:
        # Init scanner
        ep = get_scanner()
        scanner_digits = ''
        # Init timers
        timer_label_1 = time.time()
        timer_label_2 = time.time()
        timer_label_3 = time.time()
        # Create mysql table for qr data if not exist
        create_table(DB_CONFIG)
        # Init memory list
        fifo_video_1 = []
        fifo_video_2 = []
        fifo_scanner = []
        # Init ThreadPool and tasks
        pool = ThreadPool(processes=thread_num)
        video_tasks_1 = deque()
        video_tasks_2 = deque()
        scanner_tasks = deque()
        sound_tasks = deque()
        # Get video
        cap_video_1 = cv2.VideoCapture(VIDEO_SOURCE_2)
        cap_video_2 = cv2.VideoCapture(VIDEO_SOURCE_1)
        while True:
            # Fix time
            last_time = time.time()
            # Add tasks
            if (len(video_tasks_1) + len(video_tasks_2) + len(scanner_tasks)) < thread_num:
                add_video_task(video_tasks_1, pool, cap_video_1, fifo_video_1, DB_LABEL_1, qr_logger)
                add_video_task(video_tasks_2, pool, cap_video_2, fifo_video_2, DB_LABEL_2, qr_logger)
                add_scanner_task(scanner_tasks, pool, ep, qr_logger)
            if len(sound_tasks)<1:
                add_sound_task(sound_tasks, pool)
            while len(video_tasks_1) > 0 and video_tasks_1[0].ready() and\
                    len(video_tasks_2) > 0 and video_tasks_2[0].ready():
                # Get frames and info about frames from tasks
                frame_show_1, fifo_video_1, decodedObjects_1, sound_1 = get_frame(video_tasks_1, last_time, FPS=True)
                frame_show_2, fifo_video_2, decodedObjects_2, sound_2 = get_frame(video_tasks_2, last_time, FPS=True)
                if (sound_1 or sound_2) and len(sound_tasks)>0:
                    get_sound(sound_tasks)
                scanner_digit = get_scanner_ch(scanner_tasks)
                if scanner_digit:
                    scanner_digits = scanner_digits + str(scanner_digit)
                    fifo_scanner, scanner_digits, sound_3 =\
                        process_obj(scanner_digits, fifo_scanner, 'scanner', qr_logger)
                else:
                    sound_3 = False

                if sound_1: timer_label_1 = time.time()
                if (last_time - timer_label_1) < LIGHT_DELAY: light_label_1 = True
                else: light_label_1 = False
                if sound_2: timer_label_2 = time.time()
                if (last_time - timer_label_2) < LIGHT_DELAY: light_label_2 = True
                else: light_label_2 = False
                if sound_3: timer_label_3 = time.time()
                if (last_time - timer_label_3) < LIGHT_DELAY: light_label_3 = True
                else: light_label_3 = False

                show_data = [fifo_video_1, fifo_video_2, fifo_scanner,\
                             decodedObjects_1, decodedObjects_2, scanner_digits,\
                             light_label_1, light_label_2, light_label_3]
                # Draw final window
                frames = cv2.hconcat([frame_show_1, frame_show_2])
                if sound_3:
                    # Make screen
                    now = datetime.now()
                    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                    # qr_logger.info('current_time_for_name: ', current_time)
                    # qr_logger.info('os.listdir : ', os.listdir())
                    cv2.imwrite(f'/home/super/qr_detection/screen/screen_{current_time}.jpg', frames)
                im = cv2.resize(create_bord(frames, show_data), (1024, 768))
                cv2.namedWindow('QR_registrator', cv2.WND_PROP_FULLSCREEN)
                cv2.moveWindow('QR_registrator', 0, 0)
                cv2.setWindowProperty('QR_registrator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('QR_registrator', im)
            # Press q to quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        qr_logger.debug('End program')
    except Exception as e:
        qr_logger.error(e, exc_info=True)


if __name__ == '__main__':
    main()
