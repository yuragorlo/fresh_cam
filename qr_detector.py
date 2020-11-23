# simple optical character recognition (OCR)
# import pytesseract
# img = '~/work/digits/real_source_2/21.jpg'
# custom_config = r'--oem 3 --psm 6 outputbase digits'
# print(pytesseract.image_to_string(img, config=custom_config))

# read hid device python
# https://www.ontrak.net/pythonhidapi.htm
# https://pypi.org/project/hid/

# full screen opencv
# https://answers.opencv.org/question/198479/display-a-streamed-video-in-full-screen-opencv3/

import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import ZBarSymbol
import numpy as np
import mysql.connector
import cv2
import time
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from contextlib import closing
from collections import deque
from multiprocessing.pool import ThreadPool


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


def push_to_db(qr_dict, config, qr_logger):
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    placeholders = ', '.join(['%s'] * len(qr_dict))
    columns = ', '.join(qr_dict.keys())
    sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % ('qr_table', columns, placeholders)
    cursor.execute(sql, list(qr_dict.values()))
    conn.commit()
    cursor.close()
    conn.close()
    qr_logger.info(f'New qr data insert in db. Data: {qr_dict}')


# Semantic functions
def process_qr(decodedObjects, fifo, camera_label, db_config, qr_logger):
    # configure fifo
    min_size_fifo = 3
    max_size_fifo = 7
    default_check_flag = 0
    # get list of current qr objects from video stream
    current_data_list = [x.data.decode("utf-8") for x in decodedObjects]
    # push new data to db
    if not any(item in current_data_list for item in fifo):
        for obj in decodedObjects:
            entrails_data = obj.data.decode("utf-8")
            # mask for 94**********
            if len(entrails_data) == 12 and entrails_data[:2] == '94':
                # init dict with current data
                now = datetime.now()
                new_data = {'data': entrails_data,
                            'time': now.strftime("%Y/%m/%d %H:%M:%S"),
                            'route': camera_label,
                            'check_flag': default_check_flag}
                # insert new qr code info with time registration to qr_table qr database
                push_to_db(new_data, db_config, qr_logger)
    # add new data to fifo
    for obj in decodedObjects:
        if obj.data.decode("utf-8") not in fifo:
            fifo.append(obj.data.decode("utf-8"))
            qr_logger.info(
                f'Add {obj.data.decode("utf-8")} to fifo. Current fifo: {fifo}')
    # clean fifo and pass only min_size_stock_last_data elements
    if len(fifo) > max_size_fifo:
        fifo = fifo[-min_size_fifo:]
        qr_logger.info(f'Free some of fifo. Current fifo: {fifo}')

    return fifo


def add_qr_to_frame(frame, decodedObjects):
    for decodedObject in decodedObjects:
        # draw square around qr
        points = decodedObject.polygon
        # if the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
        # number of points in the convex hull
        n = len(hull)
        # draw the convex hull
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (0, 0, 255), 3)
        # show text
        x = decodedObject.rect.left
        y = decodedObject.rect.top
        barCode = str(decodedObject.data.decode("utf-8"))
        cv2.putText(frame, barCode, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def process_frame(frame, fifo, db_label, db_config, qr_logger):
    # read and transform frame to grey
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find qr code on frame
    decodedObjects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])
    if decodedObjects:
        # update fifo and push to db if not in fifo
        fifo = process_qr(decodedObjects, fifo, db_label, db_config, qr_logger)
        # add square around qr and text with data on frame
        add_qr_to_frame(frame, decodedObjects)
    return frame, fifo


def main():

    # input stream video, example 'https://www.radiantmediaplayer.com/media/big-buck-bunny-360p.mp4'
    VIDEO_SOURCE_1 = 'rtsp://user:pass@ip:port/cam/realmonitor?channel=1&subtype=1'
    VIDEO_SOURCE_2 = 'rtsp://user:pass@ip:port/cam/realmonitor?channel=1&subtype=1'

    # DB_LABEL will be use for 'route' row in db and name window
    DB_LABEL_1 = '1'
    DB_LABEL_2 = '2'
    # database configuration
    DB_CONFIG = {
        'host': "********",
        'user': "********",
        'password': "********",
        'database': "********"}

    # init logger
    qr_logger = get_logger('qr_detector')
    qr_logger.debug('Start program')

    try:
        # create mysql table for qr data if not exist
        create_table(DB_CONFIG)
        # init memory list
        fifo_1 = []
        fifo_2 = []
        # init ThreadPool
        thread_num = cv2.getNumberOfCPUs()
        pool_1 = ThreadPool(processes=thread_num)
        pool_2 = ThreadPool(processes=thread_num)
        pending_task_1 = deque()
        pending_task_2 = deque()
        # get and resize video
        cap_1 = cv2.VideoCapture(VIDEO_SOURCE_1)
        cap_1.set(3, 640)
        cap_1.set(4, 480)
        cap_2 = cv2.VideoCapture(VIDEO_SOURCE_2)
        cap_2.set(3, 640)
        cap_2.set(4, 480)

        while True:
            # fix time
            last_time = time.time()
            # populate the queue
            if len(pending_task_1) < thread_num:
                frame_got, frame = cap_1.read()
                if frame_got:
                    task = pool_1.apply_async(process_frame, (frame.copy(), fifo_1, DB_LABEL_1, DB_CONFIG, qr_logger,))
                    pending_task_1.append(task)
            if len(pending_task_2) < thread_num:
                frame_got, frame = cap_2.read()
                if frame_got:
                    task = pool_2.apply_async(process_frame, (frame.copy(), fifo_2, DB_LABEL_2, DB_CONFIG, qr_logger,))
                    pending_task_2.append(task)
            # consume the queue
            while len(pending_task_1) > 0 and pending_task_1[0].ready():
                res = pending_task_1.popleft().get()
                frame_show = res[0]
                # display FPS:
                cv2.putText(frame_show, "FPS: %f" % (1.0 / (time.time() - last_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(f'{DB_LABEL_1}_windows', frame_show)
            while len(pending_task_2) > 0 and pending_task_2[0].ready():
                res = pending_task_2.popleft().get()
                frame_show = res[0]
                # display FPS:
                cv2.putText(frame_show, "FPS: %f" % (1.0 / (time.time() - last_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(f'{DB_LABEL_2}_windows', frame_show)
            # control keys
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):  # press 's' key to save
                cv2.imwrite('Capture.png', frame)
        cv2.destroyAllWindows()
        qr_logger.debug('End program')

    except Exception as e:
        qr_logger.error(e, exc_info=True)



if __name__ == '__main__':
    main()
