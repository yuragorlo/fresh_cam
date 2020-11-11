# import pytesseract
# img = '/home/mnemonic/work/digits/real_source_2/21.jpg'
# custom_config = r'--oem 3 --psm 6 outputbase digits'
# print(pytesseract.image_to_string(img, config=custom_config))

# read hid device python
# https://www.ontrak.net/pythonhidapi.htm
# https://pypi.org/project/hid/

# full screen opencv
# https://answers.opencv.org/question/198479/display-a-streamed-video-in-full-screen-opencv3/

import pyzbar.pyzbar as pyzbar
import numpy as np
import mysql.connector
import cv2
import time
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from contextlib import closing


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
    db_conn = mysql.connector.connect(**config)
    with closing(db_conn) as conn:
        with conn.cursor() as cursor:
            cursor.execute('''CREATE TABLE IF NOT EXISTS qr_table 
            (id INT NOT NULL AUTO_INCREMENT, 
            data VARCHAR(100) NOT NULL, 
            time TIMESTAMP NOT NULL, 
            route VARCHAR(10) NOT NULL, 
            check_flag INT NOT NULL, 
            PRIMARY KEY (id));''')
            conn.commit()


def push_to_db(qr_dict, config, qr_logger):
    db_conn = mysql.connector.connect(**config)
    with closing(db_conn) as conn:
        with conn.cursor() as cursor:
            placeholders = ', '.join(['%s'] * len(qr_dict))
            columns = ', '.join(qr_dict.keys())
            sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % ('qr_table', columns, placeholders)
            cursor.execute(sql, list(qr_dict.values()))
            conn.commit()
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
            # init dict with current data
            now = datetime.now()
            new_data = {'data': obj.data.decode("utf-8"),
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
                f'Add {obj.data.decode("utf-8")} to fifo_{camera_label}. Current fifo_{camera_label}: {fifo}')
    # clean fifo and pass only min_size_stock_last_data elements
    if len(fifo) > max_size_fifo:
        fifo = fifo[-min_size_fifo:]
        qr_logger.info(f'Free some of fifo_{camera_label}. Current fifo_{camera_label}: {fifo}')

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


def main():
    # create logger
    qr_logger = get_logger('qr_detector')
    qr_logger.debug('Start program')
    # db config. You should create qr database, but qr_table table will create automatically if it not exist
    db_config = {
        'host': "localhost",
        'user': "mysql_user",
        'password': "mysql_password",
        'database': "qr"}
    # labels will be use for 'route' row in db and naming window
    label_1 = '1'
    label_2 = '2'
    # example rtsp source "rtsp://192.168.1.2:8080/out.h264"
    source_1 = 'https://www.radiantmediaplayer.com/media/big-buck-bunny-360p.mp4'
    source_2 = 0

    try:
        # create mysql table for qr data if not exist
        create_table(db_config)
        # get streams
        cap_1 = cv2.VideoCapture(source_1)
        cap_2 = cv2.VideoCapture(source_2)
        # resize caps (example size 1024.0 x 768.0; 1280.0 x 1024.0)
        cap_1.set(3, 640)
        cap_1.set(4, 480)
        cap_2.set(3, 640)
        cap_2.set(4, 480)
        time.sleep(2)
        # init fifo lists - short memory for qr data
        fifo_1 = []
        fifo_2 = []
        # endless cycle
        while (cap_1.isOpened()) or (cap_2.isOpened()):
            # fix time
            last_time = time.time()
            # capture frame-by-frame
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            # convert to grey
            im_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
            im_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
            # detecting qr in frames
            decodedObjects_1 = pyzbar.decode(im_1)
            decodedObjects_2 = pyzbar.decode(im_2)
            if decodedObjects_1:
                # update fifo and push to db if not in fifo
                fifo_1 = process_qr(decodedObjects_1, fifo_1, label_1, db_config, qr_logger)
                # add square around qr and text with data on frame
                add_qr_to_frame(frame_1, decodedObjects_1)
            if decodedObjects_2:
                # update stock and push to db if not in fifo
                fifo_2 = process_qr(decodedObjects_2, fifo_2, label_2, db_config, qr_logger)
                # add square around qr and text with data on frame
                add_qr_to_frame(frame_2, decodedObjects_2)
            # display FPS:
            cv2.putText(frame_1, "FPS: %f" % (1.0 / (time.time() - last_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_2, "FPS: %f" % (1.0 / (time.time() - last_time)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # display the resulting frame
            cv2.imshow('1', frame_1)
            cv2.imshow('2', frame_2)
            # control keys
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):  # press 's' key to save
                cv2.imwrite('Capture.png', frame_1)
            elif key & 0xFF == ord('z'):  # press 'z' key to save
                cv2.imwrite('Capture.png', frame_2)
        # when everything done, release the capture
        cap_1.release()
        cap_2.release()
        cv2.destroyAllWindows()
        qr_logger.debug('End program')

    except Exception as e:
        qr_logger.error(e, exc_info=True)


if __name__ == '__main__':
    main()
